"""``ii run --remote``: run the pipeline on a hosted GPU and fill the local run folder as it goes.

The flow, start to finish:

1. Price the container (``pricing.rate_for``) and turn ``--max-cost`` into a hard timeout, so the
   cap holds on the provider's side even if this machine disconnects. Both are printed first.
2. Upload the clips once, keyed by content.
3. Start the worker. Each finished stage arrives as a file list; files whose sha256 differs from
   the local copy are downloaded, checked and moved into place.
4. As soon as the folder has what the viewer needs (after ``features``), serve it locally. Later
   stages bump the ``generation`` in the viewer's ``--live`` status file, the viewer loads the
   folder again, and the page refreshes itself (``view_launch``).
5. Write ``remote.json`` (what ran, where, for how long, and the estimated cost) next to the run.

The local folder is a mirror of the remote one. Run outputs are the pipeline's own, unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from interpretableintelligence.remote.backends import ModalBackend, clip_files
from interpretableintelligence.remote.pricing import Rate, budget_timeout, format_duration, format_usd, rate_for
from interpretableintelligence.remote.worker import run_path, sha256_file

REMOTE_RECORD = "remote.json"
_PREFIX = "  remote | "


def clips_key(clips_dir: Path) -> str:
    """Content key for a clip directory: the same files upload once."""
    clips_dir = Path(clips_dir)
    digest = hashlib.sha256()
    for path in clip_files(clips_dir):
        digest.update(path.relative_to(clips_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:24]


def default_run_key(out: Path) -> str:
    return hashlib.sha256(str(Path(out).resolve()).encode("utf-8")).hexdigest()[:16]


def check_portable_adapter(adapter_name: str) -> None:
    """A remote container cannot read a model from a directory on this machine."""
    if ":" not in adapter_name:
        return
    model = adapter_name.split(":", 1)[1].split("#", 1)[0]
    if model.startswith(("/", "~", ".")) or Path(model).expanduser().exists():
        raise ValueError(
            f"--remote needs a Hugging Face Hub model id, not a local directory ({model}); "
            "push the weights to the Hub (a private repo works with HF_TOKEN) or run locally"
        )


def _read_record(out: Path) -> dict | None:
    path = Path(out) / REMOTE_RECORD
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def check_out(out: Path) -> dict | None:
    """``--out`` must be new, empty, or a folder an earlier remote run filled. Returns its record."""
    out = Path(out)
    if out.exists() and not out.is_dir():
        raise ValueError(f"--out {out} is a file")
    record = _read_record(out)
    if out.is_dir() and record is None and any(out.iterdir()):
        raise ValueError(
            f"{out} already holds a local run. --remote keeps its own copy of the run and mirrors it "
            "here, so pick a new --out (or an earlier --remote run's folder to resume it)"
        )
    return record


def _safe_rel(rel: str) -> PurePosixPath:
    path = PurePosixPath(rel)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"remote listed an unsafe path: {rel!r}")
    return path


class _Mirror:
    """Download changed files into ``out`` and drop files an earlier sync wrote that are gone remotely.

    The pipeline goes straight on to the next stage, which may rewrite or delete a file the last
    stage listed before it is fetched. A per-stage sync skips such a file (the next listing has
    its new version). The final listing is taken after the pipeline's last write, and that sync
    is ``strict``: every file has to arrive with the listed hash.
    """

    def __init__(self, out: Path, prefix: str, synced: list[str]) -> None:
        self.out = Path(out)
        self.prefix = prefix
        self.synced = set(synced)
        self._hashes: dict[str, tuple[int, int, str]] = {}

    def _local_sha(self, rel: str, path: Path) -> str | None:
        if not path.is_file():
            return None
        stat = path.stat()
        known = self._hashes.get(rel)
        if known is not None and known[:2] == (stat.st_size, stat.st_mtime_ns):
            return known[2]
        digest = sha256_file(path)
        self._hashes[rel] = (stat.st_size, stat.st_mtime_ns, digest)
        return digest

    def apply(self, session, files: list[dict], strict: bool = False) -> tuple[int, int]:
        changed = 0
        nbytes = 0
        listed: set[str] = set()
        for item in files:
            rel = _safe_rel(str(item["path"]))
            key = rel.as_posix()
            listed.add(key)
            dest = self.out.joinpath(*rel.parts)
            if self._local_sha(key, dest) == item["sha256"]:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            try:
                session.fetch(f"{self.prefix}/{key}", part)
                got = sha256_file(part)
                if got != item["sha256"]:
                    raise RuntimeError(f"{key} arrived with sha256 {got[:12]}, expected {item['sha256'][:12]}")
                os.replace(part, dest)
            except Exception:
                if strict:
                    raise
                listed.discard(key)  # moved on remotely; the next listing has it
                continue
            finally:
                part.unlink(missing_ok=True)
            changed += 1
            nbytes += int(item["size"])
        for key in sorted(self.synced - listed):
            path = self.out.joinpath(*_safe_rel(key).parts)
            if path.is_file():
                path.unlink()
                changed += 1
            self._hashes.pop(key, None)
        self.synced = listed
        return changed, nbytes


class _Viewer:
    """The viewer on a run that is still filling in: status and a reload ``generation`` go to the
    ``--live`` file the Rust binary watches (``view_launch``)."""

    def __init__(self, out: Path, clips_dir: Path, live: dict, host: str, port: int, open_browser: bool, log) -> None:
        from interpretableintelligence.view_launch import open_in_browser, start_viewer  # noqa: PLC0415

        self.viewer = start_viewer(out, clips_dir, host=host, port=int(port), live=live, log=log)
        self.url = self.viewer.url
        log(f"viewer {self.url}  (updates live as the remote run finishes stages)")
        if open_browser:
            open_in_browser(self.url)

    def reload(self, log) -> None:
        try:
            self.viewer.reload()
        except Exception as exc:  # noqa: BLE001 (a half-synced folder; the next stage fixes it)
            log(f"remote: viewer kept the previous state ({type(exc).__name__}: {exc})")

    def update(self, **fields) -> None:
        self.viewer.update(**fields)

    def wait(self) -> None:
        self.viewer.wait()

    def close(self) -> None:
        self.viewer.close()


def _bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds")


def execute_remote(
    *,
    adapter_name: str,
    clips_dir: Path,
    out_dir: Path,
    layer: int,
    seed: int,
    heldout_frac: float,
    tiny: bool = False,
    prefilter: int | None = None,
    extra_effects: tuple[str, ...] = (),
    gpu_cost_per_hour: float | None = None,
    sae_features: int | None = None,
    concepts_file: Path | None = None,
    score: bool = True,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = False,
    gpu: str = "L4",
    max_cost: float = 2.0,
    use_cache: bool = True,
    backend=None,
    wait: bool = True,
    log=print,
) -> dict:
    """Run the pipeline remotely, mirror the run folder into ``out_dir``, serve it live.

    With ``serve`` and ``wait`` (the CLI), block on the viewer until Ctrl-C. With ``wait=False``
    the viewer keeps running in the background and comes back as ``result["viewer"]`` (it has
    ``url`` and ``close()``), for a notebook or a test.
    """
    if not 0.0 < float(heldout_frac) < 1.0:
        raise ValueError(f"heldout_frac must be in (0, 1), got {heldout_frac}")
    check_portable_adapter(adapter_name)
    clips_dir = Path(clips_dir).resolve()
    out = Path(out_dir)
    previous = check_out(out)
    rate = rate_for(gpu, gpu_cost_per_hour)
    timeout_s = budget_timeout(max_cost, rate)
    if backend is None:
        backend = ModalBackend(rate.gpu, timeout_s, groot=adapter_name.startswith("groot:"))
    backend.check()

    log(f"remote: {backend.name} {rate.describe()}")
    log(
        f"remote: spending cap {format_usd(max_cost)} (--max-cost) -> the container is stopped after "
        f"{format_duration(timeout_s)}. A first run also builds the image, which is not counted here."
    )

    files = clip_files(clips_dir)
    key = clips_key(clips_dir)
    if backend.has_clips(key):
        log(f"remote: clips already uploaded ({len(files)} files)")
    else:
        total = sum(path.stat().st_size for path in files)
        log(f"remote: uploading {len(files)} clip files ({_bytes(total)})")
        backend.upload_clips(clips_dir, key)

    run_key = str(previous.get("run_key")) if previous and previous.get("run_key") else default_run_key(out)
    out.mkdir(parents=True, exist_ok=True)
    request = {
        "adapter": adapter_name,
        "clips_key": key,
        "run_key": run_key,
        "run_name": out.resolve().name,
        "layer": int(layer),
        "seed": int(seed),
        "heldout_frac": float(heldout_frac),
        "tiny": bool(tiny),
        "prefilter": prefilter,
        "extra_effects": list(extra_effects),
        # The pipeline's own cost report prices an hour of footage at the remote rate.
        "gpu_cost_per_hour": rate.usd_per_hour,
        "sae_features": sae_features,
        "concepts_text": None if concepts_file is None else Path(concepts_file).read_text(encoding="utf-8"),
        "score": bool(score),
        "use_cache": bool(use_cache),
        "local_clips_dir": str(clips_dir),
    }
    prefix = run_path(Path(""), run_key, request["run_name"]).as_posix()
    mirror = _Mirror(out, prefix, list((previous or {}).get("files") or []))

    started = time.time()
    clock = time.monotonic()
    live = {
        "provider": backend.name,
        "gpu": rate.gpu,
        "usd_per_hour": rate.usd_per_hour,
        "max_cost_usd": float(max_cost),
        "started_at": started,
        "state": "running",
        "stage": None,
    }
    record = {
        "provider": backend.name,
        "gpu": rate.gpu,
        "usd_per_hour": rate.usd_per_hour,
        "price_source": rate.source,
        "max_cost_usd": float(max_cost),
        "timeout_s": timeout_s,
        "run_key": run_key,
        "clips_key": key,
        "started_at": _iso(started),
    }
    state = {"status": "running", "message": None, "container_s": None, "report": None}
    last_files: list[dict] = []
    viewer: _Viewer | None = None

    def write_record() -> None:
        wall = time.monotonic() - clock
        container = state["container_s"]
        payload = {
            **record,
            "status": state["status"],
            "message": state["message"],
            "wall_s": round(wall, 3),
            "container_s": container,
            # Time inside the GPU function, at the container rate: the estimate.
            "estimated_cost_usd": None if container is None else round(rate.cost(container), 6),
            # Wall time since launch, at the same rate: covers cold start, an upper bound.
            "upper_bound_cost_usd": round(rate.cost(wall), 6),
            "files": sorted(mirror.synced),
        }
        if state["status"] != "running":
            payload["finished_at"] = _iso(time.time())
        (out / REMOTE_RECORD).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        with backend.session() as session:
            for event in session.events(request):
                kind = event.get("type")
                if kind == "log":
                    log(f"{_PREFIX}{event.get('line', '')}")
                elif kind == "stage":
                    name = str(event["name"])
                    last_files = list(event["files"])
                    changed, nbytes = mirror.apply(session, last_files)
                    state["container_s"] = float(event.get("elapsed_s") or 0.0)
                    spent = rate.cost(time.monotonic() - clock)
                    log(
                        f"remote: {name} ready, pulled {changed} files ({_bytes(nbytes)}); "
                        f"~{format_usd(spent)} so far of {format_usd(max_cost)}"
                    )
                    write_record()
                    if serve:
                        if viewer is None:
                            from interpretableintelligence.view import viewer_ready  # noqa: PLC0415

                            if viewer_ready(out):
                                try:
                                    viewer = _Viewer(out, clips_dir, {**live, "stage": name}, host, port, open_browser, log)
                                except OSError as exc:
                                    # The run is paid for; finish it and let `ii view` show it.
                                    log(f"remote: viewer could not start ({exc}); the run continues")
                                    serve = False
                        elif changed:
                            viewer.reload(log)
                        if viewer is not None:
                            viewer.update(stage=name)
                elif kind == "done":
                    changed, _nbytes = mirror.apply(session, last_files, strict=True)
                    if viewer is not None and changed:
                        viewer.reload(log)
                    state["status"] = "done"
                    state["report"] = event.get("report")
                    state["container_s"] = float(event.get("elapsed_s") or 0.0)
                elif kind == "error":
                    state["status"] = "failed"
                    state["message"] = str(event.get("message"))
                    state["container_s"] = float(event.get("elapsed_s") or 0.0)
        if state["status"] == "running":
            state["status"] = "failed"
            state["message"] = "the remote run ended without reporting a result"
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["message"] = "stopped from this machine (Ctrl-C); the remote container was stopped too"
        raise
    except Exception as exc:  # noqa: BLE001 (provider errors: timeouts, auth, network)
        timed_out = type(exc).__name__ == "FunctionTimeoutError"  # modal.exception, by name: no import
        state["status"] = "stopped at the --max-cost cap" if timed_out else "failed"
        state["message"] = (
            f"the container reached its {format_duration(timeout_s)} timeout, which is the {format_usd(max_cost)} cap"
            if timed_out
            else f"{type(exc).__name__}: {exc}"
        )
    finally:
        write_record()
        _final_cost_line(rate, state, time.monotonic() - clock, max_cost, log)
        if viewer is not None:
            viewer.update(state=state["status"], finished_at=time.time())

    if state["status"] != "done":
        if viewer is not None:
            viewer.close()
        kept = f"; stages that finished are in {out}" if mirror.synced else ""
        raise RuntimeError(f"remote run {state['status']}: {state['message']}{kept}")
    if viewer is None and serve:
        log("viewer not started: the run folder is missing features/feature_acts.parquet")
    elif viewer is not None and wait:
        viewer.wait()
        viewer = None
    return {"report": state["report"], "remote": _read_record(out), "out": out, "viewer": viewer}


def _final_cost_line(rate: Rate, state: dict, wall_s: float, max_cost: float, log) -> None:
    container = state["container_s"]
    parts = [f"remote: {state['status']}"]
    if container is not None:
        parts.append(f"GPU container {format_duration(container)} ~{format_usd(rate.cost(container))}")
    parts.append(f"wall {format_duration(wall_s)} (at most ~{format_usd(rate.cost(wall_s))} with startup)")
    parts.append(f"cap {format_usd(max_cost)}")
    log(", ".join(parts) + ". The provider's usage page has the exact bill.")
