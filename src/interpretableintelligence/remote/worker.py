"""The container side of ``ii run --remote``: run the ordinary pipeline, report as it goes.

Nothing here imports Modal. ``run_events`` runs ``execute_pipeline`` unchanged against a
storage root (the mounted Modal Volume in the container, a temporary directory in tests)
and yields plain dict events:

- ``{"type": "log", "line": ...}`` for every pipeline log line,
- ``{"type": "stage", "name": ..., "files": [...], "elapsed_s": ...}`` once a stage's outputs
  are on disk and committed, listing every run-folder file with its size and sha256,
- ``{"type": "done", "report": ..., "elapsed_s": ...}`` or ``{"type": "error", "message": ...}``.

The client downloads the files whose hashes changed, so the local folder fills in stage by stage.
"""

from __future__ import annotations

import hashlib
import os
import queue
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

# Resume caches. Large, rebuilt on demand, and not part of the run-folder contract.
SKIP_DIRS = frozenset({"stream"})

_END = object()
# Pipeline lines about this container's own viewer, which never starts; the client serves it.
_LOCAL_ONLY_LINES = frozenset({"viewer not started"})


def clips_path(root: Path, clips_key: str) -> Path:
    return Path(root) / "clips" / clips_key


def run_path(root: Path, run_key: str, run_name: str) -> Path:
    """Remote run folder. Its last component is the local folder's name, because that is the run_id."""
    return Path(root) / "runs" / run_key / run_name


def list_run_files(out: Path, cache: dict[str, tuple[int, int, str]] | None = None) -> list[dict]:
    """Every run-folder file except resume caches, with size and sha256.

    ``cache`` maps a path to ``(size, mtime_ns, sha256)`` so a file that did not change since the
    last stage is not hashed again.
    """
    out = Path(out)
    cache = {} if cache is None else cache
    files: list[dict] = []
    if not out.is_dir():
        return files
    for dirpath, dirnames, filenames in os.walk(out):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRS)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            rel = path.relative_to(out).as_posix()
            stat = path.stat()
            known = cache.get(rel)
            if known is not None and known[0] == stat.st_size and known[1] == stat.st_mtime_ns:
                digest = known[2]
            else:
                digest = sha256_file(path)
                cache[rel] = (stat.st_size, stat.st_mtime_ns, digest)
            files.append({"path": rel, "size": int(stat.st_size), "sha256": digest})
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_events(request: dict, root: Path, commit: Callable[[], None] | None = None) -> Iterator[dict]:
    """Run the pipeline described by ``request`` under ``root`` and yield progress events.

    ``commit`` makes the files written so far visible to the client (``Volume.commit``).
    """
    commit = commit or (lambda: None)
    started = time.monotonic()
    events: queue.Queue = queue.Queue()
    hashes: dict[str, tuple[int, int, str]] = {}
    clips_dir = clips_path(root, request["clips_key"])
    out = run_path(root, request["run_key"], request["run_name"])
    out.parent.mkdir(parents=True, exist_ok=True)

    def elapsed() -> float:
        return round(time.monotonic() - started, 3)

    def log(line: object = "") -> None:
        if str(line) in _LOCAL_ONLY_LINES:
            return
        events.put({"type": "log", "line": str(line)})

    def on_stage(name: str) -> None:
        commit()
        events.put({"type": "stage", "name": name, "files": list_run_files(out, hashes), "elapsed_s": elapsed()})

    def work() -> None:
        from interpretableintelligence.pipeline import execute_pipeline  # noqa: PLC0415 (torch import)

        concepts_file = None
        try:
            if request.get("concepts_text") is not None:
                handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
                with handle:
                    handle.write(request["concepts_text"])
                concepts_file = Path(handle.name)
            result = execute_pipeline(
                adapter_name=request["adapter"],
                clips_dir=clips_dir,
                out_dir=out,
                layer=int(request["layer"]),
                seed=int(request["seed"]),
                heldout_frac=float(request["heldout_frac"]),
                tiny=bool(request.get("tiny", False)),
                prefilter=request.get("prefilter"),
                extra_effects=tuple(request.get("extra_effects") or ()),
                gpu_cost_per_hour=request.get("gpu_cost_per_hour"),
                sae_features=request.get("sae_features"),
                concepts_file=concepts_file,
                score=bool(request.get("score", True)),
                serve=False,
                open_browser=False,
                log=log,
                on_stage=on_stage,
                manifest_clips_dir=request.get("local_clips_dir"),
            )
            commit()
            events.put({"type": "done", "report": result["report"], "elapsed_s": elapsed()})
        except Exception as exc:  # noqa: BLE001 (every failure has to reach the client as an event)
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}", "elapsed_s": elapsed()})
        finally:
            if concepts_file is not None:
                concepts_file.unlink(missing_ok=True)
            events.put(_END)

    thread = threading.Thread(target=work, name="ii-remote-pipeline", daemon=True)
    thread.start()
    yield {"type": "started", "elapsed_s": elapsed()}
    while True:
        event = events.get()
        if event is _END:
            break
        yield event
    thread.join(timeout=5)
