"""``ii run --remote``: pricing, the mirror, the live viewer, and a full run on the in-process backend.

Nothing here talks to Modal. ``LocalBackend`` runs the same worker the Modal function runs, so a
remote run here is the real pipeline plus the real sync, minus the network.
"""

from __future__ import annotations

import json
import tomllib
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.remote import pricing
from interpretableintelligence.remote.backends import LocalBackend
from interpretableintelligence.remote.client import (
    REMOTE_RECORD,
    _Mirror,
    check_out,
    check_portable_adapter,
    clips_key,
    execute_remote,
)
from interpretableintelligence.remote.image import RUNTIME_DEPENDENCIES
from interpretableintelligence.remote.worker import list_run_files, sha256_file
from interpretableintelligence.view import LiveRun

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _remote(clips: Path, out: Path, backend, **extra) -> dict:
    lines: list[str] = []
    options = {
        "adapter_name": "fake",
        "clips_dir": clips,
        "out_dir": out,
        "layer": 16,
        "seed": 0,
        "heldout_frac": 0.2,
        "serve": False,
        "backend": backend,
        "log": lines.append,
    }
    options.update(extra)
    result = execute_remote(**options)
    result["lines"] = lines
    return result


@pytest.fixture(scope="module")
def remote_run(demo, tmp_path_factory):
    root = tmp_path_factory.mktemp("remote")
    backend = LocalBackend(root / "volume")
    out = root / "runs" / "demo"
    return _remote(demo.clips, out, backend), backend, out


# --- pricing ----------------------------------------------------------------------------------


def test_rate_is_gpu_list_price_plus_the_cpu_and_memory_reservation():
    rate = pricing.rate_for("l4")
    gpu = pricing.GPU_USD_PER_S["L4"] * 3600
    rest = (pricing.CPU_CORES * pricing.CPU_CORE_USD_PER_S + pricing.MEMORY_GIB * pricing.MEMORY_GIB_USD_PER_S) * 3600
    assert rate.gpu == "L4"
    assert rate.gpu_usd_per_hour == pytest.approx(gpu)
    assert rate.usd_per_hour == pytest.approx(gpu + rest)
    assert rate.cost(1800) == pytest.approx((gpu + rest) / 2)
    assert pricing.rate_for("A10G").gpu == "A10"
    with pytest.raises(ValueError, match="unknown --gpu"):
        pricing.rate_for("TPU")


def test_gpu_cost_per_hour_replaces_the_list_price():
    rate = pricing.rate_for("H100", 7.5)
    assert rate.usd_per_hour == 7.5
    assert rate.source == "--gpu-cost-per-hour"
    with pytest.raises(ValueError):
        pricing.rate_for("H100", -1)


def test_max_cost_becomes_a_timeout_at_the_rate():
    rate = pricing.Rate("L4", 1.0, 1.0, "test")
    assert pricing.budget_timeout(0.5, rate) == 1800
    assert pricing.budget_timeout(10_000, rate) == pricing.MAX_TIMEOUT_S
    with pytest.raises(ValueError, match="at least"):
        pricing.budget_timeout(0.001, rate)
    with pytest.raises(ValueError):
        pricing.budget_timeout(0, rate)


def test_remote_image_installs_the_project_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert list(RUNTIME_DEPENDENCIES) == project["dependencies"]


# --- guards -----------------------------------------------------------------------------------


def test_remote_refuses_a_folder_holding_a_local_run(demo):
    with pytest.raises(ValueError, match="already holds a local run"):
        check_out(demo.run)


def test_remote_refuses_a_model_on_this_disk(tmp_path):
    check_portable_adapter("hf:facebook/dinov2-small")
    check_portable_adapter("fake")
    with pytest.raises(ValueError, match="Hub model id"):
        check_portable_adapter(f"hf:{tmp_path}")


def test_gpu_and_max_cost_need_remote(demo, tmp_path):
    result = runner.invoke(app, ["run", "fake", "--clips", str(demo.clips), "--out", str(tmp_path / "x"), "--gpu", "T4"])
    assert result.exit_code == 1
    assert "only apply with --remote" in result.output


def test_remote_without_modal_says_how_to_install(demo, tmp_path):
    try:
        import modal  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("modal is installed")
    result = runner.invoke(
        app, ["run", "fake", "--clips", str(demo.clips), "--out", str(tmp_path / "x"), "--remote", "--no-serve"]
    )
    assert result.exit_code == 2
    assert "uv sync --extra remote" in result.output
    assert not (tmp_path / "x").exists()


# --- a full run on the in-process backend --------------------------------------------------------


def test_remote_run_mirrors_a_valid_run_folder(remote_run, demo):
    result, _backend, out = remote_run
    checked = runner.invoke(app, ["validate", str(out)])
    assert checked.exit_code == 0, checked.output
    local = json.loads((demo.run / "scoreboard.json").read_text(encoding="utf-8"))
    mirrored = json.loads((out / "scoreboard.json").read_text(encoding="utf-8"))
    assert mirrored == local
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "demo"
    # The manifest points at the clips on this machine, not the container's copy.
    assert manifest["clips_dir"] == str(demo.clips.resolve())
    assert not (out / "acts" / "stream").exists()


def test_remote_run_reports_its_cost(remote_run):
    result, _backend, out = remote_run
    record = json.loads((out / REMOTE_RECORD).read_text(encoding="utf-8"))
    assert record["status"] == "done"
    assert record["gpu"] == pricing.DEFAULT_GPU
    assert record["usd_per_hour"] == pytest.approx(pricing.rate_for(pricing.DEFAULT_GPU).usd_per_hour)
    assert record["estimated_cost_usd"] == pytest.approx(record["container_s"] * record["usd_per_hour"] / 3600, rel=1e-3)
    assert record["upper_bound_cost_usd"] >= record["estimated_cost_usd"]
    assert "scoreboard.json" in record["files"]
    lines = result["lines"]
    assert lines[0].startswith("remote: local L4 at $")
    assert "spending cap $2.00" in lines[1]
    assert any(line.startswith("remote: score ready") and "so far of $2.00" in line for line in lines)
    assert any(line.startswith("remote: done, GPU container") for line in lines)
    # The pipeline's own report prices footage at the remote rate.
    assert result["report"]["gpu_cost_per_hour"] == pytest.approx(record["usd_per_hour"])


def test_second_remote_run_reuses_clips_and_resumes(remote_run, demo):
    _result, backend, out = remote_run
    again = _remote(demo.clips, out, backend)
    assert backend.uploads == 1
    assert any(line.startswith("remote: clips already uploaded") for line in again["lines"])
    for stage in ("capture", "sae", "features", "score"):
        assert f"  remote | skip {stage}" in again["lines"]
    assert any(line.startswith("remote: capture ready, pulled 0 files") for line in again["lines"])


def test_clips_key_follows_content(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "clip.npy").write_bytes(b"one")
    first = clips_key(tmp_path / "a")
    (tmp_path / "a" / ".DS_Store").write_bytes(b"ignored")
    assert clips_key(tmp_path / "a") == first
    (tmp_path / "a" / "clip.npy").write_bytes(b"two")
    assert clips_key(tmp_path / "a") != first


# --- the mirror -------------------------------------------------------------------------------


class _DirSession:
    def __init__(self, root: Path, missing: set[str] = frozenset()) -> None:
        self.root = root
        self.missing = missing
        self.fetched: list[str] = []

    def fetch(self, path: str, dest: Path) -> None:
        if path in self.missing:
            raise FileNotFoundError(path)
        self.fetched.append(path)
        dest.write_bytes((self.root / path).read_bytes())


def test_mirror_skips_a_file_that_moved_on_until_the_final_sync(tmp_path):
    remote = tmp_path / "remote"
    (remote / "run" / "features").mkdir(parents=True)
    (remote / "run" / "manifest.json").write_text("{}", encoding="utf-8")
    (remote / "run" / "features" / "candidates.json").write_text("[]", encoding="utf-8")
    files = list_run_files(remote / "run")
    out = tmp_path / "out"
    mirror = _Mirror(out, "run", [])
    changed, _ = mirror.apply(_DirSession(remote, {"run/features/candidates.json"}), files)
    assert changed == 1
    assert (out / "manifest.json").is_file()
    assert not (out / "features" / "candidates.json").exists()
    with pytest.raises(FileNotFoundError):
        mirror.apply(_DirSession(remote, {"run/features/candidates.json"}), files, strict=True)
    session = _DirSession(remote)
    mirror.apply(session, files, strict=True)
    assert session.fetched == ["run/features/candidates.json"]  # the manifest was already current
    assert sha256_file(out / "features" / "candidates.json") == sha256_file(remote / "run" / "features" / "candidates.json")


def test_mirror_deletes_only_files_it_synced(tmp_path):
    remote = tmp_path / "remote"
    (remote / "run").mkdir(parents=True)
    (remote / "run" / "scoreboard.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    mirror = _Mirror(out, "run", [])
    mirror.apply(_DirSession(remote), list_run_files(remote / "run"))
    (out / "rollouts").mkdir()
    (out / "rollouts" / "summary.json").write_text("{}", encoding="utf-8")
    (remote / "run" / "scoreboard.json").unlink()
    mirror.apply(_DirSession(remote), list_run_files(remote / "run"))
    assert not (out / "scoreboard.json").exists()
    assert (out / "rollouts" / "summary.json").is_file()


def test_mirror_refuses_paths_outside_the_run(tmp_path):
    mirror = _Mirror(tmp_path / "out", "run", [])
    with pytest.raises(RuntimeError, match="unsafe path"):
        mirror.apply(_DirSession(tmp_path), [{"path": "../escape", "size": 1, "sha256": "0"}])


# --- failures ---------------------------------------------------------------------------------


class _ScriptedBackend(LocalBackend):
    def __init__(self, root: Path, events=None, raises: Exception | None = None) -> None:
        super().__init__(root)
        self.script = events or []
        self.raises = raises

    @contextmanager
    def session(self):
        backend = self

        class Session:
            def events(self, request):
                yield from backend.script
                if backend.raises is not None:
                    raise backend.raises

            def fetch(self, path, dest):
                raise AssertionError("nothing to fetch")

        yield Session()


def test_remote_error_is_reported_and_recorded(demo, tmp_path):
    backend = _ScriptedBackend(tmp_path / "volume", [{"type": "error", "message": "CUDA out of memory", "elapsed_s": 4.0}])
    with pytest.raises(RuntimeError, match="remote run failed: CUDA out of memory"):
        _remote(demo.clips, tmp_path / "run", backend)
    record = json.loads((tmp_path / "run" / REMOTE_RECORD).read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["container_s"] == 4.0


class FunctionTimeoutError(Exception):
    """Stands in for modal.exception.FunctionTimeoutError."""


def test_hitting_the_timeout_is_reported_as_the_cost_cap(demo, tmp_path):
    backend = _ScriptedBackend(tmp_path / "volume", raises=FunctionTimeoutError("timed out"))
    with pytest.raises(RuntimeError, match=r"stopped at the --max-cost cap.*\$1\.00 cap"):
        _remote(demo.clips, tmp_path / "run", backend, max_cost=1.0)


# --- the live viewer --------------------------------------------------------------------------


def test_live_run_reloads_and_reports_status(demo):
    live = LiveRun(demo.run, demo.clips, {"provider": "modal", "gpu": "L4", "state": "running"})
    payload = live.run_payload
    assert payload["run_id"] == "demo"
    assert payload["live"] == {"provider": "modal", "gpu": "L4", "state": "running", "generation": 0}
    assert live.feature(0) is not None  # everything else is the loaded run's
    live.reload()
    live.update(state="done")
    assert live.run_payload["live"]["generation"] == 1
    assert live.run_payload["live"]["state"] == "done"


def test_remote_run_serves_the_viewer_live(demo, tmp_path):
    result = _remote(demo.clips, tmp_path / "live", LocalBackend(tmp_path / "volume"), serve=True, wait=False)
    viewer = result["viewer"]
    try:
        with urllib.request.urlopen(viewer.url + "api/run", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["run_id"] == "live"
        assert payload["live"]["state"] == "done"
        assert payload["live"]["stage"] == "view"
        # Opened after features, reloaded when score and view landed.
        assert payload["live"]["generation"] >= 1
        assert payload["scoreboard"] is not False
        assert any(line.startswith("viewer http://") for line in result["lines"])
        index = urllib.request.urlopen(viewer.url, timeout=5).read().decode("utf-8")
        assert 'data-testid="live-status"' in index
    finally:
        viewer.close()
