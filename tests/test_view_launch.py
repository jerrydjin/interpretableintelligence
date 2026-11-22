"""``ii view``, ``ii try``, ``ii run`` and ``ii run --remote`` start the Rust viewer through one launcher.

The binary serves the run; ``--steer`` goes through a small Python steering service behind
``--steer-url``; a live run's status and reload ``generation`` go through the ``--live`` file. The
Python server is used only when no binary is found, when the run's clips are video files (the binary
does not decode video frames yet), or with ``II_VIEWER=python``.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import pytest

from interpretableintelligence import server_bin, view_launch
from interpretableintelligence.artifacts import read_json, write_json
from interpretableintelligence.steer import Steerer
from interpretableintelligence.view_launch import ViewerError, start_viewer
from test_contract import _server_bin
from test_view import _tiny_run

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(url: str) -> dict:
    with _NO_PROXY.open(url, timeout=30) as response:
        return json.loads(response.read())


def _post(url: str, body: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with _NO_PROXY.open(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def binary(monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _server_bin()
    monkeypatch.setenv("II_SERVER", str(path))
    monkeypatch.delenv("II_VIEWER", raising=False)
    return path


def test_the_binary_serves_the_run_and_steering_goes_through_the_service(demo, binary) -> None:
    steerer = Steerer(demo.run, demo.clips)
    viewer = start_viewer(demo.run, demo.clips, port=0, steerer=steerer)
    try:
        assert viewer.backend == "rust"
        assert viewer.proc.args[0] == str(binary)
        payload = _get(viewer.url + "api/run")
        assert payload["run_id"] == "demo"
        assert payload["steer"] == {"enabled": True, "sae_usable": steerer.sae_usable}
        clip = payload["clips"][0]["clip_id"]
        feature = payload["features"][0]["id"]
        request = {"clip_id": clip, "window": 0, "features": [feature], "kind": "ablate"}
        status, reply = _post(viewer.url + "api/steer", json.dumps(request).encode())
        assert status == 200, reply
        assert reply == steerer.steer(request)
        status, reply = _post(viewer.url + "api/steer", b"not json")
        assert status == 400 and reply["error"] == "request body is not valid JSON"
        status, reply = _post(viewer.url + "api/steer", json.dumps({**request, "kind": "melt"}).encode())
        assert status == 400 and reply["error"].startswith("kind must be one of")
    finally:
        viewer.close()
    assert viewer.proc.poll() is not None


def test_without_steer_the_binary_refuses_steering(demo, binary) -> None:
    viewer = start_viewer(demo.run, demo.clips, port=0)
    try:
        assert "steer" not in _get(viewer.url + "api/run") or _get(viewer.url + "api/run")["steer"] is None
        status, reply = _post(viewer.url + "api/steer", b"{}")
        assert status == 404 and "--steer" in reply["error"]
    finally:
        viewer.close()


def test_a_live_run_reloads_when_its_generation_moves_on(demo, binary, tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(demo.run, run)
    viewer = start_viewer(run, demo.clips, port=0, live={"provider": "modal", "state": "running"})
    try:
        assert viewer.backend == "rust"
        assert _get(viewer.url + "api/run")["live"] == {"provider": "modal", "state": "running", "generation": 0}
        # A stage lands: the run id changes on disk, but the binary keeps the loaded run until reload.
        manifest = read_json(run / "manifest.json")
        write_json(run / "manifest.json", {**manifest, "run_id": "demo-later"})
        viewer.update(stage="score")
        payload = _get(viewer.url + "api/run")
        assert payload["live"]["stage"] == "score" and payload["live"]["generation"] == 0
        assert payload["run_id"] == "demo"
        viewer.reload()
        payload = _get(viewer.url + "api/run")
        assert payload["live"]["generation"] == 1
        assert payload["run_id"] == "demo-later"
    finally:
        viewer.close()


def test_video_runs_use_the_python_server(binary, tmp_path: Path) -> None:
    run, clips = _tiny_run(tmp_path)
    (clips / "clip-000.npy").unlink()
    (clips / "clip-000.mp4").write_bytes(b"not decoded here")
    manifest = read_json(run / "manifest.json")
    manifest["clips"][0].pop("sha256")
    write_json(run / "manifest.json", manifest)
    lines: list[str] = []
    viewer = start_viewer(run, clips, port=0, log=lines.append)
    try:
        assert viewer.backend == "python"
        assert any("video files" in line for line in lines), lines
        assert _get(viewer.url + "api/run")["clips"][0]["media"] == "video"
    finally:
        viewer.close()


def test_no_binary_or_ii_viewer_python_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run, clips = _tiny_run(tmp_path)

    def missing(**_kwargs):
        raise server_bin.ServerNotFound("no prebuilt viewer binary for plan9")

    monkeypatch.delenv("II_VIEWER", raising=False)
    monkeypatch.setattr(server_bin, "find_server", missing)
    lines: list[str] = []
    viewer = start_viewer(run, clips, port=0, log=lines.append)
    try:
        assert viewer.backend == "python"
        assert "plan9" in lines[0]
        assert _get(viewer.url + "api/run")["run_id"] == "tiny"
    finally:
        viewer.close()
    monkeypatch.setenv("II_VIEWER", "python")
    lines.clear()
    viewer = start_viewer(run, clips, port=0, log=lines.append)
    viewer.close()
    assert viewer.backend == "python" and "II_VIEWER=python" in lines[0]


def test_a_bad_run_folder_is_an_error_not_a_hang(binary, tmp_path: Path) -> None:
    (tmp_path / "real-run").mkdir()
    with pytest.raises(ViewerError, match="run folder not found"):
        start_viewer(tmp_path / "typo", None, port=0)


def test_live_status_is_written_whole(tmp_path: Path) -> None:
    status = view_launch.LiveStatus(tmp_path / "live.json", {"state": "running", "generation": 9})
    assert json.loads(status.path.read_text()) == {"state": "running", "generation": 0}
    status.bump()
    status.update(state="done", generation=5)
    assert json.loads(status.path.read_text()) == {"state": "done", "generation": 1}
    assert [path.name for path in tmp_path.iterdir()] == ["live.json"]
