"""The Rust read API (``server/``) against pinned payloads and the run folder on disk.

``tests/fixtures/read-api/`` is a small run whose every response is pinned in ``expected.json``
(floats to 1e-9, key order ignored). The fake-run case checks a fresh ``ii run fake`` folder
against its own scoreboard, manifest and clip files. Python serves no read routes: ``ii serve``
is only ``POST /explain``.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "read-api"
STATIC = ROOT / "src" / "interpretableintelligence" / "static"
FLOAT_TOLERANCE = 1e-9

def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_until(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code < 500:
                return
            last = f"{response.status_code} {response.text[:200]}"
        except httpx.HTTPError as exc:
            last = str(exc)
        time.sleep(0.05)
    raise AssertionError(f"server at {url} did not come up: {last}")


class _RustServer:
    def __init__(self, runs: Path, clips: Path | None) -> None:
        self.port = _free_port()
        command = [
            str(_server_bin()),
            "serve",
            "--runs",
            str(runs),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        if clips is not None:
            command.extend(["--clips", str(clips)])
        self.proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def __enter__(self) -> str:
        url = f"http://127.0.0.1:{self.port}"
        try:
            _wait_until(url + "/runs")
        except Exception:
            self._stop()
            raise
        return url

    def __exit__(self, *exc) -> None:
        self._stop()

    def _stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _server_bin() -> Path:
    override = os.environ.get("II_SERVER")
    if override:
        return Path(override)
    debug = ROOT / "target" / "debug" / "interpretableintelligence-server"
    release = ROOT / "target" / "release" / "interpretableintelligence-server"
    if debug.is_file():
        return debug
    if release.is_file():
        return release
    if shutil.which("cargo") is None:
        # The Python install (uv sync) needs no Rust; only this Rust/Python parity check does.
        pytest.skip("cargo is not installed (https://rustup.rs); the Rust read-API tests need it")
    subprocess.check_call(["cargo", "build", "-p", "interpretableintelligence-server"], cwd=ROOT)
    return debug


def _canon(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return ("num", float(value))
    if isinstance(value, list):
        return [_canon(item) for item in value]
    if isinstance(value, dict):
        return {key: _canon(value[key]) for key in sorted(value)}
    raise AssertionError(f"unexpected JSON value {value!r}")


def _assert_json_equal(left, right, path: str = "$") -> None:
    left_c = _canon(left)
    right_c = _canon(right)
    if isinstance(left_c, tuple) and isinstance(right_c, tuple):
        if abs(left_c[1] - right_c[1]) > FLOAT_TOLERANCE:
            raise AssertionError(f"{path}: {left_c[1]} != {right_c[1]} (tol {FLOAT_TOLERANCE})")
        return
    if isinstance(left_c, dict) and isinstance(right_c, dict):
        if left_c.keys() != right_c.keys():
            missing = sorted(set(left_c) ^ set(right_c))
            raise AssertionError(f"{path}: key mismatch {missing}")
        for key in left_c:
            _assert_json_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left_c, list) and isinstance(right_c, list):
        if len(left_c) != len(right_c):
            raise AssertionError(f"{path}: length {len(left_c)} != {len(right_c)}")
        for index, (item_l, item_r) in enumerate(zip(left, right, strict=True)):
            _assert_json_equal(item_l, item_r, f"{path}[{index}]")
        return
    if left_c != right_c:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def _get_json(client: httpx.Client, url: str) -> tuple[int, object]:
    response = client.get(url)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def test_rust_read_api_matches_pinned_payloads(tmp_path: Path) -> None:
    root = tmp_path / "read-api"
    shutil.copytree(FIXTURE / "runs", root / "runs")
    shutil.copytree(FIXTURE / "clips", root / "clips")
    expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    clip = (root / "clips" / "clip-000.npy").read_bytes()
    with _RustServer(root / "runs", root / "clips") as url, httpx.Client(timeout=60.0) as client:
        for path, want in expected.items():
            status, body = _get_json(client, url + path)
            assert status == want["status"], (path, status, body)
            _assert_json_equal(body, want["body"], path)
        whole = client.get(url + "/clips/clip-000/file?run_id=mini")
        assert whole.status_code == 200
        assert whole.content == clip
        part = client.get(url + "/clips/clip-000/file?run_id=mini", headers={"Range": "bytes=0-15"})
        assert part.status_code == 206, part.text[:200]
        assert part.content == clip[:16]


def test_rust_read_api_on_a_fake_run(demo_run: Path) -> None:
    scoreboard = json.loads((demo_run / "scoreboard.json").read_text())
    manifest = json.loads((demo_run / "manifest.json").read_text())
    real = [feature for feature in scoreboard["features"] if feature["real"]]
    assert {feature["label"] for feature in real} == {"stripe", "pulse", "corner"}
    clips = demo_run.parent.parent / "clips"
    run_id = demo_run.name
    trace = {"run_id": run_id, "model_sha": scoreboard["run"]["model_sha"], "layer": scoreboard["run"]["layer"]}
    clip_ids = {str(clip["clip_id"]) for clip in manifest["clips"]}
    with _RustServer(demo_run.parent, clips) as url, httpx.Client(timeout=60.0) as client:
        runs = client.get(url + "/runs").json()["runs"]
        assert [row["run_id"] for row in runs] == [run_id]
        assert runs[0]["usable"] is True
        assert runs[0]["model_sha"] == trace["model_sha"]

        detail = client.get(url + f"/runs/{run_id}").json()
        assert sorted(row["id"] for row in detail["features"]) == sorted(f["id"] for f in scoreboard["features"])
        assert {str(clip["clip_id"]) for clip in detail["clips"]} == clip_ids

        for record in scoreboard["features"]:
            got = client.get(url + f"/features/{record['id']}?run_id={run_id}")
            assert got.status_code == 200, got.text
            body = got.json()
            assert {key: body[key] for key in trace} == trace
            assert body["feature"]["label"] == record["label"]
            assert body["feature"]["real"] is record["real"]
            assert body["feature"]["ablation_effect_kind"] == "ablation"
            _assert_json_equal(body["feature"]["ablation_effect"], record["ablation_effect"])
            top = client.get(url + f"/features/{record['id']}/top_clips?run_id={run_id}").json()
            assert top["badge"] == ("real" if record["real"] else "unverified")
            assert {row["clip_id"] for row in top["top_clips"]} <= clip_ids

        missing = client.get(url + f"/features/999?run_id={run_id}")
        assert missing.status_code == 404
        assert {key: missing.json()["detail"][key] for key in trace} == trace
        assert client.get(url + "/features/0?run_id=../demo").status_code == 400
        assert client.get(url + "/runs/does-not-exist").status_code == 404

        for clip in manifest["clips"]:
            n_frames = int(clip["n_frames"])
            for frame in (0, n_frames - 1):
                row = client.get(url + f"/clips/{clip['clip_id']}/frame/{frame}?run_id={run_id}")
                assert row.status_code == 200, row.text
                assert row.json()["frame"] == frame
            past = client.get(url + f"/clips/{clip['clip_id']}/frame/{n_frames}?run_id={run_id}")
            assert past.status_code == 404

        first = manifest["clips"][0]["clip_id"]
        data = (clips / f"{first}.npy").read_bytes()
        assert client.get(url + f"/clips/{first}/file?run_id={run_id}").content == data
        part = client.get(url + f"/clips/{first}/file?run_id={run_id}", headers={"Range": "bytes=4-19"})
        assert part.status_code == 206
        assert part.content == data[4:20]


def test_rust_serves_the_static_viewer_bytes(tmp_path: Path) -> None:
    root = tmp_path / "read-api"
    shutil.copytree(FIXTURE / "runs", root / "runs")
    with _RustServer(root / "runs", None) as url, httpx.Client(timeout=30.0) as client:
        assert client.get(url + "/").content == (STATIC / "index.html").read_bytes()
        for name in ("guide.js", "app.js", "style.css", "fonts/geist-400.woff2"):
            assert client.get(url + f"/static/{name}").content == (STATIC / name).read_bytes(), name
