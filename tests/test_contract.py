"""The Rust read API returns the same JSON as the Python read API.

Floats may differ by at most 1e-9. Key order is ignored. The fake-run case is a
fresh ``interpretableintelligence run fake`` folder; a smaller synthetic folder fails faster when
the shapes drift.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import uvicorn

from interpretableintelligence.api import create_app

ROOT = Path(__file__).resolve().parents[1]
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


class _PythonServer:
    def __init__(self, runs: Path, clips: Path | None) -> None:
        self.port = _free_port()
        config = uvicorn.Config(
            create_app(runs, clips),
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        url = f"http://127.0.0.1:{self.port}"
        _wait_until(url + "/runs")
        return url

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


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


def _get(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url)
    return response


def _compare(python_url: str, rust_url: str, path: str, *, raw: bool = False) -> None:
    with httpx.Client(timeout=60.0) as client:
        left = _get(client, python_url + path)
        right = _get(client, rust_url + path)
    assert left.status_code == right.status_code, (
        f"{path}: python {left.status_code} {left.text[:400]} rust {right.status_code} {right.text[:400]}"
    )
    if raw:
        assert left.content == right.content, path
        return
    _assert_json_equal(left.json(), right.json(), path)


def _read_paths(run_id: str) -> list[str]:
    paths = [
        "/runs",
        f"/runs/{run_id}",
        f"/features/0?run_id={run_id}",
        f"/features/1?run_id={run_id}",
        f"/features/999?run_id={run_id}",
        "/features/0?run_id=../demo",
        f"/features/0/top_clips?run_id={run_id}",
        f"/features/1/top_clips?run_id={run_id}",
        f"/runs/missing-run",
    ]
    return paths


def _compare_run(runs: Path, clips: Path | None, run_id: str, frames: list[tuple[str, int]]) -> None:
    with _PythonServer(runs, clips) as python_url, _RustServer(runs, clips) as rust_url:
        for path in _read_paths(run_id):
            _compare(python_url, rust_url, path)
        for feature_id in range(2):
            _compare(python_url, rust_url, f"/features/{feature_id}?run_id={run_id}")
            _compare(python_url, rust_url, f"/features/{feature_id}/top_clips?run_id={run_id}")
        for clip_id, n_frames in frames:
            for frame in range(n_frames):
                _compare(
                    python_url,
                    rust_url,
                    f"/clips/{clip_id}/frame/{frame}?run_id={run_id}",
                )
            if clips is not None:
                _compare(python_url, rust_url, f"/clips/{clip_id}/file?run_id={run_id}", raw=True)
                with httpx.Client(timeout=30.0) as client:
                    left = client.get(
                        python_url + f"/clips/{clip_id}/file?run_id={run_id}",
                        headers={"Range": "bytes=0-15"},
                    )
                    right = client.get(
                        rust_url + f"/clips/{clip_id}/file?run_id={run_id}",
                        headers={"Range": "bytes=0-15"},
                    )
                assert left.status_code == right.status_code == 206, (left.status_code, right.status_code, right.text[:200])
                assert left.content == right.content


def _write_min_run(root: Path) -> tuple[Path, Path]:
    clips = root / "clips"
    clips.mkdir()
    array = np.zeros((2, 2, 2, 1), dtype=np.float32)
    array[0, 0, 1, 0] = 1.0
    np.save(clips / "clip-000.npy", array)
    (clips / "clips.json").write_text(
        json.dumps({"clips": [{"clip_id": "clip-000", "file": "clip-000.npy", "fps": 8.0}]}) + "\n",
        encoding="utf-8",
    )
    run = root / "runs" / "mini"
    (run / "features").mkdir(parents=True)
    sha = __import__("hashlib").sha256((clips / "clip-000.npy").read_bytes()).hexdigest()
    manifest = {
        "run_id": "mini",
        "tool_version": "0.1.0",
        "model": {"adapter": "fake", "name": "fake", "sha": "cd" * 32, "random_init": False},
        "layer": 3,
        "d_model": 4,
        "grid": {"t": 2, "h": 2, "w": 2, "tubelet": 1, "patch_px": 1},
        "split": {"seed": 0, "heldout_frac": 0.5, "sha": "ab" * 32},
        "clips": [
            {"clip_id": "clip-000", "sha256": sha, "fps": 8.0, "n_frames": 2, "split": "train"},
        ],
    }
    scoreboard = {
        "run": {
            "model_sha": "cd" * 32,
            "layer": 3,
            "split_sha": "ab" * 32,
            "seed": 0,
            "n_train_clips": 1,
            "n_heldout_clips": 1,
        },
        "sae": {"ev_heldout": 0.91, "loss_recovered": 0.8, "dead_frac": 0.0, "usable": True},
        "features": [
            {
                "id": 0,
                "label": "stripe",
                "distinct_clips_top20": 1,
                "label_holds_heldout": True,
                "ablation_effect": 0.2,
                "random_percentile": 100.0,
                "n_random": 100,
                "absent_in_random_init": True,
                "real": True,
            },
            {
                "id": 1,
                "label": "flicker",
                "distinct_clips_top20": 1,
                "label_holds_heldout": False,
                "ablation_effect": 0.01,
                "random_percentile": 10.0,
                "n_random": 100,
                "absent_in_random_init": True,
                "real": False,
            },
        ],
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "scoreboard.json").write_text(json.dumps(scoreboard), encoding="utf-8")
    jsonl = []
    for feature in scoreboard["features"]:
        jsonl.append({**feature, "label_model": "fake-cue-v1", "top": []})
    (run / "features" / "features.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in jsonl),
        encoding="utf-8",
    )
    acts = pa.table(
        {
            "feature_id": pa.array([0, 0, 1], type=pa.int32()),
            "clip_id": pa.array(["clip-000", "clip-000", "clip-000"]),
            "window": pa.array([0, 0, 0], type=pa.int16()),
            "t": pa.array([0, 0, 1], type=pa.int16()),
            "h": pa.array([0, 0, 1], type=pa.int16()),
            "w": pa.array([1, 1, 0], type=pa.int16()),
            "value": pa.array([1.25, 0.5, 0.75], type=pa.float32()),
        }
    )
    effects = pa.table(
        {
            "feature_id": pa.array([0, 0, 1], type=pa.int32()),
            "clip_id": pa.array(["clip-000", "clip-000", "clip-000"]),
            "window": pa.array([0, 0, 0], type=pa.int16()),
            "t": pa.array([0, 0, 1], type=pa.int16()),
            "effect": pa.array([0.2, -0.01, 0.05], type=pa.float32()),
            "effect_kind": pa.array(["ablation", "estimate", "estimate"]),
        }
    )
    pq.write_table(acts, run / "features" / "feature_acts.parquet")
    pq.write_table(effects, run / "features" / "clip_effects.parquet")
    return root / "runs", clips


def test_synthetic_read_endpoints_match(tmp_path: Path) -> None:
    runs, clips = _write_min_run(tmp_path)
    _compare_run(runs, clips, "mini", [("clip-000", 2)])


def test_fake_run_read_endpoints_match(demo_run: Path) -> None:
    scoreboard = json.loads((demo_run / "scoreboard.json").read_text())
    real = [feature for feature in scoreboard["features"] if feature["real"]]
    assert len(real) == 3
    assert {feature["label"] for feature in real} == {"stripe", "pulse", "corner"}
    manifest = json.loads((demo_run / "manifest.json").read_text())
    clips = demo_run.parent.parent / "clips"
    frames = [(clip["clip_id"], int(clip["n_frames"])) for clip in manifest["clips"]]
    # Every feature, not just the two the helper always requests.
    with _PythonServer(demo_run.parent, clips) as python_url, _RustServer(demo_run.parent, clips) as rust_url:
        for path in (
            "/runs",
            f"/runs/{demo_run.name}",
            "/features/0?run_id=../demo",
            f"/features/999?run_id={demo_run.name}",
            "/runs/does-not-exist",
        ):
            _compare(python_url, rust_url, path)
        for feature in scoreboard["features"]:
            feature_id = feature["id"]
            _compare(python_url, rust_url, f"/features/{feature_id}?run_id={demo_run.name}")
            _compare(python_url, rust_url, f"/features/{feature_id}/top_clips?run_id={demo_run.name}")
        for clip_id, n_frames in frames:
            for frame in range(n_frames):
                _compare(
                    python_url,
                    rust_url,
                    f"/clips/{clip_id}/frame/{frame}?run_id={demo_run.name}",
                )
        first = frames[0][0]
        _compare(python_url, rust_url, f"/clips/{first}/file?run_id={demo_run.name}", raw=True)
        with httpx.Client(timeout=30.0) as client:
            left = client.get(
                python_url + f"/clips/{first}/file?run_id={demo_run.name}",
                headers={"Range": "bytes=4-19"},
            )
            right = client.get(
                rust_url + f"/clips/{first}/file?run_id={demo_run.name}",
                headers={"Range": "bytes=4-19"},
            )
        assert left.status_code == right.status_code == 206
        assert left.content == right.content


def test_static_asset_bytes_match(tmp_path: Path) -> None:
    runs, clips = _write_min_run(tmp_path)
    with _PythonServer(runs, clips) as python_url, _RustServer(runs, clips) as rust_url:
        for path in ("/", "/static/app.js", "/static/style.css", "/static/fonts/geist-400.woff2"):
            _compare(python_url, rust_url, path, raw=True)
