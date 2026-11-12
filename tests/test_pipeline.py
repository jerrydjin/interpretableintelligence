"""The fake adapter run is resumable, streamed, and ends at a live viewer."""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from interpretableintelligence.pipeline import REPORT_BEGIN, REPORT_END, footage_cost
from interpretableintelligence.store import STREAM_CHUNK, ActivationWriter, RowSource


def test_footage_cost_is_measured_stage_time_over_video_time():
    # (10 GPU-seconds / 3600) * $3 / (20 footage-seconds / 3600) = 10 * 3 / 20.
    assert footage_cost(10, 20, 3) == pytest.approx(10 * 3 / 20)
    assert footage_cost(0, 60, 2.5) == 0
    with pytest.raises(ValueError):
        footage_cost(1, 0, 1)
    with pytest.raises(ValueError):
        footage_cost(1, 10, -1)


def test_activation_cache_yields_bounded_chunks(tmp_path: Path):
    acts = tmp_path / "acts"
    width = 8
    with ActivationWriter(acts, width) as writer:
        for index in range(5):
            array = np.full((2, 2, 2, width), index, dtype=np.float32)
            writer.append(array, clip_index=index % 2, window=0, split_code=index % 2)
        meta = writer.finish()
    assert meta["n"] == 40
    with RowSource(acts) as source:
        seen = 0
        widest = 0
        for start, stop in source.chunks(3):
            block = np.array(source.rows[start:stop], copy=True)
            widest = max(widest, int(block.shape[0]))
            seen += int(block.shape[0])
        assert seen == meta["n"]
        assert widest <= 3
        assert widest < STREAM_CHUNK
        batch = source.gather_train(np.array([0, 1], dtype=np.int64))
        assert batch.shape == (2, width)
        assert batch.shape[0] < meta["n"]


def test_fake_run_serves_then_skips_every_stage(tmp_path: Path):
    clips = tmp_path / "clips"
    out = tmp_path / "demo"
    generated = subprocess.run(
        [sys.executable, "-m", "interpretableintelligence", "generate-clips", "--out", str(clips), "--n", "30", "--seed", "0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    command = [
        sys.executable,
        "-m",
        "interpretableintelligence",
        "run",
        "fake",
        "--clips",
        str(clips),
        "--out",
        str(out),
        "--layer",
        "16",
        "--seed",
        "0",
        "--heldout-frac",
        "0.2",
        "--gpu-cost-per-hour",
        "3.5",
        "--port",
        "0",
        "--no-open",
    ]
    lines: list[str] = []
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert proc.stdout is not None

    def _drain() -> None:
        for line in proc.stdout:
            lines.append(line)

    drain = threading.Thread(target=_drain, daemon=True)
    drain.start()
    deadline = time.time() + 180
    while time.time() < deadline:
        if any(REPORT_END in line for line in lines):
            break
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    text = "".join(lines)
    assert REPORT_END in text, text
    assert proc.poll() is None, text
    url = re.search(r"viewer (http://127\.0\.0\.1:\d+/)", text)
    assert url, text
    assert "health /api/run ok" in text
    with urllib.request.urlopen(url.group(1) + "api/run", timeout=30) as response:
        assert response.status == 200
        payload = json.load(response)
    assert payload["run_id"] == out.name
    assert len(payload["clips"]) == 30
    assert payload["scoreboard"] is True
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"].keys() >= {"capture", "sae", "features", "score", "view"}
    for name, stage in manifest["stages"].items():
        assert stage["wall_s"] >= 0
        assert stage["peak_ram_bytes"] > 0
        assert stage["device"]
        assert stage["rows"] >= 0
        assert stage["tokens"] >= 0
        assert stage["peak_ram_bytes"] < 4 * 1024**3, name
    assert manifest["stages"]["capture"]["tokens"] == 30 * 8 * 4 * 4
    report = _parse_report(text)
    assert report["clips"] == 30
    assert report["video_s"] == pytest.approx(30 * 16 / 8)
    assert report["tokens"] == manifest["stages"]["capture"]["tokens"]
    elapsed = sum(float(stage["wall_s"]) for stage in manifest["stages"].values())
    # The report prints cost_per_hour_footage with 6 decimals, so allow half of that last digit.
    assert report["cost"] == pytest.approx(footage_cost(elapsed, report["video_s"], 3.5), abs=5e-7)
    stamped = {
        path: path.stat().st_mtime_ns
        for path in (
            out / "manifest.json",
            out / "sae" / "weights.safetensors",
            out / "features" / "feature_acts.parquet",
            out / "scoreboard.json",
        )
    }
    manifest_bytes = (out / "manifest.json").read_bytes()
    second = subprocess.run(
        command + ["--no-serve"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    for stage in ("capture", "sae", "features", "score", "view"):
        assert any(line.strip() == f"skip {stage}" for line in second.stdout.splitlines()), second.stdout
    assert (out / "manifest.json").read_bytes() == manifest_bytes
    for path, mtime in stamped.items():
        assert path.stat().st_mtime_ns == mtime
    second_report = _parse_report(second.stdout)
    assert second_report["cost"] == pytest.approx(report["cost"])
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _parse_report(text: str) -> dict:
    begin = text.index(REPORT_BEGIN)
    end = text.index(REPORT_END, begin)
    body = {}
    for line in text[begin:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        body[key.strip()] = value.strip()
    return {
        "clips": int(body["clips"]),
        "video_s": float(body["video_s"]),
        "tokens": int(body["tokens"]),
        "cost": float(body["cost_per_hour_footage"]),
    }
