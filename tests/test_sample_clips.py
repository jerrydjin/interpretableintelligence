"""``ii get-clips``: cut per-episode mp4 clips out of one concatenated LeRobot video, offline."""

from __future__ import annotations

import json

import av
import huggingface_hub
import numpy as np
from typer.testing import CliRunner

from interpretableintelligence import sample_clips
from interpretableintelligence.cli import app
from interpretableintelligence.clips import list_clip_meta


def _write_source(path, seconds: int, fps: int = 15) -> None:
    with av.open(str(path), "w") as dst:
        stream = dst.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(seconds * fps):
            image = np.full((48, 64, 3), (i * 3) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)


def _fake_dataset(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    _write_source(source, seconds=12)
    episodes = [
        {"episode": i, "task": f"task {i}", "file": (0, 0), "start": 3.0 * i, "end": 3.0 * (i + 1)}
        for i in range(4)
    ]
    monkeypatch.setattr(sample_clips, "_episodes", lambda: episodes)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(source))


def test_fetch_cuts_one_clip_per_episode(monkeypatch, tmp_path):
    _fake_dataset(monkeypatch, tmp_path)
    out = tmp_path / "clips"
    entries = sample_clips.fetch_droid_clips(out, n=3, seconds=2.0, say=lambda _: None)

    assert [e["episode"] for e in entries] == [0, 2, 3]  # spread over the dataset
    assert all(e["frames"] == 30 for e in entries)
    index = json.loads((out / "clips.json").read_text())
    assert index["source"] == sample_clips.DATASET and index["license"] == "MIT"
    meta = list_clip_meta(out)
    assert [m["clip_id"] for m in meta] == ["droid-000", "droid-002", "droid-003"]
    assert all(m["n_frames"] == 30 and m["fps"] == 15 for m in meta)
    assert sample_clips.clips_ready(out, 3)


def test_short_episode_keeps_whole_episode(monkeypatch, tmp_path):
    _fake_dataset(monkeypatch, tmp_path)
    entries = sample_clips.fetch_droid_clips(tmp_path / "c", n=1, seconds=10.0, say=lambda _: None)
    assert entries[0]["frames"] == 45


def test_cli_get_clips_points_at_next_step(monkeypatch, tmp_path):
    _fake_dataset(monkeypatch, tmp_path)
    out = tmp_path / "droid"
    result = CliRunner().invoke(app, ["get-clips", "--out", str(out), "--n", "2", "--seconds", "1"])
    assert result.exit_code == 0, result.output
    assert "wrote 2 clips" in result.output
    assert f"--clips {out}" in result.output
    # A second call reuses what is there.
    again = CliRunner().invoke(app, ["get-clips", "--out", str(out), "--n", "2", "--seconds", "1"])
    assert again.exit_code == 0 and "Downloading" not in again.output


def test_too_many_clips_is_a_clear_error(monkeypatch, tmp_path):
    _fake_dataset(monkeypatch, tmp_path)
    result = CliRunner().invoke(app, ["get-clips", "--out", str(tmp_path / "x"), "--n", "9"])
    assert result.exit_code == 1
    assert "between 1 and 4" in result.output
