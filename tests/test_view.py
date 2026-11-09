"""Local viewer: run-folder data only, and a browser check of a fresh fake run."""

from __future__ import annotations

import ast
import json
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from physint.artifacts import write_clip_effects, write_feature_acts, write_json, write_jsonl
from physint.cli import app
from physint.clips import file_sha256
from physint.view import (
    STATIC_DIR,
    badge_for,
    bind_server,
    encode_png,
    frame_index,
    frame_to_token,
    load_run,
    render_frame,
    token_to_frame,
    window_starts,
)

CAUSAL = {"stripe", "pulse", "corner"}
DECOY = {"flicker", "band", "speckle"}
runner = CliRunner()


def test_badge_is_real_only_for_json_true():
    assert badge_for(True) == "real"
    assert badge_for(False) == "unverified"
    assert badge_for(None) == "unverified"
    assert badge_for(1) == "unverified"


def test_frame_index_follows_the_grid():
    grid = {"t": 8, "h": 4, "w": 4, "tubelet": 2, "patch_px": 4}
    assert frame_index(0, grid) == (0, 0)
    assert frame_index(1, grid) == (0, 0)
    assert frame_index(2, grid) == (0, 1)
    assert frame_index(15, grid) == (0, 7)
    assert frame_index(16, grid) == (1, 0)


def test_window_start_plus_tubelet_selects_the_source_frame():
    """V-JEPA: t indexes tubelets, so frame = window_start + t * tubelet."""
    grid = {"t": 8, "h": 2, "w": 2, "tubelet": 2, "patch_px": 4}
    clip = {"window_starts": [4]}
    starts = window_starts(clip, grid, n_windows=0)
    assert starts == [4]
    assert token_to_frame(0, 3, starts, grid) == 10
    assert frame_to_token(10, starts, grid) == (0, 3)
    assert frame_to_token(11, starts, grid) == (0, 3)
    assert frame_to_token(6, starts, grid) == (0, 1)
    assert frame_to_token(0, starts, grid) == (0, None)
    assert window_starts({}, {"t": 8, "tubelet": 2, "window_start": 4}, 0) == [4]


def test_viewer_source_does_not_import_adapters():
    source = Path(inspect_view_path()).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    forbidden = [name for name in modules if name.startswith("physint.adapters") or name == "torch"]
    assert forbidden == []


def inspect_view_path() -> Path:
    import physint.view as view

    return Path(view.__file__)


def test_static_shell_is_packaged():
    assert (STATIC_DIR / "index.html").is_file()
    assert (STATIC_DIR / "app.js").is_file()
    assert (STATIC_DIR / "fonts" / "geist-400.woff2").is_file()


def test_render_sparse_multichannel_frame_marks_the_hot_patch():
    frame = np.zeros((8, 8, 6), dtype=np.float32)
    frame[0:4, 4:8, 0] = 1.0
    rgb = render_frame(frame, peak=1.0)
    assert rgb.shape == (8, 8, 3)
    assert int(rgb[1, 5].sum()) > int(rgb[6, 1].sum()) + 40
    assert encode_png(rgb).startswith(b"\x89PNG\r\n\x1a\n")


def test_non_real_feature_stays_unverified_and_effects_stay_distinct(tmp_path: Path):
    run, clips = _tiny_run(tmp_path)
    model = load_run(run, clips)
    stripe = next(feature for feature in model.run_payload["features"] if feature["label"] == "stripe")
    flicker = next(feature for feature in model.run_payload["features"] if feature["label"] == "flicker")
    assert stripe["badge"] == "real"
    assert flicker["badge"] == "unverified"
    assert flicker["random_percentile"] == 100.0
    assert stripe["top_clips"][0]["clip_id"] == "clip-000"
    assert stripe["top_clips"][0]["frame"] == 2

    payload = model.clips["clip-000"]
    assert payload["frames"][0]["features"] == []
    assert payload["frames"][2]["features"] == payload["frames"][3]["features"]
    fired = {feature["label"]: feature for feature in payload["frames"][2]["features"]}
    assert set(fired) == {"stripe", "flicker"}
    kinds = [effect["effect_kind"] for effect in fired["stripe"]["effects"]]
    assert kinds == ["ablation", "estimate"]
    assert fired["stripe"]["effects"][0]["effect"] == pytest.approx(0.2)
    assert fired["stripe"]["effects"][1]["effect"] == pytest.approx(-0.01)
    assert [effect["effect_kind"] for effect in fired["flicker"]["effects"]] == ["estimate"]
    assert fired["stripe"]["heatmap"][0][1] == pytest.approx(1.25)
    assert fired["stripe"]["heatmap"][1][0] == 0
    assert fired["flicker"]["badge"] == "unverified"
    assert payload["pixels"] == "ok"
    png = model.frame_png("clip-000", 2)
    assert png is not None and png.startswith(b"\x89PNG")


def test_clip_path_comes_from_the_manifest_or_the_flag(tmp_path: Path):
    run, clips = _tiny_run(tmp_path, manifest_path=True)
    from_manifest = load_run(run, None)
    assert from_manifest.clips["clip-000"]["pixels"] == "ok"

    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.zeros((4, 4, 4, 6), dtype=np.float32))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["path"] = str(wrong)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    overridden = load_run(run, clips)
    assert overridden.clips["clip-000"]["pixels"] == "ok"
    mismatched = load_run(run, None)
    assert mismatched.clips["clip-000"]["pixels"] == "sha_mismatch"
    assert mismatched.frame_png("clip-000", 0) is None


def test_stem_match_wins_and_sha_rescues_a_mismatched_stem(tmp_path: Path):
    clips = tmp_path / "clips"
    stem = clips / "clip-000.mp4"
    other = clips / "elsewhere.mp4"
    named = clips / "clip-001.mp4"
    decoy = clips / "decoy.mp4"
    _encode_video(stem, color="red", frames=2, size="32x32")
    _encode_video(other, color="blue", frames=2, size="32x32")
    _encode_video(named, color="green", frames=2, size="32x32")
    _encode_video(decoy, color="white", frames=2, size="32x32")
    run = _run_for_media(
        tmp_path / "run",
        [
            ("clip-000", file_sha256(other)),
            ("clip-001", file_sha256(named)),
        ],
    )
    model = load_run(run, clips)
    assert model.loaded["clip-000"].path == other
    assert model.clips["clip-000"]["pixels"] == "ok"
    assert model.clips["clip-000"]["media"] == "video"
    assert model.loaded["clip-001"].path == named


def test_missing_scoreboard_effects_and_clip_files_do_not_crash(tmp_path: Path):
    run, clips = _video_run(tmp_path)
    model = load_run(run, clips)
    grasp = model.clips["robot-grasp"]
    by_frame = {frame["frame"]: frame for frame in grasp["frames"]}
    assert by_frame[0]["features"] == []
    assert by_frame[6]["t"] == 1
    assert [feature["label"] for feature in by_frame[6]["features"]] == []
    assert by_frame[4]["features"] == by_frame[5]["features"]
    assert [feature["label"] for feature in by_frame[4]["features"]] == ["reach"]
    fired = {feature["label"]: feature for feature in by_frame[10]["features"]}
    assert set(fired) == {"grasp"}
    assert by_frame[10]["t"] == 3
    assert by_frame[10]["features"] == by_frame[11]["features"]
    assert fired["grasp"]["heatmap"][1][0] == pytest.approx(3.0)
    assert fired["grasp"]["effects"][0]["effect_kind"] == "ablation"
    assert model.clips["robot-absent"]["pixels"] == "missing"
    assert model.clips["robot-absent"]["media"] == "none"
    assert grasp["pixels"] == "ok"
    grasp_row = next(feature for feature in model.run_payload["features"] if feature["label"] == "grasp")
    reach_row = next(feature for feature in model.run_payload["features"] if feature["label"] == "reach")
    assert grasp_row["badge"] == "real"
    assert grasp_row["top_clips"][0]["frame"] == 10

    short = json.loads((run / "scoreboard.json").read_text(encoding="utf-8"))
    short["features"] = [row for row in short["features"] if row["label"] == "grasp"]
    (run / "scoreboard.json").write_text(json.dumps(short), encoding="utf-8")
    partial = load_run(run, clips)
    badges = {feature["label"]: feature["badge"] for feature in partial.run_payload["features"]}
    assert badges["grasp"] == "real"
    assert badges["reach"] == "unverified"

    (run / "scoreboard.json").unlink()
    (run / "features" / "clip_effects.parquet").unlink()
    bare = load_run(run, clips)
    assert bare.run_payload["scoreboard"] is False
    assert bare.run_payload["sae"] is None
    assert {feature["badge"] for feature in bare.run_payload["features"]} == {"unverified"}
    bare_grasp = {feature["label"]: feature for feature in bare.clips["robot-grasp"]["frames"][10]["features"]}
    assert bare_grasp["grasp"]["activation"] == pytest.approx(3.0)
    assert bare_grasp["grasp"]["effects"] == []
    assert bare.frame_png("robot-grasp", 10).startswith(b"\x89PNG")
    assert bare.clips["robot-absent"]["pixels"] == "missing"


def test_avi_transcodes_and_mp4_accepts_a_byte_range(tmp_path: Path):
    run, clips = _video_run(tmp_path)
    avi = clips / "robot-arm.avi"
    _encode_video(avi, codec="mpeg4", color="blue", frames=4, size="32x32")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"].append(
        {
            "clip_id": "robot-arm",
            "sha256": file_sha256(avi),
            "fps": 8.0,
            "n_frames": 4,
            "split": "train",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    model = load_run(run, clips)
    assert model.clips["robot-arm"]["media"] == "video"
    played, content_type = model.media_file("robot-arm")
    assert content_type == "video/mp4"
    assert played.suffix == ".mp4"
    assert played.is_file() and played.stat().st_size > 0
    native, native_type = model.media_file("robot-grasp")
    assert native_type == "video/mp4"
    assert native == clips / "robot-grasp.mp4"

    httpd = bind_server(model, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}/api/clips/robot-grasp/media"
    try:
        request = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read()
            assert response.status == 206
            assert response.headers["Content-Range"].startswith("bytes 0-7/")
            assert len(body) == 8
            assert body[4:8] == b"ftyp"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_serves_the_shell_and_refuses_path_traversal(tmp_path: Path):
    run, clips = _tiny_run(tmp_path)
    model = load_run(run, clips)
    httpd = bind_server(model, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://{httpd.server_address[0]}:{httpd.server_address[1]}"
    try:
        _wait_http(url + "/")
        index = _get(url + "/")
        assert b"data-testid=\"run-header\"" in index
        script = _get(url + "/static/app.js")
        assert b"ablation" in script
        font = _get(url + "/static/fonts/geist-400.woff2")
        assert font[:4] == b"wOF2"
        payload = json.loads(_get(url + "/api/run"))
        assert payload["layer"] == 3
        assert payload["sae"]["usable"] is True
        missing = urllib.request.Request(url + "/static/%2e%2e/%2e%2e/view.py")
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(missing, timeout=2)
        assert raised.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_view_help_and_missing_run():
    help_text = runner.invoke(app, ["view", "--help"])
    assert help_text.exit_code == 0
    assert "--clips" in help_text.stdout
    assert "--no-open" in help_text.stdout
    missing = runner.invoke(app, ["view", "runs/does-not-exist", "--no-open"])
    assert missing.exit_code == 1
    assert "not found" in missing.stderr


def _score_row(feature_id: int, label: str, real: bool, ablation: float) -> dict:
    return {
        "id": feature_id,
        "label": label,
        "distinct_clips_top20": 1 if label else 0,
        "label_holds_heldout": True if label else False,
        "ablation_effect": ablation,
        "random_percentile": 100.0,
        "n_random": 100,
        "absent_in_random_init": True if label else False,
        "real": real,
    }


def _tiny_run(tmp_path: Path, manifest_path: bool = False) -> tuple[Path, Path]:
    clips = tmp_path / "clips"
    clips.mkdir()
    frames = np.zeros((4, 4, 4, 6), dtype=np.float32)
    frames[:, 0:2, 2:4, 0] = 1.0
    clip_path = clips / "clip-000.npy"
    np.save(clip_path, frames)
    digest = file_sha256(clip_path)
    run = tmp_path / "run"
    (run / "features").mkdir(parents=True)
    clip_entry = {
        "clip_id": "clip-000",
        "sha256": digest,
        "fps": 8.0,
        "n_frames": 4,
        "split": "train",
    }
    if manifest_path:
        clip_entry["path"] = str(clip_path)
    write_json(
        run / "manifest.json",
        {
            "run_id": "tiny",
            "tool_version": "0.1.0",
            "model": {"adapter": "fake", "name": "fake", "sha": "ab" * 32, "random_init": False},
            "layer": 3,
            "d_model": 4,
            "grid": {"t": 2, "h": 2, "w": 2, "tubelet": 2, "patch_px": 2},
            "split": {"seed": 0, "heldout_frac": 0.5, "sha": "cd" * 32},
            "clips": [clip_entry],
        },
    )
    write_json(
        run / "scoreboard.json",
        {
            "run": {
                "model_sha": "ab" * 32,
                "layer": 3,
                "split_sha": "cd" * 32,
                "seed": 0,
                "n_train_clips": 1,
                "n_heldout_clips": 1,
            },
            "sae": {"ev_heldout": 0.91, "loss_recovered": 0.95, "dead_frac": 0.0, "usable": True},
            "features": [
                _score_row(0, "stripe", True, 0.2),
                _score_row(1, "flicker", False, 0.0),
            ],
        },
    )
    write_jsonl(
        run / "features" / "features.jsonl",
        [
            {"id": 0, "label": "stripe", "label_model": "fake-cue-v1"},
            {"id": 1, "label": "flicker", "label_model": "fake-cue-v1"},
        ],
    )
    write_feature_acts(
        run / "features" / "feature_acts.parquet",
        {
            "feature_id": [0, 1],
            "clip_id": ["clip-000", "clip-000"],
            "window": [0, 0],
            "t": [1, 1],
            "h": [0, 1],
            "w": [1, 0],
            "value": [1.25, 0.5],
        },
    )
    write_clip_effects(
        run / "features" / "clip_effects.parquet",
        {
            "feature_id": [0, 0, 1],
            "clip_id": ["clip-000", "clip-000", "clip-000"],
            "window": [0, 0, 0],
            "t": [1, 1, 1],
            "effect": [0.2, -0.01, 0.001],
            "effect_kind": ["ablation", "estimate", "estimate"],
        },
    )
    return run, clips


def _encode_video(
    path: Path,
    *,
    color: str = "red",
    size: str = "64x64",
    frames: int = 16,
    fps: int = 8,
    codec: str = "libx264",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:r={fps}:d={frames / fps + 1}",
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
    ]
    if path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        command.extend(["-movflags", "+faststart"])
    command.append(str(path))
    subprocess.run(command, check=True, timeout=60)


def _run_for_media(run: Path, clips: list[tuple[str, str]]) -> Path:
    (run / "features").mkdir(parents=True)
    write_json(
        run / "manifest.json",
        {
            "run_id": "media",
            "tool_version": "0.1.0",
            "model": {"adapter": "vjepa2", "name": "vjepa2", "sha": "ef" * 32, "random_init": False},
            "layer": 12,
            "d_model": 8,
            "grid": {"t": 8, "h": 2, "w": 2, "tubelet": 2, "patch_px": 16},
            "split": {"seed": 0, "heldout_frac": 0.0, "sha": "11" * 32},
            "clips": [
                {"clip_id": clip_id, "sha256": digest, "fps": 8.0, "n_frames": 2, "split": "train"}
                for clip_id, digest in clips
            ],
        },
    )
    write_feature_acts(
        run / "features" / "feature_acts.parquet",
        {
            "feature_id": [0],
            "clip_id": [clips[0][0]],
            "window": [0],
            "t": [0],
            "h": [0],
            "w": [0],
            "value": [1.0],
        },
    )
    return run


def _video_run(tmp_path: Path) -> tuple[Path, Path]:
    """A 16-frame mp4 whose first window starts at frame 4 (tubelet 2)."""
    clips = tmp_path / "clips"
    grasp = clips / "robot-grasp.mp4"
    _encode_video(grasp, color="0x2f6f4f", frames=16, size="64x64")
    run = tmp_path / "run"
    (run / "features").mkdir(parents=True)
    write_json(
        run / "manifest.json",
        {
            "run_id": "robot-demo",
            "tool_version": "0.1.0",
            "model": {"adapter": "vjepa2", "name": "vjepa2", "sha": "ab" * 32, "random_init": False},
            "layer": 24,
            "d_model": 16,
            "grid": {"t": 8, "h": 2, "w": 2, "tubelet": 2, "patch_px": 16},
            "split": {"seed": 0, "heldout_frac": 0.0, "sha": "cd" * 32},
            "clips": [
                {
                    "clip_id": "robot-grasp",
                    "sha256": file_sha256(grasp),
                    "fps": 8.0,
                    "n_frames": 16,
                    "split": "train",
                    "window_starts": [4],
                },
                {
                    "clip_id": "robot-absent",
                    "sha256": "00" * 32,
                    "fps": 8.0,
                    "n_frames": 16,
                    "split": "train",
                },
            ],
        },
    )
    write_json(
        run / "scoreboard.json",
        {
            "run": {
                "model_sha": "ab" * 32,
                "layer": 24,
                "split_sha": "cd" * 32,
                "seed": 0,
                "n_train_clips": 2,
                "n_heldout_clips": 0,
            },
            "sae": {"ev_heldout": 0.8, "loss_recovered": 0.7, "dead_frac": 0.0, "usable": True},
            "features": [
                _score_row(7, "grasp", True, 0.4),
                _score_row(8, "reach", False, 0.0),
            ],
        },
    )
    write_jsonl(
        run / "features" / "features.jsonl",
        [
            {"id": 7, "label": "grasp", "label_model": "hand-v1"},
            {"id": 8, "label": "reach", "label_model": "hand-v1"},
        ],
    )
    write_feature_acts(
        run / "features" / "feature_acts.parquet",
        {
            "feature_id": [7, 8],
            "clip_id": ["robot-grasp", "robot-grasp"],
            "window": [0, 0],
            "t": [3, 0],
            "h": [1, 0],
            "w": [0, 1],
            "value": [3.0, 1.0],
        },
    )
    write_clip_effects(
        run / "features" / "clip_effects.parquet",
        {
            "feature_id": [7],
            "clip_id": ["robot-grasp"],
            "window": [0],
            "t": [3],
            "effect": [0.4],
            "effect_kind": ["ablation"],
        },
    )
    return run, clips


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.read()


def _wait_http(url: str) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.05)
    raise RuntimeError(f"viewer did not accept connections at {url}")


def _chrome() -> str:
    for candidate in ("/usr/local/bin/google-chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"):
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Chrome is not installed; the viewer browser test cannot run")


def _moment(run_dir: Path) -> dict:
    """A frame where a planted feature has a real ablation and a decoy also fires."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    scoreboard = json.loads((run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    tubelet = int(grid["tubelet"])
    span = int(grid["t"]) * tubelet
    by_id = {feature["id"]: feature for feature in scoreboard["features"]}
    real_ids = {feature["id"] for feature in scoreboard["features"] if feature["real"]}
    decoy_ids = {feature["id"] for feature in scoreboard["features"] if feature["label"] in DECOY}
    assert {by_id[feature_id]["label"] for feature_id in real_ids} == CAUSAL

    acts = pq.read_table(run_dir / "features" / "feature_acts.parquet").to_pydict()
    effects = pq.read_table(run_dir / "features" / "clip_effects.parquet").to_pydict()
    fires: dict[tuple, set[int]] = {}
    for feature_id, clip_id, window, t in zip(
        acts["feature_id"], acts["clip_id"], acts["window"], acts["t"], strict=True
    ):
        fires.setdefault((str(clip_id), int(window), int(t)), set()).add(int(feature_id))
    ablation: dict[tuple, float] = {}
    for feature_id, clip_id, window, t, effect, kind in zip(
        effects["feature_id"],
        effects["clip_id"],
        effects["window"],
        effects["t"],
        effects["effect"],
        effects["effect_kind"],
        strict=True,
    ):
        if kind == "ablation":
            ablation[(str(clip_id), int(window), int(t), int(feature_id))] = float(effect)

    best = None
    for (clip_id, window, t), feature_ids in fires.items():
        decoys = [feature_id for feature_id in feature_ids if feature_id in decoy_ids]
        if not decoys:
            continue
        for feature_id in feature_ids:
            if feature_id not in real_ids:
                continue
            effect = ablation.get((clip_id, window, t, feature_id))
            if effect is None or effect < 1e-4:
                continue
            frame = window * span + t * tubelet
            if frame <= 0:
                continue
            if best is None or effect > best["effect"]:
                best = {
                    "clip_id": clip_id,
                    "frame": frame,
                    "feature_id": feature_id,
                    "label": by_id[feature_id]["label"],
                    "effect": effect,
                    "decoy_id": decoys[0],
                    "decoy_label": by_id[decoys[0]]["label"],
                    "score": by_id[feature_id],
                }
    assert best is not None, "no frame has both a planted ablation and a decoy"
    return best


@pytest.fixture(scope="module")
def fresh_fake_run(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("viewer-fake")
    clips = root / "clips"
    out = root / "runs" / "demo"
    generated = runner.invoke(app, ["generate-clips", "--out", str(clips), "--n", "30", "--seed", "0"])
    assert generated.exit_code == 0, generated.output
    ran = runner.invoke(
        app,
        [
            "run",
            "fake",
            "--clips",
            str(clips),
            "--layer",
            "16",
            "--seed",
            "0",
            "--heldout-frac",
            "0.2",
            "--out",
            str(out),
        ],
    )
    assert ran.exit_code == 0, ran.output
    scoreboard = json.loads((out / "scoreboard.json").read_text(encoding="utf-8"))
    real = [feature for feature in scoreboard["features"] if feature["real"]]
    assert len(real) == 3
    assert {feature["label"] for feature in real} == CAUSAL
    return clips, out


@pytest.fixture(scope="module")
def viewer_url(fresh_fake_run: tuple[Path, Path]):
    clips, out = fresh_fake_run
    err_path = out.parent / "viewer.err"
    err = err_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "physint",
            "view",
            str(out),
            "--clips",
            str(clips),
            "--no-open",
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=err,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], 30)
        err.flush()
        if not ready or proc.poll() not in (None, 0):
            raise RuntimeError(err_path.read_text(encoding="utf-8") or "viewer exited before printing a url")
        line = proc.stdout.readline().strip()
        match = re.search(r"http://127\.0\.0\.1:\d+/", line)
        assert match, line
        url = match.group(0)
        _wait_http(url)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        err.close()


def test_playwright_scrub_shows_real_ablation_and_unverified_decoy(fresh_fake_run, viewer_url: str):
    from playwright.sync_api import expect, sync_playwright

    clips, run = fresh_fake_run
    del clips
    moment = _moment(run)
    artifacts = Path("/opt/cursor/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=_chrome(),
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        page.set_default_timeout(15000)
        page.goto(viewer_url, wait_until="domcontentloaded")
        page.wait_for_selector("[data-testid=run-header]")
        expect(page.locator("[data-testid=run-model]")).to_contain_text("fake")
        expect(page.locator("[data-testid=run-layer]")).to_have_text("layer 16")
        expect(page.locator("[data-testid=split-sizes]")).to_have_text("train 24 · held-out 6")
        expect(page.locator("[data-testid=sae-summary]")).to_have_text(
            re.compile(r"^usable · ev_heldout \d\.\d{3} · loss_recovered \d\.\d{3}$")
        )

        page.locator(f"[data-testid=clip-link][data-clip-id='{moment['clip_id']}']").click()
        page.wait_for_function(
            "(id) => document.querySelector('[data-testid=clip-title]')?.textContent === id",
            arg=moment["clip_id"],
        )
        scrubber = page.locator("[data-testid=scrubber]")
        scrubber.focus()
        page.keyboard.press("Home")
        for _ in range(moment["frame"]):
            page.keyboard.press("ArrowRight")
        page.wait_for_function(
            "(frame) => document.querySelector('[data-testid=frame-features]')?.dataset.frame === frame",
            arg=str(moment["frame"]),
        )

        row = page.locator(f"[data-testid=frame-feature][data-feature-id='{moment['feature_id']}']")
        expect(row).to_be_visible()
        expect(row).to_have_attribute("data-label", moment["label"])
        expect(row.locator("[data-testid=badge]")).to_have_text("real")
        ablation = row.locator("[data-testid=effect][data-kind=ablation]")
        expect(ablation).to_be_visible()
        expect(ablation).to_contain_text("ablation")
        assert float(ablation.get_attribute("data-effect")) == pytest.approx(moment["effect"], abs=1e-5)
        shown = float(ablation.locator(".value").inner_text())
        assert shown == pytest.approx(moment["effect"], abs=1e-4)

        decoy = page.locator(f"[data-testid=frame-feature][data-feature-id='{moment['decoy_id']}']")
        expect(decoy).to_be_visible()
        expect(decoy).to_have_attribute("data-label", moment["decoy_label"])
        expect(decoy.locator("[data-testid=badge]")).to_have_text("unverified")

        row.locator("[data-testid=activation]").click()
        page.wait_for_function(
            """(id) => {
                const selected = document.querySelector('[data-testid=frame-feature][data-selected="true"]');
                return selected && selected.dataset.featureId === String(id);
            }""",
            arg=moment["feature_id"],
        )
        heat = page.locator("[data-testid=heat-cell]").evaluate_all("nodes => nodes.map((node) => Number(node.dataset.value))")
        assert heat and max(heat) > 0

        page.set_viewport_size({"width": 390, "height": 800})
        expect(row).to_be_visible()
        expect(decoy.locator("[data-testid=badge]")).to_have_text("unverified")
        page.set_viewport_size({"width": 1280, "height": 860})
        page.wait_for_function(
            """() => {
                const image = document.querySelector('[data-testid=frame-image]');
                return image && image.complete && image.naturalWidth > 0;
            }"""
        )
        page.evaluate("() => document.fonts.ready")
        page.screenshot(path=str(artifacts / "viewer-clip.png"), full_page=True)

        row.locator("[data-testid=feature-link]").click()
        page.wait_for_selector("[data-testid=feature-page]")
        feature_page = page.locator("[data-testid=feature-page]")
        expect(feature_page).to_have_attribute("data-label", moment["label"])
        expect(feature_page.locator("[data-testid=badge]")).to_have_text("real")
        score = moment["score"]
        expect(page.locator("[data-testid=distinct_clips_top20]")).to_have_text(str(score["distinct_clips_top20"]))
        expect(page.locator("[data-testid=label_holds_heldout]")).to_have_text("yes")
        assert float(page.locator("[data-testid=ablation_effect]").get_attribute("data-value")) == pytest.approx(
            score["ablation_effect"], abs=1e-6
        )
        expect(page.locator("[data-testid=random_percentile]")).to_have_text(str(int(score["random_percentile"])))
        expect(page.locator("[data-testid=n_random]")).to_have_text(str(score["n_random"]))
        expect(page.locator("[data-testid=absent_in_random_init]")).to_have_text("yes")
        expect(page.locator("[data-testid=top-clip]").first).to_be_visible()
        page.evaluate("() => document.fonts.ready")
        page.screenshot(path=str(artifacts / "viewer-feature.png"), full_page=True)
        browser.close()


@contextmanager
def _view_server(run: Path, clips: Path):
    err_path = run.parent / f"viewer-{run.name}.err"
    err = err_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "physint",
            "view",
            str(run),
            "--clips",
            str(clips),
            "--no-open",
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=err,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], 30)
        err.flush()
        if not ready or proc.poll() not in (None, 0):
            raise RuntimeError(err_path.read_text(encoding="utf-8") or "viewer exited before printing a url")
        line = proc.stdout.readline().strip()
        match = re.search(r"http://127\.0\.0\.1:\d+/", line)
        assert match, line
        url = match.group(0)
        _wait_http(url)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        err.close()


def _scrub(page, frame: int) -> None:
    page.locator("[data-testid=scrubber]").focus()
    page.keyboard.press("Home")
    for _ in range(frame):
        page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "(wanted) => document.querySelector('[data-testid=frame-features]')?.dataset.frame === wanted",
        arg=str(frame),
    )


def test_playwright_video_scrub_updates_features_at_the_tubelet_frame(tmp_path: Path):
    """mp4 clips: frame 10 is window_start 4 + t 3 * tubelet 2, with and without a scoreboard."""
    from playwright.sync_api import expect, sync_playwright

    run, clips = _video_run(tmp_path)
    bare = tmp_path / "bare"
    shutil.copytree(run, bare)
    (bare / "scoreboard.json").unlink()
    (bare / "features" / "clip_effects.parquet").unlink()
    artifacts = Path("/opt/cursor/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=_chrome(),
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--autoplay-policy=no-user-gesture-required"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        page.set_default_timeout(20000)
        with _view_server(run, clips) as url:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function(
                "() => document.querySelector('[data-testid=clip-title]')?.textContent === 'robot-grasp'"
            )
            expect(page.locator("[data-testid=frame-feature][data-feature-id='7']")).to_have_count(0)
            _scrub(page, 6)
            expect(page.locator("[data-testid=frame-readout]")).to_have_text("frame 6 · t 1 · window 0")
            expect(page.locator("[data-testid=frame-feature][data-feature-id='7']")).to_have_count(0)
            _scrub(page, 10)
            expect(page.locator("[data-testid=frame-readout]")).to_have_text("frame 10 · t 3 · window 0")
            row = page.locator("[data-testid=frame-feature][data-feature-id='7']")
            expect(row).to_have_attribute("data-label", "grasp")
            expect(row.locator("[data-testid=badge]")).to_have_text("real")
            ablation = row.locator("[data-testid=effect][data-kind=ablation]")
            assert float(ablation.get_attribute("data-effect")) == pytest.approx(0.4)
            cell = page.locator("[data-testid=heat-cell][data-h='1'][data-w='0']")
            expect(cell).to_have_attribute("data-value", "3")
            page.wait_for_function(
                """() => {
                    const video = document.querySelector('[data-testid=clip-video]');
                    const stage = document.querySelector('[data-testid=stage]');
                    return video && stage && stage.dataset.media === 'video'
                        && video.readyState >= 2
                        && Math.abs(video.currentTime - 1.25) < 0.2;
                }"""
            )
            page.evaluate("() => document.fonts.ready")
            page.screenshot(path=str(artifacts / "viewer-video.png"), full_page=True)

        with _view_server(bare, clips) as url:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function(
                "() => document.querySelector('[data-testid=clip-title]')?.textContent === 'robot-grasp'"
            )
            expect(page.locator("[data-testid=sae-summary]")).to_have_text("no scoreboard")
            _scrub(page, 10)
            row = page.locator("[data-testid=frame-feature][data-feature-id='7']")
            expect(row).to_have_attribute("data-label", "grasp")
            expect(row.locator("[data-testid=badge]")).to_have_text("unverified")
            expect(row.locator("[data-testid=activation]")).to_contain_text("3.000")
            expect(page.locator("[data-testid=effect]")).to_have_count(0)
            expect(page.locator("[data-testid=heat-cell][data-h='1'][data-w='0']")).to_have_attribute("data-value", "3")
            row.locator("[data-testid=feature-link]").click()
            page.wait_for_selector("[data-testid=feature-page]")
            expect(page.locator("[data-testid=feature-page] [data-testid=badge]")).to_have_text("unverified")
            expect(page.locator("[data-testid=scoreboard-missing]")).to_be_visible()
            expect(page.locator("[data-testid=ablation_effect]")).to_have_text("—")
            page.wait_for_function(
                """() => {
                    const image = document.querySelector('[data-testid=top-thumb-image]');
                    return image && image.complete && image.naturalWidth > 0;
                }"""
            )
            expect(page.locator("[data-testid=top-thumb]")).to_have_attribute("data-frame", "10")
            page.evaluate("() => document.fonts.ready")
            page.screenshot(path=str(artifacts / "viewer-no-scoreboard.png"), full_page=True)
        browser.close()
