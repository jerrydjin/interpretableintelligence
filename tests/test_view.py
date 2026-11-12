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

from interpretableintelligence.artifacts import write_clip_effects, write_feature_acts, write_json, write_jsonl
from interpretableintelligence.cli import app
from interpretableintelligence.clips import file_sha256
from interpretableintelligence.view import (
    STATIC_DIR,
    badge_for,
    bind_server,
    encode_png,
    frame_index,
    frame_to_token,
    _read_moments,
    load_run,
    overlay_rect,
    render_frame,
    sampled_window_starts,
    token_to_frame,
    window_starts,
)

CAUSAL = {"stripe", "pulse", "corner"}
DECOY = {"flicker", "band", "speckle"}
runner = CliRunner()


def test_badge_is_real_only_for_json_true():
    assert badge_for(True) == "real"
    assert badge_for(True, "ablation-passed, control pending") == "real"
    assert badge_for(False) == "unverified"
    assert badge_for(None) == "unverified"
    assert badge_for(1) == "unverified"
    assert badge_for(False, "ablation-passed, control pending") == "status"
    assert badge_for(False, "estimate (ablation incomplete)") == "status"
    assert badge_for(False, "real") == "unverified"
    assert badge_for(False, "  ") == "unverified"


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


def test_extra_stride_places_a_167_frame_clip_and_a_short_clip(tmp_path: Path):
    """manifest.extra stride 2: frame = window_start + t * tubelet * stride.

    167 frames, 3 windows, frames 16, stride 2 → starts [0, 68, 135].
    (window 1, t 3) with tubelet 2 is frame 68 + 12 = 80.
    A 20-frame clip is shorter than span 32, so its stride falls back to 1.
    """
    grid = {"t": 8, "h": 2, "w": 2, "tubelet": 2, "patch_px": 16}
    starts, stride = sampled_window_starts(167, 3, frames=16, stride=2)
    assert starts == [0, 68, 135]
    assert stride == 2
    assert token_to_frame(1, 3, starts, grid, stride) == 80
    assert frame_to_token(80, starts, grid, stride) == (1, 3)
    assert frame_to_token(83, starts, grid, stride) == (1, 3)
    assert frame_to_token(79, starts, grid, stride) == (1, 2)

    short_starts, short_stride = sampled_window_starts(20, 3, frames=16, stride=2)
    assert short_stride == 1
    assert short_starts == [2]
    assert token_to_frame(0, 3, short_starts, grid, short_stride) == 8

    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "droid_ep000.mp4").write_bytes(b"named-mp4")
    (clips / "0.mp4").write_bytes(b"stem-for-clip-id-0")
    (clips / "short.mp4").write_bytes(b"short-mp4")
    run = tmp_path / "run"
    (run / "features").mkdir(parents=True)
    write_json(
        run / "manifest.json",
        {
            "run_id": "vjepa-real",
            "tool_version": "0.1.0",
            "model": {"adapter": "vjepa2", "name": "vjepa2", "sha": "ab" * 32, "random_init": False},
            "layer": 24,
            "d_model": 16,
            "grid": grid,
            "extra": {"frames": 16, "stride": 2, "windows": 3, "source": "droid"},
            "split": {"seed": 0, "heldout_frac": 0.0, "sha": "cd" * 32},
            "clips": [
                {"clip_id": 0, "name": "droid_ep000.mp4", "fps": 8.0, "n_frames": 167, "n_windows": 3, "split": "train"},
                {"clip_id": 1, "name": "short.mp4", "fps": 8.0, "n_frames": 20, "n_windows": 3, "split": "train"},
            ],
        },
    )
    write_jsonl(
        run / "features" / "features.jsonl",
        [{"id": 7, "label": "grasp", "label_model": "hand-v1"}],
    )
    write_feature_acts(
        run / "features" / "feature_acts.parquet",
        {
            "feature_id": [7, 7],
            "clip_id": ["0", "1"],
            "window": [1, 0],
            "t": [3, 3],
            "h": [1, 0],
            "w": [0, 1],
            "value": [3.0, 1.0],
        },
    )
    model = load_run(run, clips)
    long = model.clips["0"]
    assert long["window_starts"] == [0, 68, 135]
    assert long["stride"] == 2
    assert long["frames"][80]["window"] == 1
    assert long["frames"][80]["t"] == 3
    assert [feature["label"] for feature in long["frames"][80]["features"]] == ["grasp"]
    assert long["frames"][80]["features"] == long["frames"][83]["features"]
    assert long["frames"][79]["t"] == 2
    assert long["frames"][68]["features"] == []
    assert model.loaded["0"].path == clips / "droid_ep000.mp4"
    grasp = next(feature for feature in model.run_payload["features"] if feature["label"] == "grasp")
    detail = model.feature(grasp["id"])
    assert detail["top_clips"][0]["clip_id"] == "0"
    assert detail["top_clips"][0]["frame"] == 80

    short = model.clips["1"]
    assert short["stride"] == 1
    assert short["window_starts"] == [2]
    assert short["frames"][8]["window"] == 0
    assert short["frames"][8]["t"] == 3
    assert [feature["label"] for feature in short["frames"][8]["features"]] == ["grasp"]
    assert short["frames"][1]["t"] is None
    assert model.loaded["1"].path == clips / "short.mp4"


def test_vjepa_overlay_is_the_centre_square_of_a_292_by_519_frame():
    """Wide DROID frames are centre-cropped. hw [292, 519] → side 292 at x 113.5."""
    hw = [292, 519]
    rect = overlay_rect(hw, None, "vjepa2")
    assert rect == {"x": 113.5, "y": 0.0, "width": 292.0, "height": 292.0}
    tall = overlay_rect([519, 292], None, "vjepa2")
    assert tall == {"x": 0.0, "y": 113.5, "width": 292.0, "height": 292.0}
    explicit = {"x": 10, "y": 4, "width": 200, "height": 180}
    assert overlay_rect(hw, {"crop": explicit}, "vjepa2") == {"x": 10.0, "y": 4.0, "width": 200.0, "height": 180.0}
    assert overlay_rect(hw, {"crop": [8, 2, 100, 90]}, "fake") == {"x": 8.0, "y": 2.0, "width": 100.0, "height": 90.0}
    assert overlay_rect(hw, None, "fake") is None


def test_wide_clip_payload_carries_the_centre_crop(tmp_path: Path):
    run = tmp_path / "run"
    (run / "features").mkdir(parents=True)
    write_json(
        run / "manifest.json",
        {
            "run_id": "droid-wide",
            "model": {"adapter": "vjepa2", "name": "vjepa2", "sha": "ab" * 32},
            "layer": 24,
            "grid": {"t": 4, "h": 2, "w": 2, "tubelet": 1},
            "clips": [
                {
                    "clip_id": 0,
                    "name": "droid_ep000.mp4",
                    "fps": 8.0,
                    "n_frames": 8,
                    "split": "train",
                    "hw": [292, 519],
                }
            ],
        },
    )
    write_feature_acts(
        run / "features" / "feature_acts.parquet",
        {"feature_id": [7], "clip_id": ["0"], "window": [0], "t": [0], "h": [0], "w": [1], "value": [1.5]},
    )
    model = load_run(run, None)
    clip = model.clips["0"]
    assert clip["hw"] == [292.0, 519.0]
    assert clip["crop"] == {"x": 113.5, "y": 0.0, "width": 292.0, "height": 292.0}


def test_playwright_heatmap_covers_the_centre_square(tmp_path: Path):
    clips = tmp_path / "clips"
    _encode_video(clips / "droid_ep000.mp4", color="0x335544", size="320x180", frames=8, fps=8)
    run = tmp_path / "run"
    (run / "features").mkdir(parents=True)
    write_json(
        run / "manifest.json",
        {
            "run_id": "droid-wide",
            "model": {"adapter": "vjepa2", "name": "vjepa2", "sha": "ab" * 32},
            "layer": 24,
            "grid": {"t": 4, "h": 2, "w": 2, "tubelet": 1},
            "clips": [
                {
                    "clip_id": 0,
                    "name": "droid_ep000.mp4",
                    "fps": 8.0,
                    "n_frames": 8,
                    "split": "train",
                    "hw": [292, 519],
                }
            ],
        },
    )
    write_jsonl(run / "features" / "features.jsonl", [{"id": 7, "label": "grasp"}])
    write_feature_acts(
        run / "features" / "feature_acts.parquet",
        {"feature_id": [7], "clip_id": ["0"], "window": [0], "t": [0], "h": [0], "w": [1], "value": [1.5]},
    )
    shots = tmp_path / "shots"
    shots.mkdir()
    from playwright.sync_api import sync_playwright

    with _view_server(run, clips) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=_chrome(),
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(url + "#/clips/0?frame=0&feature=7", wait_until="domcontentloaded")
        page.wait_for_selector("[data-testid=heatmap][data-crop-x='113.5']")
        page.wait_for_selector("[data-testid=heat-cell]")
        boxes = page.evaluate(
            """() => {
                const stage = document.querySelector('[data-testid=stage]').getBoundingClientRect();
                const heat = document.querySelector('[data-testid=heatmap]').getBoundingClientRect();
                const shade = document.querySelector('[data-testid=crop-shade]');
                const shadeBox = shade.getBoundingClientRect();
                return {
                    stage: {x: stage.x, y: stage.y, width: stage.width, height: stage.height},
                    heat: {x: heat.x, y: heat.y, width: heat.width, height: heat.height},
                    shade: {x: shadeBox.x, y: shadeBox.y, width: shadeBox.width, height: shadeBox.height},
                    shadeHidden: shade.hidden,
                    shadow: getComputedStyle(shade).boxShadow,
                };
            }"""
        )
        stage = boxes["stage"]
        heat = boxes["heat"]
        assert heat["x"] - stage["x"] == pytest.approx(stage["width"] * 113.5 / 519, abs=1.5)
        assert heat["y"] - stage["y"] == pytest.approx(0, abs=1.5)
        assert heat["width"] == pytest.approx(stage["width"] * 292 / 519, abs=1.5)
        assert heat["height"] == pytest.approx(stage["height"], abs=1.5)
        assert boxes["shadeHidden"] is False
        assert boxes["shade"]["x"] == pytest.approx(heat["x"], abs=1.0)
        assert boxes["shade"]["width"] == pytest.approx(heat["width"], abs=1.0)
        assert "rgba" in boxes["shadow"]
        page.screenshot(path=str(shots / "viewer-crop.png"), full_page=True)
        browser.close()


def test_nearest_token_keeps_frame_65_when_the_window_starts_at_55():
    """Stride 2, tubelet 2: frame 65 is t=(65-55)//4 = 2, anchor 63, not the window start."""
    grid = {"t": 8, "h": 16, "w": 16, "tubelet": 2}
    assert frame_to_token(65, [55], grid, 2) == (0, 2)
    assert token_to_frame(0, 2, [55], grid, 2) == 63
    assert token_to_frame(0, 0, [55], grid, 2) == 55
    assert frame_to_token(20, [55], grid, 2) == (0, None)

    starts, stride = sampled_window_starts(167, 3, frames=16, stride=2)
    assert starts == [0, 68, 135]
    assert stride == 2
    assert frame_to_token(20, starts, grid, stride) == (0, 5)
    assert frame_to_token(40, starts, grid, stride) == (0, 7)
    assert frame_to_token(65, starts, grid, stride) == (1, 0)
    assert token_to_frame(1, 0, starts, grid, stride) == 68


def test_playwright_url_feature_and_same_clip_frames(tmp_path: Path):
    clips = tmp_path / "clips"
    clips.mkdir()
    run = tmp_path / "run"
    grid = {"t": 8, "h": 2, "w": 2, "tubelet": 2}
    (run / "features").mkdir(parents=True)
    write_json(
        run / "manifest.json",
        {
            "run_id": "droid-url",
            "model": {"adapter": "vjepa2", "name": "vjepa2", "sha": "ab" * 32},
            "layer": 24,
            "grid": grid,
            "extra": {"frames": 16, "stride": 2, "windows": 3},
            "clips": [
                {"clip_id": 49, "fps": 8.0, "n_frames": 167, "n_windows": 3, "split": "train"},
                {
                    "clip_id": 55,
                    "fps": 8.0,
                    "n_frames": 120,
                    "n_windows": 1,
                    "split": "train",
                    "window_starts": [55],
                    "stride": 2,
                },
            ],
        },
    )
    write_jsonl(
        run / "features" / "features.jsonl",
        [{"id": 7, "label": "top"}, {"id": 3, "label": "asked"}, {"id": 270, "label": "quiet"}],
    )
    write_feature_acts(
        run / "features" / "feature_acts.parquet",
        {
            "feature_id": [7, 3, 3, 3, 3],
            "clip_id": ["49", "49", "49", "49", "55"],
            "window": [0, 0, 0, 0, 0],
            "t": [0, 0, 5, 7, 2],
            "h": [0, 0, 0, 0, 1],
            "w": [0, 1, 0, 0, 0],
            "value": [5.0, 1.0, 2.5, 4.0, 9.0],
        },
    )
    from playwright.sync_api import expect, sync_playwright

    with _view_server(run, clips) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=_chrome(),
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(url + "#/clips/49?frame=0&feature=3", wait_until="domcontentloaded")
        page.wait_for_selector("[data-testid=heatmap][data-feature-id='3']")
        expect(page.locator("[data-testid=frame-feature][data-feature-id='3']")).to_have_attribute("data-selected", "true")
        expect(page.locator("[data-testid=frame-feature][data-feature-id='7']")).to_have_attribute("data-selected", "false")
        page.wait_for_function(
            """() => {
                const cell = document.querySelector("[data-testid=heat-cell][data-h='0'][data-w='1']");
                return cell && cell.dataset.value === "1";
            }"""
        )

        page.goto(url + "#/clips/49?frame=0&feature=270", wait_until="domcontentloaded")
        expect(page.locator("[data-testid=heatmap-label]")).to_have_text("feature 270 not active at this frame")
        expect(page.locator("[data-testid=heatmap]")).to_have_attribute("data-feature-id", "270")
        expect(page.locator("[data-testid=frame-feature][data-selected='true']")).to_have_count(0)
        page.wait_for_function(
            """() => document.querySelectorAll("[data-testid=heat-cell]").length === 4
                && [...document.querySelectorAll("[data-testid=heat-cell]")].every((cell) => Number(cell.dataset.value) === 0)"""
        )

        page.evaluate("() => { location.hash = '#/clips/49?frame=20&feature=3'; }")
        page.wait_for_function(
            """() => document.querySelector("[data-testid=frame-readout]")?.dataset.frame === "20"
                && document.querySelector("[data-testid=frame-readout]")?.dataset.t === "5" """
        )
        page.wait_for_function(
            """() => {
                const cell = document.querySelector("[data-testid=heat-cell][data-h='0'][data-w='0']");
                return cell && cell.dataset.value === "2.5";
            }"""
        )

        page.evaluate("() => { location.hash = '#/clips/49?frame=40&feature=3'; }")
        page.wait_for_function(
            """() => document.querySelector("[data-testid=frame-readout]")?.dataset.frame === "40"
                && document.querySelector("[data-testid=frame-readout]")?.dataset.t === "7" """
        )
        page.wait_for_function(
            """() => document.querySelectorAll("[data-testid=heat-cell]").length === 4"""
        )
        page.wait_for_function(
            """() => {
                const cell = document.querySelector("[data-testid=heat-cell][data-h='0'][data-w='0']");
                return cell && cell.dataset.value === "4";
            }"""
        )

        page.evaluate("() => { location.hash = '#/clips/55?frame=65&feature=3'; }")
        page.wait_for_function(
            """() => document.querySelector("[data-testid=frame-readout]")?.dataset.frame === "65"
                && document.querySelector("[data-testid=frame-readout]")?.dataset.t === "2"
                && document.querySelector("[data-testid=frame-readout]")?.textContent.startsWith("frame 65")"""
        )
        page.wait_for_function(
            """() => {
                const cell = document.querySelector("[data-testid=heat-cell][data-h='1'][data-w='0']");
                return cell && cell.dataset.value === "9";
            }"""
        )
        browser.close()


def _status_run(tmp_path: Path) -> Path:
    clips = tmp_path / "clips"
    clips.mkdir()
    run = tmp_path / "run"
    (run / "features").mkdir(parents=True)
    write_json(
        run / "manifest.json",
        {
            "run_id": "status-demo",
            "model": {"adapter": "fake", "name": "fake", "sha": "ab" * 32},
            "layer": 1,
            "grid": {"t": 8, "h": 2, "w": 2, "tubelet": 2},
            "clips": [
                {"clip_id": 67, "name": "droid_ep068.mp4", "fps": 8.0, "n_frames": 40, "split": "heldout"},
            ],
        },
    )
    write_json(
        run / "scoreboard.json",
        {
            "features": [
                {"id": 1, "label": "passed", "real": False, "status": "ablation-passed, control pending"},
                {"id": 2, "label": "close", "real": False},
                {"id": 3, "label": "done", "real": True, "status": "ablation-passed, control pending"},
                {"id": 4, "label": "estimate", "real": False, "status": "estimate (ablation incomplete)"},
                {"id": 5, "label": "fake", "real": False, "status": "real"},
                {"id": 6, "label": "loud", "real": False},
            ]
        },
    )
    write_jsonl(run / "features" / "features.jsonl", [{"id": index, "label": ""} for index in range(1, 7)])
    write_feature_acts(
        run / "features" / "feature_acts.parquet",
        {
            "feature_id": [6, 1, 2, 5, 4, 3],
            "clip_id": ["67"] * 6,
            "window": [0] * 6,
            "t": [7] * 6,
            "h": [0] * 6,
            "w": [0] * 6,
            "value": [3.0, 1.19, 1.20, 1.19, 1.14, 0.2],
        },
    )
    moments = []
    for index in range(12):
        moments.append(
            {
                "clip_id": 100 + index,
                "clip_name": f"droid_ep{101 + index:03d}",
                "window": 0,
                "frame_start": 0,
                "frame_end": 4,
                "loss": float(12 - index),
                "task_text": None,
                "loss_source": "heldout",
            }
        )
    moments[1] = {
        "clip_id": 67,
        "clip_name": "droid_ep068",
        "window": 1,
        "frame_start": 10,
        "frame_end": 20,
        "loss": 11.25,
        "task_text": "pick up the cup",
        "loss_source": "heldout",
    }
    write_json(run / "heldout_loss.json", moments)
    return run


def test_heldout_loss_accepts_a_windows_object(tmp_path: Path):
    path = tmp_path / "heldout_loss.json"
    write_json(
        path,
        {
            "windows": [
                {
                    "clip_id": 67,
                    "clip_name": "droid_ep068",
                    "window": 1,
                    "frame_start": 10,
                    "frame_end": 20,
                    "loss": 1.5,
                    "task_text": "pick up the cup",
                    "loss_source": "heldout",
                }
            ]
        },
    )
    moments = _read_moments(path)
    assert len(moments) == 1
    assert moments[0]["clip_id"] == "67"
    assert moments[0]["clip_name"] == "droid_ep068"
    assert moments[0]["frame"] == 15
    assert moments[0]["loss"] == 1.5


def test_status_badges_sort_and_hardest_moments_use_clip_id(tmp_path: Path):
    model = load_run(_status_run(tmp_path), None)
    by_label = {feature["label"]: feature for feature in model.run_payload["features"]}
    assert by_label["done"]["badge"] == "real"
    assert by_label["done"]["status"] == "ablation-passed, control pending"
    assert by_label["passed"]["badge"] == "status"
    assert by_label["passed"]["status"] == "ablation-passed, control pending"
    assert by_label["estimate"]["badge"] == "status"
    assert by_label["fake"]["badge"] == "unverified"
    assert by_label["fake"]["status"] is None
    assert by_label["close"]["badge"] == "unverified"
    order = [feature["id"] for feature in model.clips["67"]["frames"][15]["features"]]
    assert order == [6, 1, 2, 5, 4, 3]
    moments = model.run_payload["moments"]
    assert len(moments) == 12
    assert moments[0]["loss"] == 12
    assert moments[1]["clip_id"] == "67"
    assert moments[1]["clip_name"] == "droid_ep068"
    assert moments[1]["frame"] == 15
    assert [moment["clip_id"] for moment in moments].count("68") == 0


def test_playwright_status_badge_and_hardest_moments(tmp_path: Path):
    from playwright.sync_api import expect, sync_playwright

    with _view_server(_status_run(tmp_path), tmp_path / "clips") as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=_chrome(),
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        page.goto(url + "#/features", wait_until="domcontentloaded")
        page.wait_for_selector("[data-testid=hardest-moments]")
        rows = page.locator("[data-testid=hardest-moment]")
        expect(rows).to_have_count(10)
        expect(rows.nth(0).locator(".moment-loss")).to_have_text("12.0000")
        moment = rows.nth(1)
        expect(moment).to_have_attribute("data-clip-id", "67")
        expect(moment).to_have_attribute("href", "#/clips/67?frame=15")
        expect(moment.locator(".moment-name")).to_have_text("droid_ep068")
        expect(moment.locator(".moment-task")).to_have_text("pick up the cup")
        expect(moment.locator(".moment-range")).to_have_text("10–20")
        expect(moment.locator(".moment-loss")).to_have_text("11.2500")
        expect(page.locator("[data-testid=feature-index-link][data-feature-id='1'] [data-testid=badge]")).to_have_text(
            "ablation-passed, control pending"
        )
        expect(page.locator("[data-testid=feature-index-link][data-feature-id='1'] [data-testid=badge]")).to_have_attribute(
            "data-kind", "status"
        )
        expect(page.locator("[data-testid=feature-index-link][data-feature-id='3'] [data-testid=badge]")).to_have_text("real")
        expect(page.locator("[data-testid=feature-index-link][data-feature-id='4'] [data-testid=badge]")).to_have_text(
            "estimate (ablation incomplete)"
        )
        expect(page.locator("[data-testid=feature-index-link][data-feature-id='5'] [data-testid=badge]")).to_have_text("unverified")

        moment.click()
        page.wait_for_function(
            """() => document.querySelector("[data-testid=clip-title]")?.textContent === "67"
                && document.querySelector("[data-testid=frame-readout]")?.dataset.frame === "15" """
        )
        expect(page.locator("[data-testid=clip-task]")).to_have_text("pick up the cup")
        mark = page.locator("[data-testid=moment-range]")
        expect(mark).to_have_attribute("data-start", "10")
        expect(mark).to_have_attribute("data-end", "20")
        ids = page.locator("[data-testid=frame-feature]").evaluate_all("nodes => nodes.map((node) => node.dataset.featureId)")
        assert ids == ["6", "1", "2", "5", "4", "3"]
        page.evaluate("() => { location.hash = '#/clips/67?frame=0'; }")
        page.wait_for_function("""() => document.querySelector("[data-testid=frame-readout]")?.dataset.frame === "0" """)
        expect(page.locator("[data-testid=clip-task]")).to_be_hidden()
        expect(page.locator("[data-testid=moment-range]")).to_have_count(1)
        browser.close()


def test_viewer_source_does_not_import_adapters():
    source = Path(inspect_view_path()).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    forbidden = [name for name in modules if name.startswith("interpretableintelligence.adapters") or name == "torch"]
    assert forbidden == []


def inspect_view_path() -> Path:
    import interpretableintelligence.view as view

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
    detail = model.feature(stripe["id"])
    assert detail["top_clips"][0]["clip_id"] == "clip-000"
    assert detail["top_clips"][0]["frame"] == 2

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
    heat = model.heatmap("clip-000", 2, stripe["id"])
    assert heat[0][1] == pytest.approx(1.25)
    assert heat[1][0] == 0
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
    assert model.heatmap("robot-grasp", 10, fired["grasp"]["id"])[1][0] == pytest.approx(3.0)
    assert fired["grasp"]["effects"][0]["effect_kind"] == "ablation"
    assert model.clips["robot-absent"]["pixels"] == "missing"
    assert model.clips["robot-absent"]["media"] == "none"
    assert grasp["pixels"] == "ok"
    grasp_row = next(feature for feature in model.run_payload["features"] if feature["label"] == "grasp")
    reach_row = next(feature for feature in model.run_payload["features"] if feature["label"] == "reach")
    assert grasp_row["badge"] == "real"
    assert model.feature(grasp_row["id"])["top_clips"][0]["frame"] == 10

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
    # Typer's rich highlighter styles the two dashes of a long option separately,
    # so a color terminal splits "--clips" with ANSI codes. Compare the visible text.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", help_text.stdout)
    assert "--clips" in plain
    assert "--no-open" in plain
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
    for candidate in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium", "/usr/local/bin/google-chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"):
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
            "--no-serve",
            "--no-open",
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
            "interpretableintelligence",
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


def test_playwright_scrub_shows_real_ablation_and_unverified_decoy(fresh_fake_run, viewer_url: str, tmp_path: Path):
    from playwright.sync_api import expect, sync_playwright

    clips, run = fresh_fake_run
    del clips
    moment = _moment(run)
    shots = tmp_path / "shots"
    shots.mkdir()

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
        page.wait_for_selector("[data-testid=heat-cell]")
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
        page.screenshot(path=str(shots / "viewer-clip.png"), full_page=True)

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
        page.screenshot(path=str(shots / "viewer-feature.png"), full_page=True)
        browser.close()


@contextmanager
def _view_server(run: Path, clips: Path):
    err_path = run.parent / f"viewer-{run.name}.err"
    err = err_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "interpretableintelligence",
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
    shots = tmp_path / "shots"
    shots.mkdir()

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
            page.wait_for_selector("[data-testid=heat-cell][data-h='1'][data-w='0']")
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
            page.screenshot(path=str(shots / "viewer-video.png"), full_page=True)

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
            page.wait_for_selector("[data-testid=heat-cell][data-h='1'][data-w='0']")
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
            page.screenshot(path=str(shots / "viewer-no-scoreboard.png"), full_page=True)
        browser.close()


def test_candidates_are_listed_before_the_other_features(tmp_path: Path):
    run = tmp_path / "run"
    (run / "features" / "viz").mkdir(parents=True)
    write_json(
        run / "manifest.json",
        {
            "run_id": "cand",
            "model": {"name": "tiny", "sha": "abc"},
            "layer": 3,
            "grid": {"t": 2, "h": 2, "w": 2, "tubelet": 1},
            "clips": [{"clip_id": "c0", "n_frames": 2, "split": "train", "fps": 8}],
        },
    )
    write_feature_acts(
        run / "features" / "feature_acts.parquet",
        {
            "feature_id": [0, 2],
            "clip_id": ["c0", "c0"],
            "window": [0, 0],
            "t": [0, 0],
            "h": [0, 0],
            "w": [0, 1],
            "value": [0.2, 0.9],
        },
    )
    write_jsonl(
        run / "features" / "features.jsonl",
        [{"id": 0, "label": "zero"}, {"id": 1, "label": "one"}],
    )
    (run / "features" / "candidates.json").write_text(
        json.dumps(
            [
                {"feature_id": 2, "distinct_clips_top20": 1, "freq": 1, "max_act": 0.9},
                {"feature_id": 0, "distinct_clips_top20": 1, "freq": 1, "max_act": 0.2},
            ]
        ),
        encoding="utf-8",
    )
    (run / "features" / "viz" / "feature_00002.png").write_bytes(b"\x89PNG\r\n")
    model = load_run(run, None)
    rows = model.run_payload["features"]
    assert [row["id"] for row in rows] == [2, 0, 1]
    assert [row["candidate"] for row in rows] == [True, True, False]
    assert "top_clips" not in rows[0]
    detail = model.feature(2)
    assert detail["viz"] is True
    assert detail["top_clips"][0]["clip_id"] == "c0"
    assert detail["top_clips"][0]["value"] == pytest.approx(0.9)


def _act_value(clip: int, window: int, tubelet: int, height: int, width: int) -> float:
    return float(((clip * 10007 + window * 97 + tubelet * 13 + height * 3 + width) % 997) * 0.001)


def _write_ten_million_acts(path: Path) -> int:
    """200 features × 40 clips × 4 × 8 × 8 × 5 = 10,240,000 rows, sorted by feature_id."""
    import pyarrow as pa

    n_features = 200
    clip_id, window, tubelet, height, width = (
        array.reshape(-1).astype(np.int32)
        for array in np.meshgrid(
            np.arange(40, dtype=np.int32),
            np.arange(4, dtype=np.int32),
            np.arange(8, dtype=np.int32),
            np.arange(8, dtype=np.int32),
            np.arange(5, dtype=np.int32),
            indexing="ij",
        )
    )
    base = (
        (clip_id * 10007 + window * 97 + tubelet * 13 + height * 3 + width) % 997
    ).astype(np.float32) * np.float32(0.001)
    planted = base.copy()
    planted[(clip_id == 3) & (window == 1) & (tubelet == 2) & (height == 1) & (width == 0)] = np.float32(999.0)
    schema = pa.schema(
        [
            ("feature_id", pa.int32()),
            ("clip_id", pa.int32()),
            ("window", pa.int32()),
            ("t", pa.int32()),
            ("h", pa.int32()),
            ("w", pa.int32()),
            ("value", pa.float32()),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    batch: list = []
    pending = 0
    total = 0
    for feature_id in range(n_features):
        table = pa.table(
            {
                "feature_id": pa.array(np.full(clip_id.shape[0], feature_id, dtype=np.int32), type=pa.int32()),
                "clip_id": pa.array(clip_id, type=pa.int32()),
                "window": pa.array(window, type=pa.int32()),
                "t": pa.array(tubelet, type=pa.int32()),
                "h": pa.array(height, type=pa.int32()),
                "w": pa.array(width, type=pa.int32()),
                "value": pa.array(planted if feature_id == 7 else base, type=pa.float32()),
            }
        )
        batch.append(table)
        pending += table.num_rows
        if pending >= 1_048_576:
            writer.write_table(pa.concat_tables(batch))
            total += pending
            batch, pending = [], 0
    if batch:
        writer.write_table(pa.concat_tables(batch))
        total += pending
    writer.close()
    return total


def _vmhwm_kb(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    raise AssertionError(f"VmHWM missing for pid {pid}")


@pytest.mark.slow
def test_ten_million_acts_bind_under_five_seconds_and_one_point_five_gb(tmp_path: Path):
    run = tmp_path / "run"
    acts = run / "features" / "feature_acts.parquet"
    total = _write_ten_million_acts(acts)
    assert total >= 10_000_000
    assert pq.ParquetFile(acts).metadata.num_rows >= 10_000_000
    write_json(
        run / "manifest.json",
        {
            "run_id": "large",
            "model": {"adapter": "fake", "name": "large", "sha": "deadbeef"},
            "layer": 16,
            "grid": {"t": 8, "h": 8, "w": 5, "tubelet": 2, "patch_px": 4},
            "clips": [
                {"clip_id": index, "n_frames": 64, "split": "train", "fps": 8, "name": f"clip-{index:03d}.mp4"}
                for index in range(40)
            ],
        },
    )
    label = "planted-token-" + ("x" * 2800)
    with (run / "features" / "features.jsonl").open("w", encoding="utf-8") as handle:
        for feature_id in range(8000):
            text = "planted" if feature_id == 7 else label
            handle.write(json.dumps({"id": feature_id, "label": text, "blob": "y" * 200}) + "\n")
    (run / "features" / "candidates.json").write_text(
        json.dumps(
            [
                {"feature_id": 7, "distinct_clips_top20": 40, "freq": 1, "max_act": 999},
                {"feature_id": 11, "distinct_clips_top20": 40, "freq": 1, "max_act": 0.01},
            ]
        ),
        encoding="utf-8",
    )
    script = tmp_path / "serve_large.py"
    script.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "from interpretableintelligence.view import bind_server, load_run\n"
        "def hwm():\n"
        "    for line in Path('/proc/self/status').read_text().splitlines():\n"
        "        if line.startswith('VmHWM:'):\n"
        "            return int(line.split()[1])\n"
        "    return 0\n"
        "started = time.perf_counter()\n"
        "model = load_run(Path(sys.argv[1]), None)\n"
        "server = bind_server(model, '127.0.0.1', 0)\n"
        "print(json.dumps({'elapsed': time.perf_counter() - started, 'hwm_kb': hwm(), 'port': server.server_address[1]}), flush=True)\n"
        "server.serve_forever()\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script), str(run)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], 20)
        assert ready, "viewer did not bind within 20s"
        line = proc.stdout.readline()
        assert proc.poll() is None, proc.stderr.read() if proc.stderr else line
        report = json.loads(line)
        assert report["elapsed"] < 5.0, report
        limit_kb = int(1.5 * 1024 * 1024)
        assert report["hwm_kb"] < limit_kb, report
        base = f"http://127.0.0.1:{report['port']}"
        with urllib.request.urlopen(base + "/api/run", timeout=30) as response:
            payload = json.load(response)
        assert [row["id"] for row in payload["features"][:2]] == [7, 11]
        assert payload["features"][0]["candidate"] is True
        assert payload["scoreboard"] is False
        with urllib.request.urlopen(base + "/api/clips/3", timeout=120) as response:
            clip = json.load(response)
        fired = {row["id"]: row for row in clip["frames"][20]["features"]}
        assert fired[7]["label"] == "planted"
        expected = sum(
            999.0 if (h, w) == (1, 0) else _act_value(3, 1, 2, h, w) for h in range(8) for w in range(5)
        )
        assert fired[7]["activation"] == pytest.approx(expected)
        with urllib.request.urlopen(base + "/api/clips/3/heatmap?frame=20&feature=7", timeout=60) as response:
            heat = json.load(response)["heatmap"]
        assert heat[1][0] == pytest.approx(999.0)
        with urllib.request.urlopen(base + "/api/features/7", timeout=60) as response:
            detail = json.load(response)
        assert detail["top_clips"][0]["clip_id"] == "3"
        assert detail["top_clips"][0]["frame"] == 20
        assert detail["top_clips"][0]["value"] == pytest.approx(999.0)
        peak_kb = _vmhwm_kb(proc.pid)
        assert peak_kb < limit_kb, peak_kb
        timing = {
            "startup_s": report["elapsed"],
            "startup_hwm_kb": report["hwm_kb"],
            "peak_hwm_kb": peak_kb,
            "rows": total,
            "parquet_bytes": acts.stat().st_size,
        }
        (tmp_path / "large-view-timing.json").write_text(json.dumps(timing), encoding="utf-8")
        print(json.dumps(timing), flush=True)
    finally:
        proc.kill()
        proc.wait(timeout=10)
