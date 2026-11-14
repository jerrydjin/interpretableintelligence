"""`ii try`: the guided first run picks a verified feature and opens the viewer on it."""

from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.first_run import (
    PLANTED,
    _clips_ready,
    best_clip,
    feature_route,
    guided_run,
    pick_feature,
)


def test_pick_feature_prefers_the_strongest_verified_feature() -> None:
    board = {
        "features": [
            {"id": 0, "real": False, "ablation_effect": 9.0},
            {"id": 1, "real": True, "ablation_effect": 0.1},
            {"id": 2, "real": True, "ablation_effect": 0.3},
        ]
    }
    assert pick_feature(board)["id"] == 2


def test_pick_feature_falls_back_when_nothing_is_verified() -> None:
    board = {
        "features": [
            {"id": 0, "real": False, "ablation_effect": None},
            {"id": 1, "real": False, "ablation_effect": 0.2},
        ]
    }
    assert pick_feature(board)["id"] == 1
    assert pick_feature({"features": []}) is None


def test_best_clip_prefers_held_out() -> None:
    tops = [
        {"clip_id": "a", "split": "train", "frame": 1},
        {"clip_id": "b", "split": "heldout", "frame": 2},
    ]
    assert best_clip(tops)["clip_id"] == "b"
    assert best_clip(tops[:1])["clip_id"] == "a"
    assert best_clip([]) is None


def test_feature_route() -> None:
    assert (
        feature_route(3, {"clip_id": "clip 1", "frame": 7})
        == "#/clips/clip%201?frame=7&feature=3"
    )
    assert feature_route(3, None) == "#/features/3"


def test_clips_ready_needs_every_clip(tmp_path: Path) -> None:
    assert not _clips_ready(tmp_path)


def test_guided_run_reuses_a_finished_run(demo, tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    out = tmp_path / "run"
    shutil.copytree(demo.clips, clips)
    shutil.copytree(demo.run, out)
    lines: list[str] = []
    result = guided_run(out, clips, serve=False, say=lines.append)
    text = "\n".join(lines)
    assert "Practice clips are already in" in text
    assert "Reusing your earlier run" in text
    assert len(result["real"]) == len(PLANTED)
    feature = result["feature"]
    assert feature["real"] is True
    assert feature["label"] in PLANTED
    assert result["route"].startswith("#/clips/")
    assert result["route"].endswith(f"&feature={feature['id']}")


def test_try_is_listed_first_in_help() -> None:
    out = CliRunner().invoke(app, ["--help"]).output
    assert "try" in out
    assert out.index("try") < out.index("generate-clips")


def test_try_points_learners_at_real_clips() -> None:
    import inspect

    from interpretableintelligence import first_run

    source = inspect.getsource(first_run.guided_run)
    assert "ii get-clips" in source
    assert "--steer" in source


def test_run_steer_opens_the_playground(demo) -> None:
    import json
    import urllib.request

    from interpretableintelligence.pipeline import _open_viewer

    lines: list[str] = []
    _model, httpd, thread, url = _open_viewer(
        demo.run, demo.clips, "127.0.0.1", 0, False, lines.append, steer=True
    )
    try:
        with urllib.request.urlopen(url + "api/run") as response:
            payload = json.loads(response.read())
        assert payload["steer"]["enabled"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
