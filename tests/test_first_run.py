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
    cli_command,
    feature_route,
    guided_run,
    next_steps,
    pick_feature,
    unverified_on_frame,
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


def test_try_points_learners_at_real_clips(monkeypatch) -> None:
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    text = "\n".join(next_steps(None, None))
    assert "  ii get-clips --out clips/droid" in text
    assert "--no-score --steer" in text


def test_hints_say_uv_run_when_started_through_uv(monkeypatch) -> None:
    """`uv sync` does not put `ii` on PATH, so a learner who typed `uv run ii try` needs `uv run` again."""
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    assert cli_command() == "uv run ii"
    assert "  uv run ii get-clips --out clips/droid" in next_steps(None, None)
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH")
    assert cli_command() == "ii"


def test_next_steps_name_a_contrast_feature_that_is_on_screen() -> None:
    text = "\n".join(next_steps({"id": 9, "label": "stripe"}, {"id": 2, "label": "band", "badge": "unverified"}))
    assert "feature 9 (stripe)" in text
    assert "click band (marked unverified)" in text
    assert "flicker" not in text
    fallback = "\n".join(next_steps({"id": 9, "label": "stripe"}, None))
    assert "Features" in fallback and "click" not in fallback.split("2.", 1)[1].split("\n", 1)[0]


def test_guided_run_suggests_an_unverified_feature_from_the_opening_frame(demo, tmp_path: Path) -> None:
    from interpretableintelligence.view import load_run

    clips = tmp_path / "clips"
    out = tmp_path / "run"
    shutil.copytree(demo.clips, clips)
    shutil.copytree(demo.run, out)
    result = guided_run(out, clips, serve=False, say=lambda _line: None)
    model = load_run(out, clips)
    top = best_clip((model.feature(int(result["feature"]["id"])) or {})["top_clips"])
    contrast = unverified_on_frame(model, top)
    assert contrast is not None and contrast["badge"] != "real"
    on_screen = model.clips.get(str(top["clip_id"]))["frames"][int(top["frame"])]["features"]
    assert contrast["id"] in {feature["id"] for feature in on_screen}
    name = contrast.get("label") or f"feature {contrast['id']}"
    assert f"click {name} (marked unverified)" in "\n".join(result["next_steps"])
    assert unverified_on_frame(model, None) is None


def test_run_steer_opens_the_playground(demo) -> None:
    import json
    import urllib.request

    from interpretableintelligence.pipeline import _open_viewer

    lines: list[str] = []
    _model, viewer, url = _open_viewer(demo.run, demo.clips, "127.0.0.1", 0, False, lines.append, steer=True)
    try:
        with urllib.request.urlopen(url + "api/run") as response:
            payload = json.loads(response.read())
        assert payload["steer"]["enabled"] is True
    finally:
        viewer.close()


def test_playground_works_before_scoring(demo, tmp_path: Path) -> None:
    """--no-score --steer: the playground loads with every feature unverified; explain still refuses."""
    import pytest

    from interpretableintelligence.explain import ExplainError, load_run
    from interpretableintelligence.steer import Steerer

    out = tmp_path / "demo"
    shutil.copytree(demo.run, out)
    (out / "scoreboard.json").unlink()
    with pytest.raises(ExplainError):
        load_run(out)
    steerer = Steerer(out, demo.clips)
    assert steerer.sae_usable is False
    assert all(record["real"] is False for record in steerer.ctx.scoreboard["features"])
    assert not (out / "scoreboard.json").exists()


def test_run_without_a_clips_folder_says_how_to_get_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    missing = tmp_path / "droid"
    result = CliRunner().invoke(
        app, ["run", "hf", "--model", "facebook/dinov2-small", "--clips", str(missing), "--out", str(tmp_path / "run")]
    )
    assert result.exit_code == 2
    assert f"clips folder not found: {missing}" in result.output
    assert f"ii get-clips --out {missing}" in result.output
    fake = CliRunner().invoke(app, ["run", "fake", "--clips", str(missing), "--out", str(tmp_path / "run")])
    assert fake.exit_code == 2 and "ii generate-clips" in fake.output


def test_too_few_clips_explains_why_and_where_to_get_more(tmp_path: Path) -> None:
    from interpretableintelligence.adapters.fake import generate_clips

    clips = tmp_path / "clips"
    generate_clips(clips, n=3, seed=0)
    result = CliRunner().invoke(app, ["run", "fake", "--clips", str(clips), "--out", str(tmp_path / "run"), "--no-serve"])
    assert result.exit_code == 1
    assert "need at least 20 clips, found 3" in result.output
    assert "held out" in result.output and "ii get-clips" in result.output


def test_closing_lines_say_what_is_and_is_not_verified(demo) -> None:
    from interpretableintelligence.pipeline import _closing_lines

    unscored = "\n".join(_closing_lines(demo.run, "http://127.0.0.1:1/", score=False, steer=True))
    assert "Viewer: http://127.0.0.1:1/ (playground on)" in unscored
    assert "Nothing is verified yet" in unscored and "without --no-score" in unscored
    scored = "\n".join(_closing_lines(demo.run, None, score=True, steer=False))
    assert "Viewer:" not in scored
    assert f"{len(PLANTED)} of " in scored and "verified" in scored


def test_view_names_nearby_runs_for_a_mistyped_folder(demo, tmp_path: Path) -> None:
    import pytest

    from interpretableintelligence.view import load_run

    runs = tmp_path / "runs"
    shutil.copytree(demo.run, runs / "try")
    with pytest.raises(FileNotFoundError, match=r"Run folders in .*: try"):
        load_run(runs / "tyr")
    with pytest.raises(FileNotFoundError, match="ii run"):
        load_run(tmp_path / "elsewhere" / "nothing")
