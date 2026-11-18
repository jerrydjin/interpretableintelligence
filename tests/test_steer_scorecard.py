"""Steering scorecard (``ii steer-scorecard``): measured against matched controls, never a badge."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.steer_scorecard import (
    EFFECT_COLUMNS,
    effects_table,
    min_controls,
    select_features,
    summarize,
)
from interpretableintelligence.validate import validate_run
from interpretableintelligence.view import load_run as load_view_model

runner = CliRunner()
FAST = ["--controls", "19", "--bootstrap", "200"]


def _copy_run(demo, tmp_path: Path) -> Path:
    """A private copy of the shared fake run, so steering/ does not leak into other tests."""
    run = tmp_path / demo.run.name
    shutil.copytree(demo.run, run)
    return run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fake_run_scores_the_verified_features_and_leaves_the_scoreboard_alone(demo, tmp_path):
    run = _copy_run(demo, tmp_path)
    before = _sha(run / "scoreboard.json"), _sha(run / "split.json")
    result = runner.invoke(app, ["steer-scorecard", str(run), "--clips", str(demo.clips), *FAST])
    assert result.exit_code == 0, result.output
    assert "not a rollout finding" in result.output
    assert (_sha(run / "scoreboard.json"), _sha(run / "split.json")) == before

    card = json.loads((run / "steering" / "scorecard.json").read_text(encoding="utf-8"))
    scoreboard = json.loads((run / "scoreboard.json").read_text(encoding="utf-8"))
    real = {feature["id"]: feature["label"] for feature in scoreboard["features"] if feature["real"] is True}
    assert card["selection"] == "verified"
    assert set(card["feature_ids"]) == set(real)
    assert card["rollout_finding"] is False and card["target"] == "masked_prediction"
    assert [(row["feature_id"], row["kind"]) for row in card["rows"]] == [
        (fid, kind) for fid in card["feature_ids"] for kind in ("remove", "boost")
    ]
    for row in card["rows"]:
        assert row["real"] is True and row["label"] == real[row["feature_id"]]
        assert row["n_windows"] > 0 and 0 < row["coverage"] <= 1
        # The planted features move the prediction more than random directions of the same size.
        assert row["beats_controls"] is True
        assert row["excess"]["ci_low"] > 0
        assert row["ratio"]["ci_low"] <= row["ratio"]["mean"] <= row["ratio"]["ci_high"]
        assert row["ratio"]["mean"] > 1
        assert row["reliability"]["rate"] > row["reliability"]["chance"] == pytest.approx(1 / 20)
        assert row["p_value"] == pytest.approx(1 / 20)
    assert validate_run(run) == []

    payload = load_view_model(run, demo.clips).run_payload
    assert payload["steering"]["rows"] == card["rows"]


def test_validate_rebuilds_the_scorecard_from_the_effects(demo, tmp_path):
    run = _copy_run(demo, tmp_path)
    result = runner.invoke(app, ["steer-scorecard", str(run), "--clips", str(demo.clips), "--top", "1", *FAST])
    assert result.exit_code == 0, result.output
    path = run / "steering" / "scorecard.json"
    card = json.loads(path.read_text(encoding="utf-8"))
    assert len(card["feature_ids"]) == 1

    edited = json.loads(json.dumps(card))
    edited["rows"][0]["beats_controls"] = not edited["rows"][0]["beats_controls"]
    path.write_text(json.dumps(edited), encoding="utf-8")
    assert any("beats_controls" in error for error in validate_run(run))

    edited = json.loads(json.dumps(card))
    edited["rollout_finding"] = True
    path.write_text(json.dumps(edited), encoding="utf-8")
    assert any("rollout_finding" in error for error in validate_run(run))

    path.write_text(json.dumps(card), encoding="utf-8")
    assert validate_run(run) == []
    (run / "steering" / "effects.parquet").unlink()
    assert any("effects.parquet" in error for error in validate_run(run))


def test_too_few_controls_is_refused_because_no_edit_could_pass(demo, tmp_path):
    assert min_controls(0.05) == 19
    result = runner.invoke(app, ["steer-scorecard", str(_copy_run(demo, tmp_path)), "--clips", str(demo.clips), "--controls", "8"])
    assert result.exit_code == 1 and "at least 19" in result.output


def test_without_verified_features_it_refuses_unless_features_are_named():
    scoreboard = {"features": [{"id": 0, "real": False, "ablation_effect": 0.5}, {"id": 1, "real": False, "ablation_effect": 0.1}]}
    with pytest.raises(ValueError, match="--features"):
        select_features(scoreboard, None, 5, 2)
    assert select_features(scoreboard, [1, 1, 0], 5, 2) == ([1, 0], "requested")
    with pytest.raises(ValueError, match="unknown feature"):
        select_features(scoreboard, [2], 5, 2)
    ranked = {"features": [{"id": 0, "real": True, "ablation_effect": 0.1}, {"id": 1, "real": True, "ablation_effect": 0.4}, {"id": 2, "real": True, "ablation_effect": None}]}
    assert select_features(ranked, None, 2, 3) == ([1, 0], "verified")


def _synthetic(feature: float, controls: list[float], clips: int = 4) -> dict:
    columns = {name: [] for name in EFFECT_COLUMNS}
    for clip in range(clips):
        for kind in ("remove", "boost"):
            for arm, control, value in [("feature", -1, feature)] + [("control", k, v) for k, v in enumerate(controls)]:
                for name, item in zip(EFFECT_COLUMNS, (0, kind, arm, control, f"c{clip}", 0, value + 0.001 * clip, 0.5)):
                    columns[name].append(item)
    meta = {
        "schema_version": 1,
        "run_id": "r",
        "model_sha": "0" * 64,
        "layer": 0,
        "split_sha": "0" * 64,
        "adapter": "fake",
        "target": "masked_prediction",
        "metric": "relative_l2_change",
        "selection": "requested",
        "control": "",
        "sae_usable": True,
        "n_episodes": clips,
        "rollout_finding": False,
        "note": "",
        "feature_ids": [0],
        "n_controls": len(controls),
        "boost_factor": 3.0,
        "seed": 0,
        "bootstrap": 300,
        "confidence": 0.95,
        "alpha": 0.05,
        "n_windows": clips * 2,
    }
    return summarize(effects_table(columns), meta, {0: {"id": 0, "label": "x", "real": False}})


def test_summary_statistics_count_ties_against_the_feature():
    strong = _synthetic(0.5, [0.1] * 19)["rows"][0]
    assert strong["beats_controls"] is True
    assert strong["p_value"] == pytest.approx(1 / 20)
    assert strong["reliability"]["rate"] == 1.0 and strong["reliability"]["chance"] == pytest.approx(1 / 20)
    assert strong["ratio"]["mean"] == pytest.approx(np.mean([0.5 + 0.001 * c for c in range(4)]) / np.mean([0.1 + 0.001 * c for c in range(4)]))
    assert strong["coverage"] == 0.5 and strong["n_episodes"] == 4
    assert strong["abs_change"] == pytest.approx(0.5)

    tied = _synthetic(0.1, [0.1] * 19)["rows"][0]
    assert tied["beats_controls"] is False
    assert tied["p_value"] == 1.0 and tied["reliability"]["rate"] == 0.0

    one_episode = _synthetic(0.5, [0.1] * 19, clips=1)["rows"][0]
    assert one_episode["excess"]["ci_low"] is None  # no interval from one episode, so no verdict
    assert one_episode["beats_controls"] is False


def test_playwright_scorecard_page_and_feature_section(demo, tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    from test_steer import _chrome
    from test_view import _view_server

    chrome = _chrome()
    if chrome is None:
        pytest.skip("Chrome is not installed; the scorecard browser test cannot run")
    run = _copy_run(demo, tmp_path)
    result = runner.invoke(app, ["steer-scorecard", str(run), "--clips", str(demo.clips), *FAST])
    assert result.exit_code == 0, result.output
    card = json.loads((run / "steering" / "scorecard.json").read_text(encoding="utf-8"))
    first = card["feature_ids"][0]

    with _view_server(run, demo.clips) as url, sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + "#/steering", wait_until="domcontentloaded")
        page.wait_for_selector("[data-testid=steering-page]")
        sync_api.expect(page.locator("[data-testid=nav-steering]")).to_be_visible()
        rows = page.locator("[data-testid=steering-row]")
        sync_api.expect(rows).to_have_count(len(card["rows"]))
        sync_api.expect(rows.first.locator("[data-testid=badge]")).to_have_text("real")
        sync_api.expect(rows.first.locator("[data-testid=steering-verdict]")).to_have_text("Beats random edits")
        sync_api.expect(rows.first.locator("svg.steer-strip circle")).to_have_count(1)
        sync_api.expect(page.locator("[data-testid=steering-note]")).to_contain_text("does not change any badge")
        # The viewer's type rules: no monospace, no uppercase labels; raw numbers wait for the switch.
        styles = page.evaluate(
            """() => [...document.querySelectorAll('[data-testid=steering-page] *')].map((node) => {
                const style = getComputedStyle(node);
                return [style.fontFamily, style.textTransform];
            })"""
        )
        assert not any("mono" in family.lower() or transform == "uppercase" for family, transform in styles)
        sync_api.expect(page.locator("[data-testid=steering-table] td.num").first).to_be_hidden()
        page.locator("[data-testid=numbers-toggle]").click()
        sync_api.expect(page.locator("[data-testid=steering-table] td.num").first).to_be_visible()

        page.goto(url + f"#/features/{first}", wait_until="domcontentloaded")
        section = page.locator("[data-testid=steering-feature]")
        section.wait_for()
        sync_api.expect(section.locator("[data-testid=steering-row]")).to_have_count(2)
        sync_api.expect(page.locator("[data-testid=feature-page]")).to_have_attribute("data-badge", "real")

        page.set_viewport_size({"width": 390, "height": 800})
        page.goto(url + "#/steering", wait_until="domcontentloaded")
        page.wait_for_selector("[data-testid=steering-page]")
        assert page.evaluate("() => document.documentElement.scrollWidth") <= 390
        assert errors == []
        browser.close()


def test_matched_feature_controls_use_live_features_and_validate(demo, tmp_path, monkeypatch):
    """``--controls-from features``: each control is another live SAE feature's decoder direction, never a steered one."""
    import interpretableintelligence.steer_scorecard as scorecard_module

    # The fake SAE has 12 features, too few for 19 matched controls; allow 3 so the path runs end to end.
    monkeypatch.setattr(scorecard_module, "min_controls", lambda alpha: 1)
    run = _copy_run(demo, tmp_path)
    card = scorecard_module.execute_steer_scorecard(run, clips_dir=demo.clips, controls=3, bootstrap=50, controls_from="features", log=lambda *_: None)
    assert card["controls_from"] == "features" and "matched on train-split" in card["control"]
    assert set(card["control_features"]) == {str(fid) for fid in card["feature_ids"]}
    for entry in card["control_features"].values():
        ids = [control["id"] for control in entry["controls"]]
        assert len(ids) == 3 and not set(ids) & set(card["feature_ids"])
        assert all(control["rate"] > 0 and control["abs_cos"] <= 0.5 for control in entry["controls"])
        assert [control["distance"] for control in entry["controls"]] == sorted(control["distance"] for control in entry["controls"])
    assert validate_run(run) == []
    assert "matched-feature controls" in scorecard_module.format_report(card)

    broken = json.loads((run / "steering" / "scorecard.json").read_text(encoding="utf-8"))
    broken["control_features"] = None
    (run / "steering" / "scorecard.json").write_text(json.dumps(broken), encoding="utf-8")
    assert any("control_features" in error for error in validate_run(run))


def test_scorecards_from_before_matched_controls_still_validate(demo, tmp_path):
    run = _copy_run(demo, tmp_path)
    assert runner.invoke(app, ["steer-scorecard", str(run), "--clips", str(demo.clips), *FAST]).exit_code == 0
    path = run / "steering" / "scorecard.json"
    card = json.loads(path.read_text(encoding="utf-8"))
    assert card["controls_from"] == "random" and card["control_features"] is None
    for key in ("controls_from", "control_features"):
        card.pop(key)
    path.write_text(json.dumps(card), encoding="utf-8")
    assert validate_run(run) == []


def test_matching_prefers_similar_live_features_and_skips_near_copies():
    import torch

    from interpretableintelligence.sae import SAE
    from interpretableintelligence.steer_scorecard import match_control_features

    sae = SAE(4, 6)
    with torch.no_grad():
        dec = torch.eye(4)[:, [0, 1, 2, 3, 1, 2]].clone()
        dec[:, 5] = torch.tensor([0.9, 0.1, 0.0, 0.0])  # nearly feature 0's direction
        sae.W_dec.copy_(dec)
    stats = {
        "rate": np.array([0.5, 0.45, 0.05, 0.0, 0.5, 0.5]),
        "mean_act": np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0]),
        "dec_norm": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.906]),
        "n_train_tokens": np.array(100),
    }
    matched = match_control_features(Path("."), sae, [0], 2, stats=stats)["0"]
    # 3 is dead and 5 is a near copy of 0; 1 is closest on rate, then 4 (mean activation off by 2x) beats 2 (rate off by 10x).
    assert [control["id"] for control in matched["controls"]] == [1, 4]
    with pytest.raises(ValueError, match="fewer than --controls"):
        match_control_features(Path("."), sae, [0], 4, stats=stats)
    with pytest.raises(ValueError, match="never fires"):
        match_control_features(Path("."), sae, [3], 1, stats=stats)
