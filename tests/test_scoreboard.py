"""Acceptance: the fake adapter's three causal features are real, and the decoys are not."""

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from physint.cli import app
from physint.validate import validate_run

CAUSAL = {"stripe", "pulse", "corner"}
DECOY = {"flicker", "band", "speckle"}
runner = CliRunner()


def test_planted_causal_features_are_real_and_decoys_are_not(demo):
    demo_run = demo.run
    scoreboard = json.loads((demo_run / "scoreboard.json").read_text())
    features = scoreboard["features"]
    rendered = json.dumps(features, indent=2)
    assert scoreboard["sae"]["usable"] is True
    assert scoreboard["sae"]["loss_recovered"] >= 0.70

    real = [feature for feature in features if feature["real"]]
    assert len(real) == 3, rendered
    assert {feature["label"] for feature in real} == CAUSAL, rendered
    for feature in real:
        assert feature["distinct_clips_top20"] >= 5
        assert feature["label_holds_heldout"] is True
        assert feature["random_percentile"] >= 95
        assert feature["absent_in_random_init"] is True
        assert feature["n_random"] >= 100

    decoys = [feature for feature in features if feature["label"] in DECOY]
    assert {feature["label"] for feature in decoys} == DECOY, rendered
    for feature in decoys:
        assert feature["real"] is False
        assert feature["distinct_clips_top20"] >= 5
        assert feature["label_holds_heldout"] is True
        assert feature["absent_in_random_init"] is True
        assert feature["random_percentile"] < 95
        assert feature["n_random"] >= 100
    assert min(feature["ablation_effect"] for feature in real) > max(
        feature["ablation_effect"] for feature in decoys
    )


def test_validate_rejects_a_loosened_real_flag(demo, tmp_path: Path):
    copied = tmp_path / "demo"
    shutil.copytree(demo.run, copied)
    scoreboard_path = copied / "scoreboard.json"
    scoreboard = json.loads(scoreboard_path.read_text())
    for feature in scoreboard["features"]:
        if feature["label"] == "flicker":
            feature["real"] = True
    scoreboard_path.write_text(json.dumps(scoreboard, indent=2) + "\n")
    errors = validate_run(copied)
    assert errors
    assert any("real" in error for error in errors)


def test_run_rejects_too_few_clips(tmp_path: Path):
    clips = tmp_path / "clips"
    generated = runner.invoke(app, ["generate-clips", "--out", str(clips), "--n", "10", "--seed", "0"])
    assert generated.exit_code == 0, generated.output
    ran = runner.invoke(
        app,
        ["run", "fake", "--clips", str(clips), "--out", str(tmp_path / "runs" / "small")],
    )
    output = ran.output + (ran.stderr or "")
    assert ran.exit_code != 0
    assert "20" in output


def test_unknown_adapter_is_rejected(tmp_path: Path):
    clips = tmp_path / "clips"
    generated = runner.invoke(app, ["generate-clips", "--out", str(clips), "--n", "20", "--seed", "0"])
    assert generated.exit_code == 0, generated.output
    ran = runner.invoke(
        app,
        ["run", "vjepa2", "--clips", str(clips), "--out", str(tmp_path / "runs" / "nope")],
    )
    output = ran.output + (ran.stderr or "")
    assert ran.exit_code != 0
    assert "fake" in output
