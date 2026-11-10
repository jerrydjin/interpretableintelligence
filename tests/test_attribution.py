"""Attribution patching ranks planted causal features above high-activation decoys."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from physint.artifacts import write_attribution, write_json
from physint.attribute import compare_run, format_report
from typer.testing import CliRunner

from physint.cli import app
from physint.validate import validate_run

runner = CliRunner()

CAUSAL = ("stripe", "pulse", "corner")
DECOY = ("flicker", "band", "speckle")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_attribution_vs_ablation.py"


def test_synthetic_spearman_and_overlap(tmp_path: Path):
    rows = {"feature_id": [], "clip_id": [], "score": []}
    # |score| order is f0, f3, f1, f2. Ablation order matches, so Spearman is 1.
    per_clip = {0: 4.0, 1: 1.0, 2: 0.0, 3: -2.0}
    for feature_id, score in per_clip.items():
        for clip_id in ("clip-a", "clip-b"):
            rows["feature_id"].append(feature_id)
            rows["clip_id"].append(clip_id)
            rows["score"].append(score)
    write_attribution(tmp_path / "features" / "attribution.parquet", rows)
    write_json(
        tmp_path / "scoreboard.json",
        {
            "features": [
                {"id": 0, "label": "stripe", "ablation_effect": 5.0},
                {"id": 1, "label": "flicker", "ablation_effect": 1.0},
                {"id": 2, "label": "", "ablation_effect": 0.0},
                {"id": 3, "label": "pulse", "ablation_effect": 3.0},
            ]
        },
    )
    result = compare_run(tmp_path, k=2)
    assert result["spearman_feature_all"] == pytest.approx(1.0)
    assert result["overlap"] == 2
    assert result["attribution_topk"] == [0, 3]
    assert result["ablation_topk"] == [0, 3]
    text = format_report(result)
    assert "Spearman" in text
    assert "top-2 overlap: 2/2" in text


def test_causal_features_outrank_active_decoys(demo):
    ranked = json.loads((demo.run / "features" / "candidates.json").read_text())["ranked"]
    causal = [row for row in ranked if row["label"] in CAUSAL]
    decoys = [row for row in ranked if row["label"] in DECOY]
    assert {row["label"] for row in causal} == set(CAUSAL)
    assert {row["label"] for row in decoys} == set(DECOY)
    assert len(causal) == 3
    assert len(decoys) == 3
    assert max(row["rank"] for row in causal) <= 3
    assert min(row["rank"] for row in decoys) > 3
    causal_floor = min(row["abs_score"] for row in causal)
    decoy_ceiling = max(row["abs_score"] for row in decoys)
    assert causal_floor > 50 * decoy_ceiling

    acts = pq.read_table(demo.run / "features" / "feature_acts.parquet").to_pydict()
    peak: dict[int, float] = {}
    for feature_id, value in zip(acts["feature_id"], acts["value"], strict=True):
        peak[int(feature_id)] = max(peak.get(int(feature_id), 0.0), float(value))
    causal_peak = min(peak[row["feature_id"]] for row in causal)
    decoy_peak = min(peak[row["feature_id"]] for row in decoys)
    assert decoy_peak > 0.5 * causal_peak

    report = format_report(compare_run(demo.run, k=3))
    print(report)
    result = compare_run(demo.run, k=3)
    assert result["overlap"] == 3
    assert result["spearman_feature_heldout"] is not None
    assert result["spearman_feature_heldout"] >= 0.9
    seconds = json.loads((demo.run / "features" / "candidates.json").read_text())["seconds"]
    print(
        f"wall attribution={seconds['attribution']:.4f}s "
        f"ablation={seconds['ablation']:.4f}s "
        f"random_controls={seconds['random_controls']:.4f}s"
    )
    assert seconds["attribution"] > 0
    assert seconds["attribution"] * 10 < seconds["random_controls"]
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(demo.run), "--k", "3"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "top-3 overlap: 3/3" in completed.stdout
    assert "Spearman" in completed.stdout


def test_rank_command_reuses_a_trained_sae(demo, tmp_path: Path):
    out = tmp_path / "ranked"
    ranked = runner.invoke(
        app,
        [
            "rank",
            "fake",
            "--clips",
            str(demo.clips),
            "--from-run",
            str(demo.run),
            "--prefilter",
            "3",
            "--out",
            str(out),
        ],
    )
    assert ranked.exit_code == 0, ranked.output
    document = json.loads((out / "features" / "candidates.json").read_text())
    assert document["prefilter_k"] == 3
    assert document["seconds"]["ablation"] is None
    labels = [row["label"] for row in document["ranked"] if row["feature_id"] in document["candidates"]]
    assert set(labels) == set(CAUSAL)
    fresh = pq.read_table(out / "features" / "attribution.parquet").to_pydict()
    original = pq.read_table(demo.run / "features" / "attribution.parquet").to_pydict()
    assert fresh["feature_id"] == original["feature_id"]
    assert fresh["clip_id"] == original["clip_id"]
    for left, right in zip(fresh["score"], original["score"], strict=True):
        assert left == right


def test_prefilter_ablates_only_the_shortlist(demo, tmp_path: Path):
    out = tmp_path / "prefilter"
    ran = runner.invoke(
        app,
        [
            "run",
            "fake",
            "--clips",
            str(demo.clips),
            "--layer",
            "16",
            "--seed",
            "0",
            "--heldout-frac",
            "0.2",
            "--prefilter",
            "3",
            "--out",
            str(out),
        ],
    )
    assert ran.exit_code == 0, ran.output
    checked = runner.invoke(app, ["validate", str(out)])
    assert checked.exit_code == 0, checked.output
    scoreboard = json.loads((out / "scoreboard.json").read_text())
    measured = [row for row in scoreboard["features"] if row["ablation_effect"] is not None]
    skipped = [row for row in scoreboard["features"] if row["ablation_effect"] is None]
    assert {row["label"] for row in measured} == set(CAUSAL)
    assert all(row["real"] for row in measured)
    assert all(row["n_random"] >= 100 for row in measured)
    decoy_rows = [row for row in skipped if row["label"] in DECOY]
    assert {row["label"] for row in decoy_rows} == set(DECOY)
    assert all(row["real"] is False and row["n_random"] == 0 and row["random_percentile"] is None for row in skipped)
    full = json.loads((demo.run / "features" / "candidates.json").read_text())["seconds"]
    short = json.loads((out / "features" / "candidates.json").read_text())["seconds"]
    print(
        f"full ablation={full['ablation']:.4f}s controls={full['random_controls']:.4f}s; "
        f"prefilter ablation={short['ablation']:.4f}s controls={short['random_controls']:.4f}s "
        f"attribution={short['attribution']:.4f}s"
    )
    assert short["random_controls"] < full["random_controls"]


def test_validate_rejects_a_broken_ranking(demo, tmp_path: Path):
    copied = tmp_path / "demo"
    shutil.copytree(demo.run, copied)
    path = copied / "features" / "candidates.json"
    document = json.loads(path.read_text())
    document["ranked"][0]["feature_id"], document["ranked"][1]["feature_id"] = (
        document["ranked"][1]["feature_id"],
        document["ranked"][0]["feature_id"],
    )
    path.write_text(json.dumps(document, indent=2) + "\n")
    errors = validate_run(copied)
    assert errors
    assert any("sorted" in error for error in errors)
