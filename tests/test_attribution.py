"""Attribution patching ranks planted causal features above high-activation decoys."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import torch

from physint.adapters.base import Clip
from physint.adapters.fake import (
    POLICY_LOUD_SCALE,
    POLICY_QUIET_SCALE,
    FakeAdapter,
    render_clip,
)
from physint.artifacts import read_attribution_objective, write_attribution, write_feature_acts, write_json
from physint.attribute import (
    attribution_by_clip,
    compare_run,
    format_report,
    rank_feature_ids,
    summarize_attribution,
)
from physint.sae import SAE
from physint.score import ablate_edit
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
    write_attribution(tmp_path / "features" / "attribution.parquet", rows, objective="prediction_loss")
    # Feature 2 is dead for the ablation and the attribution, but it is the loudest activation.
    acts = {"feature_id": [], "clip_id": [], "window": [], "t": [], "h": [], "w": [], "value": []}
    for feature_id, value in ((0, 0.2), (1, 3.0), (2, 9.0), (3, 0.4)):
        for clip_id in ("clip-a", "clip-b"):
            acts["feature_id"].append(feature_id)
            acts["clip_id"].append(clip_id)
            acts["window"].append(0)
            acts["t"].append(0)
            acts["h"].append(0)
            acts["w"].append(0)
            acts["value"].append(value)
    write_feature_acts(tmp_path / "features" / "feature_acts.parquet", acts)
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
    assert result["magnitude_overlap"] == 0
    assert result["magnitude_topk"] == [2, 1]
    assert result["spearman_magnitude"] < result["spearman_feature_all"]
    text = format_report(result)
    assert "Spearman" in text
    assert "top-2 overlap: 2/2" in text
    assert "attribution top-2 overlap: 2/2" in text
    assert "magnitude baseline Spearman" in text
    assert "magnitude baseline top-2 overlap: 0/2" in text


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
    assert result["spearman_magnitude"] is not None
    assert result["spearman_feature_heldout"] > result["spearman_magnitude"]
    assert "magnitude baseline Spearman" in report
    assert "magnitude baseline top-3 overlap:" in report
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
    assert "Spearman |attribution| vs ablation" in completed.stdout
    assert "magnitude baseline Spearman" in completed.stdout
    assert "magnitude baseline top-3 overlap:" in completed.stdout


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
    assert document["objective"] == "prediction_loss"
    assert document["shortlist_only"] is True
    assert document["seconds"]["ablation"] is None
    assert read_attribution_objective(out / "features" / "attribution.parquet") == "prediction_loss"
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
            "--no-serve",
            "--no-open",
        ],
    )
    assert ran.exit_code == 0, ran.output
    checked = runner.invoke(app, ["validate", str(out)])
    assert checked.exit_code == 0, checked.output
    scoreboard = json.loads((out / "scoreboard.json").read_text())
    measured = [row for row in scoreboard["features"] if row["ablation_effect"] is not None]
    skipped = [row for row in scoreboard["features"] if row["ablation_effect"] is None]
    document = json.loads((out / "features" / "candidates.json").read_text())
    assert document["shortlist_only"] is True
    assert {row["id"] for row in measured} == set(document["candidates"])
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


def test_action_loglik_beats_a_loud_decoy():
    """Magnitude ranks a planted loud direction first. Action log-likelihood does not.

    The quiet direction is the only one the taken action depends on. Ablating it
    moves the log-likelihood; ablating the loud direction does not. The shortlist
    is the quiet feature, and that ablation is what confirms the pick.
    """
    adapter = FakeAdapter(layer=16, device=torch.device("cpu"))
    adapter.load(random_init=False)
    frames = render_clip(0, seed=0)
    clip = Clip(clip_id="clip-000", path=Path("clip-000.npy"), frames=frames, fps=8.0, sha256="0" * 64)
    window = adapter.windows(clip)[0]
    before, _target = adapter.predict_masked(clip, window)
    adapter.plant_policy_case()
    after, _target = adapter.predict_masked(clip, window)
    assert torch.allclose(before, after, atol=1e-5)

    assert adapter.loud_dir is not None and adapter.quiet_dir is not None
    sae = SAE(adapter.d_model, 2)
    with torch.no_grad():
        sae.W_dec.zero_()
        sae.W_enc.zero_()
        sae.b_enc.zero_()
        sae.b_dec.zero_()
        sae.W_dec[:, 0] = adapter.loud_dir
        sae.W_dec[:, 1] = adapter.quiet_dir
        sae.W_enc[0] = adapter.loud_dir
        sae.W_enc[1] = adapter.quiet_dir
    sae.eval()

    clips = [clip]
    attr_mean = summarize_attribution(attribution_by_clip(adapter, sae, clips, objective="action_loglik"))[1]
    mag_mean = summarize_attribution(attribution_by_clip(adapter, sae, clips, objective="activation"))[1]
    assert rank_feature_ids(mag_mean)[0] == 0
    assert rank_feature_ids(attr_mean)[0] == 1
    assert float(mag_mean[0]) > 10 * float(mag_mean[1])
    assert abs(float(attr_mean[1])) > 20 * abs(float(attr_mean[0]))
    assert float(mag_mean[0]) / float(POLICY_LOUD_SCALE) > float(mag_mean[1]) / float(POLICY_QUIET_SCALE) * 0.5

    def mean_loglik(feature_id: int | None) -> float:
        edit = None if feature_id is None else ablate_edit(sae, feature_id)
        return float(adapter.attribution_objective(clip, window, "action_loglik", edit=edit).detach())

    clean = mean_loglik(None)
    quiet_drop = clean - mean_loglik(1)
    loud_drop = clean - mean_loglik(0)
    assert quiet_drop > 0.2
    assert quiet_drop > 20 * abs(loud_drop)
    assert rank_feature_ids(attr_mean)[:1] == [1]


def test_rank_records_objective_metadata(demo, tmp_path: Path):
    out = tmp_path / "loglik"
    ranked = runner.invoke(
        app,
        [
            "rank",
            "fake",
            "--clips",
            str(demo.clips),
            "--from-run",
            str(demo.run),
            "--objective",
            "action_loglik",
            "--prefilter",
            "3",
            "--out",
            str(out),
        ],
    )
    assert ranked.exit_code == 0, ranked.output
    assert read_attribution_objective(out / "features" / "attribution.parquet") == "action_loglik"
    document = json.loads((out / "features" / "candidates.json").read_text())
    assert document["objective"] == "action_loglik"
    assert document["shortlist_only"] is True
    rejected = runner.invoke(
        app,
        ["rank", "fake", "--clips", str(demo.clips), "--from-run", str(demo.run), "--objective", "reward", "--out", str(tmp_path / "nope")],
    )
    assert rejected.exit_code != 0
    assert "objective" in (rejected.output + (rejected.stderr or ""))


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
