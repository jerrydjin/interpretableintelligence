"""`ii probe` and the random-weights SAE control, on the fake adapter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from interpretableintelligence.adapters.fake import FAILURE_LABELS_FILE, failure_labels
from interpretableintelligence.cli import app
from interpretableintelligence.probe import (
    PROBE_DIR,
    RESULT_FILE,
    auroc,
    fit_logistic,
)
from interpretableintelligence.sae import SAEConfig, random_weights_sae, train_sae
from interpretableintelligence.validate import validate_run

runner = CliRunner()
N_CLIPS = 200  # 40 held-out episodes: enough for a p < 0.05 paired bootstrap


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(run: Path, *args: str) -> dict:
    result = runner.invoke(app, ["probe", str(run), *args])
    assert result.exit_code == 0, result.output
    return json.loads((run / PROBE_DIR / RESULT_FILE).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def probe_run(tmp_path_factory) -> tuple[Path, Path, Path]:
    """A 200-clip fake run (no scoring stage) with planted labels in its clip dir and null labels beside it."""
    os.environ.setdefault("II_DEVICE", "cpu")
    root = tmp_path_factory.mktemp("probe")
    clips = root / "clips"
    out = root / "runs" / "probe"
    made = runner.invoke(app, ["generate-clips", "--out", str(clips), "--n", str(N_CLIPS), "--failure-labels", "planted"])
    assert made.exit_code == 0, made.output
    ran = runner.invoke(app, ["run", "fake", "--clips", str(clips), "--out", str(out), "--no-score", "--no-serve", "--no-open"])
    assert ran.exit_code == 0, ran.output
    null = root / "null.json"
    null.write_text(json.dumps(failure_labels(N_CLIPS, 0, "null")), encoding="utf-8")
    return out, clips / FAILURE_LABELS_FILE, null


# ---------------------------------------------------------------- random-weights SAE


@pytest.mark.parametrize("activation,k", [("relu", None), ("topk", 3)])
def test_random_weights_sae_matches_shape_and_is_seeded(activation, k):
    config = SAEConfig(d_model=16, n_features=24, activation=activation, k=k, seed=5)
    mean = torch.linspace(-1.0, 1.0, 16)
    first = random_weights_sae(config, mean, seed=11, device=torch.device("cpu"))
    again = random_weights_sae(config, mean, seed=11, device=torch.device("cpu"))
    other = random_weights_sae(config, mean, seed=12, device=torch.device("cpu"))
    assert (first.n_features, first.d_model, first.activation, first.k) == (24, 16, activation, k)
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, again.state_dict()[name]), name
    assert not torch.equal(first.W_dec, other.W_dec)
    assert torch.allclose(first.W_dec.norm(dim=0), torch.ones(24))
    assert torch.equal(first.b_dec, mean)
    if activation == "topk":
        assert int((first.encode(torch.randn(5, 16)) > 0).sum(dim=-1).max()) <= k


def test_random_weights_sae_is_the_trainer_before_its_first_step():
    """Same initializer as training: at the SAE's own seed it equals ``train_sae`` with zero steps."""
    data = torch.randn(64, 8, generator=torch.Generator().manual_seed(0))
    config = SAEConfig(d_model=8, n_features=10, seed=3, steps=0)
    untrained = train_sae(data, config, torch.device("cpu"))
    control = random_weights_sae(config, data.mean(dim=0), seed=3, device=torch.device("cpu"))
    for name, tensor in untrained.state_dict().items():
        assert torch.equal(tensor, control.state_dict()[name]), name


def test_random_weights_sae_leaves_global_rng_alone():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    random_weights_sae(SAEConfig(d_model=4, n_features=6), torch.zeros(4), seed=9, device=torch.device("cpu"))
    assert torch.equal(torch.rand(3), expected)


# ---------------------------------------------------------------- helpers


def test_auroc_counts_ties_half():
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auroc(np.array([0.0]), np.array([1.0])) == 0.0
    assert auroc(np.array([1.0]), np.array([1.0])) == 0.5


def test_logistic_probe_is_finite_on_separable_data():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    weights, bias = fit_logistic(x, np.array([0.0, 0.0, 1.0, 1.0]), l2=1.0)
    assert np.isfinite(weights).all() and np.isfinite(bias) and weights[0] > 0


def test_fake_failure_labels_are_seeded():
    planted = failure_labels(40, 0, "planted")
    assert planted == failure_labels(40, 0, "planted")
    assert planted != failure_labels(40, 1, "planted")
    failed = [ep for ep in planted["episodes"] if ep["failure"]]
    assert failed and len(failed) < 40
    assert all(ep["onset_s"] == 2.0 for ep in failed)
    assert all(ep["onset_s"] is None for ep in planted["episodes"] if not ep["failure"])


# ---------------------------------------------------------------- ii probe


def test_planted_failure_passes_against_both_baselines(probe_run):
    run, planted, _null = probe_run
    split_before = _sha(run / "split.json")
    result = _probe(run, "--labels", str(planted))
    assert result["sae_gate"] == "pass" and result["positive"] is True, result["reason"]
    trained, control, raw = (result["inputs"][name] for name in ("trained_sae", "random_sae", "raw"))
    assert trained["auroc"] > control["auroc"] and trained["auroc"] > raw["auroc"]
    for name in ("random_sae", "raw"):
        comparison = result["comparisons"][name]
        assert comparison["beats"] is True and comparison["p_value"] < 0.05 and comparison["delta"] > 0
    assert trained["lead_time_s"] >= 1.0
    # The control matches the trained SAE's width and is never scored as a dictionary.
    assert control["n_dims"] == trained["n_dims"] == 12 and raw["n_dims"] == 64
    assert control["sae_usable"] is None and control["loss_recovered"] is None and control["clip_effects"] is None
    assert control["real"] is False and result["real"] is False
    # No scoreboard in this run: the trained SAE's fields are null, not 0.0.
    assert trained["sae_usable"] is None and trained["loss_recovered"] is None
    assert _sha(run / "split.json") == split_before
    assert not (run / "scoreboard.json").exists()


def test_null_failure_fails_against_random_baseline(probe_run):
    run, _planted, null = probe_run
    result = _probe(run, "--labels", str(null))
    assert result["comparisons"]["random_sae"]["beats"] is False
    assert result["sae_gate"] == "fail" and result["positive"] is False
    assert result["real"] is False


def test_probe_is_deterministic(probe_run):
    run, planted, _null = probe_run
    first = _probe(run, "--labels", str(planted), "--bootstrap", "200")
    second = _probe(run, "--labels", str(planted), "--bootstrap", "200")
    assert first == second


def test_labels_default_to_the_clip_dir(probe_run):
    run, planted, _null = probe_run
    result = _probe(run)
    assert result["labels"]["sha256"] == _sha(planted)


def test_lead_below_one_second_skips_the_gate(probe_run):
    run, planted, _null = probe_run
    result = _probe(run, "--labels", str(planted), "--lead-s", "0.5")
    assert result["sae_gate"] == "not_run" and result["positive"] is False
    assert "below" in result["reason"]
    assert result["inputs"]["trained_sae"]["auroc"] is not None  # still measured, just not gated


def test_one_class_train_split_skips_the_gate(probe_run, tmp_path):
    run, planted, _null = probe_run
    labels = json.loads(planted.read_text(encoding="utf-8"))
    for episode in labels["episodes"]:
        episode.update(failure=False, onset_s=None)
    path = tmp_path / "all-ok.json"
    path.write_text(json.dumps(labels), encoding="utf-8")
    result = _probe(run, "--labels", str(path))
    assert result["sae_gate"] == "not_run" and result["positive"] is False
    assert all(result["inputs"][name]["auroc"] is None for name in ("trained_sae", "random_sae", "raw"))
    assert all(result["comparisons"][name]["p_value"] is None for name in ("random_sae", "raw"))


def test_labels_must_cover_every_clip(probe_run, tmp_path):
    run, planted, _null = probe_run
    labels = json.loads(planted.read_text(encoding="utf-8"))
    labels["episodes"] = labels["episodes"][1:]
    path = tmp_path / "short.json"
    path.write_text(json.dumps(labels), encoding="utf-8")
    result = runner.invoke(app, ["probe", str(run), "--labels", str(path)])
    assert result.exit_code == 1
    assert "no label" in result.output


def test_validate_rebuilds_probe_on_a_scored_run(demo, tmp_path):
    run = tmp_path / demo.run.name
    shutil.copytree(demo.run, run)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(failure_labels(30, 0, "planted")), encoding="utf-8")
    scoreboard_before = _sha(run / "scoreboard.json")
    result = _probe(run, "--labels", str(labels))
    scoreboard = json.loads((run / "scoreboard.json").read_text(encoding="utf-8"))
    assert result["inputs"]["trained_sae"]["sae_usable"] is scoreboard["sae"]["usable"]
    assert result["inputs"]["trained_sae"]["loss_recovered"] == scoreboard["sae"]["loss_recovered"]
    assert _sha(run / "scoreboard.json") == scoreboard_before
    assert validate_run(run) == []

    path = run / PROBE_DIR / RESULT_FILE
    for tamper in (
        lambda d: d.update(real=True),
        lambda d: d["inputs"]["random_sae"].update(loss_recovered=0.0),
        lambda d: d.update(positive=not d["positive"]),
        lambda d: d["inputs"]["trained_sae"].update(auroc=0.123),
    ):
        stored = json.loads(path.read_text(encoding="utf-8"))
        tamper(stored)
        path.write_text(json.dumps(stored), encoding="utf-8")
        assert validate_run(run), "tampered probe.json should not validate"
        _probe(run, "--labels", str(labels))


def test_pooling_does_not_depend_on_chunk_size(probe_run, monkeypatch):
    import interpretableintelligence.probe as probe_module
    from interpretableintelligence.sae import load_sae

    run, _planted, _null = probe_run
    sae, _config = load_sae(run / "sae", torch.device("cpu"))
    encoders = {"trained_sae": sae, "raw": None}
    keys, pooled = probe_module.pooled_bins(run / "acts", encoders, torch.device("cpu"))
    monkeypatch.setattr(probe_module, "ROWS_PER_CHUNK", 7)  # not a multiple of the 16 tokens per bin
    keys_small, pooled_small = probe_module.pooled_bins(run / "acts", encoders, torch.device("cpu"))
    assert np.array_equal(keys, keys_small)
    # Raw activations go through no arithmetic, so any bin split or mixed across chunks shows
    # up as a mismatch here.
    assert np.array_equal(pooled["raw"], pooled_small["raw"])
    # The SAE encoder is a matmul, and BLAS may round the same row differently for a different
    # batch size (macOS Accelerate does, by one float32 ulp), so compare to float32 precision.
    np.testing.assert_allclose(pooled["trained_sae"], pooled_small["trained_sae"], rtol=1e-6, atol=1e-6)
    assert keys.size == N_CLIPS * 8  # one bin per (clip, t): fake has one window of 8 time bins
