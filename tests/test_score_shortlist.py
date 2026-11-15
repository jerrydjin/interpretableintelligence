"""The causal checks go to features that can still be verified, and finished controls are reused."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from interpretableintelligence import pipeline
from interpretableintelligence.stats import percentile_le
from interpretableintelligence.thresholds import N_RANDOM_FEATURES

from test_score_batching import _control_setup


class _Adapter:
    name = "hf:stub"


def _labels(holding: set[int], n: int = 8) -> list[dict]:
    return [{"label": f"f{i}", "label_model": "clip", "label_holds_heldout": i in holding} for i in range(n)]


def _shortlist(labels, ranked, k, free_ok, tmp_path, monkeypatch, client=None, label=None):
    monkeypatch.setattr(pipeline, "_autointerp_client", lambda adapter: client)
    if label is not None:
        monkeypatch.setattr(pipeline, "_autointerp_label", label)
    logs: list[str] = []
    picked = pipeline._causal_shortlist(_Adapter(), None, [], tmp_path, labels, ranked, k, free_ok, None, 0, logs.append)
    return picked, logs


def test_shortlist_goes_to_features_that_can_still_be_verified(tmp_path, monkeypatch):
    ranked = [5, 2, 7, 0, 1, 3, 4, 6]
    free_ok = [True, True, False, True, True, True, True, True]
    # 5 leads the ranking but its label does not hold; 2 holds but exists at random init.
    picked, logs = _shortlist(_labels({2, 7, 1, 4}), ranked, 5, free_ok, tmp_path, monkeypatch)
    assert picked == {7, 1}
    assert any("[7, 1]" in line for line in logs)


def test_shortlist_falls_back_to_attribution_order_when_nothing_can_verify(tmp_path, monkeypatch):
    ranked = [5, 2, 7, 0, 1, 3, 4, 6]
    picked, logs = _shortlist(_labels(set()), ranked, 3, [True] * 8, tmp_path, monkeypatch)
    assert picked == {5, 2, 7}
    assert any("none can be verified" in line for line in logs)


def test_no_shortlist_scores_every_feature(tmp_path, monkeypatch):
    picked, _logs = _shortlist(_labels({1}), list(range(8)), None, [True] * 8, tmp_path, monkeypatch)
    assert picked is None


def test_vision_labels_come_before_the_causal_checks(tmp_path, monkeypatch):
    ranked = [5, 2, 7, 0, 1, 3, 4, 6]
    free_ok = [True, True, False, True, True, True, True, True]
    good = {7, 0, 2}
    asked: list[list[int]] = []

    def label(client, adapter, sae, meta, out, labels, targets, device, seed, log):
        asked.append(list(targets))
        for fid in targets:
            labels[fid] = {"label": "x", "label_model": "autointerp-v1:stub", "label_holds_heldout": fid in good}
        return [{"version": "autointerp-v1", "model": "stub", "features": []}] + [{"id": fid} for fid in targets]

    class Client:
        name = "stub/vision"

    (tmp_path / "features").mkdir()
    picked, _logs = _shortlist(_labels(set()), ranked, 4, free_ok, tmp_path, monkeypatch, client=Client(), label=label)
    # The shortlist is labelled once; 2 holds but exists at random init, 5 does not hold.
    assert asked == [[5, 2, 7, 0]]
    assert picked == {7, 0}
    evidence = json.loads((tmp_path / "features" / "autointerp.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in evidence["features"]] == [0, 2, 5, 7]


def test_control_store_answers_from_finished_controls(tmp_path: Path):
    path = tmp_path / "control" / "random_controls.json"
    inputs = {"kind": "random_controls", "seed": 0}
    store = pipeline._ControlStore(path, inputs)
    assert store.lookup(3, 0.5) is None
    samples = [float(v) for v in np.linspace(0.0, 1.0, N_RANDOM_FEATURES)]
    store.record(3, 0.5, percentile_le(0.5, np.asarray(samples)), samples)
    store.record(4, 0.1, 40.0, [])  # stopped early: only an upper bound

    again = pipeline._ControlStore(path, inputs)
    # A full set of controls answers any effect, exactly.
    for effect in (0.5, 0.97, 0.0):
        assert again.lookup(3, effect) == percentile_le(effect, np.asarray(samples))
    # An upper bound is only the answer for the effect it was measured against.
    assert again.lookup(4, 0.1) == 40.0
    assert again.lookup(4, 0.2) is None
    # Controls measured for another SAE, seed or model are never reused.
    assert pipeline._ControlStore(path, {"kind": "random_controls", "seed": 1}).lookup(3, 0.5) is None
    path.write_text("not json", encoding="utf-8")
    assert pipeline._ControlStore(path, inputs).lookup(3, 0.5) is None


def test_kept_controls_reproduce_the_percentile(tmp_path: Path):
    adapter, sae, meta, heldout, pool, fires, n_tokens = _control_setup(tmp_path)

    def run(effect, keep):
        return pipeline._random_percentile(
            adapter=adapter, sae=sae, feature_id=1, real_effect=effect, fires=fires, pool=pool,
            meta=meta, heldout_meta=heldout, seed=0, n_tokens=n_tokens, keep=keep,
        )

    kept: list[float] = []
    exact = run(10.0, kept)
    assert len(kept) == N_RANDOM_FEATURES
    assert percentile_le(10.0, np.asarray(kept)) == exact
    stopped: list[float] = []
    assert run(0.0, stopped) < pipeline.RANDOM_PERCENTILE_MIN
    assert stopped == []  # a stopped feature has no full set of controls to keep


def test_rescoring_with_new_concepts_reuses_the_controls(tmp_path, monkeypatch):
    import torch
    from transformers import AutoModel, ViTConfig

    from interpretableintelligence.adapters.vjepa2 import generate_tiny_clips
    from interpretableintelligence.cli import app
    from interpretableintelligence.labelers import ClipConceptSource
    from interpretableintelligence.validate import validate_run

    torch.manual_seed(0)
    model_dir = tmp_path / "vit"
    AutoModel.from_config(
        ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    ).save_pretrained(model_dir)
    monkeypatch.setattr(
        ClipConceptSource, "_dense_similarity", lambda self, frames: torch.zeros(frames.shape[0], len(self.names), 4, 4)
    )
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clips = tmp_path / "clips"
    generate_tiny_clips(clips, 20)
    out = tmp_path / "run"
    base = ["run", "hf", "--model", str(model_dir), "--clips", str(clips), "--out", str(out), "--prefilter", "3", "--no-serve"]
    first = CliRunner().invoke(app, base)
    assert first.exit_code == 0, first.output
    board = json.loads((out / "scoreboard.json").read_text(encoding="utf-8"))
    assert (out / "control" / "random_controls.json").is_file()

    calls = {"n": 0}
    real = pipeline._random_percentile

    def counting(**kwargs):
        calls["n"] += 1
        return real(**kwargs)

    monkeypatch.setattr(pipeline, "_random_percentile", counting)
    concepts = tmp_path / "concepts.txt"
    concepts.write_text("a robot arm\na table\n", encoding="utf-8")
    second = CliRunner().invoke(app, [*base, "--concepts", str(concepts)])
    assert second.exit_code == 0, second.output
    assert "run score" in second.output and "reused the random controls of 3 features" in second.output
    assert calls["n"] == 0
    again = json.loads((out / "scoreboard.json").read_text(encoding="utf-8"))
    key = lambda rows: [(f["id"], f["ablation_effect"], f["random_percentile"], f["n_random"], f["real"]) for f in rows]
    assert key(again["features"]) == key(board["features"])
    assert validate_run(out) == []


def test_absent_flags_match_the_pairwise_loop():
    from interpretableintelligence.thresholds import DEAD_ACT_EPS, PATTERN_ABSENT_MAX_CORR

    rng = np.random.default_rng(0)
    n, k = 400, 24
    main = np.maximum(rng.normal(size=(n, k)), 0.0)
    ctrl = np.maximum(rng.normal(size=(n, k)), 0.0)
    ctrl[:, :6] = main[:, :6] * 0.7 + 0.3 * ctrl[:, :6]  # some features exist at random init
    ctrl[:, 3] = main[:, 9] * rng.uniform(0.9, 1.1, n)
    main[:, 20] = 0.0  # dead feature
    ctrl[:, 21] = 0.0  # dead control
    main[:, 22] = 1e-7 * rng.random(n)
    sums = (main.sum(0), np.square(main).sum(0), ctrl.sum(0), np.square(ctrl).sum(0), main.T @ ctrl)

    want = []
    for f in range(k):
        if pipeline._std(sums[0][f], sums[1][f], n) <= DEAD_ACT_EPS:
            want.append(False)
            continue
        best = 0.0
        for o in range(k):
            if pipeline._std(sums[2][o], sums[3][o], n) <= DEAD_ACT_EPS:
                continue
            best = max(best, pipeline._corr(sums[0][f], sums[2][o], sums[1][f], sums[3][o], sums[4][f, o], n))
        want.append(bool(best < PATTERN_ABSENT_MAX_CORR))
    assert pipeline._absent_flags(*sums, n) == want
    assert True in want and False in want
