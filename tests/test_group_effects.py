"""Group loss ablations, and the false-flag rate when nothing is planted.

The default cue model is untouched: single-feature cosine ablations still decide
``real`` on the scoreboard. These tests cover the optional loss / group scorer.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from physint.adapters.base import Clip
from physint.adapters.fake import FakeAdapter, render_clip
from physint.artifacts import write_clip_effects
from physint.sae import SAE
from physint.score import (
    ablate_edit,
    ablate_features_edit,
    capture_activations,
    clean_predictions,
    coactive_group,
    loss_frame_rows,
    mean_cosine_shift,
    random_percentile,
    score_loss_ablation,
)
from physint.stats import monte_carlo_pvalue, wilson_interval
from physint.thresholds import EFFECT_ALPHA, EFFECT_POS_EPS, N_RANDOM_FEATURES, RANDOM_PERCENTILE_MIN


def _clips(n: int) -> list[Clip]:
    clips = []
    width = max(3, len(str(n - 1)))
    for index in range(n):
        clip_id = f"clip-{index:0{width}d}"
        clips.append(
            Clip(
                clip_id=clip_id,
                path=Path(f"{clip_id}.npy"),
                frames=render_clip(index, seed=0),
                fps=8.0,
                sha256="0" * 64,
            )
        )
    return clips


def _sae_from_dirs(dirs: torch.Tensor) -> SAE:
    n_features, d_model = dirs.shape
    model = SAE(int(d_model), int(n_features))
    with torch.no_grad():
        model.W_dec.copy_(dirs.T.contiguous())
        model.W_enc.copy_(dirs.contiguous())
        model.b_enc.zero_()
        model.b_dec.zero_()
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def _prepared(plant: str, n_clips: int):
    torch.set_num_threads(1)
    adapter = FakeAdapter(layer=16, plant=plant)
    adapter.load()
    clips = _clips(n_clips)
    membership = {clip.clip_id: "heldout" for clip in clips}
    tokens, acts = capture_activations(adapter, clips, membership)
    assert adapter.feature_dirs is not None
    sae = _sae_from_dirs(adapter.feature_dirs.detach().cpu())
    with torch.no_grad():
        features = sae.encode(acts).numpy()
    return adapter, sae, clips, tokens, features


def test_default_adapter_stays_single_window():
    adapter = FakeAdapter(layer=16)
    assert adapter.plant == "cues"
    adapter.load()
    assert len(adapter.windows(_clips(1)[0])) == 1


def test_coactive_group_prefers_shared_sites():
    features = np.zeros((20, 5), dtype=np.float64)
    features[:8, 0] = 1.0
    features[:8, 1] = 1.0
    features[:8, 2] = 1.0
    features[8:14, 3] = 1.0
    features[14:, 4] = 1.0
    assert coactive_group(features, k=3) == [0, 1, 2]


def test_tied_zeros_are_not_a_real_effect():
    controls = np.zeros(N_RANDOM_FEATURES, dtype=np.float64)
    assert monte_carlo_pvalue(0.0, controls) == 1.0
    low, high = wilson_interval(0, 200)
    assert low == 0.0
    assert high <= EFFECT_ALPHA


def test_group_is_real_singles_miss_and_random_groups_are_not(capsys):
    adapter, sae, clips, tokens, features = _prepared("group", n_clips=6)
    assert len(adapter.windows(clips[0])) == 2
    group = coactive_group(features, k=3)
    assert set(group) == {0, 1, 2}, group

    clean = clean_predictions(adapter, clips)
    single_rows = []
    for feature_id in group:
        cosine = mean_cosine_shift(adapter, clips, ablate_edit(sae, feature_id), clean)
        cosine_pct = random_percentile(
            adapter=adapter,
            sae=sae,
            feature_id=feature_id,
            real_effect=cosine,
            tokens=tokens,
            features=features,
            heldout_clips=clips,
            clean=clean,
            seed=0,
            n_random=N_RANDOM_FEATURES,
        )
        single = score_loss_ablation(
            adapter=adapter,
            sae=sae,
            feature_ids=[feature_id],
            tokens=tokens,
            features=features,
            clips=clips,
            seed=0,
            n_random=N_RANDOM_FEATURES,
            scope="video",
        )
        assert single.effect_kind == "loss_video"
        assert single.n_random == N_RANDOM_FEATURES
        assert cosine_pct < RANDOM_PERCENTILE_MIN
        assert single.real is False
        assert single.random_percentile < RANDOM_PERCENTILE_MIN
        assert single.p_value > EFFECT_ALPHA
        assert abs(single.effect) < 0.01
        single_rows.append(
            {
                "feature_id": feature_id,
                "cosine_effect": cosine,
                "cosine_percentile": cosine_pct,
                "loss_effect": single.effect,
                "loss_percentile": single.random_percentile,
                "loss_p": single.p_value,
            }
        )

    grouped = score_loss_ablation(
        adapter=adapter,
        sae=sae,
        feature_ids=sorted(group),
        tokens=tokens,
        features=features,
        clips=clips,
        seed=0,
        n_random=N_RANDOM_FEATURES,
        scope="video",
    )
    assert grouped.effect_kind == "group_loss_video"
    assert grouped.real is True
    assert grouped.random_percentile >= RANDOM_PERCENTILE_MIN
    assert grouped.p_value <= EFFECT_ALPHA
    assert grouped.effect > 0.05
    assert float(np.max(grouped.control_effects)) < grouped.effect
    assert not bool(grouped.controls_are_real().any())

    seen: list[int] = []
    original = adapter.predict_masked

    def _spy(clip, window, edit=None):
        seen.append(int(window.index))
        return original(clip, window, edit=edit)

    adapter.predict_masked = _spy  # type: ignore[method-assign]
    score_loss_ablation(
        adapter=adapter,
        sae=sae,
        feature_ids=sorted(group),
        tokens=tokens,
        features=features,
        clips=clips,
        seed=1,
        n_random=1,
        scope="video",
    )
    adapter.predict_masked = original  # type: ignore[method-assign]
    assert set(seen) == {0, 1}

    rows = loss_frame_rows(
        adapter,
        clips,
        ablate_features_edit(sae, sorted(group)),
        sorted(group),
        "group_loss_video",
    )
    single_loss_rows = loss_frame_rows(
        adapter,
        clips,
        ablate_edit(sae, group[0]),
        [group[0]],
        "loss_video",
    )
    for key in rows:
        rows[key].extend(single_loss_rows[key])
    path = Path("/tmp/physint-group-effects.parquet")
    write_clip_effects(path, rows)
    table = pq.read_table(path)
    kinds = set(table.column("effect_kind").to_pylist())
    assert kinds == {"group_loss_video", "loss_video"}
    frame = table.to_pydict()
    member_effects = [
        effect
        for feature_id, kind, effect in zip(
            frame["feature_id"], frame["effect_kind"], frame["effect"], strict=True
        )
        if kind == "group_loss_video" and feature_id == sorted(group)[0]
    ]
    assert member_effects
    assert abs(float(np.mean(member_effects)) - grouped.effect) < 1e-5

    summary = {
        "group_feature_ids": sorted(group),
        "group_loss_effect": grouped.effect,
        "group_loss_percentile": grouped.random_percentile,
        "group_loss_p": grouped.p_value,
        "group_control_max": float(np.max(grouped.control_effects)),
        "n_random": grouped.n_random,
        "alpha": EFFECT_ALPHA,
        "singles": single_rows,
        "matched_controls_real": int(grouped.controls_are_real().sum()),
    }
    print("GROUP_EFFECT_SUMMARY " + json.dumps(summary))
    captured = capsys.readouterr()
    assert "GROUP_EFFECT_SUMMARY" in captured.out


def test_run_helper_writes_both_loss_kinds():
    from physint.run import _extra_effect_rows

    adapter, sae, clips, _tokens, features = _prepared("group", n_clips=2)
    ablated = {(clips[0].clip_id, 0), (clips[1].clip_id, 1)}
    rows = _extra_effect_rows(adapter, sae, clips, features, ablated, ("loss", "group"), lambda _message: None)
    assert "loss_video" in rows["effect_kind"]
    assert "group_loss_video" in rows["effect_kind"]
    assert set(rows["effect_kind"]) <= {"loss_video", "group_loss_video"}


def test_unknown_extra_effect_is_rejected(tmp_path: Path):
    from physint.run import execute_run

    with pytest.raises(ValueError, match="extra_effects"):
        execute_run("fake", tmp_path, tmp_path / "out", extra_effects=("nope",))


def test_null_fake_false_flag_rate(capsys):
    adapter, sae, clips, tokens, features = _prepared("null", n_clips=4)
    group = coactive_group(features, k=3)
    assert len(group) == 3
    n_tests = 200
    flags = 0
    for seed in range(n_tests):
        score = score_loss_ablation(
            adapter=adapter,
            sae=sae,
            feature_ids=group,
            tokens=tokens,
            features=features,
            clips=clips,
            seed=seed,
            n_random=N_RANDOM_FEATURES,
            scope="video",
        )
        assert score.effect_kind == "group_loss_video"
        assert abs(score.effect) <= EFFECT_POS_EPS
        flags += int(score.real)
    low, high = wilson_interval(flags, n_tests)
    rate = flags / n_tests
    summary = {
        "n_tests": n_tests,
        "flags": flags,
        "rate": rate,
        "wilson_low": low,
        "wilson_high": high,
        "alpha": EFFECT_ALPHA,
        "n_random": N_RANDOM_FEATURES,
    }
    print("NULL_FALSE_FLAG_SUMMARY " + json.dumps(summary))
    assert n_tests >= 200
    assert rate <= EFFECT_ALPHA
    assert high <= EFFECT_ALPHA
    captured = capsys.readouterr()
    assert "NULL_FALSE_FLAG_SUMMARY" in captured.out
