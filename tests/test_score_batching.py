"""Batched scoring must give the same numbers as the one-at-a-time loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from interpretableintelligence import pipeline
from interpretableintelligence.adapters.fake import FakeAdapter, render_clip
from interpretableintelligence.clips import list_clip_meta
from interpretableintelligence.sae import SAE


def test_control_edit_matches_one_token_at_a_time():
    torch.manual_seed(0)
    acts = torch.randn(4, 8, 8, 32)
    direction = torch.randn(32)
    rng = np.random.default_rng(0)
    flat = rng.choice(4 * 8 * 8, size=150, replace=False)
    positions = [(int(i // 64), int(i // 8 % 8), int(i % 8), float(rng.random() * 3)) for i in flat]
    looped = acts.clone()
    for ti, hi, wi, value in positions:
        looped[ti, hi, wi] = looped[ti, hi, wi] - value * direction
    assert torch.equal(pipeline._control_edit(positions, direction)(acts), looped)
    assert torch.equal(pipeline._control_edit([], direction)(acts), acts)


class _ManyFake(FakeAdapter):
    """The fake adapter with a ``predict_masked_many``, so the grouped path runs."""

    calls = 0

    def predict_masked_many(self, clip, window, edits):
        type(self).calls += 1
        return [self.predict_masked(clip, window, edit=edit)[0] for edit in edits]


def _ablate(adapter, meta, sae, out: Path, only):
    membership = {item["clip_id"]: {"split": "heldout" if index % 2 else "train"} for index, item in enumerate(meta)}
    chosen = {item["clip_id"]: [0, 3, 5] for item in meta}
    writer = pipeline._ColumnBatch(out, pipeline._EFFECT_SCHEMA)
    total, count = pipeline._ablation_rows(adapter, sae, meta, membership, chosen, sae.n_features, writer, only=only)
    writer.close()
    return total, count, pq.read_table(out).to_pydict()


@pytest.mark.parametrize("only", [None, {1, 2, 5}])
def test_grouped_ablations_match_the_loop(tmp_path: Path, monkeypatch, only):
    clips = tmp_path / "clips"
    clips.mkdir()
    for index in range(4):
        np.save(clips / f"clip-{index:03d}.npy", render_clip(index, seed=0))
    meta = list_clip_meta(clips)
    plain = FakeAdapter(layer=16, device=torch.device("cpu"))
    plain.load(random_init=False)
    many = _ManyFake(layer=16, device=torch.device("cpu"))
    many.load(random_init=False)
    torch.manual_seed(0)
    sae = SAE(plain.d_model, 7)
    for param in sae.parameters():
        torch.nn.init.normal_(param, std=0.3)
        param.requires_grad_(False)

    want = _ablate(plain, meta, sae, tmp_path / "loop.parquet", only)
    monkeypatch.setattr(pipeline, "ABLATION_BATCH", 3)
    got = _ablate(many, meta, sae, tmp_path / "grouped.parquet", only)

    assert _ManyFake.calls > 0
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    assert got[2] == want[2]


def _control_setup(tmp_path: Path):
    clips = tmp_path / "clips"
    clips.mkdir()
    for index in range(4):
        np.save(clips / f"clip-{index:03d}.npy", render_clip(index, seed=0))
    meta = list_clip_meta(clips)
    adapter = FakeAdapter(layer=16, device=torch.device("cpu"))
    adapter.load(random_init=False)
    grid = adapter.grid()
    heldout = meta[2:]
    cells = [(ci, t, h, w) for ci in (2, 3) for t in range(grid.t) for h in range(grid.h) for w in range(grid.w)]
    pool = tuple(np.asarray(column, dtype=np.int32 if k == 0 else np.int16) for k, column in enumerate(zip(*cells, strict=True)))
    torch.manual_seed(0)
    sae = SAE(adapter.d_model, 5)
    for param in sae.parameters():
        torch.nn.init.normal_(param, std=0.3)
        param.requires_grad_(False)
    fires = np.random.default_rng(1).random(12) * 2.0
    # Every held-out window is scored, so the token count is what one trial sees.
    n_tokens = 0
    for item in heldout:
        clip = pipeline.load_clip(item)
        for window in adapter.windows(clip):
            n_tokens += int(adapter.predict_masked(clip, window)[0].shape[0])
        pipeline._clear_cache(adapter)
    return adapter, sae, meta, heldout, pool, fires, n_tokens


def test_controls_stop_only_when_a_fail_is_certain(tmp_path: Path):
    adapter, sae, meta, heldout, pool, fires, n_tokens = _control_setup(tmp_path)

    def percentile(effect, n):
        return pipeline._random_percentile(
            adapter=adapter, sae=sae, feature_id=1, real_effect=effect, fires=fires, pool=pool,
            meta=meta, heldout_meta=heldout, seed=0, n_tokens=n,
        )

    exact_low = percentile(0.0, None)
    early_low = percentile(0.0, n_tokens)
    assert exact_low < pipeline.RANDOM_PERCENTILE_MIN
    assert exact_low <= early_low < pipeline.RANDOM_PERCENTILE_MIN

    # A feature that can pass runs every control and keeps its exact percentile.
    assert percentile(10.0, n_tokens) == percentile(10.0, None) == 100.0

    # Across the range of effects, stopping never changes which side of the gate it lands on.
    samples = []
    for effect in np.linspace(0.0, 0.05, 6):
        exact = percentile(float(effect), None)
        early = percentile(float(effect), n_tokens)
        assert (exact >= pipeline.RANDOM_PERCENTILE_MIN) == (early >= pipeline.RANDOM_PERCENTILE_MIN)
        assert early >= exact
        if exact >= pipeline.RANDOM_PERCENTILE_MIN:
            assert early == exact
        samples.append(exact)
    assert len(set(samples)) > 1, "effects should span the control distribution"
