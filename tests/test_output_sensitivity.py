"""The attribution prefilter on adapters scored against their own clean output (hf, groot).

There ``prediction_loss`` is zero with a zero gradient at the clean point, so
activation x gradient ranks float noise. ``output_sensitivity`` must instead
track the cosine shift a real ablation causes, which is what the scoreboard
measures for the shortlisted features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from interpretableintelligence.adapters import get_adapter
from interpretableintelligence.adapters.base import Clip
from interpretableintelligence.adapters.hf import make_spec
from interpretableintelligence.attribute import (
    GRADIENT_OBJECTIVES,
    OUTPUT_SENSITIVITY,
    PREDICTION_LOSS,
    SCORED_OBJECTIVES,
    attribution_by_clip,
    rank_description,
)
from interpretableintelligence.sae import SAE
from interpretableintelligence.score import (
    ablate_edit,
    cosine_distance,
    default_objective,
    estimate_effects,
    output_sensitivity_effects,
)
from interpretableintelligence.stats import spearman

CPU = torch.device("cpu")
N_FEATURES = 48


@pytest.fixture(scope="module")
def vit(tmp_path_factory) -> Path:
    from transformers import AutoModel, ViTConfig

    torch.manual_seed(0)
    config = ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=3, num_attention_heads=2, intermediate_size=64)
    root = tmp_path_factory.mktemp("hf-sensitivity") / "vit"
    AutoModel.from_config(config).save_pretrained(root)
    return root


def _adapter(path: Path):
    adapter = get_adapter(make_spec(str(path)))(layer=1, device=CPU)
    adapter.load()
    return adapter


def _sae(d_model: int) -> SAE:
    torch.manual_seed(1)
    sae = SAE(d_model, N_FEATURES, activation="topk", k=6)
    with torch.no_grad():
        sae.W_enc.normal_(0.0, 0.3)
        # Spread decoder norms over two orders of magnitude so true effects differ.
        scales = torch.logspace(-1.5, 0.5, N_FEATURES)[torch.randperm(N_FEATURES)]
        sae.W_dec.normal_(0.0, 1.0)
        sae.W_dec.mul_(scales / sae.W_dec.norm(dim=0))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def _clip(seed: int) -> Clip:
    frames = np.random.default_rng(seed).random((10, 40, 48, 3), dtype=np.float32)
    return Clip(clip_id=f"c{seed}", path=Path(f"c{seed}.npy"), frames=frames, fps=8.0, sha256="ab" * 32)


def _true_shift(adapter, sae, clip) -> np.ndarray:
    window = adapter.windows(clip)[0]
    with torch.no_grad():
        base, _ = adapter.predict_masked(clip, window)
        return np.array(
            [
                float(cosine_distance(base, adapter.predict_masked(clip, window, edit=ablate_edit(sae, i))[0]).mean())
                for i in range(N_FEATURES)
            ]
        )


def test_hf_defaults_to_output_sensitivity_and_others_keep_prediction_loss(vit):
    assert default_objective(_adapter(vit)) == OUTPUT_SENSITIVITY
    assert default_objective(get_adapter("fake")(layer=16, device=CPU)) == PREDICTION_LOSS
    assert OUTPUT_SENSITIVITY in GRADIENT_OBJECTIVES and OUTPUT_SENSITIVITY in SCORED_OBJECTIVES
    assert "random-probe" in rank_description(OUTPUT_SENSITIVITY)


def test_prediction_loss_attribution_is_float_noise_on_hf(vit):
    """The bug this objective fixes: zero loss, zero gradient, at the clean point."""
    adapter = _adapter(vit)
    sae = _sae(adapter.d_model)
    clip = _clip(0)
    noise = estimate_effects(adapter, sae, clip, adapter.windows(clip)[0], objective=PREDICTION_LOSS)
    true = _true_shift(adapter, sae, clip)
    assert float(noise.reshape(-1, N_FEATURES).sum(0).abs().max()) < 1e-3 * float(true.max())


def test_output_sensitivity_tracks_the_real_ablation(vit):
    adapter = _adapter(vit)
    sae = _sae(adapter.d_model)
    for seed in (0, 1):
        clip = _clip(seed)
        scores = attribution_by_clip(adapter, sae, [clip])[clip.clip_id]
        assert np.all(scores >= 0.0)
        rho = spearman(scores, _true_shift(adapter, sae, clip))
        assert rho > 0.8, rho


def test_shares_sum_to_the_rms_score_and_are_reproducible(vit):
    adapter = _adapter(vit)
    sae = _sae(adapter.d_model)
    clip = _clip(2)
    window = adapter.windows(clip)[0]
    first = output_sensitivity_effects(adapter, sae, clip, window, probes=4)
    again = output_sensitivity_effects(adapter, sae, clip, window, probes=4)
    assert first.shape[-1] == N_FEATURES
    assert torch.equal(first, again)
    totals = first.reshape(-1, N_FEATURES).sum(0)
    assert bool((totals >= -1e-6).all())
    # Features that never fire on this window score exactly zero.
    with torch.no_grad():
        fired = (sae.encode(adapter.raw_activations(clip, window).reshape(-1, adapter.d_model)) > 0).any(0)
    assert bool((totals[~fired] == 0).all())
    with pytest.raises(ValueError):
        output_sensitivity_effects(adapter, sae, clip, window, probes=0)
