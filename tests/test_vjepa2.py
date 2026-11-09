"""V-JEPA 2 adapter on a tiny random config. No checkpoint download.

Real ``facebook/vjepa2-*`` weights are not required. Mark any future test that
downloads them with ``@pytest.mark.slow``; the default pytest addopts skip that
marker.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from physint.adapters.base import Clip
from physint.adapters.vjepa2 import WEIGHT_ALLOW_PATTERNS, VJEPA2Adapter, fetch_checkpoint, generate_tiny_clips
from physint.clips import load_clips
from physint.device import ensure_mps_fallback
from physint.sae import SAE
from physint.score import estimate_effects, mean_predictor_loss
from physint.stats import loss_recovered

CPU = torch.device("cpu")


def _adapter(layer: int = 1) -> VJEPA2Adapter:
    adapter = VJEPA2Adapter(layer=layer, device=CPU, tiny=True)
    adapter.load(random_init=False)
    return adapter


def _frames(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((4, 32, 32, 3), dtype=np.float32)


def _clip(frames: np.ndarray, clip_id: str) -> Clip:
    return Clip(clip_id=clip_id, path=Path(f"{clip_id}.npy"), frames=frames, fps=8.0, sha256="ab" * 32)


def _paint(frames: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
    painted = np.array(frames, copy=True)
    tubelet, patch = 2, 16
    for t, h, w in zip(*np.nonzero(mask), strict=True):
        painted[
            t * tubelet : (t + 1) * tubelet,
            h * patch : (h + 1) * patch,
            w * patch : (w + 1) * patch,
        ] = value
    return painted


def test_weight_fetch_skips_the_original_checkpoint(monkeypatch):
    seen: dict = {}

    def snap(repo_id, allow_patterns=None, **kwargs):
        del kwargs
        seen["repo"] = repo_id
        seen["patterns"] = list(allow_patterns)
        return "cached"

    monkeypatch.setattr("huggingface_hub.snapshot_download", snap)
    assert fetch_checkpoint() == "cached"
    blob = " ".join(seen["patterns"])
    assert seen["repo"] == "facebook/vjepa2-vitl-fpc64-256"
    assert seen["patterns"] == WEIGHT_ALLOW_PATTERNS
    assert "pth" not in blob
    assert "original" not in blob
    assert "model.safetensors" in seen["patterns"]


def test_tiny_load_does_not_download(monkeypatch):
    def boom(*args, **kwargs):
        del args, kwargs
        raise AssertionError("tiny load must not download weights")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    adapter = VJEPA2Adapter(layer=1, device=CPU, tiny=True)
    adapter.load(random_init=False)
    adapter.load(random_init=True)


def test_tiny_rejects_a_layer_the_encoder_does_not_have():
    adapter = VJEPA2Adapter(layer=16, device=CPU, tiny=True)
    with pytest.raises(ValueError, match="1 and 2"):
        adapter.load(random_init=False)


def test_model_sha_is_stable_and_changes_with_random_init():
    first = _adapter()
    second = _adapter()
    control = VJEPA2Adapter(layer=1, device=CPU, tiny=True)
    control.load(random_init=True)
    assert first.model_sha() == second.model_sha()
    assert len(first.model_sha()) == 64
    assert control.model_sha() != first.model_sha()


def test_tiny_clips_match_the_grid(tmp_path: Path):
    generate_tiny_clips(tmp_path, 2, seed=0)
    clips = load_clips(tmp_path)
    adapter = _adapter()
    assert clips[0].frames.shape == (4, 32, 32, 3)
    window = adapter.windows(clips[0])[0]
    assert np.array_equal(window.frames, clips[0].frames)
    grid = adapter.grid()
    assert grid.as_dict() == {"t": 2, "h": 2, "w": 2, "tubelet": 2, "patch_px": 16}
    pred, target = adapter.predict_masked(clips[0], window)
    mask = adapter.mask(window)
    assert tuple(pred.shape) == tuple(target.shape)
    assert pred.shape[0] == int(mask.sum())
    assert bool(mask.any()) and not bool(mask.all())
    assert torch.equal(mask[0], mask[1])


def test_capture_matches_hidden_states_and_does_not_track_gradients():
    adapter = _adapter(layer=1)
    clip = _clip(_frames(), "cap")
    window = adapter.windows(clip)[0]
    with adapter.hook(adapter.layer, mode="capture") as handle:
        encoded = adapter.encode(clip, window)
    assert len(handle.captured) == 1
    assert torch.equal(handle.captured[0], encoded)
    captured = handle.captured[0]
    assert captured.grad_fn is None
    assert not captured.requires_grad
    assert not captured.is_inference()

    pixels = adapter.pixel_values(clip, window)
    with torch.inference_mode():
        hidden = adapter.model(
            pixel_values_videos=pixels,
            skip_predictor=True,
            output_hidden_states=True,
        ).hidden_states
    got = captured.reshape(-1, adapter.d_model)
    ref = hidden[adapter.layer][0]
    assert torch.equal(got, ref)

    # A parameter that requires grad must still not attach a graph during capture.
    weight = adapter.model.encoder.layer[0].mlp.fc2.weight
    weight.requires_grad_(True)
    try:
        with adapter.hook(adapter.layer, mode="capture") as again:
            adapter.encode(clip, window)
        assert again.captured[0].grad_fn is None
        assert not again.captured[0].requires_grad
    finally:
        weight.requires_grad_(False)


def test_capture_at_the_last_block_matches_hidden_states():
    adapter = _adapter(layer=2)
    clip = _clip(_frames(1), "cap2")
    window = adapter.windows(clip)[0]
    with adapter.hook(2, mode="capture") as handle:
        adapter.encode(clip, window)
    pixels = adapter.pixel_values(clip, window)
    with torch.inference_mode():
        hidden = adapter.model(
            pixel_values_videos=pixels,
            skip_predictor=True,
            output_hidden_states=True,
        ).hidden_states
    assert torch.equal(handle.captured[0].reshape(-1, adapter.d_model), hidden[2][0])


def test_target_applies_the_training_layer_norm():
    """transformers' target_hidden_state omits the no-affine LN training uses.

    Without that norm the predictor is scored in the encoder's affine space,
    where it is no better than guessing the mean of the visible patches. The
    returned target has to be the normalised tensor, which is a different
    vector once the encoder layernorm affine is not identity.
    """
    adapter = _adapter()
    with torch.no_grad():
        norm = adapter.model.encoder.layernorm
        norm.weight.uniform_(0.3, 2.2)
        norm.bias.uniform_(-1.0, 1.0)
    clip = _clip(_frames(2), "ln")
    window = adapter.windows(clip)[0]
    _pred, target = adapter.predict_masked(clip, window)
    raw = adapter.encoder_output(clip, window)[adapter.mask(window)]
    assert raw.shape == target.shape
    assert torch.equal(target, F.layer_norm(raw, (raw.shape[-1],)))
    assert not torch.allclose(target, raw, atol=1e-3, rtol=0)
    # The training target is normalised per token. The raw HF target is not.
    assert float(target.mean(dim=-1).abs().max()) < 1e-5
    assert float(raw.mean(dim=-1).abs().max()) > 1e-2


def test_masked_pixels_do_not_change_the_prediction():
    """Changing pixels only inside the mask must not change the predictor.

    The target encoder still sees those pixels, so the target moves. Changing a
    visible patch must move the prediction, or the assertion would be vacuous.
    """
    adapter = _adapter()
    clip = _clip(_frames(3), "base")
    window = adapter.windows(clip)[0]
    base = window.frames
    mask = adapter.mask(window).detach().cpu().numpy()
    pred, target = adapter.predict_masked(clip, window)

    masked = _clip(_paint(base, mask, 0.0), "masked")
    pred_masked, target_masked = adapter.predict_masked(masked, adapter.windows(masked)[0])
    assert torch.equal(pred, pred_masked)
    assert not torch.equal(target, target_masked)

    visible_clip = _clip(_paint(base, np.logical_not(mask), 1.0), "visible")
    pred_visible, _target_visible = adapter.predict_masked(visible_clip, adapter.windows(visible_clip)[0])
    assert not torch.equal(pred, pred_visible)


def test_identity_edit_changes_the_output_by_exactly_zero():
    adapter = _adapter()
    clip = _clip(_frames(4), "ident")
    window = adapter.windows(clip)[0]
    pred, target = adapter.predict_masked(clip, window)
    pred_id, target_id = adapter.predict_masked(clip, window, edit=lambda acts: acts)
    assert torch.equal(pred, pred_id)
    assert torch.equal(target, target_id)
    assert float((pred - pred_id).abs().max()) == 0.0


def test_zero_and_identity_are_the_loss_recovered_endpoints():
    adapter = _adapter()
    clips = [_clip(_frames(5), "end")]
    clean = mean_predictor_loss(adapter, clips)
    zero = mean_predictor_loss(adapter, clips, zero=True)
    identity = mean_predictor_loss(adapter, clips, edit=lambda acts: acts)
    zero_edit = mean_predictor_loss(adapter, clips, edit=lambda acts: torch.zeros_like(acts))
    assert identity == clean
    assert zero_edit == zero
    assert loss_recovered(zero, identity, clean) == 1.0
    assert loss_recovered(zero, zero_edit, clean) == 0.0


def test_straight_through_estimate_reaches_the_predictor():
    adapter = _adapter()
    clip = _clip(_frames(6), "est")
    sae = SAE(adapter.d_model, 4)
    with torch.no_grad():
        sae.W_dec.normal_()
        sae.W_enc.copy_(sae.W_dec.T)
        sae.b_enc.zero_()
        sae.b_dec.zero_()
        sae.unit_norm_decoder()
    effect = estimate_effects(adapter, sae, clip, adapter.windows(clip)[0])
    assert effect.shape[-1] == 4
    assert torch.isfinite(effect).all()


def test_mps_fallback_is_enabled_for_an_mps_device(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    ensure_mps_fallback(torch.device("mps"))
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    ensure_mps_fallback(torch.device("mps"))
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
