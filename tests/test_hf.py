"""Generic Hugging Face adapter on tiny random configs saved to disk. No network.

``test_real_dinov2_small_end_to_end`` downloads facebook/dinov2-small and is
marked ``slow``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from interpretableintelligence.adapters import get_adapter
from interpretableintelligence.adapters.base import Clip
from interpretableintelligence.adapters.hf import make_spec, parse_spec
from interpretableintelligence.adapters.vjepa2 import generate_tiny_clips
from interpretableintelligence.cli import app
from interpretableintelligence.steer import Steerer
from interpretableintelligence.validate import validate_run

CPU = torch.device("cpu")
_SMALL = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)


def _configs() -> dict:
    from transformers import Dinov2WithRegistersConfig, VideoMAEConfig, ViTConfig, VJEPA2Config

    return {
        "vit": (ViTConfig(image_size=32, patch_size=8, **_SMALL), (8, 4, 4, 1), 1),
        "dinov2-registers": (Dinov2WithRegistersConfig(image_size=32, patch_size=8, num_register_tokens=4, **_SMALL), (8, 4, 4, 1), 5),
        "videomae": (VideoMAEConfig(image_size=32, patch_size=8, num_frames=4, tubelet_size=2, **_SMALL), (2, 4, 4, 2), 0),
        "vjepa2": (
            VJEPA2Config(
                crop_size=32,
                patch_size=8,
                frames_per_clip=4,
                tubelet_size=2,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                pred_hidden_size=32,
                pred_num_hidden_layers=1,
                pred_num_attention_heads=2,
            ),
            (2, 4, 4, 2),
            0,
        ),
    }


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory) -> dict[str, Path]:
    from transformers import AutoModel

    root = tmp_path_factory.mktemp("hf-models")
    out = {}
    for name, (config, _grid, _prefix) in _configs().items():
        torch.manual_seed(0)
        AutoModel.from_config(config).save_pretrained(root / name)
        out[name] = root / name
    return out


def _clip(seed: int = 0, shape=(10, 40, 48, 3)) -> Clip:
    frames = np.random.default_rng(seed).random(shape, dtype=np.float32)
    return Clip(clip_id=f"c{seed}", path=Path(f"c{seed}.npy"), frames=frames, fps=8.0, sha256="ab" * 32)


def _adapter(path: Path, layer: int = 1, random_init: bool = False):
    adapter = get_adapter(make_spec(str(path)))(layer=layer, device=CPU)
    adapter.load(random_init=random_init)
    return adapter


def test_spec_round_trips_and_rejects_bad_names():
    assert parse_spec("hf:facebook/dinov2-small") == ("facebook/dinov2-small", None)
    assert parse_spec(make_spec("org/m", "encoder.layer")) == ("org/m", "encoder.layer")
    with pytest.raises(ValueError):
        parse_spec("hf:")
    with pytest.raises(ValueError):
        make_spec("a#b")
    with pytest.raises(ValueError, match="hf:<model>"):
        get_adapter("nope")
    with pytest.raises(ValueError, match="--tiny"):
        get_adapter("hf:org/m")(layer=1, device=CPU, tiny=True)


@pytest.mark.parametrize("name", list(_configs()))
def test_grid_prefix_and_edits(tiny_models, name):
    _config, (t, h, w, tubelet), prefix = _configs()[name]
    adapter = _adapter(tiny_models[name])
    grid = adapter.grid()
    assert (grid.t, grid.h, grid.w, grid.tubelet, grid.patch_px) == (t, h, w, tubelet, 8)
    assert adapter._prefix == prefix
    clip = _clip()
    window = adapter.windows(clip)[0]

    with adapter.hook(1, mode="capture") as handle:
        acts = adapter.encode(clip, window)
    assert tuple(acts.shape) == (t, h, w, 32)
    assert torch.equal(handle.captured[0], acts)
    assert bool(adapter.mask(window).all())

    pred, target = adapter.predict_masked(clip, window)
    assert tuple(target.shape) == (t * h * w, 32)
    identity, _ = adapter.predict_masked(clip, window, edit=lambda x: x)
    assert torch.allclose(identity, target, atol=1e-5)
    with adapter.hook(1, mode="zero"):
        zeroed, _ = adapter.predict_masked(clip, window)
    assert float(((zeroed - target) ** 2).mean()) > 1e-3
    baseline = adapter.visible_mean_baseline(clip, window)
    assert baseline.shape == target.shape


def test_edit_only_touches_patch_tokens_and_carries_gradient(tiny_models):
    adapter = _adapter(tiny_models["vit"])
    clip = _clip(1)
    window = adapter.windows(clip)[0]
    scale = torch.ones((), requires_grad=True)
    loss = adapter.attribution_objective(clip, window, "prediction_loss", edit=lambda x: x * scale)
    loss.backward()
    assert scale.grad is not None
    # Editing a single cell moves that frame and leaves the other frames alone (image models batch frames).
    def poke(x):
        x = x.clone()
        x[0, 0, 0] += 5.0
        return x

    edited, target = adapter.predict_masked(clip, window, edit=poke)
    moved = ((edited - target).abs().sum(dim=-1) > 1e-6).reshape(8, 16)
    assert bool(moved[0].any()) and not bool(moved[1:].any())


def test_random_init_changes_the_sha_and_layers_are_checked(tiny_models):
    trained = _adapter(tiny_models["vit"])
    control = _adapter(tiny_models["vit"], random_init=True)
    assert trained.model_sha() != control.model_sha()
    assert control.random_init
    with pytest.raises(ValueError, match="layer 3"):
        _adapter(tiny_models["vit"], layer=3)


def test_ii_run_hf_end_to_end_then_steer(tiny_models, tmp_path):
    clips = tmp_path / "clips"
    generate_tiny_clips(clips, 20)
    out = tmp_path / "runs" / "dino"
    result = CliRunner().invoke(
        app,
        ["run", "hf", "--model", str(tiny_models["dinov2-registers"]), "--clips", str(clips), "--out", str(out), "--prefilter", "2", "--no-serve"],
    )
    assert result.exit_code == 0, result.output
    assert validate_run(out) == []
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["adapter"] == make_spec(str(tiny_models["dinov2-registers"].resolve()))
    assert manifest["layer"] == 1  # middle block of two
    scoreboard = json.loads((out / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["sae"]["loss_visible_mean"] > 0

    steerer = Steerer(out, clips)
    clip_id = manifest["clips"][0]["clip_id"]
    same = steerer.steer({"clip_id": clip_id, "features": [0], "kind": "scale", "value": 1.0})
    moved = steerer.steer({"clip_id": clip_id, "features": [0], "kind": "clamp", "value": 3.0})
    assert same["summary"]["mean_shift"] < 1e-5
    assert moved["summary"]["mean_shift"] > 1e-3


def test_cli_rejects_hf_without_model_and_model_without_hf(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    runner = CliRunner()
    missing = runner.invoke(app, ["run", "hf", "--clips", str(clips), "--out", str(tmp_path / "r"), "--no-serve"])
    assert missing.exit_code == 1 and "--model" in missing.output
    stray = runner.invoke(app, ["run", "fake", "--model", "x", "--clips", str(clips), "--out", str(tmp_path / "r"), "--no-serve"])
    assert stray.exit_code == 1 and "only apply to the hf adapter" in stray.output


@pytest.mark.slow
def test_real_dinov2_small_end_to_end(tmp_path):
    clips = tmp_path / "clips"
    generate_tiny_clips(clips, 20)
    out = tmp_path / "runs" / "dinov2-small"
    result = CliRunner().invoke(
        app,
        ["run", "hf", "--model", "facebook/dinov2-small", "--clips", str(clips), "--out", str(out), "--prefilter", "4", "--no-serve"],
    )
    assert result.exit_code == 0, result.output
    assert validate_run(out) == []
