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


@pytest.mark.parametrize("name", list(_configs()))
def test_batched_edits_match_one_forward_each(tiny_models, name, monkeypatch):
    import interpretableintelligence.adapters.hf as hf

    monkeypatch.setattr(hf, "MANY_BATCH_ITEMS", 16)  # forces several chunks
    adapter = _adapter(tiny_models[name])
    clip = _clip(2)
    window = adapter.windows(clip)[0]
    edits = [(lambda x, k=k: x * (1.0 - 0.1 * k)) for k in range(5)]
    with torch.no_grad():
        batched = adapter.predict_masked_many(clip, window, edits)
        looped = [adapter.predict_masked(clip, window, edit=edit)[0] for edit in edits]
    assert len(batched) == len(looped)
    for one, many in zip(looped, batched, strict=True):
        assert torch.allclose(one, many, atol=1e-5)
    # The swapped-out blocks are restored afterwards.
    again, target = adapter.predict_masked(clip, window, edit=lambda x: x)
    assert torch.allclose(again, target, atol=1e-5)


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


def test_input_size_follows_the_preprocessor_when_the_model_accepts_it(tmp_path):
    """DINOv2's config says 518 px but its preprocessor crops to 224. Feed what the model card feeds."""
    from transformers import AutoModel, Dinov2Config, VideoMAEConfig

    torch.manual_seed(0)
    dino = tmp_path / "dino"
    AutoModel.from_config(Dinov2Config(image_size=64, patch_size=8, **_SMALL)).save_pretrained(dino)
    (dino / "preprocessor_config.json").write_text(json.dumps({"crop_size": {"height": 32, "width": 32}, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}))
    adapter = _adapter(dino)
    assert (adapter._size, adapter.grid().h) == (32, 4)
    assert adapter._mean == (0.5, 0.5, 0.5)

    # VideoMAE's position table is fixed to its config size, so it keeps 32 px.
    vmae = tmp_path / "vmae"
    AutoModel.from_config(VideoMAEConfig(image_size=32, patch_size=8, num_frames=4, tubelet_size=2, **_SMALL)).save_pretrained(vmae)
    (vmae / "preprocessor_config.json").write_text(json.dumps({"crop_size": {"height": 16, "width": 16}}))
    adapter = _adapter(vmae)
    assert (adapter._size, adapter.grid().h) == (32, 4)


def test_sae_width_scales_with_the_model_for_hf_only():
    from interpretableintelligence.pipeline import sae_dictionary_size

    assert sae_dictionary_size("fake", 64) == 12
    assert sae_dictionary_size("vjepa2", 1024) == 12
    assert sae_dictionary_size("hf:facebook/dinov2-small", 384) == 8 * 384
    assert sae_dictionary_size("hf:x", 384, requested=100) == 100
    with pytest.raises(ValueError):
        sae_dictionary_size("hf:x", 384, requested=0)


def test_random_init_changes_the_sha_and_layers_are_checked(tiny_models):
    trained = _adapter(tiny_models["vit"])
    control = _adapter(tiny_models["vit"], random_init=True)
    assert trained.model_sha() != control.model_sha()
    assert control.random_init
    with pytest.raises(ValueError, match="layer 3"):
        _adapter(tiny_models["vit"], layer=3)


def _colour_similarity(self, frames):
    """Stand-in for CLIP: concept i is the mean of colour channel i % 3 on a 4x4 patch grid."""
    pixels = torch.from_numpy(np.ascontiguousarray(frames)).float().permute(0, 3, 1, 2)
    pooled = torch.nn.functional.adaptive_avg_pool2d(pixels, 4)  # [frames, 3, 4, 4]
    return torch.stack([pooled[:, i % 3] for i in range(len(self.names))], dim=1)


def test_ii_run_hf_end_to_end_then_steer(tiny_models, tmp_path, monkeypatch):
    from interpretableintelligence.labelers import CLIP_LABEL_MODEL, ClipConceptSource

    monkeypatch.setattr(ClipConceptSource, "_dense_similarity", _colour_similarity)
    concepts = tmp_path / "concepts.txt"
    concepts.write_text("red things\n# a comment\ngreen things\nblue things\n")
    clips = tmp_path / "clips"
    generate_tiny_clips(clips, 20)
    out = tmp_path / "runs" / "dino"
    result = CliRunner().invoke(
        app,
        [
            "run", "hf", "--model", str(tiny_models["dinov2-registers"]), "--clips", str(clips), "--out", str(out),
            "--prefilter", "2", "--concepts", str(concepts), "--no-serve",
        ],
    )
    assert result.exit_code == 0, result.output
    assert validate_run(out) == []
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["adapter"] == make_spec(str(tiny_models["dinov2-registers"].resolve()))
    assert manifest["layer"] == 1  # middle block of two
    scoreboard = json.loads((out / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["sae"]["loss_visible_mean"] > 0
    sae_config = json.loads((out / "sae" / "config.json").read_text(encoding="utf-8"))
    assert sae_config["n_features"] == 8 * 32
    assert sae_config["activation"] == "topk"
    features = [json.loads(line) for line in (out / "features" / "features.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {feature.get("label_model") for feature in features} == {CLIP_LABEL_MODEL}
    assert {feature["label"] for feature in features} <= {"", "red things", "green things", "blue things"}

    steerer = Steerer(out, clips)
    clip_id = manifest["clips"][0]["clip_id"]
    same = steerer.steer({"clip_id": clip_id, "features": [0], "kind": "scale", "value": 1.0})
    moved = steerer.steer({"clip_id": clip_id, "features": [0], "kind": "clamp", "value": 3.0})
    assert same["summary"]["mean_shift"] < 1e-5
    assert moved["summary"]["mean_shift"] > 1e-3


def test_concept_maps_follow_the_token_grid(tiny_models):
    from transformers import CLIPConfig, CLIPModel

    from interpretableintelligence.labelers import ClipConceptSource, clip_text_embeddings, load_concepts

    assert len(load_concepts()) > 50
    for name, frames in (("vit", 8), ("videomae", 4)):
        adapter = _adapter(tiny_models[name])
        source = ClipConceptSource(adapter, ["a", "b", "c"])
        torch.manual_seed(0)
        config = CLIPConfig(
            text_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2},
            vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": 32, "patch_size": 8},
            projection_dim=8,
        )
        source._model = CLIPModel(config).eval()  # random weights; bypasses the download and tokenizer
        ids = torch.randint(0, 100, (3, 5))
        text = clip_text_embeddings(source._model, ids, torch.ones_like(ids))
        assert tuple(text.shape) == (3, 8)
        source._text = torch.nn.functional.normalize(text, dim=-1)
        window = adapter.windows(_clip(3))[0]
        maps = source.concept_maps(window.frames)
        grid = adapter.grid()
        assert window.frames.shape[0] == frames
        assert maps.shape == (3, grid.t, grid.h, grid.w)
        assert np.isfinite(maps).all()


def test_cli_rejects_hf_without_model_and_model_without_hf(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    runner = CliRunner()
    missing = runner.invoke(app, ["run", "hf", "--clips", str(clips), "--out", str(tmp_path / "r"), "--no-serve"])
    assert missing.exit_code == 1 and "--model" in missing.output
    stray = runner.invoke(app, ["run", "fake", "--model", "x", "--clips", str(clips), "--out", str(tmp_path / "r"), "--no-serve"])
    assert stray.exit_code == 1 and "only apply to the hf" in stray.output


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


def test_auroc_labels_a_sparse_feature_that_pearson_would_miss(monkeypatch):
    import interpretableintelligence.pipeline as pipeline

    rng = np.random.default_rng(0)
    n = 4000
    concepts = rng.random((n, 2)).astype(np.float32)
    split = (rng.random(n) < 0.25).astype(np.uint8)
    planted = concepts[:, 0] > 0.97  # fires on ~3% of tokens, exactly where concept A is strong
    noise = rng.random(n) < 0.03
    acts = np.stack([planted, noise], axis=1).astype(np.float32)

    class Rows:
        def __init__(self, _path):
            self.rows = acts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def chunks(self, size):
            for start in range(0, n, size):
                yield start, min(n, start + size)

    class Identity:
        def encode(self, x):
            return x

    monkeypatch.setattr(pipeline, "RowSource", Rows)
    labels = pipeline._AurocLabels(2, 2)
    labels.add_concepts(concepts[:1500], split[:1500])
    labels.add_concepts(concepts[1500:], split[1500:])
    out = labels.labels(None, Identity(), torch.device("cpu"), 700, ["A", "B"], "m")
    assert out[0] == {"label": "A", "label_model": "m", "label_holds_heldout": True}
    assert out[1]["label"] == "" and out[1]["label_holds_heldout"] is False
    assert np.corrcoef(planted, concepts[:, 0])[0, 1] < 0.5  # the Pearson gate would not label it


def test_score_stage_holds_one_decoded_clip_at_a_time(tiny_models, tmp_path, monkeypatch):
    """A 100-episode DROID run was killed for memory when attribution decoded every clip at once."""
    import weakref

    import interpretableintelligence.pipeline as pipeline
    from interpretableintelligence.labelers import ClipConceptSource

    monkeypatch.setattr(ClipConceptSource, "_dense_similarity", _colour_similarity)
    live = {"now": 0, "peak": 0}
    real_load = pipeline.load_clip

    def counted(item):
        clip = real_load(item)
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        weakref.finalize(clip, lambda: live.__setitem__("now", live["now"] - 1))
        return clip

    monkeypatch.setattr(pipeline, "load_clip", counted)
    clips = tmp_path / "clips"
    generate_tiny_clips(clips, 20)
    out = tmp_path / "runs" / "lazy"
    result = CliRunner().invoke(
        app,
        ["run", "hf", "--model", str(tiny_models["vit"]), "--clips", str(clips), "--out", str(out), "--prefilter", "2", "--no-serve"],
    )
    assert result.exit_code == 0, result.output
    assert live["peak"] <= 2, live
