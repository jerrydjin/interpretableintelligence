"""Auto-interp labeller: explain on train clips, simulate on held-out clips, same AUROC gate.

No network: a stub vision client stands in for the provider.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest
from typer.testing import CliRunner

from interpretableintelligence import autointerp
from interpretableintelligence.autointerp import (
    AUTOINTERP_VERSION,
    CachedClient,
    Tokens,
    auroc,
    client_from_env,
    parse_label,
    parse_scores,
    pick_top,
    png_bytes,
    provider_from_env,
    render,
    simulation_set,
)
from interpretableintelligence.thresholds import LABEL_HOLD_MIN_AUROC


class Stub:
    name = "stub/vision"

    def __init__(self, label: str = "the gripper", score: float = 5.0) -> None:
        self.label = label
        self.score = score
        self.calls: list[tuple[str, int]] = []

    def complete(self, prompt: str, images: list[bytes]) -> str:
        self.calls.append((prompt, len(images)))
        if prompt.startswith("These"):
            return json.dumps({"label": self.label, "description": "The robot's two-finger gripper."})
        return json.dumps([self.score] * len(images))


def test_png_is_valid_and_lossless():
    image = np.random.default_rng(0).integers(0, 255, (7, 5, 3), dtype=np.uint8)
    data = png_bytes(image)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    idat = data[data.index(b"IDAT") + 4 : data.index(b"IEND") - 8]
    raw = zlib.decompress(idat)
    rows = [raw[i * 16 + 1 : (i + 1) * 16] for i in range(7)]
    assert np.array_equal(np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(7, 5, 3), image)


def test_render_boxes_the_cell_in_red():
    frame = np.full((40, 60, 3), 128, dtype=np.uint8)
    data = render(frame, (0, 0), (4, 4), heat=np.eye(4))
    assert data.startswith(b"\x89PNG")


def test_auroc_counts_ties_half():
    assert auroc(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert auroc(np.array([5, 5, 5, 5]), np.array([0, 0, 1, 1], bool)) == 0.5
    assert auroc(np.array([4, 3, 2, 1]), np.array([0, 0, 1, 1], bool)) == 0.0
    assert auroc(np.array([1, 2]), np.array([1, 1], bool)) == 0.5


def test_replies_are_parsed_defensively():
    assert parse_label('Sure! {"label": "red  mug", "description": "A mug."}') == ("red mug", "A mug.")
    assert parse_label("table edge\nmore") == ("table edge", "")
    assert parse_scores("scores: [1, 12, -3]", 3) == [1.0, 10.0, 0.0]
    assert parse_scores("[1, 2]", 3) is None
    assert parse_scores("no list", 1) is None


def test_provider_follows_the_environment():
    assert provider_from_env({}) is None
    assert provider_from_env({"FIREWORKS_API_KEY": "k"})[0] == "fireworks"
    assert provider_from_env({"FIREWORKS_API_KEY": "k", "ANTHROPIC_API_KEY": "k"})[0] == "anthropic"
    assert provider_from_env({"FIREWORKS_API_KEY": "k", "II_LABEL_PROVIDER": "none"}) is None
    assert provider_from_env({"OPENAI_API_KEY": "k", "II_LABEL_MODEL": "m"}) == ("openai", "m")
    with pytest.raises(ValueError):
        provider_from_env({"II_LABEL_PROVIDER": "fireworks"})
    with pytest.raises(ValueError):
        provider_from_env({"II_LABEL_PROVIDER": "nope", "FIREWORKS_API_KEY": "k"})
    client = client_from_env({"FIREWORKS_API_KEY": "k", "II_LABEL_CACHE": "/tmp/x"})
    assert isinstance(client, CachedClient) and client.name.startswith("fireworks/")


def test_cache_answers_repeats_without_calling(tmp_path):
    stub = Stub()
    cached = CachedClient(stub, tmp_path)
    first = cached.complete("These 1 images", [b"a"])
    again = cached.complete("These 1 images", [b"a"])
    other = cached.complete("These 1 images", [b"b"])
    assert first == again == other
    assert len(stub.calls) == 2


def _tokens() -> tuple[Tokens, np.ndarray]:
    # 6 clips (4 train, 2 held-out) x 4 frames x 4 x 4 cells. The feature fires in the top-left cell.
    clip, t, h, w = np.meshgrid(np.arange(6), np.arange(4), np.arange(4), np.arange(4), indexing="ij")
    clip, t, h, w = (a.ravel() for a in (clip, t, h, w))
    split = (clip >= 4).astype(np.uint8)
    column = np.where((h == 0) & (w == 0), 1.0 + t + clip, 0.0).astype(np.float32)
    tokens = Tokens(clip=clip, window=np.zeros_like(clip), t=t, h=h, w=w, split=split, acts=column[:, None])
    return tokens, column


def test_examples_come_from_train_and_simulation_from_held_out():
    tokens, column = _tokens()
    train = np.flatnonzero(tokens.split == 0)
    top = pick_top(tokens, column, train, k=8, per_clip=2)
    assert len(top) == 8 and all(tokens.split[row] == 0 for row in top)
    assert len({(tokens.clip[r], tokens.t[r]) for r in top}) == 8
    rows, truth = simulation_set(tokens, column, np.random.default_rng(0), n=6)
    assert all(tokens.split[row] == 1 for row in rows)
    assert sum(truth) == len(truth) - sum(truth) > 0
    for row, fires in zip(rows, truth, strict=True):
        assert (column[row] > 0) == fires


def test_a_constant_simulator_cannot_pass_the_gate(monkeypatch):
    tokens, _column = _tokens()

    class Grid:
        t, h, w, tubelet = 4, 4, 4, 1

    class Adapter:
        def grid(self):
            return Grid()

    monkeypatch.setattr(autointerp, "gather_tokens", lambda *a, **k: tokens)
    monkeypatch.setattr(autointerp._Frames, "get", lambda self, c, wi, t: np.zeros((8, 8, 3), np.uint8))
    stub = Stub(score=7)
    labels, evidence = autointerp.label_features(
        client=stub, adapter=Adapter(), meta=[], acts_dir=None, sae=None, feature_ids=[3], device=None, seed=0, log=lambda _m: None
    )
    assert labels[3]["label"] == "the gripper"
    assert labels[3]["label_model"] == f"{AUTOINTERP_VERSION}:stub/vision"
    assert labels[3]["label_holds_heldout"] is False
    assert evidence["features"][0]["auroc_heldout"] == 0.5 < LABEL_HOLD_MIN_AUROC


def test_ii_run_hf_uses_the_vision_labeller_when_a_key_is_set(tmp_path, monkeypatch):
    from transformers import AutoModel, ViTConfig

    from interpretableintelligence.adapters.vjepa2 import generate_tiny_clips
    from interpretableintelligence.cli import app
    from interpretableintelligence.labelers import ClipConceptSource
    from interpretableintelligence.validate import validate_run

    import torch

    torch.manual_seed(0)
    model_dir = tmp_path / "vit"
    AutoModel.from_config(
        ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    ).save_pretrained(model_dir)
    monkeypatch.setattr(
        ClipConceptSource, "_dense_similarity", lambda self, frames: torch.zeros(frames.shape[0], len(self.names), 4, 4)
    )
    stub = Stub(score=5)
    monkeypatch.setattr(autointerp, "client_from_env", lambda env=None: stub)
    monkeypatch.setenv("II_LABEL_PROVIDER", "fireworks")
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    clips = tmp_path / "clips"
    generate_tiny_clips(clips, 20)
    out = tmp_path / "run"
    result = CliRunner().invoke(
        app, ["run", "hf", "--model", str(model_dir), "--clips", str(clips), "--out", str(out), "--prefilter", "2", "--no-serve"]
    )
    assert result.exit_code == 0, result.output
    assert validate_run(out) == []
    evidence = json.loads((out / "features" / "autointerp.json").read_text(encoding="utf-8"))
    assert evidence["model"] == "stub/vision" and len(evidence["features"]) == 2
    features = [json.loads(line) for line in (out / "features" / "features.jsonl").read_text(encoding="utf-8").splitlines()]
    relabelled = [f for f in features if f["label_model"].startswith(AUTOINTERP_VERSION)]
    assert sorted(f["id"] for f in relabelled) == sorted(f["id"] for f in evidence["features"])
    assert not any(f["real"] for f in relabelled)  # a constant simulator never verifies anything
