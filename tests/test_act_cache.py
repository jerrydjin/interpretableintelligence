"""Content-addressed activation cache: keys, damaged entries, reruns, and the playground."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from interpretableintelligence.act_cache import (
    ActivationCache,
    cache_usage,
    captured_layer,
    prime_clean,
    resolve_root,
)
from interpretableintelligence.adapters.base import Clip, Window
from interpretableintelligence.adapters.fake import FakeAdapter
from interpretableintelligence.adapters.vjepa2 import VJEPA2Adapter
from interpretableintelligence.cli import app

CPU = torch.device("cpu")
runner = CliRunner()


def _fake(layer: int = 16) -> FakeAdapter:
    adapter = FakeAdapter(layer=layer, device=CPU)
    adapter.load(random_init=False)
    return adapter


def _fake_clip(seed: int = 0, clip_id: str = "clip_000") -> Clip:
    frames = np.random.default_rng(seed).random((16, 16, 16, 8), dtype=np.float32)
    return Clip(clip_id=clip_id, path=Path(f"{clip_id}.npy"), frames=frames, fps=8.0, sha256=f"{seed:064x}")


def _window(adapter, clip: Clip) -> Window:
    return adapter.windows(clip)[0]


def test_resolve_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("II_ACTS_CACHE", "off")
    assert resolve_root() is None
    assert ActivationCache.open().enabled is False
    monkeypatch.setenv("II_ACTS_CACHE", str(tmp_path / "c"))
    assert resolve_root() == tmp_path / "c"
    monkeypatch.delenv("II_ACTS_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert resolve_root() == tmp_path / "xdg" / "interpretableintelligence" / "acts"
    assert ActivationCache.open(tmp_path / "c", enabled=False).root is None


def test_the_key_moves_with_everything_that_decides_the_activations(tmp_path):
    cache = ActivationCache(tmp_path)
    adapter = _fake()
    sha = adapter.model_sha()
    clip = _fake_clip(0)
    window = _window(adapter, clip)
    base = cache.key("layer", adapter, sha, clip, window)
    assert base is not None
    assert cache.key("layer", adapter, sha, clip, window) == base  # deterministic

    other_frames = Window(index=0, start_frame=0, frames=window.frames + 1.0)
    renamed = Clip(clip.clip_id, clip.path, clip.frames, clip.fps, "f" * 64)
    other_id = Clip("clip_001", clip.path, clip.frames, clip.fps, clip.sha256)  # fake noise is seeded by the id
    variants = {
        "weights": cache.key("layer", adapter, "0" * 64, clip, window),
        "layer": cache.key("layer", _fake(layer=3), sha, clip, window),
        "clip sha": cache.key("layer", adapter, sha, renamed, window),
        "frames": cache.key("layer", adapter, sha, clip, other_frames),
        "kind": cache.key("clean", adapter, sha, clip, window),
        "preprocessing": cache.key("layer", adapter, sha, other_id, window),
    }
    for name, key in variants.items():
        assert key is not None and key.digest != base.digest, name
    assert cache.key("layer", adapter, "", clip, window) is None
    unhashed = Clip(clip.clip_id, clip.path, clip.frames, clip.fps, "")
    assert cache.key("layer", adapter, sha, unhashed, window) is None


def test_adapters_without_a_cache_spec_are_never_cached(tmp_path):
    class Bare:
        name = "bare"
        layer = 1

    cache = ActivationCache(tmp_path)
    clip = _fake_clip(0)
    window = Window(index=0, start_frame=0, frames=clip.frames)
    assert cache.key("layer", Bare(), "ab" * 32, clip, window) is None
    assert ActivationCache(None).key("layer", _fake(), "ab" * 32, clip, window) is None


def test_store_and_load_are_bit_exact_and_damaged_entries_are_misses(tmp_path):
    cache = ActivationCache(tmp_path)
    adapter = _fake()
    sha = adapter.model_sha()
    clip = _fake_clip(0)
    window = _window(adapter, clip)

    first = captured_layer(adapter, clip, window, cache=cache, model_sha=sha)
    assert (cache.hits, cache.misses, cache.stored) == (0, 1, 1)
    again = captured_layer(adapter, clip, window, cache=cache, model_sha=sha)
    assert cache.hits == 1
    assert again.dtype == torch.float32 and torch.equal(first, again)
    with adapter.hook(adapter.layer, mode="capture") as handle:
        adapter.encode(clip, window)
    assert torch.equal(handle.captured[0].float(), again)

    key = cache.key("layer", adapter, sha, clip, window)
    path = cache.path(key)
    path.write_bytes(path.read_bytes()[:20])  # truncated, as after a crash
    assert cache.load(key) is None and not path.exists()

    # A file whose stored key differs (a collision, or a copied file) is never returned.
    other = cache.key("layer", adapter, sha, _fake_clip(1), window)
    cache.store(other, {"acts": first})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cache.path(other).read_bytes())
    assert cache.load(key) is None


def _run(clips: Path, out: Path, *extra: str):
    return runner.invoke(
        app,
        ["run", "fake", "--clips", str(clips), "--out", str(out), "--layer", "16", "--seed", "0",
         "--heldout-frac", "0.2", "--no-serve", "--no-open", *extra],
    )


def _stream(out: Path) -> bytes:
    return (out / "acts" / "stream" / "rows.f32").read_bytes()


def test_a_rerun_reads_the_cache_and_writes_the_same_run(demo, tmp_path, monkeypatch):
    monkeypatch.setenv("II_ACTS_CACHE", str(tmp_path / "cache"))
    first = _run(demo.clips, tmp_path / "a")
    assert first.exit_code == 0, first.output
    usage = cache_usage(tmp_path / "cache")
    assert usage["kinds"]["layer"]["entries"] == 60  # 30 clips, trained model and random-init control
    second = _run(demo.clips, tmp_path / "b")
    assert second.exit_code == 0, second.output
    assert "60 hit(s), 0 miss(es)" in second.output
    for name in ("scoreboard.json", "split.json"):
        assert json.loads((tmp_path / "a" / name).read_text()) == json.loads((tmp_path / "b" / name).read_text())
    assert _stream(tmp_path / "a") == _stream(tmp_path / "b") == _stream(demo.run)
    assert _stream(tmp_path / "a" / "control") == _stream(tmp_path / "b" / "control")
    assert cache_usage(tmp_path / "cache")["entries"] == 60


def test_no_cache_reads_and_writes_nothing(demo, tmp_path, monkeypatch):
    monkeypatch.setenv("II_ACTS_CACHE", str(tmp_path / "cache"))
    ran = _run(demo.clips, tmp_path / "a", "--no-score", "--no-cache")
    assert ran.exit_code == 0, ran.output
    assert "activation cache" not in ran.output
    assert cache_usage(tmp_path / "cache")["entries"] == 0


def test_cache_command_reports_and_clears(tmp_path, monkeypatch):
    monkeypatch.setenv("II_ACTS_CACHE", str(tmp_path / "cache"))
    adapter = _fake()
    cache = ActivationCache.open()
    for seed in range(3):
        clip = _fake_clip(seed, f"clip_{seed:03d}")
        captured_layer(adapter, clip, _window(adapter, clip), cache=cache, model_sha=adapter.model_sha())
    shown = runner.invoke(app, ["cache", "--json"])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output)["kinds"]["layer"]["entries"] == 3
    cleared = runner.invoke(app, ["cache", "--clear"])
    assert cleared.exit_code == 0 and "removed 3" in cleared.output
    assert cache_usage(tmp_path / "cache")["entries"] == 0
    monkeypatch.setenv("II_ACTS_CACHE", "off")
    assert "off" in runner.invoke(app, ["cache"]).output


def _vjepa2() -> VJEPA2Adapter:
    adapter = VJEPA2Adapter(layer=1, device=CPU, tiny=True)
    adapter.load(random_init=False)
    return adapter


def test_vjepa2_clean_state_replaces_the_forward_passes(tmp_path, monkeypatch):
    frames = np.random.default_rng(0).random((4, 32, 32, 3), dtype=np.float32)
    clip = Clip(clip_id="c0", path=Path("c0.npy"), frames=frames, fps=8.0, sha256="ab" * 32)
    cache = ActivationCache(tmp_path)

    cold = _vjepa2()
    window = cold.windows(clip)[0]
    assert prime_clean(cold, clip, window, cache=cache, model_sha=cold.model_sha()) is False
    acts = cold.encode(clip, window)
    pred, target = cold.predict_masked(clip, window)
    edit = lambda x: x * 0.5  # noqa: E731
    edited, _ = cold.predict_masked(clip, window, edit=edit)

    warm = _vjepa2()
    calls = {"n": 0}
    embeddings = warm.model.encoder.embeddings
    original = embeddings.forward

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(embeddings, "forward", counted)
    assert prime_clean(warm, clip, window, cache=cache, model_sha=warm.model_sha()) is True
    assert torch.equal(warm.encode(clip, window), acts)
    warm_pred, warm_target = warm.predict_masked(clip, window)
    warm_edited, _ = warm.predict_masked(clip, window, edit=edit)
    assert torch.equal(warm_pred, pred) and torch.equal(warm_target, target) and torch.equal(warm_edited, edited)
    assert calls["n"] == 0  # nothing ran through the encoder embeddings: every pre-layer pass came from disk


def test_a_cache_entry_that_does_not_fit_is_recomputed(tmp_path):
    frames = np.random.default_rng(0).random((4, 32, 32, 3), dtype=np.float32)
    clip = Clip(clip_id="c0", path=Path("c0.npy"), frames=frames, fps=8.0, sha256="ab" * 32)
    cache = ActivationCache(tmp_path)
    adapter = _vjepa2()
    window = adapter.windows(clip)[0]
    key = cache.key("clean", adapter, adapter.model_sha(), clip, window)
    cache.store(key, {"layer": torch.zeros(1, 2, 3), "visible": torch.zeros(1), "encoder": torch.zeros(1)})
    assert prime_clean(adapter, clip, window, cache=cache, model_sha=adapter.model_sha()) is False
    assert prime_clean(_vjepa2(), clip, window, cache=cache, model_sha=adapter.model_sha()) is True


def test_playground_cold_start_matches_and_tweaks_reuse_the_clean_pass(demo, tmp_path, monkeypatch):
    from interpretableintelligence.steer import Steerer

    monkeypatch.setenv("II_ACTS_CACHE", "off")
    plain = Steerer(demo.run, demo.clips)
    clip_id = json.loads((demo.run / "manifest.json").read_text())["clips"][0]["clip_id"]
    request = {"clip_id": clip_id, "window": 0, "features": [0], "kind": "scale", "value": 3.0}
    expected = plain.steer(request)

    calls = {"n": 0}
    original = plain._compute_clean

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(plain, "_compute_clean", counted)
    for value in (0.5, 2.0, 3.0):
        result = plain.steer({**request, "value": value})
    assert result == expected
    assert calls["n"] == 0  # the clean pass was computed once, on the first request


@pytest.mark.parametrize("name", ["vit", "dinov2-registers"])
def test_hf_clean_state_round_trips(tmp_path, name):
    from transformers import AutoModel, Dinov2WithRegistersConfig, ViTConfig

    from interpretableintelligence.adapters import get_adapter
    from interpretableintelligence.adapters.hf import make_spec

    small = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    config = (
        ViTConfig(image_size=32, patch_size=8, **small)
        if name == "vit"
        else Dinov2WithRegistersConfig(image_size=32, patch_size=8, num_register_tokens=4, **small)
    )
    torch.manual_seed(0)
    AutoModel.from_config(config).save_pretrained(tmp_path / name)

    def adapter():
        loaded = get_adapter(make_spec(str(tmp_path / name)))(layer=1, device=CPU)
        loaded.load()
        return loaded

    frames = np.random.default_rng(0).random((10, 40, 48, 3), dtype=np.float32)
    clip = Clip(clip_id="c0", path=Path("c0.npy"), frames=frames, fps=8.0, sha256="ab" * 32)
    cache = ActivationCache(tmp_path / "cache")
    cold = adapter()
    window = cold.windows(clip)[0]
    assert prime_clean(cold, clip, window, cache=cache, model_sha=cold.model_sha()) is False
    acts, (pred, target) = cold.encode(clip, window), cold.predict_masked(clip, window)

    warm = adapter()
    forwards = {"n": 0}
    hook = warm.model.register_forward_pre_hook(lambda *_: forwards.__setitem__("n", forwards["n"] + 1))
    assert prime_clean(warm, clip, window, cache=cache, model_sha=warm.model_sha()) is True
    assert torch.equal(warm.encode(clip, window), acts)
    warm_pred, warm_target = warm.predict_masked(clip, window)
    hook.remove()
    assert torch.equal(warm_pred, pred) and torch.equal(warm_target, target)
    assert forwards["n"] == 0
