"""RoPE theta verification: a mismatch must raise; repair of an already-loaded policy still works."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v8"))

import low_ram_policy_load as L  # noqa: E402
import inproc_v8 as v8  # noqa: E402


class _Cfg(types.SimpleNamespace):
    pass


class _Rope(torch.nn.Module):
    """Minimal RoPE stand-in: inv_freq from config.rope_parameters['rope_theta']."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        theta = float(config.rope_parameters["rope_theta"])
        n = int(getattr(config, "n_freq", 16))
        inv = 1.0 / (theta ** (torch.arange(0, n, dtype=torch.float32) / n))
        self.inv_freq = torch.nn.Buffer(inv, persistent=False)
        self.original_inv_freq = torch.nn.Buffer(inv.clone(), persistent=False)


class _BadRope(_Rope):
    """Ignores config theta and always builds inv_freq with 10000 — the v7 failure mode."""

    def __init__(self, config):
        super().__init__(config)
        n = int(self.inv_freq.numel())
        wrong = 1.0 / (10000.0 ** (torch.arange(0, n, dtype=torch.float32) / n))
        self.inv_freq = torch.nn.Buffer(wrong, persistent=False)
        self.original_inv_freq = torch.nn.Buffer(wrong.clone(), persistent=False)


def _text_cfg(theta=5_000_000.0, n_freq=16):
    return _Cfg(rope_parameters={"rope_theta": theta, "rope_type": "default"}, n_freq=n_freq)


def _wrap(module, name="language_model.rotary_emb"):
    root = torch.nn.Module()
    root.add_module("rotary_emb", module)
    # expose the same dotted name the reports use
    holder = torch.nn.Module()
    holder.add_module("language_model", root)
    return holder


def test_verify_ok_for_matching_theta():
    m = _Rope(_text_cfg(5_000_000.0))
    assert L.verify_rotary_inv_freq(m, "language_model.rotary_emb") <= 1e-5
    assert v8.verify_rotary_inv_freq(m, "language_model.rotary_emb") <= 1e-5


def test_verify_raises_on_mismatched_theta():
    m = _Rope(_text_cfg(5_000_000.0))
    n = m.inv_freq.numel()
    m.inv_freq = torch.nn.Buffer(1.0 / (10000.0 ** (torch.arange(0, n, dtype=torch.float32) / n)), persistent=False)
    with pytest.raises(RuntimeError, match=r"language_model.rotary_emb.*expected theta=5000000"):
        L.verify_rotary_inv_freq(m, "language_model.rotary_emb")
    with pytest.raises(RuntimeError, match=r"language_model.rotary_emb.*expected theta=5000000"):
        v8.verify_rotary_inv_freq(m, "language_model.rotary_emb")


def test_rebuild_raises_if_class_ignores_config_theta():
    m = _BadRope(_text_cfg(5_000_000.0))
    with pytest.raises(RuntimeError, match=r"expected theta=5000000"):
        L.rebuild_rotary_inv_freq(m, "cpu")
    with pytest.raises(RuntimeError, match=r"expected theta=5000000"):
        v8._reference_inv_freq(m)


def test_repair_fixes_already_loaded_then_verifies():
    """v7-style corruption is repaired in place; a second call is a no-op; verify passes."""
    m = _Rope(_text_cfg(5_000_000.0))
    n = m.inv_freq.numel()
    correct = m.inv_freq.detach().clone()
    m.inv_freq = torch.nn.Buffer(1.0 / (10000.0 ** (torch.arange(0, n, dtype=torch.float32) / n)), persistent=False)
    pol = _wrap(m)
    # point named_modules at the actual module with config
    # _wrap nests as language_model.rotary_emb
    rep = L.repair_rotary_buffers(pol)
    hit = [r for r in rep if r["module"].endswith("rotary_emb")]
    assert hit and hit[0]["repaired"] is True and hit[0]["theta"] == 5_000_000.0
    assert torch.allclose(m.inv_freq.float(), correct)
    L.verify_rotary_inv_freq(m, "language_model.rotary_emb")
    rep2 = L.repair_rotary_buffers(pol)
    assert all(not r["repaired"] for r in rep2)


def test_v8_repair_rope_raises_if_repair_cannot_match_theta():
    m = _BadRope(_text_cfg(5_000_000.0))
    pol = _wrap(m)
    with pytest.raises(RuntimeError, match=r"expected theta=5000000"):
        v8.repair_rope(pol, log=lambda s: None)


def test_rope_theta_missing_raises_not_default_10000():
    with pytest.raises(RuntimeError, match="no rope_theta"):
        L._rope_theta(_Cfg())
    with pytest.raises(RuntimeError, match="no config"):
        L._rope_theta(None)


def test_vision_and_text_use_their_own_theta():
    text = _Rope(_text_cfg(5_000_000.0))
    vision = _Rope(_Cfg(rope_parameters={"rope_theta": 10_000.0, "rope_type": "axial"}, n_freq=8))
    assert L.verify_rotary_inv_freq(text, "text.rotary_emb") <= 1e-5
    assert L.verify_rotary_inv_freq(vision, "visual.rotary_emb") <= 1e-5
    # text buffer built with vision's theta must fail against the text config
    n = text.inv_freq.numel()
    text.inv_freq = torch.nn.Buffer(
        1.0 / (10_000.0 ** (torch.arange(0, n, dtype=torch.float32) / n)), persistent=False
    )
    with pytest.raises(RuntimeError, match=r"text.rotary_emb.*expected theta=5000000"):
        L.verify_rotary_inv_freq(text, "text.rotary_emb")
