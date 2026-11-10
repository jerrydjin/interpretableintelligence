"""Regression: re-exec'ing physint_inproc_v9 in one runtime must re-point the sample_noise patch at
the new module instance (Colab capture failed with 'sample_noise called 0x' after a second exec)."""
import importlib.util, sys, types
from pathlib import Path
import torch

SRC = Path(__file__).resolve().parents[1] / "v9" / "physint_inproc_v9.py"


def _fake_lerobot():
    gm = types.ModuleType("lerobot.policies.groot.groot_n1_7")
    gm.sample_noise = lambda shape, device, dtype=torch.float32, distribution="normal": torch.zeros(shape)
    for name in ["lerobot", "lerobot.policies", "lerobot.policies.groot"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[gm.__name__] = gm
    sys.modules["lerobot.policies.groot"].groot_n1_7 = gm
    return gm


def _load(tag):
    spec = importlib.util.spec_from_file_location(f"v9_{tag}", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_reexec_routes_to_new_instance():
    gm = _fake_lerobot()
    if hasattr(gm, "_v9_orig_sample_noise"):
        del gm._v9_orig_sample_noise
    orig = gm.sample_noise
    a = _load("a"); a.install_noise_router()
    b = _load("b"); b.install_noise_router()
    b._ROUTE.gens, b._ROUTE.calls = [torch.Generator().manual_seed(0)], 0
    gm.sample_noise((1, 4, 3), "cpu")
    assert b._ROUTE.calls == 1 and a._ROUTE.calls == 0
    b._ROUTE.gens = None
    assert b._ORIG_SAMPLE_NOISE is orig  # never wraps a router


def test_unwraps_stale_prefix_router():
    gm = _fake_lerobot()
    orig = gm.sample_noise
    old = _load("old")
    # simulate the pre-fix install: router in place, no saved original on gm
    old._ORIG_SAMPLE_NOISE = orig
    old._routed_sample_noise._v9_routed = True
    gm.sample_noise = old._routed_sample_noise
    new = _load("new"); new.install_noise_router()
    assert new._ORIG_SAMPLE_NOISE is orig
    new._ROUTE.gens, new._ROUTE.calls = [torch.Generator().manual_seed(0)], 0
    gm.sample_noise((1, 4, 3), "cpu")
    assert new._ROUTE.calls == 1
    new._ROUTE.gens = None
