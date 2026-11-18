"""``ii map-feature``: carry a GR00T SAE feature into another dictionary by what it does on the same tokens."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from test_groot_random_sae import D, PLANTED, _write_capture
from typer.testing import CliRunner

from interpretableintelligence.adapters.groot import map_feature as MF
from interpretableintelligence.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def capture(tmp_path_factory) -> Path:
    return _write_capture(tmp_path_factory.mktemp("map-capture"))


def _permuted(capture: Path, out: Path, perm: np.ndarray, *, channel_scale: bool) -> Path:
    """The capture SAE's ``len(perm)`` features, reordered; optionally refolded with a per-channel norm_scale.

    The fixture's first D features are orthonormal (the rest are random directions that overlap them), so keeping only
    those gives a dictionary whose other features do not track the planted one."""
    weights = load_file(str(capture / "sae" / "weights.safetensors"))
    idx = torch.as_tensor(perm)
    new = {
        "W_enc": weights["W_enc"][:, idx].contiguous(),
        "W_dec": weights["W_dec"][idx].contiguous(),
        "b_enc": weights["b_enc"][idx].contiguous(),
        "b_dec": weights["b_dec"].clone(),
        "norm_mu": weights["norm_mu"].clone(),
        "norm_scale": weights["norm_scale"].clone(),
        "threshold": weights["threshold"].clone(),
    }
    if channel_scale:
        # Same encoder in raw units: divide normalised inputs by s, so multiply W_enc rows by s and keep b_dec / s.
        d = int(new["W_dec"].shape[1])
        scale = torch.linspace(0.5, 2.0, d)
        new["W_enc"] = (new["W_enc"] * scale[:, None]).contiguous()
        new["b_dec"] = (new["b_dec"] / scale).contiguous()
        new["W_dec"] = (new["W_dec"] / scale[None, :]).contiguous()
        new["norm_scale"] = scale
    sae = out / "sae"
    sae.mkdir(parents=True)
    save_file(new, str(sae / "weights.safetensors"))
    (sae / "config.json").write_text((capture / "sae" / "config.json").read_text())
    return sae


def test_a_reordered_dictionary_maps_one_to_one_and_passes_the_direction_check(capture: Path, tmp_path: Path):
    from_sae = _permuted(capture, tmp_path / "old", np.arange(D), channel_scale=False)
    perm = np.random.default_rng(3).permutation(D)
    target = int(np.flatnonzero(perm == PLANTED)[0])
    to_sae = _permuted(capture, tmp_path / "new", perm, channel_scale=True)
    out = tmp_path / "map"
    result = runner.invoke(app, ["map-feature", str(capture), "--from-sae", str(from_sae), "--to-sae", str(to_sae),
                                 "--feature", str(PLANTED), "--out", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads((out / "mapping.json").read_text())
    assert report["decision"]["outcome"] == "one" and report["decision"]["features"] == [target]
    assert report["top"][0]["id"] == target
    assert report["top"][0]["corr"] == pytest.approx(1.0, abs=1e-4)
    assert report["top"][0]["cos"] == pytest.approx(1.0, abs=1e-4)
    # The planted feature rises before every failure, through either dictionary.
    direction = report["direction"]
    assert direction["passed"] is True
    assert direction["to"]["features"][str(target)]["d"] > 0
    assert direction["from"]["features"][str(PLANTED)]["d"] > 0
    # A second run refuses to overwrite the report.
    again = runner.invoke(app, ["map-feature", str(capture), "--from-sae", str(from_sae), "--to-sae", str(to_sae),
                                "--feature", str(PLANTED), "--out", str(out)])
    assert again.exit_code == 1 and "exists" in again.output


def test_an_unrelated_dictionary_maps_to_nothing(capture: Path, tmp_path: Path):
    weights = load_file(str(capture / "sae" / "weights.safetensors"))
    rng = np.random.default_rng(11)
    d, m = weights["W_dec"].shape[1], 24
    w_dec = rng.standard_normal((m, d)).astype(np.float32)
    w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    sae = tmp_path / "unrelated" / "sae"
    sae.mkdir(parents=True)
    save_file({"W_enc": torch.tensor(w_dec.T.copy()), "W_dec": torch.tensor(w_dec), "b_enc": torch.zeros(m),
               "b_dec": torch.zeros(d), "norm_mu": torch.zeros(d), "norm_scale": torch.ones(1),
               "threshold": torch.tensor(2.5)}, str(sae / "weights.safetensors"))
    report = MF.map_feature(capture, capture / "sae", sae, PLANTED, tmp_path / "out", direction=False, log=lambda _m: None)
    assert report["decision"]["outcome"] == "none" and report["decision"]["features"] == []
    assert report["direction"] is None


def test_a_feature_split_in_two_is_found_as_a_group():
    rng = np.random.default_rng(0)
    n = 4000
    first = np.where(rng.uniform(size=n) < 0.2, rng.uniform(1, 2, n), 0.0)
    second = np.where((first == 0) & (rng.uniform(size=n) < 0.25), rng.uniform(1, 2, n), 0.0)
    noise = [np.where(rng.uniform(size=n) < 0.2, rng.uniform(1, 2, n), 0.0) for _ in range(4)]
    old = first + second
    columns = np.column_stack([noise[0], first, noise[1], second, noise[2], noise[3]])
    ids = [40, 7, 41, 9, 42, 43]
    corr = np.array([np.corrcoef(columns[:, j], old)[0, 1] for j in range(len(ids))])
    full_corr = np.zeros(50)
    full_corr[ids] = corr
    cos = np.full(50, 0.6)
    assert full_corr.max() < MF.MATCH_CORR  # no single feature carries it
    path = MF.greedy_group(old, columns, ids)
    decision = MF.decide(full_corr, cos, path)
    assert decision["outcome"] == "split" and decision["features"] == [7, 9]
    assert all(w > 0 for w in decision["weights"])


def test_one_needs_a_clear_runner_up_and_an_aligned_decoder():
    corr = np.array([0.9, 0.6, 0.1])
    cos = np.array([0.95, 0.2, 0.0])
    path = [{"features": [0], "r": 0.9, "weights": [1.0]}]
    assert MF.decide(corr, cos, path)["outcome"] == "none"  # runner-up 0.6 >= 0.5 and no group passes
    assert MF.decide(np.array([0.9, 0.3, 0.1]), cos, path)["outcome"] == "one"
    assert MF.decide(np.array([0.9, 0.3, 0.1]), np.array([0.5, 0.2, 0.0]), path)["outcome"] == "none"  # cos < 0.7
