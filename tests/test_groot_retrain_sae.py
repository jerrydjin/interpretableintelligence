"""``ii retrain-sae``: a replacement GR00T capture SAE that the rest of the GR00T lane reads unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from test_import import HELDOUT, N_EPISODES, make_v9_capture
from typer.testing import CliRunner

from interpretableintelligence.adapters.groot import random_sae
from interpretableintelligence.adapters.groot import retrain_sae as R
from interpretableintelligence.adapters.groot.v9 import ablate_v9
from interpretableintelligence.cli import app
from interpretableintelligence.importers import import_run
from interpretableintelligence.validate import validate_run

runner = CliRunner()
FAST = ["--n-latents", "32", "--k", "2", "--steps", "60", "--batch-tokens", "64", "--lr", "3e-3", "--aux-k", "8", "--device", "cpu"]


@pytest.fixture(scope="module")
def retrained(tmp_path_factory) -> tuple[Path, Path, dict]:
    root = tmp_path_factory.mktemp("retrain")
    capture = make_v9_capture(root)
    # An SAE-derived output of the old dictionary, which the new folder must not carry over.
    (capture / "candidates_final.json").write_text("{}", encoding="utf-8")
    out = root / "retrained"
    result = runner.invoke(app, ["retrain-sae", str(capture), "--out", str(out), *FAST])
    assert result.exit_code == 0, result.output
    assert "alive" in result.output and "heldout" in result.output
    return capture, out, json.loads((out / R.REPORT).read_text())


def test_writes_the_capture_sae_layout(retrained):
    capture, out, report = retrained
    weights = random_sae.load_tensors(out / "sae" / "weights.safetensors")
    config = json.loads((out / "sae" / "config.json").read_text())
    d = int(load_file(str(capture / "sae" / "weights.safetensors"))["W_enc"].shape[0])
    assert weights["W_enc"].shape == (d, 32) and weights["W_dec"].shape == (32, d)
    assert torch.allclose(weights["W_dec"].norm(dim=1), torch.ones(32), atol=1e-5)
    assert config["arch"] == "batchtopk" and config["k"] == 2 and config["d_sae"] == 32
    assert config["inference_threshold"] == pytest.approx(float(weights["threshold"]))
    # norm="keep": the old SAE's input statistics, unchanged.
    old = random_sae.load_tensors(capture / "sae" / "weights.safetensors")
    assert torch.equal(weights["norm_mu"], old["norm_mu"]) and torch.equal(weights["norm_scale"], old["norm_scale"].reshape(1))
    # The v9 hook's loader and the picker's reader take it as is.
    subset = ablate_v9.load_sae(out / "sae" / "weights.safetensors", [0, 5])
    assert subset.latent_ids == [0, 5]
    assert report["old"] is not None and set(report["new"]) == {"train", "heldout"}


def test_threshold_gives_k_per_token_on_train(retrained):
    _capture, _out, report = retrained
    assert report["new"]["train"]["l0_mean"] == pytest.approx(2.0, abs=0.05)  # strict > at the (k * n)-th value drops that one activation
    assert report["new"]["train"]["n_episodes"] == N_EPISODES - len(HELDOUT)
    assert report["new"]["heldout"]["n_episodes"] == len(HELDOUT)


def test_links_only_sae_independent_files(retrained):
    capture, out, report = retrained
    assert (out / "split.json").read_bytes() == (capture / "split.json").read_bytes()
    for name in ("manifest.json", "episodes.jsonl", "acts"):
        assert (out / name).is_symlink()
    assert not (out / "candidates_final.json").exists()
    assert "candidates_final.json" in report["capture_links"]["not_linked"]
    assert (out / "sae").is_dir() and not (out / "sae").is_symlink()


def test_random_weights_control_starts_where_training_started(retrained):
    """random:<seed> for the training seed is the retrained SAE's own init (research rule 4)."""
    capture, out, _report = retrained
    train_split, _held = R.load_capture(capture, torch.device("cpu"), log=lambda _m: None)
    weights = random_sae.load_tensors(out / "sae" / "weights.safetensors")
    control, _record = random_sae.build_untrained_sae(out / "sae" / "weights.safetensors", 0, train_split.chunks, k=2)
    template = {"W_enc": torch.empty(weights["W_enc"].shape), "norm_mu": weights["norm_mu"], "norm_scale": torch.ones(1)}
    init = random_sae.untrained_weights(template, 0)
    assert torch.equal(control["W_enc"], init["W_enc"]) and torch.equal(control["W_dec"], init["W_dec"])
    assert not torch.allclose(control["W_dec"], weights["W_dec"])


def test_ii_import_reads_the_retrained_folder(retrained, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _capture, out, _report = retrained
    run = tmp_path / "run"
    import_run("groot-v9", out, run, log=lambda _m: None, model_sha="ab" * 32)
    assert validate_run(run) == []


def test_refuses_to_overwrite(retrained):
    capture, out, _report = retrained
    result = runner.invoke(app, ["retrain-sae", str(capture), "--out", str(out), *FAST])
    assert result.exit_code == 1 and "already exists" in result.output


def _anisotropic(n: int = 4000, d: int = 16, true: int = 64, k: int = 2, seed: int = 0) -> R.Split:
    """Sparse codes over ``true`` directions plus two outlier channels with 20x the spread."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((true, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    codes = np.zeros((n, true))
    for row in range(n):
        codes[row, rng.choice(true, k, replace=False)] = rng.uniform(1.0, 3.0, k)
    x = codes @ dirs
    x[:, :2] += 20.0 * rng.standard_normal((n, 2))
    return R.Split(torch.tensor(x, dtype=torch.float16), torch.zeros(n, dtype=torch.long), torch.arange(n) // 100)


def test_evaluation_reports_variance_concentration():
    split = _anisotropic()
    mu, scale = R.norm_stats(split, None, "fit")
    weights, _curve = R.train(split, mu, scale, R.RetrainConfig(n_latents=32, k=2, steps=20, batch_tokens=128, norm="fit"), log=lambda _m: None)
    result = R.evaluate(weights, split)
    assert result["variance_share_top_dims"] > 0.9
    assert set(result["top_dims"][:2]) == {0, 1}
    assert result["ev_per_dim_mean"] < result["ev_normalised"]


@pytest.mark.parametrize("revive", ["auxk", "resample"])
def test_revival_gives_never_fired_latents_a_gradient(revive):
    """Without revival a latent that never fires never moves; AuxK and resampling both move it."""
    split = _anisotropic()
    mu, scale = R.norm_stats(split, None, "fit")
    config = dict(n_latents=128, k=2, steps=120, batch_tokens=256, lr=3e-3, aux_k=32, dead_tokens=512, norm="fit")
    plain, _ = R.train(split, mu, scale, R.RetrainConfig(revive="none", **config), log=lambda _m: None)
    template = {"W_enc": torch.empty(16, 128), "norm_mu": mu, "norm_scale": torch.ones(1)}
    init = random_sae.untrained_weights(template, 0)
    untouched = (plain["W_enc"] == init["W_enc"]).all(dim=0)
    assert untouched.any(), "the fixture should leave some latents that never fire without revival"
    revived, curve = R.train(split, mu, scale, R.RetrainConfig(revive=revive, **config), log=lambda _m: None)
    moved = ~(revived["W_enc"] == init["W_enc"]).all(dim=0)
    assert moved[untouched].float().mean() > 0.5
    if revive == "resample":
        assert curve[-1]["resampled"] > 0


def test_colab_notebook_builds_and_its_cells_parse(tmp_path):
    from interpretableintelligence.adapters.groot.colab import build_retrain_notebook as nb

    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb.build("some-branch")))
    notebook = json.loads(path.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code
    for cell in code:
        # IPython shell lines (!...) are not Python; everything else must compile.
        source = "".join(line for line in cell["source"] if not line.lstrip().startswith("!"))
        compile(source, "<cell>", "exec")
    assert all(revive in R.REVIVE for _name, _n, revive in nb.SWEEP)
    assert "--branch $BRANCH" in "".join(code[0]["source"]) and "some-branch" in "".join(code[0]["source"])
