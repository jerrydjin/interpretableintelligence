"""Untrained GR00T SAEs for the random-weights SAE control (AGENTS.md research rule 4).

The GR00T capture SAE (``groot-1028-v9-n100/sae``) is a BatchTopK SAE, ``k`` = 32, run at inference as a JumpReLU
at a learned threshold on ``x_n = (x - norm_mu) / norm_scale`` (``ablate_v9.SAESubset.encode``). Its trainer's init,
recorded in ``sae/SAE_SPEC_for_random_weights_control.json``, is a ``randn(D, M)`` matrix with unit-norm columns used
as the encoder and, transposed, as the decoder, with zero biases.

An untrained control keeps everything that is not learned and redraws the rest:

- ``norm_mu`` and ``norm_scale`` are copied: they are data statistics fixed before training, not weights.
- ``W_enc`` / ``W_dec`` come from that init with ``torch.Generator().manual_seed(seed)``; ``b_enc`` and ``b_dec`` are 0.
- The threshold is the one learned parameter BatchTopK inference needs that the init does not give. An untrained SAE
  run at the trained threshold fires about 1,900 latents per token instead of 32, so it would not be the same
  architecture at the same sparsity. The threshold is set so the untrained SAE keeps exactly ``k * n`` activations
  over the ``n`` calibration tokens (the trained SAE's train-split tokens), the BatchTopK rule over one batch of all
  of them. Its mean L0 there is ``k``.

The result has the trained SAE's keys and shapes, so ``ablate_v9.load_sae``, ``ablate_v9.write_subset`` and the
picker's ``rollout_io.PolicySAE`` read it unchanged. Nothing is optimised.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

SAE_KEYS = ("W_enc", "W_dec", "b_enc", "b_dec", "norm_mu", "norm_scale", "threshold")
INIT_RULE = (
    "W = randn(D, M, generator=manual_seed(seed)); columns unit-normalised; W_enc = W, W_dec = W.T; b_enc = b_dec = 0; "
    "norm_mu, norm_scale copied from the trained SAE; threshold = the (k * n)-th largest pre-activation over the n "
    "calibration tokens (BatchTopK over one batch), so mean L0 there is k"
)


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    missing = [key for key in SAE_KEYS if key not in tensors]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    return tensors


def untrained_weights(trained: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """The trainer's init drawn with ``seed``, with the trained SAE's normalisation. ``threshold`` is not set yet."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    d_in, n_latents = (int(item) for item in trained["W_enc"].shape)
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(d_in, n_latents, generator=generator, dtype=torch.float32)
    matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return {
        "W_enc": matrix.contiguous(),
        "W_dec": matrix.T.contiguous(),
        "b_enc": torch.zeros(n_latents, dtype=torch.float32),
        "b_dec": torch.zeros(d_in, dtype=torch.float32),
        "norm_mu": trained["norm_mu"].float().clone(),
        "norm_scale": trained["norm_scale"].float().reshape(-1).clone(),
    }


def pre_activations(weights: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """ReLU pre-activations ``relu(((x - norm_mu) / norm_scale - b_dec) @ W_enc + b_enc)`` of raw rows ``x`` [n, D]."""
    scale = float(weights["norm_scale"].reshape(-1)[0])
    xn = (x.float() - weights["norm_mu"]) / scale
    return torch.relu((xn - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"])


def batchtopk_threshold(weights: dict[str, torch.Tensor], chunks, k: int) -> tuple[float, int]:
    """The (k * n)-th largest pre-activation over the n rows ``chunks()`` yields, and n.

    ``chunks`` is called twice (count, then select), so it must return a fresh iterable each time. Exact: the global
    top (k * n) values are among each chunk's own top (k * n), so a buffer of that size suffices.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rows = sum(int(x.shape[0]) for x in chunks())
    if rows == 0:
        raise ValueError("no calibration rows")
    target = int(k) * rows
    best = torch.empty(0, dtype=torch.float32)
    seen = 0
    for x in chunks():
        if x.numel() == 0:
            continue
        seen += int(x.shape[0])
        merged = torch.cat([best, pre_activations(weights, x).reshape(-1)])
        best = torch.topk(merged, min(target, merged.numel()), sorted=False).values
    if seen != rows or best.numel() < target:
        raise RuntimeError(f"calibration rows changed between passes ({rows} then {seen})")
    return float(best.min()), rows


def build_untrained_sae(trained_path: Path, seed: int, chunks, *, k: int) -> tuple[dict[str, torch.Tensor], dict]:
    """Untrained weights for ``seed`` with a BatchTopK threshold over ``chunks()`` (a callable returning a fresh iterable
    of raw rows); and a provenance record."""
    trained = load_tensors(Path(trained_path))
    weights = untrained_weights(trained, int(seed))
    threshold, rows = batchtopk_threshold(weights, chunks, int(k))
    weights["threshold"] = torch.tensor(threshold, dtype=torch.float32)
    record = {
        "dictionary": f"random:{int(seed)}",
        "seed": int(seed),
        "k": int(k),
        "threshold": threshold,
        "calibration_rows": rows,
        "init": INIT_RULE,
        "trained_threshold": float(trained["threshold"].float().reshape(-1)[0]),
    }
    return weights, record


def save_sae_dir(weights: dict[str, torch.Tensor], directory: Path, config: dict) -> Path:
    """Write ``weights.safetensors`` and ``config.json`` in the capture's ``sae/`` layout. Returns the weights path."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "weights.safetensors"
    save_file({key: weights[key].contiguous() for key in SAE_KEYS}, str(path))
    (directory / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    return path
