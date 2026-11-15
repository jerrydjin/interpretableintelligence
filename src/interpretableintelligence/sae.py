"""ReLU sparse autoencoder. Decoder columns stay unit-norm."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class SAEConfig:
    d_model: int
    n_features: int = 12
    l1_coeff: float = 1e-3
    # Penalize aligned decoder columns so one planted direction does not split
    # into two features that both look causal.
    ortho_coeff: float = 1e-3
    lr: float = 3e-3
    steps: int = 2500
    batch_size: int = 256
    seed: int = 0
    normalize_decoder: bool = True
    activation: str = "relu"
    # Active features per token for ``activation="topk"`` (Gao et al. 2024).
    k: int | None = None
    # Inference cutoff for ``activation="jumprelu"``: a feature is kept only
    # where its ReLU pre-activation is strictly above this value. This is how a
    # BatchTopK SAE is evaluated, so imported BatchTopK dictionaries use it.
    threshold: float | None = None

    def as_dict(self) -> dict:
        out = {
            "d_model": self.d_model,
            "n_features": self.n_features,
            "l1_coeff": self.l1_coeff,
            "ortho_coeff": self.ortho_coeff,
            "lr": self.lr,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "normalize_decoder": self.normalize_decoder,
            "activation": self.activation,
        }
        if self.activation == "topk":
            out["k"] = self.k
        if self.activation == "jumprelu":
            out["threshold"] = self.threshold
        return out


ACTIVATIONS = ("relu", "topk", "jumprelu")


class SAE(torch.nn.Module):
    """``relu``: L1-sparse ReLU SAE. ``topk``: keep the ``k`` largest ReLU pre-activations per token.
    ``jumprelu``: keep the ReLU pre-activations strictly above ``threshold``."""

    def __init__(
        self,
        d_model: int,
        n_features: int,
        activation: str = "relu",
        k: int | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"SAE activation must be one of {ACTIVATIONS}, got {activation!r}")
        if activation == "topk" and (k is None or not 1 <= int(k) <= n_features):
            raise ValueError(f"topk SAE needs 1 <= k <= n_features, got k={k}")
        if activation == "jumprelu" and (threshold is None or not math.isfinite(float(threshold)) or float(threshold) < 0):
            raise ValueError(f"jumprelu SAE needs a finite threshold >= 0, got {threshold}")
        self.d_model = d_model
        self.n_features = n_features
        self.activation = activation
        self.k = None if k is None else int(k)
        self.threshold = None if threshold is None else float(threshold)
        self.W_enc = torch.nn.Parameter(torch.empty(n_features, d_model))
        self.b_enc = torch.nn.Parameter(torch.empty(n_features))
        self.W_dec = torch.nn.Parameter(torch.empty(d_model, n_features))
        self.b_dec = torch.nn.Parameter(torch.empty(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        if self.activation == "jumprelu":
            return pre * (pre > self.threshold)
        if self.activation != "topk":
            return pre
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, top.indices, top.values)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec.T + self.b_dec

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return features, self.decode(features)

    @torch.no_grad()
    def unit_norm_decoder(self) -> None:
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)


def _decoder_ortho(decoder: torch.Tensor) -> torch.Tensor:
    """Squared off-diagonal cosine similarity of decoder columns."""
    norms = decoder.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cosine = (decoder / norms).T @ (decoder / norms)
    identity = torch.eye(cosine.shape[0], device=cosine.device, dtype=cosine.dtype)
    return ((cosine - identity) ** 2).sum()


def _loss(mse: torch.Tensor, features: torch.Tensor, model: SAE, config: SAEConfig) -> torch.Tensor:
    loss = mse
    if config.l1_coeff:
        loss = loss + config.l1_coeff * features.abs().sum(dim=-1).mean()
    if config.ortho_coeff:
        loss = loss + config.ortho_coeff * _decoder_ortho(model.W_dec)
    return loss


def untrained_sae(config: SAEConfig, mean: torch.Tensor, device: torch.device) -> SAE:
    """The SAE that training starts from, seeded by ``config.seed``, with no optimiser step.

    A unit-norm Gaussian decoder, the encoder tied to its transpose, a zero encoder bias, and
    ``b_dec`` at ``mean`` (the train-row mean). :func:`train_sae` and
    :func:`train_sae_fetched` start here, so an untrained SAE is the trained one's own
    initialisation, not a lookalike. The rollout lane's random-weights SAE control
    (``ii rollouts --random-sae``) edits with these weights. It leaves torch's global RNG
    at the same point the training loop then continues from.
    """
    if int(mean.shape[0]) != config.d_model:
        raise ValueError(f"d_model {config.d_model} != mean {tuple(mean.shape)}")
    torch.manual_seed(config.seed)
    model = SAE(config.d_model, config.n_features, config.activation, config.k).to(device)
    with torch.no_grad():
        decoder = torch.randn(config.d_model, config.n_features, device=device)
        decoder = decoder / decoder.norm(dim=0, keepdim=True).clamp(min=1e-8)
        model.W_dec.copy_(decoder)
        model.W_enc.copy_(decoder.T)
        model.b_enc.zero_()
        model.b_dec.copy_(mean.detach().to(device=device, dtype=torch.float32))
    return model


def random_weights_sae(config: SAEConfig, mean: torch.Tensor, seed: int, device: torch.device) -> SAE:
    """Untrained control SAE: the trained SAE's architecture, width and ``k``, zero optimiser steps.

    Deterministic in ``seed``: the same ``(config, mean, seed)`` gives bit-identical
    weights. Weights come from :func:`untrained_sae` with ``seed`` in place of
    ``config.seed``, so ``seed == config.seed`` is the trained SAE before its first step.
    ``mean`` is the float32 mean of the train rows (``b_dec``'s initial value), the only
    thing the control takes from the data. Nothing here is optimised, and the caller's
    torch RNG is left untouched.
    """
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        model = untrained_sae(replace(config, seed=int(seed)), mean, device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def train_sae(
    activations: torch.Tensor,
    config: SAEConfig,
    device: torch.device,
    log: Callable[[str], None] | None = None,
) -> SAE:
    """Train on rows of ``activations`` (the train split only)."""
    if activations.ndim != 2:
        raise ValueError(f"expected [N, d], got {tuple(activations.shape)}")
    if activations.shape[1] != config.d_model:
        raise ValueError(f"d_model {config.d_model} != {activations.shape[1]}")
    if activations.shape[0] < 2:
        raise ValueError("need at least 2 activation rows to train an SAE")

    data = activations.detach().to(device=device, dtype=torch.float32)
    model = untrained_sae(config, data.mean(dim=0), device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    n = data.shape[0]
    batch = min(config.batch_size, n)
    model.train()
    last = ""
    for step in range(config.steps):
        idx = torch.randint(0, n, (batch,), device=device)
        batch_x = data[idx]
        features, recon = model(batch_x)
        mse = ((recon - batch_x) ** 2).mean()
        loss = _loss(mse, features, model, config)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if config.normalize_decoder:
            model.unit_norm_decoder()
        if log is not None and (step + 1 == config.steps or step % 500 == 0):
            l0 = float((features.detach() > 0).float().sum(dim=-1).mean())
            last = f"step {step + 1}/{config.steps} mse={float(mse.detach()):.6f} l0={l0:.2f}"
            log(last)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model._train_log = last  # type: ignore[attr-defined]
    return model


def train_sae_fetched(
    n_rows: int,
    mean: torch.Tensor,
    fetch,
    config: SAEConfig,
    device: torch.device,
    log: Callable[[str], None] | None = None,
) -> SAE:
    """Train like :func:`train_sae`, pulling one batch at a time.

    ``fetch(local_indices)`` returns float32 ``[batch, d_model]`` for train-row
    positions in ``0 .. n_rows``. The full activation matrix is never a tensor.
    ``mean`` is the float32 mean of those train rows (computed by the caller).
    The seed sequence matches :func:`train_sae`: decoder init, then one
    ``randint`` per step.
    """
    if n_rows < 2:
        raise ValueError("need at least 2 activation rows to train an SAE")
    if int(mean.shape[0]) != config.d_model:
        raise ValueError(f"d_model {config.d_model} != mean {tuple(mean.shape)}")

    model = untrained_sae(config, mean, device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    batch = min(config.batch_size, n_rows)
    model.train()
    last = ""
    for step in range(config.steps):
        idx = torch.randint(0, n_rows, (batch,), device=device)
        fetched = np.ascontiguousarray(fetch(idx.detach().cpu().numpy()), dtype=np.float32)
        if fetched.shape != (batch, config.d_model):
            raise RuntimeError(f"fetch returned {fetched.shape}, expected {(batch, config.d_model)}")
        batch_x = torch.from_numpy(fetched).to(device=device, dtype=torch.float32)
        features, recon = model(batch_x)
        mse = ((recon - batch_x) ** 2).mean()
        loss = _loss(mse, features, model, config)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if config.normalize_decoder:
            model.unit_norm_decoder()
        if log is not None and (step + 1 == config.steps or step % 500 == 0):
            l0 = float((features.detach() > 0).float().sum(dim=-1).mean())
            last = f"step {step + 1}/{config.steps} mse={float(mse.detach()):.6f} l0={l0:.2f}"
            log(last)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model._train_log = last  # type: ignore[attr-defined]
    return model


def save_sae(model: SAE, config: SAEConfig, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(tensors, str(directory / "weights.safetensors"))
    (directory / "config.json").write_text(json.dumps(config.as_dict(), indent=2) + "\n", encoding="utf-8")


def load_sae(directory: Path, device: torch.device) -> tuple[SAE, SAEConfig]:
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config = SAEConfig(
        d_model=int(raw["d_model"]),
        n_features=int(raw["n_features"]),
        l1_coeff=float(raw["l1_coeff"]),
        ortho_coeff=float(raw["ortho_coeff"]),
        lr=float(raw["lr"]),
        steps=int(raw["steps"]),
        batch_size=int(raw["batch_size"]),
        seed=int(raw["seed"]),
        normalize_decoder=bool(raw["normalize_decoder"]),
        activation=str(raw["activation"]),
        k=None if raw.get("k") is None else int(raw["k"]),
        threshold=None if raw.get("threshold") is None else float(raw["threshold"]),
    )
    model = SAE(config.d_model, config.n_features, config.activation, config.k, config.threshold)
    state = load_file(str(directory / "weights.safetensors"), device="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config
