"""Reuse an SAE trained somewhere else (``ii run --sae-from``).

A run normally trains its own SAE on its train split. ``--sae-from`` instead
copies a frozen dictionary into the run, so that feature ids found on another
dataset keep their meaning on new clips. The GR00T case: the task-1028 capture
SAE (``groot-1028-v9-n100/sae``) picked features 6262, 7353 and 7775 on the v9
capture; the steering scorecard has to measure those same latents on freshly
recorded LIBERO-plus episodes.

Two layouts are read:

* an ``ii`` SAE (``sae/config.json`` with ``activation``), from a run folder or
  its ``sae/`` directory;
* a GR00T v9 BatchTopK SAE (``sae/config.json`` with ``arch: batchtopk``), from a
  capture folder, a ``ii retrain-sae`` output, or its ``sae/`` directory. Its
  input normalisation is folded in exactly as ``ii import --format groot-v9``
  does, so the active set equals the source's and feature ids are unchanged.

The run records where the SAE came from in ``manifest.sae_source``. Such a run
is never scored: the scoreboard gates assume an SAE trained on the run's own
train split, which this one was not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from interpretableintelligence.sae import SAE, SAEConfig, load_sae

FORMATS = ("ii", "groot-v9")
SOURCE_KEYS = {"path", "format", "config_sha256", "weights_sha256", "n_features", "notes"}


@dataclass
class ExternalSAE:
    model: SAE
    config: SAEConfig
    record: dict


def resolve_sae_dir(path: Path) -> Path:
    """``path`` itself when it holds ``config.json`` and ``weights.safetensors``, else ``path/sae``."""
    path = Path(path)
    for candidate in (path, path / "sae"):
        if (candidate / "config.json").is_file() and (candidate / "weights.safetensors").is_file():
            return candidate
    raise FileNotFoundError(f"no SAE at {path}: expected config.json and weights.safetensors there or in {path / 'sae'}")


def load_external_sae(path: Path, d_model: int) -> ExternalSAE:
    """Load the SAE at ``path`` (see the module docstring) for a layer of width ``d_model``."""
    directory = resolve_sae_dir(path)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{directory / 'config.json'} must hold a JSON object")
    if "arch" in raw:
        from interpretableintelligence.importers.groot_v9 import _fold_sae  # noqa: PLC0415 (pulls safetensors.numpy)

        model, config, notes = _fold_sae(directory, int(d_model))
        fmt = "groot-v9"
    elif "activation" in raw:
        model, config = load_sae(directory, torch.device("cpu"))
        notes = {}
        fmt = "ii"
    else:
        raise ValueError(f"{directory / 'config.json'}: neither an ii SAE (activation) nor a GR00T v9 SAE (arch)")
    if int(config.d_model) != int(d_model):
        raise ValueError(f"the SAE at {directory} reads d_model {config.d_model}, but this layer has {d_model}")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    record = {
        "path": str(directory.resolve()),
        "format": fmt,
        "config_sha256": _sha256(directory / "config.json"),
        "weights_sha256": _sha256(directory / "weights.safetensors"),
        "n_features": int(config.n_features),
        "notes": {str(key): value for key, value in notes.items()},
    }
    return ExternalSAE(model=model, config=config, record=record)


def check_source(value: object, errors: list[str]) -> None:
    """Contract errors for ``manifest.sae_source``."""
    where = "manifest.sae_source"
    if not isinstance(value, dict):
        errors.append(f"{where} must be an object")
        return
    if set(value) != SOURCE_KEYS:
        errors.append(f"{where} keys mismatch missing={sorted(SOURCE_KEYS - set(value))} extra={sorted(set(value) - SOURCE_KEYS)}")
        return
    if not isinstance(value["path"], str) or not value["path"]:
        errors.append(f"{where}.path must be a non-empty string")
    if value["format"] not in FORMATS:
        errors.append(f"{where}.format must be one of {list(FORMATS)}")
    for key in ("config_sha256", "weights_sha256"):
        text = value[key]
        if not isinstance(text, str) or len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
            errors.append(f"{where}.{key} must be 64 lowercase hex characters")
    if not isinstance(value["n_features"], int) or isinstance(value["n_features"], bool) or value["n_features"] < 1:
        errors.append(f"{where}.n_features must be a positive integer")
    if not isinstance(value["notes"], dict):
        errors.append(f"{where}.notes must be an object")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
