"""Pick a device that exists on this machine. CPU and MPS are both first-class."""

from __future__ import annotations

import os

import torch


def ensure_mps_fallback(device: torch.device) -> None:
    """Allow unimplemented MPS ops to run on CPU. Set before the first MPS kernel."""
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device() -> torch.device:
    forced = os.environ.get("II_DEVICE")
    if forced:
        device = torch.device(forced)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ensure_mps_fallback(device)
    return device
