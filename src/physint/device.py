"""Pick a device that exists on this machine. CPU and MPS are both first-class."""

from __future__ import annotations

import os

import torch


def pick_device() -> torch.device:
    forced = os.environ.get("PHYSINT_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
