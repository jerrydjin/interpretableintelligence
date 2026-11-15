"""What a remote run costs, worked out before it starts and shown while it runs.

Modal bills each container per second for its GPU, its CPU cores and its memory. The
container behind ``ii run --remote`` reserves a fixed ``CPU_CORES`` and ``MEMORY_GIB``, so its
price per hour is known up front: the GPU's list price plus that reservation. The table is
Modal's published list price (USD, 2026). Plans, credits and region multipliers can change
what you pay; ``--gpu-cost-per-hour`` replaces the whole rate when you know your own, and the
Modal dashboard is the bill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# GPU list price, USD per second.
GPU_USD_PER_S: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
}
GPU_ALIASES = {"A10G": "A10", "A100": "A100-40GB"}
CPU_CORE_USD_PER_S = 0.0000131
MEMORY_GIB_USD_PER_S = 0.00000222

# What the remote container reserves. Billing is the larger of reservation and use.
CPU_CORES = 4.0
MEMORY_GIB = 16

DEFAULT_GPU = "L4"
DEFAULT_MAX_COST = 2.00
# Modal's longest function timeout.
MAX_TIMEOUT_S = 24 * 3600
# Below this a run cannot get past model loading, so a cap this small is a mistake.
MIN_TIMEOUT_S = 60

PRICE_SOURCE = "Modal list price (2026)"


@dataclass(frozen=True)
class Rate:
    gpu: str
    usd_per_hour: float
    gpu_usd_per_hour: float | None
    source: str

    def cost(self, seconds: float) -> float:
        return max(0.0, float(seconds)) * self.usd_per_hour / 3600.0

    def describe(self) -> str:
        if self.gpu_usd_per_hour is None:
            return f"{self.gpu} at ${self.usd_per_hour:.2f}/h ({self.source})"
        rest = self.usd_per_hour - self.gpu_usd_per_hour
        return (
            f"{self.gpu} at ${self.usd_per_hour:.2f}/h: GPU ${self.gpu_usd_per_hour:.2f}/h + "
            f"{CPU_CORES:g} CPU cores and {MEMORY_GIB} GiB ${rest:.2f}/h ({self.source})"
        )


def normalize_gpu(gpu: str) -> str:
    name = str(gpu).strip().upper()
    name = GPU_ALIASES.get(name, name)
    if name not in GPU_USD_PER_S:
        known = ", ".join(GPU_USD_PER_S)
        raise ValueError(f"unknown --gpu {gpu!r}; choose one of {known}")
    return name


def rate_for(gpu: str, override_per_hour: float | None = None) -> Rate:
    """Price per hour of the remote container. ``override_per_hour`` replaces the list price."""
    name = normalize_gpu(gpu)
    if override_per_hour is not None:
        if not math.isfinite(override_per_hour) or override_per_hour < 0:
            raise ValueError("gpu_cost_per_hour must be a non-negative finite number")
        return Rate(name, float(override_per_hour), None, "--gpu-cost-per-hour")
    gpu_hour = GPU_USD_PER_S[name] * 3600.0
    extra = (CPU_CORES * CPU_CORE_USD_PER_S + MEMORY_GIB * MEMORY_GIB_USD_PER_S) * 3600.0
    return Rate(name, gpu_hour + extra, gpu_hour, PRICE_SOURCE)


def budget_timeout(max_cost: float, rate: Rate) -> int:
    """Container timeout in seconds that keeps the run under ``max_cost`` at ``rate``.

    Modal stops the function at its timeout, so the cap holds even if this machine goes away.
    """
    if not math.isfinite(max_cost) or max_cost <= 0:
        raise ValueError("--max-cost must be a positive number of dollars")
    if rate.usd_per_hour <= 0:
        return MAX_TIMEOUT_S
    seconds = int(math.floor(float(max_cost) / rate.usd_per_hour * 3600.0))
    if seconds < MIN_TIMEOUT_S:
        need = rate.cost(MIN_TIMEOUT_S)
        raise ValueError(
            f"--max-cost ${max_cost:.2f} buys {seconds}s on {rate.gpu}; "
            f"allow at least ${need:.2f} (one minute) or pick a cheaper --gpu"
        )
    return min(seconds, MAX_TIMEOUT_S)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_usd(value: float) -> str:
    return f"${value:.3f}" if value < 1 else f"${value:.2f}"
