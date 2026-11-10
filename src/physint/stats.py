"""Pure scoring helpers. Null never counts as a pass."""

from __future__ import annotations

import math

import numpy as np

from physint.thresholds import (
    DISTINCT_CLIPS_TOP20,
    RANDOM_PERCENTILE_MIN,
    USABLE_LOSS_RECOVERED,
)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation. Returns 0 when either side has no variance."""
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    if a64.shape != b64.shape:
        raise ValueError(f"pearson shape mismatch {a64.shape} vs {b64.shape}")
    a64 = a64 - a64.mean()
    b64 = b64 - b64.mean()
    da = float(np.linalg.norm(a64))
    db = float(np.linalg.norm(b64))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(np.dot(a64, b64) / (da * db))


def loss_recovered(loss_zero: float, loss_sae: float, loss_clean: float) -> float:
    """Scale predictor loss from 0 (layer zeroed) to 1 (untouched model)."""
    denom = loss_zero - loss_clean
    if not math.isfinite(denom) or denom <= 1e-12:
        raise RuntimeError(
            "cannot scale loss_recovered: zeroed loss is not above the clean loss "
            f"(zero={loss_zero}, clean={loss_clean}, sae={loss_sae})"
        )
    value = (loss_zero - loss_sae) / denom
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite loss_recovered from {loss_zero}, {loss_sae}, {loss_clean}")
    return float(value)


def is_usable(loss_recovered_value: float) -> bool:
    return bool(loss_recovered_value >= USABLE_LOSS_RECOVERED)


def percentile_le(value: float, samples: np.ndarray) -> float:
    """Percent of samples <= value, in [0, 100]. Ties count as not exceeding the feature."""
    arr = np.asarray(samples, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("percentile requires at least one random feature")
    if not math.isfinite(value):
        return 0.0
    return float(100.0 * np.mean(arr <= value))


def monte_carlo_pvalue(effect: float, controls: np.ndarray) -> float:
    """One-sided Monte Carlo p-value of ``effect`` against matched controls.

    ``(1 + #{controls >= effect}) / (n + 1)``. Ties count against the candidate,
    so an all-zero comparison is not significant. Under exchangeability and no
    ties this is at most ``alpha`` when the threshold is ``alpha``.
    """
    arr = np.asarray(controls, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("pvalue requires at least one control")
    if not math.isfinite(float(effect)):
        return 1.0
    return float((1 + np.count_nonzero(arr >= effect)) / (arr.size + 1))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(low, high)`` clipped to ``[0, 1]``. ``confidence`` is two-sided.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must lie in [0, trials], got {successes} of {trials}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    n = float(trials)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def feature_is_real(
    *,
    usable: bool,
    distinct_clips_top20: int,
    label_holds_heldout: bool | None,
    random_percentile: float,
    absent_in_random_init: bool | None,
) -> bool:
    """A feature is real only if the SAE is usable and all four gates pass.

    ``None`` means the check did not run and never counts as a pass.
    """
    return bool(
        usable
        and distinct_clips_top20 >= DISTINCT_CLIPS_TOP20
        and label_holds_heldout is True
        and random_percentile >= RANDOM_PERCENTILE_MIN
        and absent_in_random_init is True
    )
