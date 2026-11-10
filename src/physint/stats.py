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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """0-based average ranks. Ties share the mean of the occupied ranks."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. Ties use average ranks.

    Returns 0 when either ranking has no variance, including a constant input.
    Needs at least two pairs.
    """
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError(f"spearman shape mismatch {left.shape} vs {right.shape}")
    if left.shape[0] < 2:
        raise ValueError("spearman needs at least two pairs")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("spearman inputs must be finite")
    return pearson(_average_ranks(left), _average_ranks(right))


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


def is_usable(loss_recovered_value: float, loss_sae: float, loss_visible_mean: float) -> bool:
    """Usable when the splice recovers enough loss and beats the visible-mean floor.

    ``USABLE_LOSS_RECOVERED`` is unchanged. ``loss_sae < loss_visible_mean`` is an
    additional requirement: a patched predictor that does not beat the mean of
    the visible patches is not usable, regardless of ``loss_recovered``.
    """
    return bool(
        loss_recovered_value >= USABLE_LOSS_RECOVERED and loss_sae < loss_visible_mean
    )


def percentile_le(value: float, samples: np.ndarray) -> float:
    """Percent of samples <= value, in [0, 100]. Ties count as not exceeding the feature."""
    arr = np.asarray(samples, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("percentile requires at least one random feature")
    if not math.isfinite(value):
        return 0.0
    return float(100.0 * np.mean(arr <= value))


def feature_is_real(
    *,
    usable: bool,
    distinct_clips_top20: int,
    label_holds_heldout: bool | None,
    random_percentile: float | None,
    absent_in_random_init: bool | None,
) -> bool:
    """A feature is real only if the SAE is usable and all four gates pass.

    ``None`` means the check did not run and never counts as a pass.
    """
    percentile_ok = (
        isinstance(random_percentile, (int, float))
        and not isinstance(random_percentile, bool)
        and random_percentile >= RANDOM_PERCENTILE_MIN
    )
    return bool(
        usable
        and distinct_clips_top20 >= DISTINCT_CLIPS_TOP20
        and label_holds_heldout is True
        and percentile_ok
        and absent_in_random_init is True
    )
