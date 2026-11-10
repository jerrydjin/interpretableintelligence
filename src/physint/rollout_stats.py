"""Paired tests and Wilson bounds for rollout causal scores.

No scipy. The early-stop rule is Newcombe's Wilson score interval on the
difference of two success rates (method 10 in Newcombe 1998). A control stops
once that interval excludes zero against every candidate.
"""

from __future__ import annotations

import math

import numpy as np

from physint.stats import percentile_le
from physint.thresholds import CAUSAL_CONTROL_PERCENTILE, CAUSAL_DELTA_MIN


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF, Acklam's approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"normal_ppf expects p in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def wilson_interval(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("wilson_interval needs at least one trial")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    z = normal_ppf(1.0 - alpha / 2.0)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def newcombe_diff_ci(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float,
) -> tuple[float, float, float]:
    """Newcombe Wilson interval for ``p_a - p_b``. Returns ``(delta, low, high)``."""
    if n_a <= 0 or n_b <= 0:
        raise ValueError("newcombe_diff_ci needs at least one trial on each arm")
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    low_a, high_a = wilson_interval(successes_a, n_a, alpha)
    low_b, high_b = wilson_interval(successes_b, n_b, alpha)
    delta = p_a - p_b
    low = delta - math.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2)
    high = delta + math.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2)
    return float(delta), float(low), float(high)


def gap_is_separated(
    control: list[bool] | np.ndarray,
    candidates: list[list[bool] | np.ndarray],
    alpha: float,
) -> bool:
    """True when the control's success rate is separated from every candidate.

    The gap is control minus candidate. Separated means the two-sided
    Newcombe interval at ``alpha`` excludes 0.
    """
    control_flags = [bool(item) for item in control]
    n = len(control_flags)
    if n <= 0 or not candidates:
        return False
    k_control = sum(control_flags)
    for candidate in candidates:
        cand_flags = [bool(item) for item in list(candidate)[:n]]
        if len(cand_flags) < n:
            raise ValueError("candidate successes are shorter than the control prefix")
        _delta, low, high = newcombe_diff_ci(k_control, n, sum(cand_flags), n, alpha)
        if low <= 0.0 <= high:
            return False
    return True


def binding_gap(
    control: list[bool] | np.ndarray,
    candidates: list[list[bool] | np.ndarray],
    alpha: float,
) -> tuple[float, float, float]:
    """Gap against the candidate whose lower bound is closest to crossing 0."""
    control_flags = [bool(item) for item in control]
    n = len(control_flags)
    if n <= 0 or not candidates:
        raise ValueError("binding_gap needs a control prefix and at least one candidate")
    k_control = sum(control_flags)
    best: tuple[float, float, float] | None = None
    for candidate in candidates:
        cand_flags = [bool(item) for item in list(candidate)[:n]]
        estimate = newcombe_diff_ci(k_control, n, sum(cand_flags), n, alpha)
        if best is None or estimate[1] < best[1]:
            best = estimate
    assert best is not None
    return best


def mcnemar_exact_pvalue(n_baseline_only: int, n_intervention_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs.

    ``n_baseline_only`` is baseline success and intervention failure.
    ``n_intervention_only`` is the opposite. Concordant pairs do not enter.
    """
    if n_baseline_only < 0 or n_intervention_only < 0:
        raise ValueError("discordant counts must be non-negative")
    n = n_baseline_only + n_intervention_only
    if n == 0:
        return 1.0
    k = min(n_baseline_only, n_intervention_only)
    if 2 * k == n:
        return 1.0
    log_half_n = -n * math.log(2.0)
    total = 0.0
    for i in range(k + 1):
        log_c = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        total += math.exp(log_c + log_half_n)
    return float(min(1.0, 2.0 * total))


def paired_success_delta_ci(
    baseline: list[bool] | np.ndarray,
    intervention: list[bool] | np.ndarray,
    alpha: float,
) -> tuple[float, float, float]:
    """Wald interval for the mean paired difference. Returns ``(delta, low, high)``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    base = [bool(item) for item in baseline]
    edited = [bool(item) for item in intervention]
    if len(base) != len(edited):
        raise ValueError(f"paired lengths differ: {len(base)} vs {len(edited)}")
    if not base:
        raise ValueError("paired_success_delta_ci needs at least one pair")
    diffs = [float(left) - float(right) for left, right in zip(base, edited, strict=True)]
    n = len(diffs)
    delta = sum(diffs) / n
    if n < 2:
        return float(delta), float(delta), float(delta)
    var = sum((item - delta) ** 2 for item in diffs) / (n - 1)
    se = math.sqrt(var / n)
    z = normal_ppf(1.0 - alpha / 2.0)
    return float(delta), float(delta - z * se), float(delta + z * se)


def discordant_counts(
    baseline: list[bool] | np.ndarray,
    intervention: list[bool] | np.ndarray,
) -> tuple[int, int]:
    base = [bool(item) for item in baseline]
    edited = [bool(item) for item in intervention]
    if len(base) != len(edited):
        raise ValueError(f"paired lengths differ: {len(base)} vs {len(edited)}")
    n_baseline_only = sum(1 for left, right in zip(base, edited, strict=True) if left and not right)
    n_intervention_only = sum(1 for left, right in zip(base, edited, strict=True) if right and not left)
    return n_baseline_only, n_intervention_only


def feature_is_causal(
    *,
    delta: float,
    p_value: float,
    control_percentile: float | None,
    alpha: float,
    delta_min: float = CAUSAL_DELTA_MIN,
    percentile_min: float = CAUSAL_CONTROL_PERCENTILE,
) -> bool:
    """Ablation hurt, the exact test clears ``alpha``, and controls do not match it.

    ``control_percentile is None`` means the comparison was not run and does
    not pass.
    """
    if control_percentile is None:
        return False
    return bool(
        delta >= delta_min
        and p_value <= alpha
        and control_percentile >= percentile_min
    )


def control_percentile(delta: float, control_deltas: list[float]) -> float | None:
    if not control_deltas:
        return None
    return percentile_le(delta, np.asarray(control_deltas, dtype=np.float64))
