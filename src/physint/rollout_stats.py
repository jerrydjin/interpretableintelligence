"""Paired tests and the control early-stop SPRT.

No scipy. A control stops when a two-sided sequential probability ratio test
on its discordant pairs separates it from every candidate. The threshold is
a Ville bound, so repeated looks keep the type I error at ``alpha``. The
final causal call is still a single McNemar test on the episodes that ran.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from physint.thresholds import CAUSAL_CONTROL_PERCENTILE, CAUSAL_DELTA_MIN, SPRT_LAMBDA

# Deltas closer than this are the same effect for the percentile.
_PERCENTILE_TIE_ATOL = 1e-9


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


def sprt_threshold(alpha: float) -> float:
    """Log likelihood-ratio boundary for a two-sided test at level ``alpha``.

    Each one-sided martingale is compared with ``2 / alpha``. Ville's
    inequality bounds that side by ``alpha / 2``, so the union is at most
    ``alpha`` at every look, including looks that stop early.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.log(2.0 / alpha)


def _check_lambda(lam: float) -> None:
    if not 0.0 < lam < 1.0:
        raise ValueError(f"sprt lambda must be in (0, 1), got {lam}")


def sprt_log_lr(
    control: list[bool],
    candidate: list[bool],
    lam: float = SPRT_LAMBDA,
) -> float:
    """Path-maximum two-sided log likelihood ratio on discordant pairs.

    A pair is ``+1`` when the control succeeds and the candidate fails, and
    ``-1`` the other way. Concordant pairs are skipped. Under the null that
    the arms are exchangeable, the signs are fair. The alternative for the
    positive martingale is ``P(sign = +1) = (1 + lam) / 2``; the negative
    martingale swaps that. The return value is the largest log wealth either
    martingale reached on this prefix. It starts at 0 when there are no
    discordant pairs.
    """
    _check_lambda(lam)
    control_flags = [bool(item) for item in control]
    cand_flags = [bool(item) for item in candidate]
    if len(control_flags) != len(cand_flags):
        raise ValueError(
            f"sprt lengths differ: control {len(control_flags)} vs candidate {len(cand_flags)}"
        )
    log_pos = 0.0
    log_neg = 0.0
    best = 0.0
    log_up = math.log(1.0 + lam)
    log_down = math.log(1.0 - lam)
    for left, right in zip(control_flags, cand_flags, strict=True):
        if left == right:
            continue
        if left and not right:
            log_pos += log_up
            log_neg += log_down
        else:
            log_pos += log_down
            log_neg += log_up
        best = max(best, log_pos, log_neg)
    return best


def sprt_separated(
    control: list[bool],
    candidates: list[list[bool]],
    alpha: float,
    lam: float = SPRT_LAMBDA,
) -> bool:
    """True when the control is separated from every candidate at level ``alpha``.

    Separation is ``sprt_log_lr >= sprt_threshold(alpha)``. An empty prefix
    or an empty candidate list does not separate.
    """
    if not control or not candidates:
        return False
    n = len(control)
    threshold = sprt_threshold(alpha)
    for candidate in candidates:
        if len(candidate) < n:
            raise ValueError("candidate successes are shorter than the control prefix")
        if sprt_log_lr(list(control), list(candidate)[:n], lam) < threshold:
            return False
    return True


def binding_sprt(
    control: list[bool],
    candidates: list[list[bool]],
    alpha: float,
    lam: float = SPRT_LAMBDA,
) -> tuple[float, float, float]:
    """Gap and log likelihood ratio against the hardest candidate.

    The hardest candidate is the one with the smallest path-maximum log
    likelihood ratio. The gap is the control success rate minus that
    candidate's success rate on the shared prefix. The third value is the
    log threshold ``log(2 / alpha)``.
    """
    if not control or not candidates:
        raise ValueError("binding_sprt needs a control prefix and at least one candidate")
    n = len(control)
    threshold = sprt_threshold(alpha)
    best_lr: float | None = None
    best_gap = 0.0
    control_rate = sum(bool(item) for item in control) / n
    for candidate in candidates:
        if len(candidate) < n:
            raise ValueError("candidate successes are shorter than the control prefix")
        prefix = [bool(item) for item in candidate[:n]]
        log_lr = sprt_log_lr([bool(item) for item in control], prefix, lam)
        if best_lr is None or log_lr < best_lr:
            best_lr = log_lr
            best_gap = control_rate - (sum(prefix) / n)
    assert best_lr is not None
    return float(best_gap), float(best_lr), float(threshold)


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
    baseline: Sequence[bool],
    intervention: Sequence[bool],
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
    baseline: Sequence[bool],
    intervention: Sequence[bool],
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
    """Average-rank percentile of ``delta`` among the comparison deltas.

    ``100 * (n_strictly_below + 0.5 * n_tied) / n``. A delta tied with every
    comparison scores 50. A delta strictly above every comparison scores 100.
    A delta strictly below every comparison scores 0. Values within ``1e-9``
    count as tied. An empty comparison returns None, which does not pass the
    causal gate.

    This is not ``percentile_le``. That scoreboard helper counts a tie as
    ``<=``, so a row of equal deltas all score 100.
    """
    if not control_deltas:
        return None
    below = 0
    tied = 0
    for other in control_deltas:
        if math.isclose(float(delta), float(other), rel_tol=0.0, abs_tol=_PERCENTILE_TIE_ATOL):
            tied += 1
        elif float(other) < float(delta):
            below += 1
    return 100.0 * (below + 0.5 * tied) / len(control_deltas)
