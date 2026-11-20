"""Rollout statistics: paired tests, the control early-stop SPRT, and the Sobel-Wald rise/fall SPRT.

This is the one stats module for rollouts. The fake lane and ii rollouts use the first part; the GR00T runner
(adapters/groot/v9/ablate_v9.py) and research rule 1 on --pick plans use the rise/fall SPRT and its exact
sequential inference at the end of the file.

No scipy. A control stops only for efficacy: a one-sided sequential
probability ratio test on discordant pairs, in the direction where the
control succeeds and the candidate fails. There is no futility boundary.
The threshold is the Wald boundary at a Bonferroni share of ``alpha`` and at
the per-control beta that keeps family-wise power at ``SPRT_FAMILY_POWER``.
The final causal call is still a single McNemar test on the episodes that ran.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from interpretableintelligence.thresholds import (
    CAUSAL_CONTROL_PERCENTILE,
    CAUSAL_DELTA_MIN,
    SPRT_FAMILY_POWER,
    SPRT_LAMBDA,
)

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


def family_sprt_error_rates(alpha: float, n_controls: int) -> tuple[float, float]:
    """Per-control ``(alpha, beta)`` inside a family of ``n_controls``.

    Alpha is split by Bonferroni, so the family-wise type I error of stopping
    a control is at most ``alpha``. Beta solves
    ``(1 - beta) ** n_controls = SPRT_FAMILY_POWER``, so the probability that
    every control stops under the design alternative is the family-wise power.
    For 10 controls that beta is about 0.022.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_controls < 1:
        raise ValueError(f"n_controls must be positive, got {n_controls}")
    alpha_per = alpha / n_controls
    beta = 1.0 - SPRT_FAMILY_POWER ** (1.0 / n_controls)
    if not 0.0 < beta < 1.0:
        raise ValueError(f"per-control beta must be in (0, 1), got {beta}")
    return alpha_per, beta


def sprt_threshold(alpha: float, n_controls: int) -> float:
    """Log likelihood-ratio boundary for one control, efficacy only.

    ``log((1 - beta) / alpha_per)``. A control crosses it only in the
    efficacy direction. Paths that never cross run to the episode cap;
    there is no boundary that accepts the null early.
    """
    alpha_per, beta = family_sprt_error_rates(alpha, n_controls)
    return math.log((1.0 - beta) / alpha_per)


def _check_lambda(lam: float) -> None:
    if not 0.0 < lam < 1.0:
        raise ValueError(f"sprt lambda must be in (0, 1), got {lam}")


def sprt_log_lr(
    control: list[bool],
    candidate: list[bool],
    lam: float = SPRT_LAMBDA,
) -> float:
    """Path-maximum one-sided log likelihood ratio for efficacy.

    A discordant pair is ``+1`` when the control succeeds and the candidate
    fails, and ``-1`` the other way. Concordant pairs are skipped. The
    alternative is ``P(+1) = (1 + lam) / 2`` against the null ``1/2``. Only
    the efficacy direction can cross the boundary: a run of ``-1`` pairs
    drives the likelihood ratio down. The return value is the largest log
    likelihood ratio reached on this prefix, and it is 0 when the path
    never rose above its starting value.
    """
    _check_lambda(lam)
    control_flags = [bool(item) for item in control]
    cand_flags = [bool(item) for item in candidate]
    if len(control_flags) != len(cand_flags):
        raise ValueError(
            f"sprt lengths differ: control {len(control_flags)} vs candidate {len(cand_flags)}"
        )
    theta1 = (1.0 + lam) / 2.0
    log_up = math.log(theta1 / 0.5)
    log_down = math.log((1.0 - theta1) / 0.5)
    llr = 0.0
    best = 0.0
    for left, right in zip(control_flags, cand_flags, strict=True):
        if left == right:
            continue
        if left and not right:
            llr += log_up
        else:
            llr += log_down
        if llr > best:
            best = llr
    return best


def sprt_separated(
    control: list[bool],
    candidates: list[list[bool]],
    alpha: float,
    lam: float = SPRT_LAMBDA,
    *,
    n_controls: int,
) -> bool:
    """True when efficacy has separated this control from every candidate.

    Separation is ``sprt_log_lr >= sprt_threshold(alpha, n_controls)``.
    An empty prefix or an empty candidate list does not separate. The
    opposite discordant direction never separates a control.
    """
    if not control or not candidates:
        return False
    n = len(control)
    threshold = sprt_threshold(alpha, n_controls)
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
    *,
    n_controls: int,
) -> tuple[float, float, float]:
    """Gap and efficacy log likelihood ratio against the hardest candidate.

    The hardest candidate is the one with the smallest path-maximum log
    likelihood ratio. The gap is the control success rate minus that
    candidate's success rate on the shared prefix. The third value is the
    efficacy log threshold.
    """
    if not control or not candidates:
        raise ValueError("binding_sprt needs a control prefix and at least one candidate")
    n = len(control)
    threshold = sprt_threshold(alpha, n_controls)
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


# --------------------------------------------------------------------------- Sobel-Wald rise/fall SPRT (GR00T plans)
# Two one-sided Wald tests on one arm's success sequence (H0 p=p0 vs H1 p=p1_rise, and vs p1_fall), exact inference
# under that stopping rule, Holm, and the arm-vs-matched-controls comparison. Plans carry the design in a ``sprt`` dict
# (p0, p1_rise, p1_fall, alpha, beta, cap, stop_rule). Used by the GR00T runner (adapters/groot/v9/ablate_v9.py) and by
# ii rollouts for research rule 1 on --pick plans.
def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Wald boundaries on the log-likelihood ratio: accept H0 at <= lo, accept H1 at >= hi."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def llr_step(success: bool, p0: float, p1: float) -> float:
    return math.log(p1 / p0) if success else math.log((1.0 - p1) / (1.0 - p0))


def sequential_decision(outcomes: list[bool], sprt: dict) -> dict:
    """Run the rise and fall SPRTs over `outcomes` (episode-index order); a test freezes once it crosses a boundary.

    stop_rule (sprt["stop_rule"]):
      "h1_or_both_h0" (default; Sobel-Wald 3-way SPRT): stop when either test accepts its H1 (rise / fall), or when both
          have accepted H0 (no effect either way), or at the cap. Since any H1 crossing happens after the other test has
          already accepted H0 (see below) this equals "both_tests_decide".
      "either_test_decides": stop at the first crossing of either boundary by either test. With p0=0.56, p1=0.86/0.26 and
          alpha=beta=0.05 this can never return rise or fall: reaching rise-H1 needs S >= 6.86 + 2.67 F successes, at which
          the fall LLR is <= -5.26 - 1.53 F < log(beta/(1-alpha)) = -2.94, i.e. the fall test has already accepted H0
          (symmetrically for fall-H1). Kept only so the literal reading can be checked.
      "both_tests_decide": stop when both tests have decided."""
    p0, a, b, cap = float(sprt["p0"]), float(sprt["alpha"]), float(sprt["beta"]), int(sprt["cap"])
    lo, hi = sprt_bounds(a, b)
    rule = sprt.get("stop_rule", "h1_or_both_h0")
    if rule not in ("h1_or_both_h0", "either_test_decides", "both_tests_decide"):
        raise ValueError(f"unknown stop_rule {rule!r}")
    tests = {name: {"p0": p0, "p1": float(sprt[key]), "llr": 0.0, "decision": None, "n_decided": None}
             for name, key in (("rise", "p1_rise"), ("fall", "p1_fall"))}
    n = 0
    stopped, reason = False, None
    for y in outcomes[:cap]:
        n += 1
        for t in tests.values():
            if t["decision"] is not None:
                continue
            t["llr"] += llr_step(bool(y), t["p0"], t["p1"])
            if t["llr"] >= hi:
                t["decision"], t["n_decided"] = "H1", n
            elif t["llr"] <= lo:
                t["decision"], t["n_decided"] = "H0", n
        decided = [k for k, t in tests.items() if t["decision"] is not None]
        h1 = [k for k in decided if tests[k]["decision"] == "H1"]
        if ((rule == "either_test_decides" and decided) or (rule == "both_tests_decide" and len(decided) == 2)
                or (rule == "h1_or_both_h0" and (h1 or len(decided) == 2))):
            stopped, reason = True, "sprt:" + ",".join(f"{k}={tests[k]['decision']}" for k in decided)
            break
    if not stopped and n >= cap:
        stopped, reason = True, "cap"
    for t in tests.values():
        t["llr"] = round(t["llr"], 6)
    if tests["rise"]["decision"] == "H1":
        verdict = "rise"
    elif tests["fall"]["decision"] == "H1":
        verdict = "fall"
    elif tests["rise"]["decision"] == "H0" and tests["fall"]["decision"] == "H0":
        verdict = "no_effect"
    elif stopped and reason != "cap":
        verdict = "no_" + ("rise" if tests["rise"]["decision"] == "H0" else "fall")  # only under "either_test_decides"
    else:
        verdict = "cap_undecided" if stopped else "running"
    return {"n_used": n, "stopped": stopped, "stop_reason": reason, "verdict": verdict, "bounds": [round(lo, 6), round(hi, 6)],
            "stop_rule": rule, "tests": tests}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


# --------------------------------------------------------------------------- sequential inference (pure)
# The Wilson interval above treats n as fixed. Here n is chosen by the SPRT, so a condition that stopped early on a run
# of failures is over-represented among "fall" stops, and Wilson / fixed-n binomial tests are not valid. Below: exact
# inference under the plan's own stopping rule. The SPRT's log-likelihood ratio after n episodes depends only on
# (n, S_n), so the probability that a run stops at (n, s) is paths(n, s) * p^s (1-p)^(n-s), where paths(n, s) counts the
# success/failure orderings that reach (n, s) without having stopped before (each test freezes once it crosses). We
# enumerate those exactly (dynamic programme over n <= cap), order outcomes by the sample proportion s/n (the MLE /
# sample-mean ordering), and invert the two one-sided exact tests: an equal-tailed confidence set that is valid however
# early the run stopped, and one-sided p-values against the plan's p0.
_VERDICTS_STOPPED = ("rise", "fall", "no_effect", "no_rise", "no_fall", "cap_undecided")


def _sprt_key(sprt: dict) -> tuple:
    return (float(sprt["p0"]), float(sprt["p1_rise"]), float(sprt["p1_fall"]), float(sprt["alpha"]), float(sprt["beta"]),
            int(sprt["cap"]), sprt.get("stop_rule", "h1_or_both_h0"))


def _closed_llr(n: int, s: int, p0: float, p1: float) -> float:
    return s * math.log(p1 / p0) + (n - s) * math.log((1.0 - p1) / (1.0 - p0))


_STOP_CACHE: dict[tuple, list[tuple[int, int, str, int]]] = {}


def sprt_stop_points(sprt: dict) -> list[tuple[int, int, str, int]]:
    """Every way the plan's SPRT (sequential_decision, same stop rule) can stop: [(n, s, verdict, n_paths)], exact.

    sum over points of n_paths * p^s * (1-p)^(n-s) == 1 for every p. The boundaries are compared with the closed-form
    LLR s*log(p1/p0) + (n-s)*log((1-p1)/(1-p0)); sequential_decision sums the same steps one by one, and the two agree
    unless an LLR lands within float rounding of a boundary (min_boundary_gap reports the closest one)."""
    key = _sprt_key(sprt)
    if key in _STOP_CACHE:
        return _STOP_CACHE[key]
    p0, p1r, p1f, a, b, cap, rule = key
    lo, hi = sprt_bounds(a, b)
    # state: (s, rise decision, fall decision) -> number of paths; decision in (None, "H0", "H1")
    live: dict[tuple, int] = {(0, None, None): 1}
    stops: dict[tuple[int, int, str], int] = {}
    for n in range(1, cap + 1):
        nxt: dict[tuple, int] = {}
        for (s0, dr0, df0), c in live.items():
            for y in (1, 0):
                s = s0 + y
                dec = {"rise": dr0, "fall": df0}
                for name, p1 in (("rise", p1r), ("fall", p1f)):
                    if dec[name] is None:
                        llr = _closed_llr(n, s, p0, p1)
                        dec[name] = "H1" if llr >= hi else ("H0" if llr <= lo else None)
                decided = [k for k in ("rise", "fall") if dec[k] is not None]
                h1 = [k for k in decided if dec[k] == "H1"]
                stop = ((rule == "either_test_decides" and decided) or (rule == "both_tests_decide" and len(decided) == 2)
                        or (rule == "h1_or_both_h0" and (h1 or len(decided) == 2)))
                if stop or n == cap:
                    if dec["rise"] == "H1":
                        v = "rise"
                    elif dec["fall"] == "H1":
                        v = "fall"
                    elif dec["rise"] == "H0" and dec["fall"] == "H0":
                        v = "no_effect"
                    elif stop:
                        v = "no_" + ("rise" if dec["rise"] == "H0" else "fall")
                    else:
                        v = "cap_undecided"
                    stops[(n, s, v)] = stops.get((n, s, v), 0) + c
                else:
                    k = (s, dec["rise"], dec["fall"])
                    nxt[k] = nxt.get(k, 0) + c
        live = nxt
        if not live:
            break
    out = sorted((n, s, v, c) for (n, s, v), c in stops.items())
    _STOP_CACHE[key] = out
    return out


def min_boundary_gap(sprt: dict) -> float:
    """Smallest |closed-form LLR - boundary| over all (n, s) the SPRT can visit; > ~1e-9 means sprt_stop_points and
    sequential_decision cannot disagree through float rounding."""
    p0, p1r, p1f, a, b, cap, _ = _sprt_key(sprt)
    lo, hi = sprt_bounds(a, b)
    return min(abs(_closed_llr(n, s, p0, p1) - bd) for n in range(1, cap + 1) for s in range(n + 1)
               for p1 in (p1r, p1f) for bd in (lo, hi))


def _stop_arrays(sprt: dict):
    pts = sprt_stop_points(sprt)
    n = np.array([q[0] for q in pts], float)
    s = np.array([q[1] for q in pts], float)
    logc = np.array([math.log(q[3]) for q in pts], float)
    return pts, n, s, logc


def _stop_probs(sprt: dict, p: np.ndarray) -> np.ndarray:
    """[len(p), n_points] probability of stopping at each point, for each p in [0, 1]."""
    _pts, n, s, logc = _stop_arrays(sprt)
    p = np.atleast_1d(np.asarray(p, float))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(s > 0, s * np.log(p), 0.0) + np.where(n - s > 0, (n - s) * np.log1p(-p), 0.0)
    return np.exp(logc + lp)


def _tails(sprt: dict, n_obs: int, s_obs: int, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper P_p(S/N >= s_obs/n_obs) and lower P_p(S/N <= s_obs/n_obs) at the stop, under the plan's stopping rule."""
    _pts, n, s, _ = _stop_arrays(sprt)
    P = _stop_probs(sprt, p)
    ge = s * n_obs >= s_obs * n        # exact rational comparison of s/n with s_obs/n_obs
    le = s * n_obs <= s_obs * n
    return np.minimum(1.0, P[:, ge].sum(1)), np.minimum(1.0, P[:, le].sum(1))


def _root(f, a: float, b: float, it: int = 60) -> float:
    """Boundary between f(a) True and f(b) False (bisection)."""
    for _ in range(it):
        m = 0.5 * (a + b)
        if f(m):
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def sequential_ci(n: int, s: int, sprt: dict, conf: float = 0.95, grid: int = 2001) -> list[float]:
    """Exact equal-tailed confidence interval for p that stays valid under the plan's early stopping:
    {p : P_p(S/N >= s/n) > (1-conf)/2 and P_p(S/N <= s/n) > (1-conf)/2}. Both tails are monotone in p for these
    designs (checked in tests), so the set is an interval; endpoints are refined by bisection and rounded outward."""
    if not any(q[0] == n and q[1] == s for q in sprt_stop_points(sprt)):
        raise ValueError(f"(n={n}, s={s}) is not a point where this SPRT can stop")
    half = (1.0 - conf) / 2.0
    ps = np.linspace(0.0, 1.0, grid)
    up, low = _tails(sprt, n, s, ps)
    ok = (up > half) & (low > half)
    if not ok.any():
        raise RuntimeError("empty confidence set")
    i0, i1 = int(np.argmax(ok)), int(len(ok) - 1 - np.argmax(ok[::-1]))
    lo_b = 0.0 if i0 == 0 else _root(lambda q: not (_tails(sprt, n, s, [q])[0][0] > half), ps[i0 - 1], ps[i0])
    hi_b = 1.0 if i1 == grid - 1 else _root(lambda q: _tails(sprt, n, s, [q])[1][0] > half, ps[i1], ps[i1 + 1])
    return [math.floor(lo_b * 1e4) / 1e4, math.ceil(hi_b * 1e4) / 1e4]


def sequential_pvalues(n: int, s: int, sprt: dict) -> dict:
    """Exact one-sided p-values against the plan's p0 under its stopping rule (valid however early the run stopped):
    rise = P_p0(S/N >= s/n), fall = P_p0(S/N <= s/n); two_sided = min(1, 2 * min)."""
    up, low = _tails(sprt, n, s, [float(sprt["p0"])])
    r, f = float(up[0]), float(low[0])
    return {"rise": round(r, 6), "fall": round(f, 6), "two_sided": round(min(1.0, 2 * min(r, f)), 6)}


def sequential_inference(n: int, s: int, sprt: dict, conf: float = 0.95) -> dict:
    return {"ci95": sequential_ci(n, s, sprt, conf), "p_vs_p0": sequential_pvalues(n, s, sprt), "p0": float(sprt["p0"]),
            "ordering": "sample proportion s/n at the stop (MLE ordering)",
            "method": "exact: enumerated stopping points of the plan's SPRT (h1/h0 boundaries, cap), inverted tails"}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family = the keys)."""
    ks = sorted(pvals, key=lambda k: pvals[k])
    m, out, run = len(ks), {}, 0.0
    for j, k in enumerate(ks):
        run = max(run, min(1.0, (m - j) * pvals[k]))
        out[k] = round(run, 6)
    return {k: out[k] for k in pvals}


def control_comparison(cand: tuple[int, int], controls: dict[str, tuple[int, int, str | None]]) -> dict:
    """Candidate (n, s) against its matched random controls {cid: (n, s, verdict)}.

    Statistic: the success proportion s/n at the stop. Every condition runs the same SPRT design, so ordering by s/n is
    the same as ordering by the exact sequential fall p-value. p_fall = (1 + #controls with s/n <= candidate) / (1 + K)
    is the permutation-style p-value for "a random feature does at least as badly as the candidate"; p_rise is the
    mirror image (at least as well). With K = 10 the smallest attainable value is 1/11 = 0.0909."""
    cn, cs = cand
    K = len(controls)
    worse = sorted(c for c, (n, s, _) in controls.items() if s * cn <= cs * n)
    better = sorted(c for c, (n, s, _) in controls.items() if s * cn >= cs * n)
    fell = sorted(c for c, (_n, _s, v) in controls.items() if v == "fall")
    rose = sorted(c for c, (_n, _s, v) in controls.items() if v == "rise")
    return {"n_controls": K, "controls_fell": len(fell), "controls_fell_ids": fell, "controls_rose": len(rose),
            "n_controls_at_least_as_bad": len(worse), "controls_at_least_as_bad_ids": worse,
            "frac_controls_at_least_as_bad": round(len(worse) / K, 4) if K else None,
            "p_perm_fall": round((1 + len(worse)) / (1 + K), 6), "n_controls_at_least_as_good": len(better),
            "p_perm_rise": round((1 + len(better)) / (1 + K), 6),
            "beats_every_control_rise": bool(K) and all(s * cn < cs * n for n, s, _ in controls.values()),
            "statistic": "success proportion s/n at the SPRT stop (same order as the sequential fall p-value)",
            "min_attainable_p": round(1 / (1 + K), 6)}


def arm_inference(rows: dict[str, dict], conds: dict[str, dict], alpha: float = 0.05) -> dict:
    """(b) per-arm control comparison and (c) Holm across candidates, from per-condition {cid: {n, successes, verdict}}.

    Holm families (m = number of candidates): sequential rise p, sequential fall p, permutation rise p, permutation
    fall p. Reported only; nothing here sets a verdict or `real`."""
    cands = [c for c, d in conds.items() if d["kind"] == "candidate"]
    out: dict[str, Any] = {"alpha": alpha, "candidates": {}}
    for c in cands:
        arm = conds[c]["arm"]
        ctrl = {k: (rows[k]["n"], rows[k]["successes"], rows[k].get("verdict")) for k, d in conds.items()
                if d["kind"] == "control" and d["arm"] == arm and k in rows and rows[k]["n"]}
        out["candidates"][c] = {"arm": arm, "n": rows[c]["n"], "successes": rows[c]["successes"],
                                "verdict": rows[c].get("verdict"), "p_seq": rows[c]["sequential"]["p_vs_p0"],
                                "vs_controls": control_comparison((rows[c]["n"], rows[c]["successes"]), ctrl)}
    fam = {"p_seq_rise": lambda d: d["p_seq"]["rise"], "p_seq_fall": lambda d: d["p_seq"]["fall"],
           "p_perm_rise": lambda d: d["vs_controls"]["p_perm_rise"], "p_perm_fall": lambda d: d["vs_controls"]["p_perm_fall"]}
    out["holm"] = {}
    for name, get in fam.items():
        adj = holm({c: get(d) for c, d in out["candidates"].items()})
        out["holm"][name] = adj
        for c, v in adj.items():
            out["candidates"][c].setdefault("holm", {})[name] = v
    out["holm_note"] = f"Holm step-down across the {len(cands)} candidates, separately for each p-value family"
    return out


def mcnemar_exact_p(n10: int, n01: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts (symmetric in its arguments)."""
    return mcnemar_exact_pvalue(n10, n01)


def paired_delta(cond: dict[int, bool], base: dict[int, bool]) -> dict:
    """Success rate of a condition minus the baseline's on the episode indices (= seeds) both have."""
    shared = sorted(set(cond) & set(base))
    n = len(shared)
    if not n:
        return {"n_shared": 0, "delta": None}
    c = np.array([cond[i] for i in shared], bool)
    bb = np.array([base[i] for i in shared], bool)
    n10, n01 = int((c & ~bb).sum()), int((~c & bb).sum())
    return {"n_shared": n, "rate_cond": round(float(c.mean()), 4), "rate_base": round(float(bb.mean()), 4),
            "delta": round(float(c.mean() - bb.mean()), 4), "n_cond_only_success": n10, "n_base_only_success": n01,
            "mcnemar_exact_p_two_sided": round(mcnemar_exact_p(n10, n01), 6), "shared_eps_max": shared[-1]}
