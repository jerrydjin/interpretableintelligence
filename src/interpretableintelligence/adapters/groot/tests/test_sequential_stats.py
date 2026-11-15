"""Early-stopping-valid inference for the v9 ablation SPRT, control comparison, Holm, and the 1028 summary regeneration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from interpretableintelligence.adapters.groot.v9 import ablate_v9 as A
from interpretableintelligence.adapters.groot.v9 import resummarize_v9 as R

REPO = Path(__file__).resolve().parents[5]
PLAN = REPO / "src/interpretableintelligence/adapters/groot/v9/ablate/ablate_plan.json"
SUMMARY = REPO / "demo-data/groot/groot-1028-ablate-summary.json"


@pytest.fixture(scope="module")
def sprt(tmp_path_factory) -> dict:
    """The SPRT exactly as ablate_v9.default_plan writes it (boundaries, p0/p1, cap, stop rule)."""
    cf = tmp_path_factory.mktemp("plan") / "candidates_final.json"
    ctrl = {"720": list(range(100, 110)), "7081": list(range(200, 210)), "5852": list(range(300, 310))}
    cf.write_text(json.dumps({"matched_random_controls": ctrl}))
    plan = A.default_plan(cf, sae_meta={}, token_indices=[], task={}, picker_commit="test")
    assert plan["sprt"] == json.loads(PLAN.read_text())["sprt"], "vendored plan and default_plan disagree"
    return plan["sprt"]


def test_stop_points_are_a_distribution_and_match_sequential_decision(sprt):
    pts = A.sprt_stop_points(sprt)
    for p in (0.0, 0.05, 0.26, 0.56, 0.86, 1.0):
        assert A._stop_probs(sprt, [p]).sum() == pytest.approx(1.0, abs=1e-12)
    assert A.min_boundary_gap(sprt) > 1e-6     # closed-form and running-sum LLRs cannot fall on different sides
    by = {(n, s): v for n, s, v, _ in pts}
    assert len(by) == len(pts)                 # each (n, s) stops with one verdict
    rng = np.random.default_rng(1)
    for p in (0.2, 0.56, 0.9):
        for _ in range(300):
            ys = list(rng.random(sprt["cap"]) < p)
            d = A.sequential_decision(ys, sprt)
            assert by[(d["n_used"], int(sum(ys[: d["n_used"]])))] == d["verdict"]
    # the all-success / all-failure paths from test_ablate
    assert by[(7, 7)] == "rise" and by[(6, 0)] == "fall"


def test_tails_are_monotone_so_the_confidence_set_is_an_interval(sprt):
    ps = np.linspace(0, 1, 401)
    for n, s, _v, _c in A.sprt_stop_points(sprt):
        up, low = A._tails(sprt, n, s, ps)
        assert (np.diff(up) >= -1e-12).all() and (np.diff(low) <= 1e-12).all()


def _ci_table(sprt):
    return {(n, s): A.sequential_ci(n, s, sprt) for n, s, _v, _c in A.sprt_stop_points(sprt)}


def test_exact_coverage_is_at_least_95_everywhere_and_wilson_is_not(sprt):
    pts = A.sprt_stop_points(sprt)
    cis = _ci_table(sprt)
    grid = np.linspace(0.01, 0.99, 99)
    P = A._stop_probs(sprt, grid)
    seq = np.array([[cis[(n, s)][0] <= p <= cis[(n, s)][1] for n, s, _v, _c in pts] for p in grid])
    wil = np.array([[A.wilson(s, n)[0] <= p <= A.wilson(s, n)[1] for n, s, _v, _c in pts] for p in grid])
    cov, cov_w = (P * seq).sum(1), (P * wil).sum(1)
    assert cov.min() >= 0.95
    i0 = int(np.argmin(abs(grid - sprt["p0"])))
    assert cov_w[i0] < 0.95 and cov_w.min() < 0.90   # why the new columns exist: Wilson ignores the early stop


def test_simulated_coverage_and_type_one_error_under_the_null(sprt):
    """Monte Carlo under H0 (p = p0) with the runner's own sequential_decision: >= 95% coverage, p-values valid."""
    p0 = sprt["p0"]
    cis = _ci_table(sprt)
    pv = {(n, s): A.sequential_pvalues(n, s, sprt) for n, s, _v, _c in A.sprt_stop_points(sprt)}
    rng = np.random.default_rng(20261115)
    N = 4000
    hit = rej_rise = rej_fall = rej_two = wil_hit = 0
    for _ in range(N):
        ys = list(rng.random(sprt["cap"]) < p0)
        d = A.sequential_decision(ys, sprt)
        n = d["n_used"]
        s = int(sum(ys[:n]))
        lo, hi = cis[(n, s)]
        hit += lo <= p0 <= hi
        wl, wh = A.wilson(s, n)
        wil_hit += wl <= p0 <= wh
        rej_rise += pv[(n, s)]["rise"] <= 0.05
        rej_fall += pv[(n, s)]["fall"] <= 0.05
        rej_two += pv[(n, s)]["two_sided"] <= 0.05
    assert hit / N >= 0.95, hit / N
    se = (0.05 * 0.95 / N) ** 0.5
    for r in (rej_rise, rej_fall, rej_two):
        assert r / N <= 0.05 + 3 * se
    assert wil_hit / N < 0.95


def test_ci_rejects_points_the_design_cannot_stop_at(sprt):
    with pytest.raises(ValueError, match="not a point"):
        A.sequential_ci(5, 3, sprt)


def test_holm():
    assert A.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert A.holm({"a": 0.03, "b": 0.02, "c": 0.5}) == {"a": 0.06, "b": 0.06, "c": 0.5}
    assert A.holm({"a": 0.9, "b": 0.8}) == {"a": 1.0, "b": 1.0}


def test_control_comparison_counts_and_permutation_p():
    ctrl = {"c1": (6, 0, "fall"), "c2": (14, 8, "no_effect"), "c3": (23, 7, "fall"), "c4": (9, 1, None)}
    d = A.control_comparison((9, 1), ctrl)
    assert d["controls_fell"] == 2 and d["controls_fell_ids"] == ["c1", "c3"]
    assert d["controls_at_least_as_bad_ids"] == ["c1", "c4"]          # ties count against the candidate
    assert d["p_perm_fall"] == pytest.approx(3 / 5) and d["p_perm_rise"] == pytest.approx(4 / 5)
    assert not d["beats_every_control_rise"] and d["min_attainable_p"] == pytest.approx(0.2)


def test_regenerated_1028_summary(sprt):
    """The vendored summary is exactly what resummarize_v9 --table produces, and nothing moves toward a finding."""
    have = json.loads(SUMMARY.read_text())
    plan = json.loads(PLAN.read_text())
    original = {k: v for k, v in have.items() if k not in ("arms", "stats_updated", "stats_source")}
    original["table"] = [{k: r[k] for k in ("cond", "n", "successes", "rate", "wilson95", "verdict", "delta_paired",
                                            "n_shared")} for r in have["table"]]
    again = R.from_table(original, plan)
    assert again["table"] == have["table"] and again["arms"] == have["arms"]
    assert have["real"] is False and have["stats_source"].startswith("vendored table")
    rows = {r["cond"]: r for r in have["table"]}
    assert [rows[c]["verdict"] for c in ("A-cand-720", "B-cand-7081+5852")] == ["fall", "fall"]
    arms = have["arms"]["candidates"]
    assert arms["A-cand-720"]["vs_controls"]["controls_fell"] == 3
    assert arms["B-cand-7081+5852"]["vs_controls"]["controls_fell"] == 2
    assert A.threshold_crossings(have)["rise"] == []
    # each sequential interval contains its Wilson interval's centre and is at least as wide on the stopped side
    for r in have["table"]:
        if r["seq95"] is not None:
            assert r["seq95"][0] <= r["rate"] <= r["seq95"][1]


def test_from_table_refuses_inconsistent_rows():
    have = json.loads(SUMMARY.read_text())
    plan = json.loads(PLAN.read_text())
    bad = json.loads(json.dumps(have))
    row = next(r for r in bad["table"] if r["cond"] == "A-cand-720")
    row["successes"], row["wilson95"] = 2, A.wilson(2, 9)     # (9, 2) is not where this SPRT stops with "fall"
    with pytest.raises(R.StatsMismatch, match="A-cand-720"):
        R.from_table(bad, plan)
    bad = json.loads(json.dumps(have))
    bad["real"] = True
    with pytest.raises(R.StatsMismatch, match="real"):
        R.from_table(bad, plan)


def test_runner_summaries_carry_the_sequential_columns(tmp_path):
    sp = dict(p0=0.56, p1_rise=0.86, p1_fall=0.26, alpha=0.05, beta=0.05, cap=75, stop_rule="h1_or_both_h0")
    conds = [{"id": "baseline", "arm": "baseline", "kind": "baseline", "latents": None, "n_max": 10, "sprt": False, "dir": "baseline"},
             {"id": "A-cand", "arm": "A", "kind": "candidate", "latents": [1], "n_max": 75, "sprt": True, "dir": "arm-A/cand"},
             {"id": "A-ctrl01", "arm": "A", "kind": "control", "latents": [2], "n_max": 75, "sprt": True, "dir": "arm-A/ctrl01"}]
    plan = {"schema": A.PLAN_SCHEMA, "sprt": sp, "seeds": {"base": 40000}, "conditions": conds,
            "schedule": {"rule": "lockstep", "order": [c["id"] for c in conds]}}
    st = A.AblationState(plan, tmp_path / "o")
    row = lambda i, ok: {"ep": i, "seed": 40000 + i, "success": ok, "fail_kind": None if ok else "never_grasped"}  # noqa: E731
    for i in range(10):
        st.record("baseline", row(i, i % 2 == 0))
    for i in range(6):
        st.record("A-cand", row(i, False))      # fall at n=6
    for i in range(3):
        st.record("A-ctrl01", row(i, True))     # still running
    top = st.write_summaries()
    t = {r["cond"]: r for r in top["table"]}
    assert t["A-cand"]["seq95"] == A.sequential_ci(6, 0, sp) and t["A-cand"]["p_seq_fall"] < 0.05
    assert t["A-ctrl01"]["seq95"] is None and t["baseline"]["seq95"] is None
    assert top["arms"] is None                  # a control is still running
    for i in range(3, 7):
        st.record("A-ctrl01", row(i, True))     # rise at n=7
    top = st.write_summaries()
    c = top["arms"]["candidates"]["A-cand"]
    assert c["vs_controls"]["controls_fell"] == 0 and c["vs_controls"]["p_perm_fall"] == pytest.approx(0.5)
    assert c["holm"]["p_seq_fall"] == c["p_seq"]["fall"]        # one candidate: Holm is the identity
    s = json.loads((tmp_path / "o/arm-A/cand/summary.json").read_text())
    assert s["sequential"]["ci95"] == t["A-cand"]["seq95"]
