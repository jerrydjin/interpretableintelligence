"""Regression tests for the per-pot failure detector (v9/pot_events.py, rule pot0-v1).

Replays the 100 recorded traces of groot-1028-v9-n100 (sim/ep-*.npz) through the SAME code the Colab capture uses and
requires the offline relabel in that run's episodes.jsonl exactly. Skips if the run folder is not on this machine.
"""
from box_paths import RUNS  # noqa: E402 (box roots; see box_paths.py)
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "v9"), str(ROOT / "v8")]
import pot_events as pe  # noqa: E402

RUN = Path(f"{RUNS}/groot-1028-v9-n100")
pytestmark = pytest.mark.skipif(not (RUN / "episodes.jsonl").is_file(), reason="groot-1028-v9-n100 run folder not present")


@pytest.fixture(scope="module")
def run():
    man = json.loads((RUN / "manifest.json").read_text())
    eps = [json.loads(l) for l in (RUN / "episodes.jsonl").read_text().splitlines() if l.strip()]
    sims = {e["episode_idx"]: dict(np.load(RUN / "sim" / f"ep-{e['episode_idx']:05d}.npz")) for e in eps}
    return man, eps, sims


def _detect(e, sim, task_name, region=None):
    return pe.detect(sim, success=e["success"], task_name=task_name, names=[str(n) for n in sim["obj_names"]],
                     fail_kind_active=e["fail_kind_v9_active"], fail_t_active=e["fail_t_v9_active"], region=region)


def test_reproduces_relabel_exactly(run):
    man, eps, sims = run
    assert len(eps) == 100
    for e in eps:
        r = _detect(e, sims[e["episode_idx"]], man["task_name"])  # online path: built-in STOVE_REGIONS box
        assert r is not None
        got = (r["fail_kind"], r["fail_t"], r["fail_subkind"], r["fail_rule"])
        assert got == (e["fail_kind"], e["fail_t"], e["fail_subkind"], e["fail_rule"]), (e["episode_idx"], got)
        assert {k: e[k] for k in r["fields"]} == r["fields"], e["episode_idx"]


def test_counts(run):
    _, eps, _ = run
    kinds = [e["fail_kind"] for e in eps if not e["success"]]
    assert sum(e["success"] for e in eps) == 56
    assert {k: kinds.count(k) for k in set(kinds)} == {"lifted_not_placed": 22, "never_grasped": 12,
                                                       "pot0_placed_other_failure": 6, "empty_grasp": 4}


def test_region_constant_matches_successes(run):
    man, eps, sims = run
    lo, hi = pe.stove_region_from_successes([sims[e["episode_idx"]]["obj_pos"][-1, :, :2] for e in eps if e["success"]])
    clo, chi = pe.region_for_task(man["task_name"])
    assert np.abs(lo - clo).max() < 1e-4 and np.abs(hi - chi).max() < 1e-4
    for e in eps:  # exact data-derived box gives the same labels as the rounded constant
        r = _detect(e, sims[e["episode_idx"]], man["task_name"], region=(lo, hi))
        assert (r["fail_kind"], r["fail_t"]) == (e["fail_kind"], e["fail_t"])


def test_lifted_not_placed_never_on_pot1_placement(run):
    man, eps, sims = run
    reg = pe.region_for_task(man["task_name"])
    n = 0
    for e in eps:
        if e["fail_kind"] != "lifted_not_placed":
            continue
        n += 1
        sim = sims[e["episode_idx"]]
        p1 = pe.pot_events(sim["obj_pos"], sim["obj_grasped"], sim["gripper_width"], sim["eef_pos"], 1, *reg)
        t = e["fail_t"]
        pot1_marks = [r for r in p1["release_events"] if r is not None] + \
                     [x for x in (p1["first_touchdown_in_region_t"], p1["place_t"]) if x is not None]
        assert pot1_marks, e["episode_idx"]
        assert all(abs(t - m) > 16 for m in pot1_marks), (e["episode_idx"], t, pot1_marks)  # > one action chunk away
        assert t > max(r for r in p1["release_events"] if r is not None), e["episode_idx"]   # after pot 1 is let go
        # the event is on pot 0: pot 0 has lifted by then, and the old active label (pot 1) is not reproduced
        assert e["pot0_lift_t"] is not None and t >= e["pot0_lift_t"]
        assert t != e["fail_t_v9_active"]
    assert n == 22


def test_v9_label_episode_keeps_active_label(run):
    """physint_inproc_v9.label_episode on the recorded traces: new label == relabel, active label == the v9.0 capture's."""
    v9 = pytest.importorskip("physint_inproc_v9")
    man, eps, sims = run
    for e in eps:
        ep = SimpleNamespace(success=e["success"], sim=sims[e["episode_idx"]])
        fr = v9.label_episode(ep, man["task_name"])
        assert (fr["fail_kind"], fr["fail_t"]) == (e["fail_kind"], e["fail_t"]), e["episode_idx"]
        assert (fr["extra"]["fail_kind_active"], fr["extra"]["fail_t_active"]) == (e["fail_kind_v9_active"], e["fail_t_v9_active"])
        assert fr["details"] == e["failure_details"]
        assert fr["extra"]["fail_rule"] == e["fail_rule"]


def test_unknown_task_falls_back_to_active(run):
    v9 = pytest.importorskip("physint_inproc_v9")
    _, eps, sims = run
    e = next(x for x in eps if x["fail_kind_v9_active"] == "drop")
    fr = v9.label_episode(SimpleNamespace(success=False, sim=sims[e["episode_idx"]]), "LIVING_ROOM_SCENE1_some_other_task")
    assert (fr["fail_kind"], fr["fail_t"]) == (e["fail_kind_v9_active"], e["fail_t_v9_active"])
    assert fr["extra"]["fail_kind_active"] == fr["fail_kind"] and fr["extra"]["fail_pot"] is None
