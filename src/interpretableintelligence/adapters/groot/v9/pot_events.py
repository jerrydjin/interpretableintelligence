"""Per-pot manipulation events + failure label for 'put both moka pots on the stove' (LIBERO KITCHEN_SCENE8), numpy only.

ONE implementation shared by
  * the online detector in physint_inproc_v9.capture (Colab; written next to physint_inproc_v9.py as /content/pot_events.py), and
  * the offline relabel /workspace/runs/groot-1028-v9-n100/tools/derive_pot0_fail.py (rule 'pot0-v1'),
so the two cannot drift. tests/test_pot_events.py replays the 100 recorded traces of groot-1028-v9-n100 and requires the
labels in its episodes.jsonl exactly.

Why: v8.derive_failure judges a 'drop' on whichever movable object is nearest the EEF (the 'active' object). In task 1028 the
policy always moves pot 1 (obj_names[1] = moka_pot_2) first and pot 0 (obj_names[0] = moka_pot_1) second; in 37/40 v9 'drop'
episodes the active detector fired on pot 1 being set down on the stove (normal). Its label is kept as fail_kind_active.

Inputs (one row per env step; step == arange(T)): obj_pos [T,K,3], obj_grasped [T,K] (contact flag, kept only as a marker: it
misses most pot-0 handle grasps), gripper_width [T], eef_pos [T,3]. Units: env policy steps (same as acts 'step').

Thresholds (from the 100-episode capture; README 'pot-0 failure relabel'):
  Z_LIFT 0.020  lift = pot z more than 2 cm above its step-0 z (per-episode max is <= 0.011 or >= 0.028; nothing between)
  D_NEAR 0.100  ||eef - pot|| at every successful lift onset is 0.051-0.095
  G_OPEN 0.078 / G_CLOSE 0.075 (>= CLOSE_MIN 3 steps)  open gripper reads 0.078-0.081; pot handle grasps hold at 0.014-0.075
  SEP 0.015 / SEP_CONF 0.030  release by separation: d0 leaves the carry distance by > SEP and exceeds carry + SEP_CONF within
                              10 steps (else it is a re-grip and the carry distance is reset); waived in the last 10 steps.
                              Release by gripper: after the lift gw went < G_CLOSE and then back to >= G_OPEN; onset = start of
                              that monotone rise. The earlier of the two is the release step.
  REGION (box)  stove placement region, xy: both pots' final xy over the 56 successes, padded 1 cm (STOVE_REGIONS below).
                A release is OFF the stove if the pot rests outside REGION before its next lift (or at the episode end).
  Z_REST_HI 1.012  touchdown = pot z <= 1.012 inside REGION after the lift; place_t = first touchdown that stays in REGION
                   to the end (successful episodes terminate at touchdown with the pot still held).
  V_STALL 0.002 / N_STALL 10   held-at-timeout: first 10-step run moving < 2 mm/step outside REGION after the lift.
"""
from __future__ import annotations

from typing import Any

import numpy as np

RULE = "pot0-v1"
Z_LIFT, D_NEAR, G_OPEN, G_CLOSE, CLOSE_MIN, SEP, PAD = 0.020, 0.100, 0.078, 0.075, 3, 0.015, 0.010
Z_REST_HI, V_STALL, N_STALL = 1.012, 0.002, 10
SEP_CONF = 0.030
POT_NAMES = ("moka_pot_1", "moka_pot_2")  # obj_names order in the LIBERO KITCHEN_SCENE8 moka task; pot 0 = moka_pot_1
# xy box (lo, hi) in world coordinates; derived by stove_region_from_successes() on groot-1028-v9-n100 (rounded to 0.1 mm)
STOVE_REGIONS = {"KITCHEN_SCENE8": ((-0.1292, -0.2819), (0.0349, -0.1047))}
FAIL_KINDS = ("lifted_not_placed", "never_grasped", "pot0_placed_other_failure", "empty_grasp")


def region_for_task(task_name: str | None):
    """(lo, hi) arrays for a task name, or None if the scene has no measured stove region."""
    for scene, (lo, hi) in STOVE_REGIONS.items():
        if task_name and scene in task_name and "moka_pots_on_the_stove" in task_name:
            return np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    return None


def stove_region_from_successes(final_xy: np.ndarray, pad: float = PAD):
    """final_xy [N,2] = final xy of every pot in successful episodes -> padded bounding box (lo, hi)."""
    F = np.asarray(final_xy, np.float64).reshape(-1, 2)
    return F.min(0) - pad, F.max(0) + pad


def segs(mask, min_len=1):
    m = np.r_[0, np.asarray(mask).astype(int), 0]; d = np.diff(m)
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]  # [a, b)


def _inreg(xy, lo, hi):
    return np.all((xy >= lo) & (xy <= hi), axis=-1)


def pot_events(obj_pos, obj_grasped, gripper_width, eef_pos, k: int, lo, hi) -> dict[str, Any]:
    """Events of pot k. Keys are unprefixed (lift_t, place_t, ...); see pot_event_fields()."""
    P_in = np.asarray(obj_pos)[:, k]; P = P_in.astype(np.float64); T = len(P)
    gw = np.asarray(gripper_width, np.float64).reshape(-1); eef = np.asarray(eef_pos, np.float64).reshape(T, 3)
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    z = P[:, 2] - P[0, 2]; d0 = np.linalg.norm(eef - P, axis=1)
    inreg = _inreg(P[:, :2], lo, hi)
    flag = np.flatnonzero(np.asarray(obj_grasped)[:, k] == 1); out = dict(flag_grasp_t=int(flag[0]) if len(flag) else None)
    near = np.flatnonzero(d0 < D_NEAR); out["first_near_t"] = int(near[0]) if len(near) else None
    closes = [a for a, b in segs(gw < G_CLOSE, CLOSE_MIN) if d0[a:b].min() < D_NEAR]  # close attempts near pot k
    out["close_attempts"] = [int(a) for a in closes]
    out["max_zrel"] = round(float(z.max()), 4)
    # ---- lift -> release events (a re-lift after a release starts a new event)
    lifts, releases = [], []
    t = 0
    while t < T:
        up = np.flatnonzero(z[t:] > Z_LIFT)
        if not len(up): break
        lt = t + int(up[0])
        carry = float(np.median(d0[lt:lt + 10]))
        ts, u0 = None, lt
        while True:  # separation must be confirmed (d0 > carry + SEP_CONF within 10 steps); otherwise it is a re-grip shift
            sep = np.flatnonzero(d0[u0:] > carry + SEP)
            if not len(sep): break
            c = u0 + int(sep[0])
            if d0[c:c + 10].max() > carry + SEP_CONF or c + 10 >= T: ts = c; break
            carry = float(np.median(d0[c:c + 10])); u0 = c + 1
        r_sep = None
        if ts is not None:
            back = np.flatnonzero(d0[lt:ts] <= carry + 0.005); r_sep = lt + int(back[-1]) if len(back) else lt
        # gripper-open release: after lift, gw had been < G_CLOSE and then rises to >= G_OPEN; onset = start of that monotone rise
        r_open = None; closed = np.flatnonzero(gw[lt:] < G_CLOSE)
        if len(closed):
            c0 = lt + int(closed[0]); op = np.flatnonzero(gw[c0:] >= G_OPEN)
            if len(op):
                r = c0 + int(op[0])
                while r - 1 > c0 and gw[r - 1] < gw[r] - 1e-4: r -= 1
                r_open = r
        cands = [x for x in (r_sep, r_open) if x is not None]
        if not cands:
            lifts.append(lt); releases.append(None); break
        rt = min(cands); ts = max(rt + 1, ts if ts is not None else rt + 1)
        carry = float(np.median(d0[lt:lt + 10]))
        lifts.append(lt); releases.append(rt)
        # next lift = re-grasped: d0 back to carry range AND pot rises > Z_LIFT above where it settled
        j = int(np.argmin(P[ts:ts + 15, 2])); rest_z = float(P[ts + j, 2]); nxt = None
        for u in range(ts + j + 1, T):
            if d0[u] <= carry + 0.005 and P[u, 2] - rest_z > Z_LIFT: nxt = u; break
        if nxt is None: break
        z = np.r_[z[:nxt], P[nxt:, 2] - rest_z]; t = nxt
    out["lift_t"] = lifts[0] if lifts else None
    out["lift_events"] = [int(x) for x in lifts]; out["release_events"] = [None if r is None else int(r) for r in releases]
    # OFF the stove = pot rests outside REGION before its next lift / at the end (released pots slide off or onto the stove edge)
    nxt_l = lifts[1:] + [T]
    off = [r is not None and not inreg[nl - 1] for r, nl in zip(releases, nxt_l)]
    rel_in = [r for r, o in zip(releases, off) if r is not None and not o]; rel_out = [r for r, o in zip(releases, off) if o]
    after = np.arange(T) > (lifts[0] if lifts else T)
    td = np.flatnonzero(inreg & (P[:, 2] <= Z_REST_HI) & after)
    kept = [t for t in td if inreg[t:].all()]
    out["place_t"] = int(kept[0]) if kept else None
    out["first_touchdown_in_region_t"] = int(td[0]) if len(td) else None
    out["release_in_region_t"] = int(rel_in[0]) if rel_in else None
    sp = np.r_[np.linalg.norm(np.diff(P, axis=0), axis=1), 0]
    st = [a for a, b in segs((sp < V_STALL) & ~inreg & after, N_STALL)]; out["stall_off_region_t"] = int(st[0]) if st else None
    out["n_drops"] = len(rel_out)
    out["first_drop_t"] = int(rel_out[0]) if rel_out else None
    out["drop_xyz"] = P_in[rel_out[0]].round(4).tolist() if rel_out else None
    out["final_in_region"] = bool(inreg[-1])
    out["held_at_end"] = bool(releases and releases[-1] is None)
    # grasp onset: last gripper-close onset (gw leaves fully open) within 40 steps before the first lift, eef near the pot
    g = None
    if lifts:
        lt = lifts[0]; a0 = max(0, lt - 40); op = np.flatnonzero(gw[a0:lt + 1] >= G_OPEN)
        if len(op) and a0 + int(op[-1]) + 1 <= lt:
            c = a0 + int(op[-1]) + 1
            if d0[c] < D_NEAR: g = c
    out["grasp_t"] = g
    return out


def all_pot_events(sim: dict, lo, hi) -> list[dict[str, Any]]:
    K = np.asarray(sim["obj_pos"]).shape[1]
    return [pot_events(sim["obj_pos"], sim["obj_grasped"], sim["gripper_width"], sim["eef_pos"], k, lo, hi) for k in range(K)]


def classify(success: bool, ev: list[dict], fail_kind_active: str | None, fail_t_active: int | None):
    """Episode label from pot events -> (fail_t, fail_kind, subkind, fail_pot). Rule pot0-v1: the failure is judged on pot 0
    (handled second); pot 1 (handled first) only decides empty_grasp (pot 1 never picked up -> keep the active detector's
    empty_grasp step) and the '_pot1_off_region' suffix. Successes -> (None, None, None, None)."""
    if success: return None, None, None, None
    o, o1 = ev[0], ev[1]
    if fail_kind_active == "empty_grasp":
        return fail_t_active, "empty_grasp", "pot1_empty_grasp_kept_v9", 1
    if o["lift_t"] is None:
        if o["close_attempts"]: return o["close_attempts"][0], "never_grasped", "close_attempt_failed", 0
        return o["first_near_t"], "never_grasped", "no_close_attempt_anchor_first_near", 0
    if o["first_drop_t"] is not None and (o["place_t"] is None or o["first_drop_t"] < o["place_t"]):
        return o["first_drop_t"], "lifted_not_placed", "released_off_stove", 0
    if o["place_t"] is not None:  # pot 0 reached the stove: not a pot-0 failure
        return None, "pot0_placed_other_failure", "pot0_on_stove_" + ("held_at_timeout" if o["held_at_end"] else "released") + \
            ("_pot1_off_region" if not o1["final_in_region"] else ""), None
    if o["stall_off_region_t"] is not None: return o["stall_off_region_t"], "lifted_not_placed", "held_until_timeout_stall_off_stove", 0
    return o["lift_t"], "lifted_not_placed", "held_until_timeout_anchor_lift", 0


def pot_event_fields(ev: list[dict], pots=(0,)) -> dict[str, Any]:
    """Flatten events to the manifest columns (pot0_lift_t, ...). Default pot 0 only, matching the relabel."""
    out = {}
    for k in pots:
        for key, v in ev[k].items():
            out[f"pot{k}_{key}"] = v
    if 0 in pots and len(ev) > 1:
        out["pot1_final_in_region"] = ev[1]["final_in_region"]
    return out


def detect(sim: dict, *, success: bool, task_name: str | None, names, fail_kind_active, fail_t_active,
           region=None) -> dict[str, Any] | None:
    """Online entry point. Returns {fail_t, fail_kind, fail_subkind, fail_pot, fail_rule, fields, pot1_events} or None when the
    task is not the two-moka-pot stove task, the pot names differ, or the trace has non-finite positions (caller keeps
    the active detector's label then)."""
    reg = region if region is not None else region_for_task(task_name)
    if reg is None or tuple(names) != POT_NAMES: return None
    if not np.isfinite(np.asarray(sim["obj_pos"], np.float64)).all(): return None
    ev = all_pot_events(sim, *reg)
    ft, fk, sub, pot = classify(success, ev, fail_kind_active, fail_t_active)
    return dict(fail_t=None if ft is None else int(ft), fail_kind=fk, fail_subkind=sub, fail_pot=pot,
                fail_rule=f"{RULE}: " + ("success" if success else sub), fields=pot_event_fields(ev), pot1_events=ev[1])
