"""Re-derive the failure step from pot 0 (obj_names[0] == 'moka_pot_1', the SECOND pot the policy handles) for groot-1028-v9-n100.

Why: the v9 capture's detector anchored fail_t on the *active* object; in 37/40 'drop' episodes that is pot 1 being set down on the
stove (normal behaviour). Task 1028 is 'put both moka pots on the stove': in every episode pot 1 (index 1, moka_pot_2) is handled first
(~steps 100-200) and pot 0 (index 0, moka_pot_1) second (~250-500).

Signals (sim/ep-*.npz, one row per env step; step == arange):
  obj_pos [T,2,3], obj_grasped [T,2] (contact flag), gripper_width [T], eef_pos [T,3].
  obj_grasped is NOT reliable for pot 0: 10 successes carry pot 0 to the stove with obj_grasped[:,0] never true (handle held with the
  fingers barely closed, gw ~0.066), and in others the flag first fires up to 83 steps after the pot is already in the air.
  So events are derived from pot-0 motion + eef distance + gripper width; the contact flag is recorded alongside (pot0_flag_grasp_t).

Thresholds (chosen from the distributions, see README 'pot-0 failure relabel'):
  Z_LIFT  = 0.020 m  pot-0 z above its step-0 z. Per-episode max z_rel is bimodal: <= 0.011 (pushes/tilts, 16 eps) vs >= 0.028 (84 eps).
  D_NEAR  = 0.100 m  ||eef - pot0||. d0 at successful lift onsets is 0.051-0.095 (all 84 lifts).
  G_OPEN  = 0.078 m  gripper fully open is 0.078-0.081.
  G_CLOSE = 0.075 m  a close attempt = gw < G_CLOSE for >= 3 consecutive steps (pot-0 handle grasps hold at gw 0.014-0.075).
  SEP     = 0.015 m  release = d0 rises more than SEP above the carry distance (median d0 over the 10 steps after lift onset);
                     release step = last step before that with d0 <= carry + 0.005 (where separation starts).
  release = earlier of (a) separation as above and (b) gripper opening: after lift gw was < G_CLOSE and then rises to >= G_OPEN,
            release step = start of that monotone rise (ep 6: pot parked on the stove edge while held from ~344, gripper opens at 420).
            Separation confirmation is waived if the episode ends within 10 steps (ep 93 lets go at ~506, T=520).
  SEP_CONF = 0.030 m the separation is a release only if d0 exceeds carry + 0.030 within 10 steps; a smaller step change that
                     plateaus is a re-grip shift (ep 16 success: pot slips 2 cm in the hand at step 302, d0 0.062 -> 0.081, still carried)
                     and the carry distance is reset to the new plateau.
  REGION  stove placement region = xy bounding box of BOTH pots' final positions over the 56 successes, padded 1.0 cm:
          x in [-0.129, 0.035], y in [-0.282, -0.105]. (Edge evidence: ep 93 pot 0 released at y=-0.100 slides off the stove;
          ep 68 pot 0 released at y=-0.109 stays on it at z 1.008.)
  Z_REST_HI = 1.012 m  touchdown on the stove = pot-0 absolute z <= 1.012 (success final pot-0 z 0.978-1.009) inside REGION, after lift;
                       pot0_place_t = first touchdown after which pot 0 stays in REGION to the episode end.
                       Successful episodes terminate at touchdown with the pot still in the gripper, so placement != release.
  V_STALL/N_STALL      held-until-timeout episodes (no release): fail_t = first step after lift where pot 0 is outside REGION and moves
                       < 2 mm/step for 10 consecutive steps (set down / stuck off the stove while still held); else lift step.

Implementation: /workspace/groot-colab/v9/pot_events.py (shared with the online detector in physint_inproc_v9.capture, so the
offline relabel and the Colab capture cannot drift; tests/test_pot_events.py replays these 100 traces against episodes.jsonl).
This script only loads the run, derives the stove region from this run's successes (and checks it against
pot_events.STOVE_REGIONS), applies pot_events, and writes the manifest / episodes.jsonl.
"""
import json, sys, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/groot-colab/v9")
import pot_events as pe  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/runs/groot-1028-v9-n100")
RULE = pe.RULE


def load():
    man = json.loads((RUN / "manifest.json").read_text())
    sims = {e["episode_idx"]: dict(np.load(RUN / "sim" / f"ep-{e['episode_idx']:05d}.npz")) for e in man["episodes"]}
    return man, sims


def main():
    man, sims = load()
    for name in ("episodes.jsonl", "manifest.json"):
        bak = RUN / (name + ".v9active.bak")
        if not bak.exists(): shutil.copy(RUN / name, bak)
    base = json.loads((RUN / "manifest.json.v9active.bak").read_text())  # always derive from the pristine v9 rows (idempotent)
    lo, hi = pe.stove_region_from_successes([sims[e["episode_idx"]]["obj_pos"][-1, :, :2] for e in base["episodes"] if e["success"]])
    clo, chi = pe.region_for_task(base["task_name"])
    assert np.abs(lo - clo).max() < 1e-4 and np.abs(hi - chi).max() < 1e-4, (lo, hi, clo, chi)
    rows = []
    for e in base["episodes"]:
        e = dict(e); e["fail_t_v9_active"] = e["fail_t"]; e["fail_kind_v9_active"] = e["fail_kind"]
        d = sims[e["episode_idx"]]
        assert tuple(d["obj_names"]) == pe.POT_NAMES and (d["step"] == np.arange(len(d["step"]))).all()
        ev = pe.all_pot_events(d, lo, hi); e.update(pe.pot_event_fields(ev))
        ft, fk, sub, _pot = pe.classify(e["success"], ev, e["fail_kind_v9_active"], e["fail_t_v9_active"])
        e["fail_t"], e["fail_kind"], e["fail_subkind"] = ft, fk, sub
        e["fail_rule"] = f"{RULE}: " + ("success" if e["success"] else sub)
        rows.append(e)
    man = dict(base); man["episodes"] = rows
    man["fail_relabel"] = dict(rule=RULE, script="tools/derive_pot0_fail.py", pot0="obj_names[0] = moka_pot_1 (handled second)",
                               thresholds=dict(Z_LIFT=pe.Z_LIFT, D_NEAR=pe.D_NEAR, G_OPEN=pe.G_OPEN, G_CLOSE=pe.G_CLOSE, CLOSE_MIN_STEPS=pe.CLOSE_MIN, SEP=pe.SEP, REGION_PAD=pe.PAD),
                               stove_region_xy=dict(lo=lo.round(4).tolist(), hi=hi.round(4).tolist(), source="final xy of both pots over the 56 successes"),
                               units="env policy steps, same as acts 'step' (chunk starts every 16)",
                               originals="fail_t_v9_active / fail_kind_v9_active; backups *.v9active.bak")
    (RUN / "manifest.json").write_text(json.dumps(man, indent=1))
    (RUN / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in rows))
    return man


if __name__ == "__main__":
    m = main()
    for e in m["episodes"]:
        print(e["episode_idx"], "S" if e["success"] else "F", e["fail_kind_v9_active"], e["fail_t_v9_active"], "->", e["fail_kind"], e["fail_t"], e["fail_subkind"],
              "| grasp", e["pot0_grasp_t"], "flag", e["pot0_flag_grasp_t"], "lift", e["pot0_lift_events"], "rel", e["pot0_release_events"], "place", e["pot0_place_t"], "drops", e["pot0_n_drops"])
