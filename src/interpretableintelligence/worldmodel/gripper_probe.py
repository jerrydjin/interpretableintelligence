"""``ii gripper-probe``: does V-JEPA 2-AC's goal score read the gripper? (Hypothesis H1, sharpened by H5.)

H5 found the planner's arm search well shaped but its gripper command close to noise. This test
asks the question directly, with no segmentation: take moments in recorded DROID episodes where the
gripper opens or closes while the arm stays nearly still. The current frame is just before the
change, the goal is just after it. A planner that reads the gripper should

1. score the gripper command in the right direction (close when the goal shows it closed) lower
   than the opposite command, with the arm step fixed at the real one (``prefers_correct``), and
2. return a planned gripper command with the right sign (``plan_sign_correct``).

Verdict on ``prefers_correct``, fixed before any real run, with a 95% interval from resampling
whole episodes: ``reads_gripper`` when its lower end is above 0.5, ``gripper_blind`` when its upper
end is below ``BLIND_HI`` (0.65), ``mixed`` otherwise. The same events through an untrained
predictor (``--random-predictor``) give the floor. Offline, on recorded video; not a rollout finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel, euler_xyz_to_matrix, matrix_to_euler_xyz
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.plan_probe import GRIP_MAX, MAXNORM, _boot, cem

SCHEMA = "gripper-probe/v1"
MIN_GRIP_CHANGE = 0.5
MAX_ARM_M = 0.03
MAX_ARM_RAD = 0.15
MAX_GAP = 4
BLIND_HI = 0.65
N_BOOT = 2000


def find_events(states: np.ndarray) -> list[tuple[int, int]]:
    """``(before, after)`` index pairs where the gripper changes by at least ``MIN_GRIP_CHANGE`` within
    ``MAX_GAP`` steps while the arm moves under ``MAX_ARM_M`` and ``MAX_ARM_RAD``. Non-overlapping."""
    events = []
    t = 0
    n = len(states)
    while t < n - 1:
        found = None
        for gap in range(1, MAX_GAP + 1):
            u = t + gap
            if u >= n:
                break
            if abs(states[u, 6] - states[t, 6]) >= MIN_GRIP_CHANGE:
                found = u
                break
        if found is None:
            t += 1
            continue
        # Start from the last frame before the gripper starts to move.
        t = max(k for k in range(t, found) if abs(states[k, 6] - states[t, 6]) < 0.1)
        move = np.linalg.norm(states[found, :3] - states[t, :3])
        rel = euler_xyz_to_matrix(states[found, 3:6].astype(np.float64)) @ euler_xyz_to_matrix(states[t, 3:6].astype(np.float64)).T
        turn = np.linalg.norm(matrix_to_euler_xyz(rel))
        if move <= MAX_ARM_M and turn <= MAX_ARM_RAD:
            events.append((t, found))
            t = found + 1
        else:
            t += 1
    return events


def probe(model: ACWorldModel, trajectories: list[Trajectory], *, samples: int = 100, steps: int = 6, seed: int = 0, say=print) -> list[dict]:
    gen = torch.Generator().manual_seed(seed)
    records = []
    for i, tr in enumerate(trajectories):
        events = find_events(tr.states)
        if not events:
            continue
        z = model.encode_frames(tr.frames)
        for before, after in events:
            s0, s1 = tr.states[before], tr.states[after]
            delta = float(s1[6] - s0[6])
            sign = 1.0 if delta > 0 else -1.0
            state = torch.from_numpy(s0)
            arm = np.clip(s1[:3] - s0[:3], -MAXNORM, MAXNORM)
            cand = np.zeros((3, 7), dtype=np.float32)
            cand[:, :3] = arm
            cand[0, 6] = sign * GRIP_MAX  # the right direction
            cand[1, 6] = -sign * GRIP_MAX  # the wrong one
            cand[2, 6] = 0.0
            e = model.energy(z[before], state, z[after], torch.from_numpy(cand)).tolist()
            plan = cem(model, z[before], state, z[after], maxnorm=MAXNORM, grip_max=GRIP_MAX, samples=samples, steps=steps, generator=gen)
            records.append(
                {
                    "clip_id": tr.clip_id,
                    "episode": tr.episode,
                    "before": before,
                    "after": after,
                    "grip_before": float(s0[6]),
                    "grip_delta": delta,
                    "closing": delta > 0,
                    "e_correct": e[0],
                    "e_wrong": e[1],
                    "e_none": e[2],
                    "prefers_correct": e[0] < e[1],
                    "gap_rel": (e[1] - e[0]) / e[2] if e[2] else float("nan"),
                    "plan_grip": float(plan.action[6]),
                    "plan_sign_correct": bool(np.sign(plan.action[6]) == sign),
                }
            )
        say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: {len(events)} gripper events, {len(records)} so far")
    return records


def verdict(pref: dict) -> str:
    if not np.isfinite(pref["lo"]):
        return "insufficient_data"
    if pref["lo"] > 0.5:
        return "reads_gripper"
    if pref["hi"] < BLIND_HI:
        return "gripper_blind"
    return "mixed"


def summarize(records: list[dict], *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    def frac(key):
        return lambda rs: float(np.mean([r[key] for r in rs])) if rs else float("nan")

    def mean(key):
        return lambda rs: float(np.nanmean([r[key] for r in rs])) if rs else float("nan")

    stats = {
        "prefers_correct": _boot(records, frac("prefers_correct"), n_boot=n_boot, seed=seed),
        "plan_sign_correct": _boot(records, frac("plan_sign_correct"), n_boot=n_boot, seed=seed),
        "gap_rel": _boot(records, mean("gap_rel"), n_boot=n_boot, seed=seed),
    }
    split = {}
    for name, want in (("closing", True), ("opening", False)):
        rs = [r for r in records if r["closing"] == want]
        split[name] = {"n": len(rs), "prefers_correct": frac("prefers_correct")(rs), "plan_sign_correct": frac("plan_sign_correct")(rs)}
    return {
        "n_events": len(records),
        "n_episodes": len({r["episode"] for r in records}),
        "stats": stats,
        "by_direction": split,
        "verdict": verdict(stats["prefers_correct"]),
    }


def run_gripper_probe(model: ACWorldModel, trajectories: list[Trajectory], out_dir: Path, *, samples: int = 100, steps: int = 6,
                      seed: int = 0, n_boot: int = N_BOOT, say=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = probe(model, trajectories, samples=samples, steps=steps, seed=seed, say=say)
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "seed": seed,
        "n_boot": n_boot,
        "rules": {"min_grip_change": MIN_GRIP_CHANGE, "max_arm_m": MAX_ARM_M, "max_arm_rad": MAX_ARM_RAD, "max_gap": MAX_GAP,
                  "blind_hi": BLIND_HI},
        **summarize(records, n_boot=n_boot, seed=seed),
        "events": records,
    }
    (out_dir / "gripper_probe.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
