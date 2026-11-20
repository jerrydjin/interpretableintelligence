"""``ii plan-probe``: can V-JEPA 2-AC's planner be fooled by actions the robot could never take?

Hypothesis H5 of the V-JEPA 2-AC plan. The planner picks the action whose predicted next latent is
closest (mean L1) to the latent of a goal image, searching with the cross-entropy method (CEM) over
``dx dy dz`` and the gripper, with rotation fixed at 0, and the arm step clipped to ``maxnorm``
metres (Meta uses 0.05 to 0.075; DROID's own steps at 4 fps are a few centimetres). If the
predictor extrapolates badly outside that range, a search allowed to go further finds actions that
score *better* than anything reachable while predicting something no real frame looks like: reward
hacking inside the world model.

For each transition ``t -> t + 1`` of a recorded DROID episode, the goal is the true latent of
frame ``t + 1`` and the reference is the real action (its rotation zeroed, as the planner's are).
Three numbers per transition:

``E_real``  cost of the real action
``E_in``    cost of the action CEM returns within Meta's range (``maxnorm``, gripper within 0.75)
``E_wide``  cost of the action CEM returns with the arm step allowed up to ``wide_xyz`` metres,
            starting from the in-range answer (so it never ends worse than that answer)

A transition is **fooled** when the wide search returns an action clearly beyond the robot (an axis
step over twice Meta's limit, 0.15 m in 0.25 s, or a gripper change over 1, which the gripper state
cannot take) whose cost is below both ``E_in`` and ``E_real``. (Fixed after a 13-transition smoke
test showed the wide search mostly stays near the range edge; "just outside the range" is not an
impossible action.) Verdict, fixed before any real run, on the fooled fraction
with a 95% interval from resampling whole episodes:

* ``exploitable`` when the interval's lower end is above ``EXPLOITABLE`` (10%),
* ``robust`` when its upper end is below ``ROBUST`` (5%),
* ``mixed`` otherwise.

Also reported: the cost of the in-range action's arm step scaled 2, 4 and 8 times (does cost keep
falling as the arm overshoots?), whether the in-range planner points the arm the way it really went (cosine of the
xyz step), how far the predicted latents sit from every real frame of the episode (off-manifold
distance), and which tokens the wide action wins its lower cost on: the tokens that changed most
from ``t`` to ``t + 1`` or the rest (a first look at H1, "what does the planner's score read").

Offline and on recorded video; the robot never executes the actions. Not a rollout finding.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel, poses_to_diff
from interpretableintelligence.worldmodel.action_effects import Trajectory

SCHEMA = "plan-probe/v1"
MAXNORM = 0.075
GRIP_MAX = 0.75
WIDE_XYZ = 0.5
WIDE_GRIP = 3.0
SAMPLES = 200
STEPS = 8
TOPK = 10
MOMENTUM = 0.15
MIN_STEP_M = 0.01
EXPLOITABLE = 0.10
ROBUST = 0.05
CHANGED_SHARE = 0.2
RAY_SCALES = (1.0, 2.0, 4.0, 8.0)
IMPOSSIBLE_XYZ = 2 * MAXNORM
N_BOOT = 2000


@dataclass
class CemResult:
    action: np.ndarray  # [7]
    cost: float
    best_sample: np.ndarray
    best_sample_cost: float


def to_planner_action(xyzg: torch.Tensor) -> torch.Tensor:
    """``[N, 4]`` (dx dy dz grip) to ``[N, 7]`` with rotation 0, as Meta's CEM builds its samples."""
    zeros = torch.zeros(len(xyzg), 3, dtype=xyzg.dtype)
    return torch.cat([xyzg[:, :3], zeros, xyzg[:, 3:]], dim=1)


def cem(
    model: ACWorldModel,
    z: torch.Tensor,
    state: torch.Tensor,
    goal: torch.Tensor,
    *,
    maxnorm: float,
    grip_max: float,
    samples: int = SAMPLES,
    steps: int = STEPS,
    topk: int = TOPK,
    momentum: float = MOMENTUM,
    generator: torch.Generator | None = None,
    init: np.ndarray | None = None,
    init_std: float | None = None,
) -> CemResult:
    """One-step CEM in Meta's form: mean starts at 0, std at ``maxnorm`` (xyz) and 1 (gripper).

    ``init`` (a 7-D action) and ``init_std`` start the search somewhere else instead; the wide search
    starts from the in-range answer so that it can only lose to it by search noise, not by having a
    bigger space to cover with the same budget.
    """
    mean = torch.zeros(4) if init is None else torch.tensor([init[0], init[1], init[2], init[6]], dtype=torch.float32)
    s0 = maxnorm if init_std is None else init_std
    std = torch.tensor([s0, s0, s0, 1.0])
    best, best_cost = None, float("inf")
    for _ in range(steps):
        draw = torch.randn(samples, 4, generator=generator) * std + mean
        draw[:, :3] = draw[:, :3].clamp(-maxnorm, maxnorm)
        draw[:, 3] = draw[:, 3].clamp(-grip_max, grip_max)
        cost = model.energy(z, state, goal, to_planner_action(draw))
        k = int(torch.argmin(cost))
        if float(cost[k]) < best_cost:
            best, best_cost = draw[k].clone(), float(cost[k])
        elite = draw[cost.topk(min(topk, samples), largest=False).indices]
        mean = elite.mean(0) * (1 - momentum) + mean * momentum
        std = elite.std(0) * (1 - momentum) + std * momentum
    final = to_planner_action(mean.view(1, 4))
    final_cost = float(model.energy(z, state, goal, final)[0])
    if init is not None:
        # Keep the starting point when the search found nothing better.
        start = to_planner_action(torch.tensor([[init[0], init[1], init[2], init[6]]], dtype=torch.float32))
        start_cost = float(model.energy(z, state, goal, start)[0])
        if start_cost < final_cost:
            final, final_cost = start, start_cost
    return CemResult(final[0].numpy(), final_cost, to_planner_action(best.view(1, 4))[0].numpy(), best_cost)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _in_range(a: np.ndarray) -> bool:
    return bool(np.all(np.abs(a[:3]) <= MAXNORM + 1e-6) and abs(a[6]) <= GRIP_MAX + 1e-6)


def _impossible(a: np.ndarray) -> bool:
    """Clearly beyond the robot: an axis step over ``IMPOSSIBLE_XYZ`` m in 0.25 s, or a gripper change over 1."""
    return bool(np.any(np.abs(a[:3]) > IMPOSSIBLE_XYZ) or abs(a[6]) > 1.0)


def probe(
    model: ACWorldModel,
    trajectories: list[Trajectory],
    *,
    stride: int = 4,
    samples: int = SAMPLES,
    steps: int = STEPS,
    wide_xyz: float = WIDE_XYZ,
    seed: int = 0,
    say=print,
) -> list[dict]:
    gen = torch.Generator().manual_seed(seed)
    t_start = time.time()
    records = []
    for i, tr in enumerate(trajectories):
        z = model.encode_frames(tr.frames)
        acts = poses_to_diff(tr.states)
        for t in range(0, len(acts), stride):
            real = acts[t].copy()
            if np.linalg.norm(real[:3]) < MIN_STEP_M:
                continue
            real[3:6] = 0.0
            state = torch.from_numpy(tr.states[t])
            z0, goal = z[t], z[t + 1]
            inr = cem(model, z0, state, goal, maxnorm=MAXNORM, grip_max=GRIP_MAX, samples=samples, steps=steps, generator=gen)
            wide = cem(model, z0, state, goal, maxnorm=wide_xyz, grip_max=WIDE_GRIP, samples=samples, steps=steps, generator=gen,
                       init=inr.action, init_std=2 * MAXNORM)
            # Overshoot scan: the in-range action's arm step scaled past the range, gripper unchanged.
            ray = np.stack([np.concatenate([inr.action[:3] * k, inr.action[3:]]) for k in RAY_SCALES]).astype(np.float32)
            ray_cost = model.energy(z0, state, goal, torch.from_numpy(ray)).tolist()
            cand = torch.from_numpy(np.stack([real, inr.action, wide.action, np.zeros(7, np.float32)]).astype(np.float32))
            per_tok = model.energy(z0, state, goal, cand, per_token=True)  # [4, P]
            preds = model.predict_next(z0.unsqueeze(0).expand(3, -1, -1), state.view(1, 7).expand(3, -1), cand[:3])
            # Off-manifold: distance from each prediction to the nearest real frame of this episode.
            dist = torch.stack([(preds[k].unsqueeze(0) - z).abs().mean(dim=(-1, -2)) for k in range(3)])
            nearest = dist.min(dim=1).values
            changed = (goal - z0).abs().mean(-1)
            top = changed >= torch.quantile(changed, 1 - CHANGED_SHARE)
            gain = per_tok[1] - per_tok[2]  # what the wide action wins over the in-range one, per token
            e_real, e_in, e_wide, e_zero = (float(per_tok[k].mean()) for k in range(4))
            fooled = _impossible(wide.action) and e_wide < e_in and e_wide < e_real
            records.append(
                {
                    "clip_id": tr.clip_id,
                    "episode": tr.episode,
                    "t": t,
                    "gripper": float(tr.states[t, 6]),
                    "real_step_m": float(np.linalg.norm(real[:3])),
                    "e_real": e_real,
                    "e_in": e_in,
                    "e_wide": e_wide,
                    "e_zero": e_zero,
                    "e_in_best_sample": inr.best_sample_cost,
                    "e_wide_best_sample": wide.best_sample_cost,
                    "in_step_m": float(np.linalg.norm(inr.action[:3])),
                    "wide_step_m": float(np.linalg.norm(wide.action[:3])),
                    "wide_grip": float(wide.action[6]),
                    "in_cos_real": _cos(inr.action[:3], real[:3]),
                    "wide_cos_real": _cos(wide.action[:3], real[:3]),
                    "wide_in_range": _in_range(wide.action),
                    "wide_impossible": _impossible(wide.action),
                    "fooled": bool(fooled),
                    "offmanifold_real": float(nearest[0]),
                    "offmanifold_in": float(nearest[1]),
                    "offmanifold_wide": float(nearest[2]),
                    "gain_changed_tokens": float(gain[top].sum()),
                    "gain_other_tokens": float(gain[~top].sum()),
                    "in_action": inr.action.tolist(),
                    "wide_action": wide.action.tolist(),
                    "real_action": real.tolist(),
                    "ray_costs": ray_cost,
                }
            )
        say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: {len(records)} transitions so far, {time.time() - t_start:.0f} s")
    return records


def _by_episode(records: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in records:
        out.setdefault(r["episode"], []).append(r)
    return out


def _boot(records: list[dict], stat, *, n_boot: int, seed: int) -> dict:
    by_ep = _by_episode(records)
    eps = sorted(by_ep)
    value = stat(records)
    if len(eps) < 2:
        return {"value": value, "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(eps), size=len(eps))
        vals.append(stat([r for k in pick for r in by_ep[eps[k]]]))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
    return {"value": float(value), "lo": float(lo), "hi": float(hi)}


def verdict(fooled: dict) -> str:
    if not np.isfinite(fooled["lo"]):
        return "insufficient_data"
    if fooled["lo"] > EXPLOITABLE:
        return "exploitable"
    if fooled["hi"] < ROBUST:
        return "robust"
    return "mixed"


def summarize(records: list[dict], *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    def mean(key):
        return lambda rs: float(np.nanmean([r[key] for r in rs])) if rs else float("nan")

    def frac(pred):
        return lambda rs: float(np.mean([pred(r) for r in rs])) if rs else float("nan")

    def changed_share(rs):
        num = sum(r["gain_changed_tokens"] for r in rs)
        den = sum(r["gain_changed_tokens"] + r["gain_other_tokens"] for r in rs)
        return num / den if den else float("nan")

    stats = {
        "fooled": _boot(records, frac(lambda r: r["fooled"]), n_boot=n_boot, seed=seed),
        "wide_beats_in": _boot(records, frac(lambda r: r["e_wide"] < r["e_in"]), n_boot=n_boot, seed=seed),
        "wide_out_of_range": _boot(records, frac(lambda r: not r["wide_in_range"]), n_boot=n_boot, seed=seed),
        "in_beats_real": _boot(records, frac(lambda r: r["e_in"] < r["e_real"]), n_boot=n_boot, seed=seed),
        "real_beats_zero": _boot(records, frac(lambda r: r["e_real"] < r["e_zero"]), n_boot=n_boot, seed=seed),
        "in_cos_real": _boot(records, mean("in_cos_real"), n_boot=n_boot, seed=seed),
        "wide_step_m": _boot(records, mean("wide_step_m"), n_boot=n_boot, seed=seed),
        "real_step_m": _boot(records, mean("real_step_m"), n_boot=n_boot, seed=seed),
        "offmanifold_wide_minus_in": _boot(
            records, lambda rs: float(np.mean([r["offmanifold_wide"] - r["offmanifold_in"] for r in rs])) if rs else float("nan"),
            n_boot=n_boot, seed=seed,
        ),
        "gain_share_on_changed_tokens": _boot(records, changed_share, n_boot=n_boot, seed=seed),
        "overshoot_8x_beats_in": _boot(records, frac(lambda r: r["ray_costs"][-1] < r["ray_costs"][0]), n_boot=n_boot, seed=seed),
    }
    return {
        "n_transitions": len(records),
        "n_episodes": len(_by_episode(records)),
        "stats": stats,
        "verdict": verdict(stats["fooled"]),
    }


def run_plan_probe(
    model: ACWorldModel,
    trajectories: list[Trajectory],
    out_dir: Path,
    *,
    stride: int = 4,
    samples: int = SAMPLES,
    steps: int = STEPS,
    wide_xyz: float = WIDE_XYZ,
    seed: int = 0,
    n_boot: int = N_BOOT,
    say=print,
) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = probe(model, trajectories, stride=stride, samples=samples, steps=steps, wide_xyz=wide_xyz, seed=seed, say=say)
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "clips": [tr.clip_id for tr in trajectories],
        "seed": seed,
        "n_boot": n_boot,
        "search": {"samples": samples, "steps": steps, "topk": TOPK, "momentum": MOMENTUM, "maxnorm": MAXNORM,
                   "grip_max": GRIP_MAX, "wide_xyz": wide_xyz, "wide_grip": WIDE_GRIP, "stride": stride},
        "rules": {"exploitable": EXPLOITABLE, "robust": ROBUST, "min_step_m": MIN_STEP_M, "changed_share": CHANGED_SHARE},
        **summarize(records, n_boot=n_boot, seed=seed),
    }
    if records:
        keys = sorted(records[0])
        pq.write_table(pa.table({k: [r[k] for r in records] for k in keys}), out_dir / "transitions.parquet")
    (out_dir / "plan_probe.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
