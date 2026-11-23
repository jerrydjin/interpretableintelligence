"""``ii plan-stall``: does V-JEPA 2-AC's "world froze" bias make its planner rather stand still?

``ii impossible-events`` found the predictor's next frame always closer to a frozen world than to
the real one. This command asks whether that hurts planning, and whether a goal score that looks at
the moving parts of the scene helps.

For every DROID transition where the arm moves at least ``MIN_STEP_M``, the planner gets the current
latent ``z_t`` and a goal latent ``z_{t+h}`` (``h`` in ``HORIZONS``: the next frame, and one second
ahead). It chooses from a fixed menu of one-step actions, all with the real arm step's length and
rotation 0, as Meta's planner builds them:

* ``real``     the real arm step (and gripper change)
* ``reverse``  the real step backwards
* ``zero``     doing nothing
* 26 more     the real step's length along every direction of the 3 x 3 x 3 grid

A menu instead of CEM: no search noise, and every score below ranks exactly the same predictions.
Three goal scores rank the menu, all mean L1 between prediction and goal over a set of tokens:

``mean``     Meta's score: all tokens
``moving``   only the ``MOVING_SHARE`` (20%) of tokens that differ most between ``z_t`` and the goal
             (the planner knows both, so this needs nothing it does not have)
``random``   the same number of tokens, drawn at random: the control for ``moving``

Per transition and score: ``stall`` (the planner picks ``zero``), ``zero_beats_real``,
``reverse_beats_real``, and ``cos``, the cosine between the chosen and the real arm step (0 when it
stalls). Verdicts, fixed before any real run, with 95% intervals from resampling whole episodes,
per horizon:

**Hurt** (Meta's ``mean`` score), on the stall share:

* ``hurts`` when the lower end is above ``HURTS`` (0.20): on more than 1 in 5 moving moments the
  planner would rather do nothing than any move of the right size
* ``harmless`` when the upper end is below ``HARMLESS`` (0.10)
* ``mixed`` otherwise

**Fix** (``moving`` against ``mean``, paired on the same transitions):

* ``fixes`` when ``moving`` raises ``cos`` over ``mean`` and over ``random`` (both lower ends above
  0) and does not raise the stall share or ``reverse_beats_real`` (upper ends of ``moving`` minus
  ``mean`` at most ``SLACK``, 0.02)
* ``no_fix`` when the ``cos`` gain over ``mean`` has its upper end at or below 0
* ``partial`` otherwise

Also reported: where the ``mean`` score's preference comes from. ``E(zero) - E(real)`` is split into
the moving tokens and the rest: real winning on the moving tokens but losing on the still ones means
the score is diluted by the static scene (what ``moving`` fixes); losing on the moving tokens too
means the prediction itself does not move (what no reweighting fixes).

``--random-predictor`` keeps the trained encoder and swaps in an untrained predictor: the floor.
Offline, on recorded video; the robot never executes anything. Not a rollout finding.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel, poses_to_diff
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.plan_probe import MAXNORM, _boot

SCHEMA = "plan-stall/v1"
HORIZONS = (1, 4)
SCORES = ("mean", "moving", "random")
MOVING_SHARE = 0.2
MIN_STEP_M = 0.01
HURTS = 0.20
HARMLESS = 0.10
SLACK = 0.02
N_BOOT = 2000
REAL, REVERSE, ZERO = 0, 1, 2


def directions() -> np.ndarray:
    """The 26 unit vectors of the 3 x 3 x 3 grid around the origin."""
    d = np.array([v for v in itertools.product((-1, 0, 1), repeat=3) if any(v)], dtype=np.float32)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def menu(real: np.ndarray) -> np.ndarray:
    """``[29, 7]`` candidate actions for one real 7-D action: real, reverse, zero, then 26 directions.

    Rotation is 0 everywhere (Meta's planner never rotates); every arm step has the real step's
    length, each axis clipped to ``MAXNORM``; every move keeps the real gripper change.
    """
    real = np.asarray(real, dtype=np.float32).copy()
    real[3:6] = 0.0
    real[:3] = np.clip(real[:3], -MAXNORM, MAXNORM)
    step = float(np.linalg.norm(real[:3]))
    out = np.zeros((3 + 26, 7), dtype=np.float32)
    out[REAL] = real
    out[REVERSE, :3] = -real[:3]
    out[REVERSE, 6] = real[6]
    out[3:, :3] = np.clip(directions() * step, -MAXNORM, MAXNORM)
    out[3:, 6] = real[6]
    return out


def token_masks(z: torch.Tensor, goal: torch.Tensor, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """Boolean ``[P]`` masks for each score: all, the most-changed ``MOVING_SHARE``, a random set of the same size."""
    p = z.shape[0]
    k = max(1, int(round(MOVING_SHARE * p)))
    change = (goal - z).abs().mean(-1)
    moving = torch.zeros(p, dtype=torch.bool)
    moving[change.topk(k).indices] = True
    rand = torch.zeros(p, dtype=torch.bool)
    rand[torch.from_numpy(rng.choice(p, size=k, replace=False))] = True
    return {"mean": torch.ones(p, dtype=torch.bool), "moving": moving, "random": rand}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def probe(model: ACWorldModel, trajectories: list[Trajectory], *, seed: int = 0, say=print) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    for i, tr in enumerate(trajectories):
        acts = poses_to_diff(tr.states)
        steps = [t for t in range(len(acts)) if np.linalg.norm(acts[t, :3]) >= MIN_STEP_M and t + min(HORIZONS) < len(tr.frames)]
        if not steps:
            say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: no moving transitions")
            continue
        z = model.encode_frames(tr.frames)
        for t in steps:
            cand = menu(acts[t])
            n = len(cand)
            preds = model.predict_next(z[t].unsqueeze(0).expand(n, -1, -1), torch.from_numpy(tr.states[t]).view(1, 7).expand(n, -1),
                                       torch.from_numpy(cand))
            for h in HORIZONS:
                if t + h >= len(z):
                    continue
                goal = z[t + h]
                per_tok = (preds - goal.unsqueeze(0)).abs().mean(-1)  # [n, P]
                masks = token_masks(z[t], goal, rng)
                rec = {"clip_id": tr.clip_id, "episode": tr.episode, "t": int(t), "horizon": h,
                       "real_step_m": float(np.linalg.norm(cand[REAL, :3]))}
                for name, m in masks.items():
                    cost = per_tok[:, m].mean(-1).numpy()
                    pick = int(np.argmin(cost))
                    rec[f"{name}_stall"] = pick == ZERO
                    rec[f"{name}_zero_beats_real"] = bool(cost[ZERO] < cost[REAL])
                    rec[f"{name}_reverse_beats_real"] = bool(cost[REVERSE] < cost[REAL])
                    rec[f"{name}_cos"] = _cos(cand[pick, :3], cand[REAL, :3])
                    rec[f"{name}_margin"] = float((cost[ZERO] - cost[REAL]) / cost[REAL])
                # Where Meta's score's zero-vs-real preference comes from: moving tokens vs the rest.
                d = (per_tok[ZERO] - per_tok[REAL]).numpy() / per_tok.shape[1]
                mv = masks["moving"].numpy()
                rec["split_moving"] = float(d[mv].sum())
                rec["split_still"] = float(d[~mv].sum())
                records.append(rec)
        say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: {len(steps)} moving transitions, {len(records)} records so far")
    return records


def hurt_verdict(stall: dict) -> str:
    if not np.isfinite(stall["lo"]):
        return "insufficient_data"
    if stall["lo"] > HURTS:
        return "hurts"
    if stall["hi"] < HARMLESS:
        return "harmless"
    return "mixed"


def fix_verdict(cos_vs_mean: dict, cos_vs_random: dict, stall_rise: dict, reverse_rise: dict) -> str:
    if not np.isfinite(cos_vs_mean["lo"]):
        return "insufficient_data"
    if cos_vs_mean["hi"] <= 0:
        return "no_fix"
    if cos_vs_mean["lo"] > 0 and cos_vs_random["lo"] > 0 and stall_rise["hi"] <= SLACK and reverse_rise["hi"] <= SLACK:
        return "fixes"
    return "partial"


def summarize(records: list[dict], *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    def mean(key):
        return lambda rs: float(np.mean([r[key] for r in rs])) if rs else float("nan")

    def diff(a, b):
        return lambda rs: float(np.mean([r[a] - r[b] for r in rs])) if rs else float("nan")

    out = {}
    for h in HORIZONS:
        rs = [r for r in records if r["horizon"] == h]
        b = lambda f: _boot(rs, f, n_boot=n_boot, seed=seed)  # noqa: E731
        scores = {
            s: {k: b(mean(f"{s}_{k}")) for k in ("stall", "zero_beats_real", "reverse_beats_real", "cos", "margin")} for s in SCORES
        }
        paired = {
            "cos_moving_minus_mean": b(diff("moving_cos", "mean_cos")),
            "cos_moving_minus_random": b(diff("moving_cos", "random_cos")),
            "stall_moving_minus_mean": b(diff("moving_stall", "mean_stall")),
            "reverse_moving_minus_mean": b(diff("moving_reverse_beats_real", "mean_reverse_beats_real")),
        }
        split = {"moving_tokens": b(mean("split_moving")), "still_tokens": b(mean("split_still"))}
        out[str(h)] = {
            "n_transitions": len(rs),
            "n_episodes": len({r["episode"] for r in rs}),
            "scores": scores,
            "paired": paired,
            "zero_minus_real_split": split,
            "hurt": hurt_verdict(scores["mean"]["stall"]),
            "fix": fix_verdict(paired["cos_moving_minus_mean"], paired["cos_moving_minus_random"],
                               paired["stall_moving_minus_mean"], paired["reverse_moving_minus_mean"]),
        }
    return {"horizons": out, "verdicts": {h: {"hurt": v["hurt"], "fix": v["fix"]} for h, v in out.items()}}


def run_plan_stall(model: ACWorldModel, trajectories: list[Trajectory], out_dir: Path, *, seed: int = 0, n_boot: int = N_BOOT,
                   say=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = probe(model, trajectories, seed=seed, say=say)
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "seed": seed,
        "n_boot": n_boot,
        "rules": {"horizons": list(HORIZONS), "moving_share": MOVING_SHARE, "min_step_m": MIN_STEP_M, "maxnorm": MAXNORM,
                  "hurts": HURTS, "harmless": HARMLESS, "slack": SLACK, "menu": "real, reverse, zero, 26 grid directions"},
        **summarize(records, n_boot=n_boot, seed=seed),
        "transitions": records,
    }
    (out_dir / "plan_stall.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
