"""``ii action-amp``: does V-JEPA 2-AC know the motion but hold it back?

``ii impossible-events`` found the predictor's next frame always closer to a frozen world than to
the real one, and ``ii plan-stall`` found its planner aims the arm badly (cosine to the real step
about 0.46). Two readings fit both: the predictor carries the right motion but scaled down, or the
motion is not there. Turning the action's own contribution up tells them apart. The idea is
Goodfire's model-diff amplification (``after + alpha * (after - before)``), with "before" the same
predictor under a zero action:

``action``    ``pred(a) + alpha * (pred(a) - pred(0))``: more of what this action changes
``shuffled``  ``pred(a) + alpha * (pred(a') - pred(0))``, ``a'`` a real moving action from another
              episode: more of *an* action's change, not this one's (the control)
``change``    ``pred(a) + alpha * (pred(a) - z_t)``: more of everything the prediction changes,
              action or not (reported, not part of a verdict)

Each amplified prediction is layer-normed again, as the model's own output is, so it lives on the
same scale as the encoder's targets. ``alpha`` runs over ``ALPHAS``; 0 is the model as it is.

On every DROID transition where the arm moves at least ``MIN_STEP_M``:

**Prediction.** ``err`` is the planner's distance (mean L1 over tokens) from the amplified
prediction to the real next latent; ``closer_real`` is whether it is closer to the real next frame
than to the current one (the freeze bias is ``closer_real`` near 0 at ``alpha`` 0).

**Planning.** The same menu as ``ii plan-stall`` (real step, reverse, zero, 26 directions of the
real step's length, no rotation), each candidate's prediction amplified the same way (``action`` and
``change``; ``zero`` is unchanged by ``action``), ranked by Meta's goal score against the latent
``h`` steps ahead (``HORIZONS``). ``cos`` is the cosine between the chosen and the real arm step (0
when it stalls).

``alpha`` is never chosen on the transitions it is scored on: episodes are split into two folds, each
fold's ``alpha`` is the one with the lowest mean ``err`` (prediction) or highest mean ``cos``
(planning) on the other fold, and the held-out gain over ``alpha`` 0 is pooled over both folds.
95% intervals come from resampling whole episodes. Verdicts, fixed before any real run:

**Prediction** (horizon-free):

* ``under_commits`` when the ``action`` gain in ``err`` has its lower end above 0 and beats the
  ``shuffled`` gain (lower end of the paired difference above 0): the right motion is in the
  prediction, too small
* ``absent`` when the ``action`` gain has its upper end at or below 0: no amount helps
* ``mixed`` otherwise

**Planning** (per horizon, on ``action``):

* ``aims_better`` when the gain in ``cos`` has its lower end above 0
* ``no_aim`` when its upper end is at or below 0
* ``mixed`` otherwise

``--random-predictor`` keeps the trained encoder and swaps in an untrained predictor: the floor.
Offline, on recorded video; the robot never executes anything. Not a rollout finding and never
touches a scoreboard.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel, poses_to_diff
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.plan_probe import _boot
from interpretableintelligence.worldmodel.plan_stall import REAL, ZERO, menu

SCHEMA = "action-amp/v1"
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
ARMS = ("action", "shuffled", "change")
PLAN_ARMS = ("action", "change")
HORIZONS = (1, 4)
MIN_STEP_M = 0.01
N_BOOT = 2000


def amplify(pred: torch.Tensor, base: torch.Tensor, alpha: float, delta: torch.Tensor | None = None) -> torch.Tensor:
    """``pred + alpha * (delta if given else pred - base)``, layer-normed again per token. ``alpha`` 0 is ``pred``."""
    if alpha == 0:
        return pred
    d = (pred - base) if delta is None else delta
    y = pred + alpha * d
    return F.layer_norm(y, (y.shape[-1],))


def _l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference over the last two dims: ``[..., P, D]`` pair to ``[...]``."""
    return (a - b).abs().mean(dim=(-1, -2))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def balanced_folds(episodes, salt: str = "action-amp") -> dict:
    """Two episode folds of equal size (within one): episodes ordered by a hash, then alternated."""
    eps = sorted(set(episodes), key=lambda e: (zlib.crc32(f"{salt}:{e}".encode()), str(e)))
    return {e: i % 2 for i, e in enumerate(eps)}


def _moving_steps(tr: Trajectory) -> tuple[np.ndarray, list[int]]:
    acts = poses_to_diff(tr.states)
    steps = [t for t in range(len(acts)) if np.linalg.norm(acts[t, :3]) >= MIN_STEP_M and t + 1 < len(tr.frames)]
    return acts, steps


def probe(model: ACWorldModel, trajectories: list[Trajectory], *, seed: int = 0, say=print) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool = []  # (episode, action) for every moving transition: the shuffled arm's donors
    for tr in trajectories:
        acts, steps = _moving_steps(tr)
        pool += [(tr.episode, acts[t]) for t in steps]
    folds = balanced_folds(tr.episode for tr in trajectories)
    records = []
    for i, tr in enumerate(trajectories):
        acts, steps = _moving_steps(tr)
        donors = [a for ep, a in pool if ep != tr.episode]
        if not steps or not donors:
            say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: no usable transitions")
            continue
        z = model.encode_frames(tr.frames)
        for t in steps:
            cand = menu(acts[t])
            other = donors[int(rng.integers(len(donors)))]
            batch = np.concatenate([cand, acts[t][None], other[None]]).astype(np.float32)
            n = len(batch)
            preds = model.predict_next(z[t].unsqueeze(0).expand(n, -1, -1), torch.from_numpy(tr.states[t]).view(1, 7).expand(n, -1),
                                       torch.from_numpy(batch)).to(model.device)  # amplify and score on the GPU
            menu_preds, p_real, p_other = preds[: len(cand)], preds[-2], preds[-1]
            p_zero = menu_preds[ZERO]
            rec = {"clip_id": tr.clip_id, "episode": tr.episode, "fold": folds[tr.episode], "t": int(t),
                   "real_step_m": float(np.linalg.norm(acts[t, :3]))}
            nxt, cur = z[t + 1].to(model.device), z[t].to(model.device)
            for arm in ARMS:
                errs, closer = [], []
                for a in ALPHAS:
                    if arm == "action":
                        y = amplify(p_real, p_zero, a)
                    elif arm == "shuffled":
                        y = amplify(p_real, p_zero, a, delta=p_other - p_zero)
                    else:
                        y = amplify(p_real, cur, a)
                    e = float(_l1(y, nxt))
                    errs.append(e)
                    closer.append(bool(e < float(_l1(y, cur))))
                rec[f"err_{arm}"] = errs
                rec[f"closer_real_{arm}"] = closer
            for h in HORIZONS:
                if t + h >= len(z):
                    continue
                goal = z[t + h].to(model.device)
                for arm in PLAN_ARMS:
                    cos, stall = [], []
                    for a in ALPHAS:
                        base = p_zero.unsqueeze(0) if arm == "action" else cur.unsqueeze(0)
                        y = amplify(menu_preds, base, a)
                        pick = int(torch.argmin(_l1(y, goal.unsqueeze(0))))
                        cos.append(_cos(cand[pick, :3], cand[REAL, :3]))
                        stall.append(pick == ZERO)
                    rec[f"plan{h}_cos_{arm}"] = cos
                    rec[f"plan{h}_stall_{arm}"] = stall
            records.append(rec)
        say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: {len(steps)} moving transitions, {len(records)} records so far")
    return records


def cross_fit(records: list[dict], key: str, *, lower_is_better: bool) -> dict:
    """Per fold, the ``alpha`` index that does best on the *other* fold. Returns ``{fold: index}``."""
    out = {}
    for f in (0, 1):
        train = [r[key] for r in records if r["fold"] != f and key in r]
        if not train:
            out[f] = 0
            continue
        m = np.mean(np.asarray(train, dtype=np.float64), axis=0)
        out[f] = int(np.argmin(m) if lower_is_better else np.argmax(m))
    return out


def _held_out_gain(key: str, pick: dict, *, lower_is_better: bool):
    """Statistic: mean over records of the gain at the held-out ``alpha`` over ``alpha`` 0 (positive is better)."""
    sign = -1.0 if lower_is_better else 1.0

    def stat(rs):
        rs = [r for r in rs if key in r]
        if not rs:
            return float("nan")
        return float(np.mean([sign * (r[key][pick[r["fold"]]] - r[key][0]) for r in rs]))

    return stat


def _at(key: str, pick: dict | None):
    """Statistic: mean of ``key`` at the held-out ``alpha`` (``pick``), or at ``alpha`` 0 when ``pick`` is None."""

    def stat(rs):
        rs = [r for r in rs if key in r]
        if not rs:
            return float("nan")
        return float(np.mean([float(r[key][0 if pick is None else pick[r["fold"]]]) for r in rs]))

    return stat


def prediction_verdict(gain: dict, vs_shuffled: dict) -> str:
    if not np.isfinite(gain["value"]):
        return "insufficient_data"
    if np.isfinite(gain["lo"]) and gain["lo"] > 0 and np.isfinite(vs_shuffled["lo"]) and vs_shuffled["lo"] > 0:
        return "under_commits"
    if np.isfinite(gain["hi"]) and gain["hi"] <= 0:
        return "absent"
    return "mixed"


def planning_verdict(gain: dict) -> str:
    if not np.isfinite(gain["value"]) or not np.isfinite(gain["lo"]):
        return "insufficient_data"
    if gain["lo"] > 0:
        return "aims_better"
    if gain["hi"] <= 0:
        return "no_aim"
    return "mixed"


def summarize(records: list[dict], *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    b = lambda f: _boot(records, f, n_boot=n_boot, seed=seed)  # noqa: E731
    pred = {}
    picks = {arm: cross_fit(records, f"err_{arm}", lower_is_better=True) for arm in ARMS}
    for arm in ARMS:
        key = f"err_{arm}"
        pred[arm] = {
            "alpha_per_fold": {str(f): ALPHAS[i] for f, i in picks[arm].items()},
            "err_by_alpha": [float(np.mean([r[key][j] for r in records])) if records else float("nan") for j in range(len(ALPHAS))],
            "closer_real_by_alpha": [float(np.mean([r[f"closer_real_{arm}"][j] for r in records])) if records else float("nan")
                                     for j in range(len(ALPHAS))],
            "err_alpha0": b(_at(key, None)),
            "gain": b(_held_out_gain(key, picks[arm], lower_is_better=True)),
            "closer_real_alpha0": b(_at(f"closer_real_{arm}", None)),
            "closer_real_held_out": b(_at(f"closer_real_{arm}", picks[arm])),
        }
    ga = _held_out_gain("err_action", picks["action"], lower_is_better=True)
    gs = _held_out_gain("err_shuffled", picks["shuffled"], lower_is_better=True)
    gc = _held_out_gain("err_change", picks["change"], lower_is_better=True)
    paired = {"action_minus_shuffled": b(lambda rs: ga(rs) - gs(rs)), "action_minus_change": b(lambda rs: ga(rs) - gc(rs))}
    plan = {}
    for h in HORIZONS:
        rs = [r for r in records if f"plan{h}_cos_action" in r]
        out = {"n_transitions": len(rs), "n_episodes": len({r["episode"] for r in rs})}
        for arm in PLAN_ARMS:
            key = f"plan{h}_cos_{arm}"
            pk = cross_fit(rs, key, lower_is_better=False)
            out[arm] = {
                "alpha_per_fold": {str(f): ALPHAS[i] for f, i in pk.items()},
                "cos_by_alpha": [float(np.mean([r[key][j] for r in rs])) if rs else float("nan") for j in range(len(ALPHAS))],
                "stall_by_alpha": [float(np.mean([r[f"plan{h}_stall_{arm}"][j] for r in rs])) if rs else float("nan")
                                   for j in range(len(ALPHAS))],
                "cos_alpha0": _boot(rs, _at(key, None), n_boot=n_boot, seed=seed),
                "gain": _boot(rs, _held_out_gain(key, pk, lower_is_better=False), n_boot=n_boot, seed=seed),
                "stall_held_out": _boot(rs, _at(f"plan{h}_stall_{arm}", pk), n_boot=n_boot, seed=seed),
            }
        out["verdict"] = planning_verdict(out["action"]["gain"])
        plan[str(h)] = out
    verdicts = {"prediction": prediction_verdict(pred["action"]["gain"], paired["action_minus_shuffled"]),
                **{f"planning_{h}": v["verdict"] for h, v in plan.items()}}
    return {
        "n_transitions": len(records),
        "n_episodes": len({r["episode"] for r in records}),
        "prediction": {**pred, "paired": paired},
        "planning": plan,
        "verdicts": verdicts,
    }


def run_action_amp(model: ACWorldModel, trajectories: list[Trajectory], out_dir: Path, *, seed: int = 0, n_boot: int = N_BOOT,
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
        "rules": {"alphas": list(ALPHAS), "horizons": list(HORIZONS), "min_step_m": MIN_STEP_M,
                  "alpha_choice": "two episode folds, each scored at the alpha best on the other",
                  "prediction": "under_commits if action gain lo > 0 and action-minus-shuffled lo > 0; absent if gain hi <= 0",
                  "planning": "aims_better if cos gain lo > 0; no_aim if hi <= 0"},
        **summarize(records, n_boot=n_boot, seed=seed),
        "transitions": records,
    }
    (out_dir / "action_amp.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
