"""``ii action-neurons``: do a few predictor neurons carry V-JEPA 2-AC's arm action?

Goodfire's recent neuron work (raw MLP neurons are mostly single-purpose, and small communities of
under ten neurons can carry a whole prediction) suggests looking at the predictor's own neurons
instead of an SAE. This command asks whether the arm action's effect on the prediction runs through
a small set of them.

A *neuron* is one hidden unit of a predictor block's MLP (the input of ``mlp.fc2``). On every DROID
transition where the arm moves at least ``MIN_STEP_M`` the predictor runs twice, with the real action
and with a zero action. The action's effect is ``E = pred(a) - pred(0)`` (layer-normed outputs).

1. **Pick.** Each neuron's score is ``mean_tokens |h(a) - h(0)| * ||fc2 column||``: how much the
   action moves it, times how much it can write. Scores are averaged per episode fold.
2. **Ablate.** On the *other* fold, the top ``K`` neurons (``KS``) get their zero-action values in the
   real-action run, so they cannot pass the action on. ``remain`` is ``||E_ablated|| / ||E||`` over
   the frame tokens: 0 means the action no longer changes the prediction, 1 means nothing was lost.
   ``err_rise`` is how much further the prediction lands from the real next latent (mean L1).
3. **Controls.** ``N_RAND`` sets per ``K`` of random neurons with the same count per block, each drawn
   from the same block and the same activity quintile (mean ``|h(a)|``) as the top neuron it stands
   for, ablated the same way.

Verdict, fixed before any real run, with 95% intervals from resampling whole episodes:

* ``few_carry`` when for some ``K`` the top set's ``remain`` has its upper end below ``CARRY`` (0.5)
  and is below every random set's ``remain``: removing ``K`` neurons takes away most of the arm
  action, and matched random neurons do not (``k_carry`` is the smallest such ``K``)
* ``spread`` when even the largest ``K`` leaves ``remain`` with its lower end above ``SPREAD`` (0.8)
* ``mixed`` otherwise

Offline, on recorded video; the robot never executes anything. Not a rollout finding and never
touches a scoreboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel, poses_to_diff
from interpretableintelligence.worldmodel.action_amp import balanced_folds
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.gripper_trace import _blocks
from interpretableintelligence.worldmodel.plan_probe import _boot

SCHEMA = "action-neurons/v1"
KS = (10, 100, 1000)
N_RAND = 10
MIN_STEP_M = 0.01
CARRY = 0.5
SPREAD = 0.8
N_QUANT = 5
N_BOOT = 2000


def _mlps(model: ACWorldModel) -> list:
    mlps = []
    for b, blk in enumerate(_blocks(model)):
        mlp = getattr(blk, "mlp", None)
        if mlp is None or not isinstance(getattr(mlp, "fc2", None), torch.nn.Linear):
            raise ValueError(f"{model.name}: predictor block {b} has no mlp.fc2 to read neurons from")
        mlps.append(mlp)
    return mlps


def _forward(model: ACWorldModel, z: torch.Tensor, state: np.ndarray, actions: np.ndarray, hooks: list) -> torch.Tensor:
    handles = [mod.register_forward_pre_hook(fn) for mod, fn in hooks]
    n = len(actions)
    try:
        zd = z.float().to(model.device).unsqueeze(0).expand(n, -1, -1)
        sd = torch.from_numpy(np.repeat(state[None], n, axis=0)).to(model.device, torch.float32).unsqueeze(1)
        ad = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(model.device).unsqueeze(1)
        with torch.inference_mode(), model._autocast():
            y = model.predictor(zd, ad, sd)[:, -model.tokens_per_frame :]
        return F.layer_norm(y.float(), (y.shape[-1],)).cpu()
    finally:
        for h in handles:
            h.remove()


def _moving(tr: Trajectory) -> tuple[np.ndarray, list[int]]:
    acts = poses_to_diff(tr.states)
    return acts, [t for t in range(len(acts)) if np.linalg.norm(acts[t, :3]) >= MIN_STEP_M and t + 1 < len(tr.frames)]


def score_pass(model: ACWorldModel, items: list[tuple], folds: dict, say=print) -> dict:
    """Per fold, per block: summed action score and activity of every neuron, and the transition count."""
    mlps = _mlps(model)
    wnorm = [m.fc2.weight.detach().float().norm(dim=0).cpu() for m in mlps]
    p = model.tokens_per_frame
    acc = {f: {"score": [torch.zeros_like(w) for w in wnorm], "activity": [torch.zeros_like(w) for w in wnorm], "n": 0} for f in (0, 1)}
    for j, (tr, z, acts, t) in enumerate(items):
        f = folds[tr.episode]
        cap = {}

        def hook(b):
            def fn(_m, args):
                h = args[0][:, -p:].float()
                cap[b] = ((h[0] - h[1]).abs().mean(0).cpu(), h[0].abs().mean(0).cpu())
            return fn

        _forward(model, z[t], tr.states[t], np.stack([acts[t], np.zeros(7, np.float32)]), [(m.fc2, hook(b)) for b, m in enumerate(mlps)])
        for b in range(len(mlps)):
            acc[f]["score"][b] += cap[b][0] * wnorm[b]
            acc[f]["activity"][b] += cap[b][1]
        acc[f]["n"] += 1
        if (j + 1) % 200 == 0:
            say(f"     scored {j + 1}/{len(items)} transitions")
    return acc


def pick_sets(acc: dict, *, seed: int) -> dict:
    """For each fold to be *tested*: top ``K`` neurons from the other fold and ``N_RAND`` matched random sets.

    Neurons are ``(block, index)``. Random stand-ins share the top neuron's block and activity quintile.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for test in (0, 1):
        src = acc[1 - test]
        n = max(1, src["n"])
        score = torch.cat([s / n for s in src["score"]]).numpy()
        activity = [a.numpy() / n for a in src["activity"]]
        sizes = [len(a) for a in activity]
        offs = np.cumsum([0] + sizes)
        order = np.argsort(-score, kind="stable")
        quint = []
        for a in activity:
            edges = np.quantile(a, np.linspace(0, 1, N_QUANT + 1)[1:-1]) if len(a) > 1 else np.array([])
            quint.append(np.searchsorted(edges, a, side="right"))
        sets = {}
        for big_k in KS:
            k = min(big_k, len(score))
            top = order[:k]
            blocks = np.searchsorted(offs, top, side="right") - 1
            top_n = [(int(b), int(g - offs[b])) for g, b in zip(top, blocks)]
            rands = []
            taken = set(top.tolist())
            for _ in range(N_RAND):
                chosen = []
                used = set()
                for b, i in top_n:
                    pool = np.flatnonzero(quint[b] == quint[b][i])
                    pool = [int(x) for x in pool if (b, int(x)) not in used and offs[b] + int(x) not in taken] or \
                           [int(x) for x in range(sizes[b]) if (b, int(x)) not in used]
                    c = pool[int(rng.integers(len(pool)))]
                    used.add((b, c))
                    chosen.append((b, c))
                rands.append(chosen)
            sets[str(big_k)] = {"top": top_n, "random": rands}
        out[test] = sets
    return out


def _by_block(neurons: list[tuple[int, int]], n_blocks: int) -> list[torch.Tensor]:
    idx = [[] for _ in range(n_blocks)]
    for b, i in neurons:
        idx[b].append(i)
    return [torch.tensor(x, dtype=torch.long) for x in idx]


def ablate_pass(model: ACWorldModel, items: list[tuple], sets: dict, folds: dict, say=print) -> list[dict]:
    """Rows: real, zero, then each (K, set) ablated. Each ablated row takes the zero row's values at its neurons."""
    mlps = _mlps(model)
    nb = len(mlps)
    records = []
    for j, (tr, z, acts, t) in enumerate(items):
        f = folds[tr.episode]
        arms = []  # (tag, k, neuron list)
        for k, s in sets[f].items():
            arms.append(("top", k, s["top"]))
            arms += [("random", k, r) for r in s["random"]]
        per_row = [_by_block(n, nb) for _, _, n in arms]
        rows = np.concatenate([np.stack([acts[t], np.zeros(7, np.float32)]), np.repeat(acts[t][None], len(arms), axis=0)])

        def hook(b):
            def fn(_m, args):
                h = args[0].clone()
                for r, idx in enumerate(per_row):
                    if len(idx[b]):
                        ii = idx[b].to(h.device)
                        h[2 + r, :, ii] = h[1, :, ii]
                return (h,) + tuple(args[1:])
            return fn

        y = _forward(model, z[t], tr.states[t], rows, [(m.fc2, hook(b)) for b, m in enumerate(mlps)])
        e = (y[0] - y[1]).norm()
        nxt = z[t + 1].float()
        base_err = float((y[0] - nxt).abs().mean())
        rec = {"clip_id": tr.clip_id, "episode": tr.episode, "fold": f, "t": int(t), "effect_norm": float(e)}
        for r, (tag, k, _) in enumerate(arms):
            ya = y[2 + r]
            remain = float((ya - y[1]).norm() / e) if e > 0 else float("nan")
            err_rise = float((ya - nxt).abs().mean()) - base_err
            if tag == "top":
                rec[f"top_{k}_remain"], rec[f"top_{k}_err_rise"] = remain, err_rise
                rec[f"rand_{k}_remain"], rec[f"rand_{k}_err_rise"] = [], []
            else:
                rec[f"rand_{k}_remain"].append(remain)
                rec[f"rand_{k}_err_rise"].append(err_rise)
        records.append(rec)
        if (j + 1) % 200 == 0:
            say(f"     ablated {j + 1}/{len(items)} transitions")
    return records


def verdict(per_k: dict) -> tuple[str, int | None]:
    ks = sorted(per_k, key=int)
    if not ks or not np.isfinite(per_k[ks[0]]["top_remain"]["value"]):
        return "insufficient_data", None
    for k in ks:
        top = per_k[k]["top_remain"]
        if np.isfinite(top["hi"]) and top["hi"] < CARRY and top["value"] < min(per_k[k]["random_remain_per_set"]):
            return "few_carry", int(k)
    last = per_k[ks[-1]]["top_remain"]
    if np.isfinite(last["lo"]) and last["lo"] > SPREAD:
        return "spread", None
    return "mixed", None


def summarize(records: list[dict], sets: dict, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    def mean(key, j=None):
        def stat(rs):
            v = [r[key] if j is None else r[key][j] for r in rs]
            v = [x for x in v if np.isfinite(x)]
            return float(np.mean(v)) if v else float("nan")
        return stat

    per_k = {}
    for k in sets[0]:
        rand_sets = [mean(f"rand_{k}_remain", j)(records) for j in range(N_RAND)] if records else [float("nan")]
        per_k[k] = {
            "top_remain": _boot(records, mean(f"top_{k}_remain"), n_boot=n_boot, seed=seed),
            "top_err_rise": _boot(records, mean(f"top_{k}_err_rise"), n_boot=n_boot, seed=seed),
            "random_remain_per_set": rand_sets,
            "random_remain_mean": float(np.nanmean(rand_sets)) if records else float("nan"),
            "random_err_rise_mean": float(np.nanmean([mean(f"rand_{k}_err_rise", j)(records) for j in range(N_RAND)])) if records else float("nan"),
            "top_minus_random_remain": _boot(
                records, lambda rs, k=k: mean(f"top_{k}_remain")(rs) - float(np.nanmean([mean(f"rand_{k}_remain", j)(rs) for j in range(N_RAND)])),
                n_boot=n_boot, seed=seed),
        }
    v, k_carry = verdict(per_k)
    return {"n_transitions": len(records), "n_episodes": len({r["episode"] for r in records}), "per_k": per_k,
            "verdict": v, "k_carry": k_carry}


def run_action_neurons(model: ACWorldModel, trajectories: list[Trajectory], out_dir: Path, *, seed: int = 0, n_boot: int = N_BOOT,
                       say=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mlps = _mlps(model)
    items = []
    for i, tr in enumerate(trajectories):
        acts, steps = _moving(tr)
        if not steps:
            continue
        z = model.encode_frames(tr.frames).half()
        items += [(tr, z, acts, t) for t in steps]
        say(f"     encoded {i + 1}/{len(trajectories)} {tr.clip_id}: {len(steps)} moving transitions")
    folds = balanced_folds((tr.episode for tr, *_ in items), salt="action-neurons")
    acc = score_pass(model, items, folds, say=say)
    sets = pick_sets(acc, seed=seed)
    records = ablate_pass(model, items, sets, folds, say=say)
    n_neurons = int(sum(m.fc2.weight.shape[1] for m in mlps))
    top_named = {str(f): [f"b{b}.n{i}" for b, i in s[str(min(KS))]["top"]] for f, s in sets.items()}
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "seed": seed,
        "n_boot": n_boot,
        "n_neurons": n_neurons,
        "n_blocks": len(mlps),
        "rules": {"ks": list(KS), "n_random": N_RAND, "min_step_m": MIN_STEP_M, "carry": CARRY, "spread": SPREAD,
                  "activity_bins": N_QUANT, "pick": "other episode fold", "ablation": "zero-action value at the picked neurons"},
        "top_neurons_by_test_fold": top_named,
        "top_blocks_by_test_fold": {str(f): np.bincount([b for b, _ in s[str(max(KS))]["top"]], minlength=len(mlps)).tolist()
                                    for f, s in sets.items()},
        **summarize(records, sets, n_boot=n_boot, seed=seed),
        "transitions": records,
    }
    (out_dir / "action_neurons.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
