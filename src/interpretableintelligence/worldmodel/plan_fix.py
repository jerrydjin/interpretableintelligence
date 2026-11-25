"""``ii plan-fix``: can switching one failure feature fix V-JEPA 2-AC's planner, beyond random features?

The gate test for "find why your robot model fails, with proof". ``ii plan-stall`` found the 2-AC
planner often picks the wrong move on real DROID video: doing nothing beats the real step on 38% of
moving transitions, and its pick points only loosely along the real step. This command asks whether a
feature inside the predictor explains that, in the only sense that matters for a user: switching it
makes the planner pick the right move, more often than switching random features does.

Per moving transition (arm step at least ``MIN_STEP_M``), the planner picks from ``plan_stall.menu``
(real step, reverse, zero, 26 directions of the same length) under Meta's goal score (mean L1 to the
next frame's latent). ``cos`` is the cosine between the pick and the real step (0 for a stall).

* **failing** step: ``cos <= FAIL_COS`` (0: picks nothing, or a move pointing away)
* **good** step: ``cos >= GOOD_COS`` (0.7)

Episodes are split in two halves (``gripper_trace.fold_of``).

1. **Half A.** A BatchTopK SAE (``sae_batchtopk``) is trained on the predictor's frame-token residual at
   ``--layer``, under the real action. Each latent's mean frame-token activation is compared between
   failing and good steps, within bins of real step size (``N_BINS`` quantile bins, so "the arm moved
   less" is not what gets picked). The ``N_PICK`` latents with the largest difference over its spread
   are the suspects.
2. **Half B** (episodes the SAE and the picking never saw). On failing steps each suspect is moved
   toward the good steps on every frame token at ``--layer``, in all 29 menu runs: one higher on
   failing steps is switched off (set to 0), one higher on good steps is switched on at its typical
   good-step firing level. The change goes back through the decoder, keeping the SAE's error. The
   planner then picks again. A **fix** is a new pick with ``cos >= GOOD_COS``. On good steps the same edit checks for
   damage (``kept``: the pick stays good).

Controls, through the same edit:

* ``--random-sets`` sets of ``N_PICK`` random live latents of the trained SAE, each matched to a suspect
  on firing rate (within a factor of 2), switched off or on by the same rule;
* ``--random-saes`` untrained SAEs (research rule 4's untrained dictionary: the trainer's own init,
  the same normalisation, a threshold for the same ``k``), picked and edited by the same rule.

Verdict, fixed before any real run: ``yes`` when the suspects' fix rate is at least ``MIN_FIX`` (0.10),
its lower end is above every random set's upper end, it is above every untrained SAE's fix rate, and
good steps are kept at least ``MIN_KEPT`` (0.90) of the time; ``no`` otherwise. 95% intervals resample
whole episodes. Offline, on recorded video; the robot never executes anything. Not a rollout finding.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence import sae_batchtopk as sb
from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel, poses_to_diff
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.gripper_sae import _hook, decode_delta, encode
from interpretableintelligence.worldmodel.gripper_trace import COND, fold_of
from interpretableintelligence.worldmodel.plan_probe import _boot
from interpretableintelligence.worldmodel.plan_stall import MIN_STEP_M, REAL, _cos, menu

SCHEMA = "plan-fix/v1"
FAIL_COS = 0.0
GOOD_COS = 0.7
N_PICK = 3
N_BINS = 5
MIN_FIRE = 0.01
MIN_FIX = 0.10
MIN_KEPT = 0.90
MAX_ROWS = 200_000
N_BOOT = 2000


class Clamp:
    """A per-token SAE edit at the hooked layer: ``ablate`` latents set to 0 on every frame token, ``boost`` latents
    set to a fixed value on every frame token (the change goes back through the decoder; the SAE's error is kept)."""

    def __init__(self, weights: dict, ablate: list[int], boost: dict[int, float]) -> None:
        self.w = weights
        self.ablate = [int(j) for j in ablate]
        self.boost = {int(j): float(v) for j, v in boost.items()}
        self.norms: list[float] = []

    @property
    def latents(self) -> list[int]:
        return self.ablate + list(self.boost)

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """Raw-unit change ``[B, P, D]`` for frame tokens ``h [B, P, D]``."""
        f = encode(self.w, h.float())
        df = torch.zeros_like(f)
        if self.ablate:
            df[..., self.ablate] = -f[..., self.ablate]
        for j, v in self.boost.items():
            df[..., j] = v - f[..., j]
        d = decode_delta(self.w, df)
        self.norms.append(float(d.norm(dim=-1).mean()))
        return d


def _menu_costs(model: ACWorldModel, z: torch.Tensor, state: np.ndarray, cand: np.ndarray, goal: torch.Tensor, layer: int,
                edits: list[Clamp | None], grab: list | None = None) -> np.ndarray:
    """Goal score (mean L1) of every menu action, once per edit: ``[len(edits), len(cand)]``.

    Each edit changes the frame tokens at ``layer`` in all menu runs (``None``: no edit). With ``grab``, the
    unedited real-action frame tokens at ``layer`` are appended to it (on the first ``None`` edit)."""
    n = len(cand)
    p = model.tokens_per_frame
    dev = model.device
    batch = max(1, model.predict_batch)
    out = np.zeros((len(edits), n), dtype=np.float64)
    zg = goal.to(dev).unsqueeze(0)
    for e_i, edit in enumerate(edits):
        for i in range(0, n, batch):
            sl = slice(i, min(n, i + batch))
            m = sl.stop - sl.start

            def fn(x, sl=sl, edit=edit):
                if grab is not None and edit is None and not grab and sl.start <= REAL < sl.stop:
                    grab.append(x[REAL - sl.start, COND:].detach().float().cpu())
                if edit is None:
                    return x
                x = x.clone()
                x[:, COND:] = x[:, COND:] + edit(x[:, COND:]).to(x.dtype)
                return x

            handle = _hook(model, layer, fn)
            try:
                zd = z.unsqueeze(0).expand(m, -1, -1).to(dev)
                ad = torch.from_numpy(np.asarray(cand[sl], np.float32)).to(dev).unsqueeze(1)
                sd = torch.from_numpy(np.repeat(np.asarray(state, np.float32)[None], m, axis=0)).to(dev).unsqueeze(1)
                with torch.inference_mode(), model._autocast():
                    y = model.predictor(zd, ad, sd)[:, -p:]
                y = F.layer_norm(y.float(), (y.shape[-1],))
                out[e_i, sl] = (y - zg).abs().mean((-1, -2)).double().cpu().numpy()
            finally:
                handle.remove()
    return out


def _pick_cos(costs: np.ndarray, cand: np.ndarray) -> list[float]:
    return [_cos(cand[int(np.argmin(c)), :3], cand[REAL, :3]) for c in costs]


def label(c: float) -> str | None:
    if c <= FAIL_COS:
        return "failing"
    if c >= GOOD_COS:
        return "good"
    return None


def bins_of(steps: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Quantile bin of each real step size."""
    if len(steps) == 0:
        return np.zeros(0, dtype=int)
    edges = np.quantile(steps, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, steps, side="right")


def stratified_diff(feats: np.ndarray, labels: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Good-minus-failing mean per latent within step-size bins, weighted by ``min(n_fail, n_good)``, and its
    score (difference over the pooled within-bin spread). ``feats [S, M]``, ``labels`` 1 good / 0 failing."""
    m = feats.shape[1]
    num, wsum = np.zeros(m), 0.0
    resid = []
    for b in np.unique(bins):
        g = feats[(bins == b) & (labels == 1)]
        f = feats[(bins == b) & (labels == 0)]
        if len(g) < 2 or len(f) < 2:
            continue
        w = min(len(g), len(f))
        num += w * (g.mean(0) - f.mean(0))
        wsum += w
        resid += [g - g.mean(0), f - f.mean(0)]
    if wsum == 0:
        return np.zeros(m), np.zeros(m)
    delta = num / wsum
    spread = np.concatenate(resid).std(0) + 1e-9
    return delta, delta / spread


def pick(score: np.ndarray, fire: np.ndarray, n: int = N_PICK) -> list[int]:
    ok = np.flatnonzero(fire >= MIN_FIRE)
    order = ok[np.argsort(-np.abs(score[ok]))]
    return [int(i) for i in order[:n]]


def matched_random(fire: np.ndarray, picked: list[int], n_sets: int, rng: np.random.Generator) -> list[list[int]]:
    """``n_sets`` sets of live latents, each latent matched to one pick on firing rate (within x2), never a pick."""
    live = np.flatnonzero(fire >= MIN_FIRE)
    live = live[~np.isin(live, picked)]
    sets = []
    for _ in range(n_sets):
        chosen: list[int] = []
        for p in picked:
            pool = live[(fire[live] >= fire[p] / 2) & (fire[live] <= fire[p] * 2) & ~np.isin(live, chosen)]
            if len(pool) == 0:
                pool = live[~np.isin(live, chosen)]
            if len(pool) == 0:
                break
            chosen.append(int(rng.choice(pool)))
        sets.append(chosen)
    return sets


def clamp_for(w: dict, latents: list[int], delta: np.ndarray, on_level: np.ndarray) -> Clamp:
    """Move ``latents`` toward the good steps: a latent higher on failing steps (``delta < 0``) is switched off, one
    higher on good steps is switched on at its typical good-step firing level ``on_level``."""
    ablate = [j for j in latents if delta[j] < 0]
    boost = {j: float(on_level[j]) for j in latents if delta[j] >= 0}
    return Clamp(w, ablate, boost)


def verdict(arms: dict, kept: dict) -> str:
    fix = arms["suspects"]["fix"]
    if not np.isfinite(fix["lo"]):
        return "insufficient_data"
    randoms = [v["fix"] for k, v in arms.items() if k.startswith("random-set:")]
    untrained = [v["fix"] for k, v in arms.items() if k.startswith("untrained:")]
    ok = (fix["value"] >= MIN_FIX and all(fix["lo"] > r["hi"] for r in randoms)
          and all(fix["value"] > u["value"] for u in untrained) and kept["value"] >= MIN_KEPT)
    return "yes" if ok else "no"


def run_plan_fix(model: ACWorldModel, trajectories: list[Trajectory], out_dir: Path, *, layer: int, n_latents: int = 4096,
                 k: int = 32, steps: int = 3000, random_sets: int = 10, random_saes: int = 5, max_fail: int = 150,
                 max_good: int = 60, stride: int = 1, seed: int = 0, n_boot: int = N_BOOT, say=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clock = {"start": time.time()}

    # Pass 1: every moving transition, unedited: the planner's pick, and half A's residual rows for the SAE.
    steps_all, rows, row_eps = [], [], []
    for i, tr in enumerate(trajectories):
        acts = poses_to_diff(tr.states)
        moving = [t for t in range(len(acts)) if np.linalg.norm(acts[t, :3]) >= MIN_STEP_M and t + 1 < len(tr.frames)]
        moving = moving[::stride]
        if not moving:
            continue
        z = model.encode_frames(tr.frames)
        f = fold_of(tr.episode, seed)
        for t in moving:
            cand = menu(acts[t])
            grab: list = []
            costs = _menu_costs(model, z[t], tr.states[t], cand, z[t + 1], layer, [None], grab=grab)
            c = _pick_cos(costs, cand)[0]
            s = {"clip_id": tr.clip_id, "episode": tr.episode, "t": int(t), "fold": f, "cos": c, "label": label(c),
                 "step_m": float(np.linalg.norm(cand[REAL, :3])), "state": tr.states[t], "cand": cand}
            # Keep only what later passes need: half A's residuals (SAE rows, picking), half B's latents (editing).
            if f == 0:
                h = grab[0].half()
                rows.append(h)
                row_eps.append(torch.full((h.shape[0],), tr.episode, dtype=torch.long))
                if s["label"] is not None:
                    s["h"] = h
            elif s["label"] is not None:
                s["z"], s["goal"] = z[t].clone(), z[t + 1].clone()
            steps_all.append(s)
        del z
        say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: {len(moving)} moving steps, {len(steps_all)} so far")
    if not rows:
        raise ValueError("no moving transitions in half A to train the SAE on")

    clock["pass1"] = time.time()
    # Train the SAE on half A, and build the untrained twins.
    acts_rows, eps_rows = torch.cat(rows), torch.cat(row_eps)
    if len(acts_rows) > MAX_ROWS:
        keep = torch.from_numpy(np.sort(rng.choice(len(acts_rows), size=MAX_ROWS, replace=False)))
        acts_rows, eps_rows = acts_rows[keep], eps_rows[keep]
    device = model.device
    split = sb.Split(acts=acts_rows.to(device), tok=torch.zeros(len(acts_rows), dtype=torch.long, device=device), episode=eps_rows.to(device))
    mu, scale = sb.norm_stats(split, None, "fit")
    trained, curve = sb.train(split, mu, scale, sb.RetrainConfig(n_latents=n_latents, k=k, steps=steps, seed=seed, norm="fit"),
                              log=lambda m: None)
    health = sb.evaluate(trained, split)
    sb.save_sae_dir(trained, out_dir / "sae", {"layer": layer, "n_latents": n_latents, "k": k, "model": model.name})
    dicts = {"trained": {key: v.to(device) for key, v in trained.items()}}
    for r in range(random_saes):
        w = {key: v.to(device) for key, v in sb.untrained_weights(trained, seed + 1000 + r).items()}
        threshold, _ = sb.batchtopk_threshold(w, split.chunks, k)
        w["threshold"] = torch.tensor(threshold, device=device)
        dicts[f"untrained:{seed + 1000 + r}"] = w
    n_rows = split.n
    resid_norm = float(acts_rows[:20_000].float().norm(dim=-1).mean())
    del split, acts_rows, rows
    say(f"     SAE: {n_latents} latents, k={k}, final FVU {curve[-1]['fvu']:.4f}")

    # Pick the suspects on half A's labelled steps, within step-size bins.
    sel = [s for s in steps_all if s["fold"] == 0 and s["label"] is not None]
    test_fail = [s for s in steps_all if s["fold"] == 1 and s["label"] == "failing"]
    test_good = [s for s in steps_all if s["fold"] == 1 and s["label"] == "good"]
    if len({s["label"] for s in sel}) < 2 or not test_fail:
        raise ValueError(f"need failing and good steps in half A and failing steps in half B "
                         f"(half A: {len(sel)} labelled, half B: {len(test_fail)} failing)")
    labels = np.array([1 if s["label"] == "good" else 0 for s in sel])
    bins = bins_of(np.array([s["step_m"] for s in sel]))
    edits: dict[str, Clamp | None] = {}
    suspects_info = {}
    for name, w in dicts.items():
        feats, on_sum, on_cnt = [], np.zeros(n_latents), np.zeros(n_latents)
        for s, lab in zip(sel, labels):
            f = encode(w, s["h"].to(device).float())
            feats.append(f.mean(0).cpu().numpy())
            if lab == 1:
                on_sum += f.sum(0).cpu().numpy()
                on_cnt += (f > 0).sum(0).cpu().numpy()
        feats = np.stack(feats)
        on_level = on_sum / np.maximum(on_cnt, 1)
        fire = (feats > 0).mean(0)
        delta, score = stratified_diff(feats, labels, bins)
        chosen = pick(score, fire)
        if name == "trained":
            edits["suspects"] = clamp_for(w, chosen, delta, on_level)
            for j, rset in enumerate(matched_random(fire, chosen, random_sets, rng)):
                edits[f"random-set:{j}"] = clamp_for(w, rset, delta, on_level)
            for lat in chosen:
                order = np.argsort(-feats[:, lat])
                suspects_info[str(lat)] = {
                    "edit": "off" if delta[lat] < 0 else f"on at {on_level[lat]:.3f}",
                    "score": float(score[lat]), "delta": float(delta[lat]), "fire": float(fire[lat]),
                    "mean_failing": float(feats[labels == 0, lat].mean()), "mean_good": float(feats[labels == 1, lat].mean()),
                    "top_steps": [{"clip_id": sel[o]["clip_id"], "t": sel[o]["t"], "label": sel[o]["label"]} for o in order[:8]],
                }
        else:
            edits[name] = clamp_for(w, chosen, delta, on_level)
    # Sanity arm: no edit must reproduce pass 1's picks.
    edits["no-edit"] = None
    say(f"     suspects {edits['suspects'].latents} (off {edits['suspects'].ablate}, on {list(edits['suspects'].boost)})")

    clock["sae"] = time.time()
    # Half B: edit every arm in all menu runs, on failing steps (fix) and good steps (damage).
    order = rng.permutation(len(test_fail))[:max_fail]
    order_g = rng.permutation(len(test_good))[:max_good]
    names = list(edits)
    records = []
    todo = [(test_fail[i], "failing") for i in order] + [(test_good[i], "good") for i in order_g]
    for j, (s, kind) in enumerate(todo):
        costs = _menu_costs(model, s["z"], s["state"], s["cand"], s["goal"], layer, [edits[n] for n in names])
        cs = _pick_cos(costs, s["cand"])
        rec = {"clip_id": s["clip_id"], "episode": s["episode"], "t": s["t"], "kind": kind, "cos": s["cos"], "step_m": s["step_m"]}
        for n, c in zip(names, cs):
            rec[f"cos/{n}"] = c
            rec[f"good/{n}"] = c >= GOOD_COS
        records.append(rec)
        if (j + 1) % 20 == 0:
            say(f"     edited {j + 1}/{len(todo)} steps")

    clock["edit"] = time.time()
    fail_recs = [r for r in records if r["kind"] == "failing"]
    good_recs = [r for r in records if r["kind"] == "good"]

    def frac(key):
        return lambda rs: float(np.mean([r[key] for r in rs])) if rs else float("nan")

    arms = {}
    for n in names:
        e = edits[n]
        arms[n] = {"latents": [] if e is None else e.latents, "off": [] if e is None else e.ablate,
                   "on": {} if e is None else {str(j): v for j, v in e.boost.items()},
                   "edit_norm": 0.0 if e is None or not e.norms else float(np.mean(e.norms)),
                   "fix": _boot(fail_recs, frac(f"good/{n}"), n_boot=n_boot, seed=seed),
                   "kept": _boot(good_recs, frac(f"good/{n}"), n_boot=n_boot, seed=seed)}
    kept = arms["suspects"]["kept"] if good_recs else {"value": float("nan"), "lo": float("nan"), "hi": float("nan")}
    labelled = [s for s in steps_all if s["label"] is not None]
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "layer": layer,
        "seed": seed,
        "n_boot": n_boot,
        "rules": {"fail_cos": FAIL_COS, "good_cos": GOOD_COS, "n_pick": N_PICK, "n_bins": N_BINS, "min_fire": MIN_FIRE,
                  "min_fix": MIN_FIX, "min_kept": MIN_KEPT, "min_step_m": MIN_STEP_M, "goal": "next frame, Meta's mean L1 score"},
        "stride": stride,
        "seconds": {"pass1": clock["pass1"] - clock["start"], "sae_and_pick": clock["sae"] - clock["pass1"],
                    "edit": clock["edit"] - clock["sae"], "edit_per_step": (clock["edit"] - clock["sae"]) / max(1, len(records))},
        "residual_norm": resid_norm,
        "sae": {"n_latents": n_latents, "k": k, "steps": steps, "rows": n_rows, "final_fvu": curve[-1]["fvu"], "health": health},
        "n_steps": len(steps_all),
        "n_failing": sum(s["label"] == "failing" for s in steps_all),
        "n_good": sum(s["label"] == "good" for s in steps_all),
        "n_labelled_episodes": len({s["episode"] for s in labelled}),
        "n_selection_steps": len(sel),
        "n_test_failing": len(fail_recs),
        "n_test_good": len(good_recs),
        "n_test_episodes": len({r["episode"] for r in records}),
        "suspects": suspects_info,
        "arms": arms,
        "verdict": verdict(arms, kept),
        "steps": records,
    }
    (out_dir / "plan_fix.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
