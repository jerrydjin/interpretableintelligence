"""``ii gripper-circuit``: which parts of V-JEPA 2-AC turn the gripper command down, and does undoing them help?

``ii gripper-trace`` found the command written into the frame tokens in block 1, growing until a
peak (block 9 on the real model, 1.5 times the effect of a 6 cm arm step) and shrunk to 0.28 by the
output. This command names the parts that shrink it and tests them causally, on the same gripper
events (the gripper opens or closes while the arm stays still):

1. **Suspects.** Every block adds an attention output and an MLP output to the residual. For each one
   (and each attention head, through its slice of ``attn.proj``) the right-minus-wrong command
   difference in its output is projected on the gap it receives:
   ``s = sum <d, g> / sum |g|^2`` over frame tokens. ``s = -1`` erases the gap, ``s = 0`` ignores it.
   The same is done for two arm steps 6 cm apart. A part with ``s_grip - s_arm`` well below zero
   removes the gripper difference and leaves the arm alone.
2. **Knockouts.** Episodes split into two folds (``gripper_trace.fold_of``). Each fold picks its peak
   layer and its top ``K_COMP`` parts and ``K_HEAD`` heads after the peak; the other fold is tested
   with them. A knocked-out part gets its right-command output in the wrong-command run too, so it
   cannot react to the command. ``recovered`` is how much of the drop from peak to output in the
   gripper-to-arm ratio comes back. Controls: ``N_RAND`` sets of the same number of random parts
   (or heads) after the peak. Two descriptive arms knock out every attention output, or every MLP
   output, after the peak at once. For the top heads the attention mass from frame tokens to the action
   and state tokens is recorded, under both commands.
3. **Steering.** For each candidate action, the command's own effect at the peak layer
   (``h_peak(a) - h_peak(a with gripper 0)``) is added to the last block's output, rescaled per token
   to the residual's size there (``alpha = 1`` restores the relative size it had at the peak). Then the
   goal score from ``ii gripper-probe`` is read again: does it prefer the right command? Controls:
   ``N_RAND_STEER`` random directions with the same per-token size and the same sign flip with the
   command. Checks: the score's preference for the real arm step over steps 3 cm off on x
   (``arm_pref``) and the score of the right command (``damage``) under steering.

Verdicts, fixed before any real run, on the held-out fold with 95% intervals from resampling whole
episodes: knockouts find ``erasers`` when ``recovered`` is at least ``RECOVER`` (0.5) with its lower
end above 0, above every random set, and the random sets' mean is under ``RANDOM_MAX`` (0.10);
otherwise ``none``. Steering ``fixes_gripper`` when ``prefers_correct`` at ``alpha = 1`` has its lower
end above ``FIX_LO`` (0.60) and is above every random direction, ``arm_pref`` drops by at most
``ARM_DROP`` (0.05) and ``damage`` is at most ``DAMAGE`` (2%); otherwise ``no_fix``.
Offline, on recorded video; not a rollout finding.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.gripper_probe import find_events
from interpretableintelligence.worldmodel.gripper_trace import ARM_STEP, COND, _blocks, fold_of
from interpretableintelligence.worldmodel.plan_probe import GRIP_MAX, MAXNORM, _boot

SCHEMA = "gripper-circuit/v1"
K_COMP = 4
K_HEAD = 8
N_RAND = 20
N_RAND_STEER = 10
ALPHAS = (0.5, 1.0, 2.0, 4.0)
STEER_ALPHA = 1.0
RECOVER = 0.5
RANDOM_MAX = 0.10
FIX_LO = 0.60
ARM_DROP = 0.05
DAMAGE = 0.02
N_BOOT = 2000
KINDS = ("attn", "mlp")


def _check(model: ACWorldModel) -> tuple[list, int, int]:
    blocks = _blocks(model)
    for blk in blocks:
        attn = getattr(blk, "attn", None)
        if attn is None or getattr(blk, "mlp", None) is None or not hasattr(attn, "proj") or not hasattr(attn, "num_heads"):
            raise ValueError(f"{model.name}: predictor blocks need attn (with proj, num_heads) and mlp sublayers")
    heads = int(blocks[0].attn.num_heads)
    return blocks, heads, blocks[0].attn.proj.in_features // heads


def _event_actions(arm: np.ndarray, sign: float) -> dict[str, np.ndarray]:
    base = np.zeros(7, dtype=np.float32)
    base[:3] = arm
    out = {}
    for name, grip, dx in (("R", sign, 0.0), ("W", -sign, 0.0), ("G0", 0.0, 0.0), ("A+", 0.0, ARM_STEP), ("A-", 0.0, -ARM_STEP),
                           ("Rx+", sign, ARM_STEP), ("Rx-", sign, -ARM_STEP)):
        a = base.copy()
        a[6] = grip * GRIP_MAX
        a[0] += dx
        out[name] = a
    return out


def _run(model: ACWorldModel, z: torch.Tensor, state: np.ndarray, rows: np.ndarray, hooks) -> torch.Tensor:
    """Batched forward with ``hooks`` (``(module, "pre" | "pre_kw" | "fwd", fn)``); layer-normed output on the CPU."""
    handles = []
    for mod, kind, fn in hooks:
        if kind == "pre_kw":
            handles.append(mod.register_forward_pre_hook(fn, with_kwargs=True))
        elif kind == "pre":
            handles.append(mod.register_forward_pre_hook(fn))
        else:
            handles.append(mod.register_forward_hook(fn))
    n = len(rows)
    try:
        zd = z.to(model.device).unsqueeze(0).expand(n, -1, -1)
        sd = torch.from_numpy(np.repeat(state[None], n, axis=0)).to(model.device, torch.float32).unsqueeze(1)
        ad = torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(model.device).unsqueeze(1)
        with torch.inference_mode(), model._autocast():
            y = model.predictor(zd, ad, sd)[:, -model.tokens_per_frame :]
        return F.layer_norm(y.float(), (y.shape[-1],)).cpu()
    finally:
        for h in handles:
            h.remove()


def _proj(d: torch.Tensor, g: torch.Tensor) -> float:
    den = float((g * g).sum())
    return float((d * g).sum()) / den if den > 1e-12 else float("nan")


# --------------------------------------------------------------------------- step 1: suspects


def measure_event(model: ACWorldModel, z: torch.Tensor, state: np.ndarray, arm: np.ndarray, sign: float) -> dict:
    """Rows: right, wrong, arm +x, arm -x. Per-layer drift (as ``gripper-trace``) and per-part projections."""
    blocks, heads, hd = _check(model)
    acts = _event_actions(arm, sign)
    rows = np.stack([acts["R"], acts["W"], acts["A+"], acts["A-"]])
    n_layers = len(blocks) + 1
    grip, armd = np.full(n_layers, np.nan), np.full(n_layers, np.nan)
    s = {k: np.full((len(blocks), 2), np.nan) for k in ("grip", "arm")}
    sh = {k: np.full((len(blocks), heads), np.nan) for k in ("grip", "arm")}
    cur: dict = {}

    def drift(layer, x):
        h = x.float()[:, COND:]
        size = h[:2].norm(dim=-1).mean().clamp_min(1e-9)
        grip[layer] = float((h[0] - h[1]).norm(dim=-1).mean() / size)
        armd[layer] = float((h[2] - h[3]).norm(dim=-1).mean() / size)

    def gaps(x):
        return x[0] - x[1], x[2] - x[3]

    hooks = []
    for b, blk in enumerate(blocks):
        def pre(_m, args, kwargs, b=b):
            x = args[0].float()[:, COND:]
            cur["in"] = x
            if b == 0:
                drift(0, args[0])
            return None

        def attn_out(_m, _a, out, b=b):
            o = out.float()[:, COND:]
            (dg, da), (gg, ga) = gaps(o), gaps(cur["in"])
            s["grip"][b, 0], s["arm"][b, 0] = _proj(dg, gg), _proj(da, ga)
            cur["mid"] = cur["in"] + o
            return None

        def mlp_out(_m, _a, out, b=b):
            o = out.float()[:, COND:]
            (dg, da), (gg, ga) = gaps(o), gaps(cur["mid"])
            s["grip"][b, 1], s["arm"][b, 1] = _proj(dg, gg), _proj(da, ga)
            return None

        def proj_in(mod, args, b=b):
            inp = args[0].float()[:, COND:]
            w = mod.weight.float()
            (gg, ga) = gaps(cur["in"])
            for h in range(heads):
                sl = slice(h * hd, (h + 1) * hd)
                c = inp[:, :, sl] @ w[:, sl].T
                sh["grip"][b, h], sh["arm"][b, h] = _proj(c[0] - c[1], gg), _proj(c[2] - c[3], ga)
            return None

        hooks += [(blk, "pre_kw", pre), (blk.attn.proj, "pre", proj_in), (blk.attn, "fwd", attn_out), (blk.mlp, "fwd", mlp_out),
                  (blk, "fwd", lambda _m, _a, out, b=b: drift(b + 1, out))]
    _run(model, z, state, rows, hooks)
    return {"grip": grip.tolist(), "arm": armd.tolist(), "s_grip": s["grip"].tolist(), "s_arm": s["arm"].tolist(),
            "sh_grip": sh["grip"].tolist(), "sh_arm": sh["arm"].tolist()}


def _nanmean(xs) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan columns: parts the command never reaches
        return np.nanmean(np.asarray(xs, dtype=np.float64), axis=0)


def _ratio_curve(records: list[dict]) -> np.ndarray:
    g = _nanmean([r["grip"] for r in records])
    a = _nanmean([r["arm"] for r in records])
    return g / np.maximum(a, 1e-12)


def pick(records: list[dict], *, k_comp: int, k_head: int, n_rand: int, seed: int) -> dict | None:
    """Peak layer, top parts and heads after it (most negative ``s_grip - s_arm``), and random sets."""
    if not records:
        return None
    ratio = _ratio_curve(records)
    n_layers = len(ratio)
    if not np.isfinite(ratio[1:]).any():
        return None
    peak = int(np.nanargmax(ratio[1:]) + 1)
    if peak >= n_layers - 1:
        return {"peak": peak, "comps": [], "heads": [], "rand_comps": [], "rand_heads": []}
    sc = _nanmean([np.asarray(r["s_grip"]) - np.asarray(r["s_arm"]) for r in records])
    shd = _nanmean([np.asarray(r["sh_grip"]) - np.asarray(r["sh_arm"]) for r in records])
    comp_pool = [(b, k) for b in range(peak, sc.shape[0]) for k in range(2) if np.isfinite(sc[b, k])]
    head_pool = [(b, h) for b in range(peak, shd.shape[0]) for h in range(shd.shape[1]) if np.isfinite(shd[b, h])]
    comps = sorted(comp_pool, key=lambda u: sc[u])[:k_comp]
    heads = sorted(head_pool, key=lambda u: shd[u])[:k_head]
    rng = np.random.default_rng(seed)

    def draws(pool, top, k):
        rest = [u for u in pool if u not in top]
        if len(rest) < k or k == 0:
            return []
        return [[rest[i] for i in rng.choice(len(rest), size=k, replace=False)] for _ in range(n_rand)]

    return {
        "peak": peak,
        "comps": comps,
        "heads": heads,
        "comp_scores": [float(sc[u]) for u in comps],
        "head_scores": [float(shd[u]) for u in heads],
        "rand_comps": draws(comp_pool, comps, len(comps)),
        "rand_heads": draws(head_pool, heads, len(heads)),
    }


# --------------------------------------------------------------------------- steps 2 and 3: knockouts, steering


class _FProxy:
    """Stands in for ``torch.nn.functional`` in the attention's module, so attention weights can be read."""

    def __init__(self, base, sdpa) -> None:
        self._base = base
        self.scaled_dot_product_attention = sdpa

    def __getattr__(self, name):
        return getattr(self._base, name)


def knock_and_steer(model: ACWorldModel, z: torch.Tensor, goal: torch.Tensor, state: np.ndarray, arm: np.ndarray, sign: float,
               picks: dict, *, seed: int) -> dict:
    blocks, heads, hd = _check(model)
    acts = _event_actions(arm, sign)
    peak = picks["peak"]
    spec = [dict(act="R"), dict(act="W"), dict(act="G0"), dict(act="Rx+"), dict(act="Rx-"),
            dict(act="W", comps=picks["comps"], tag="top_comps"), dict(act="W", heads=picks["heads"], tag="top_heads")]
    after = range(peak, len(blocks))
    spec += [dict(act="W", comps=[(b, 0) for b in after], tag="all_attn"), dict(act="W", comps=[(b, 1) for b in after], tag="all_mlp")]
    spec += [dict(act="W", comps=s, tag="rand_comps") for s in picks["rand_comps"]]
    spec += [dict(act="W", heads=s, tag="rand_heads") for s in picks["rand_heads"]]
    for a in ALPHAS:
        spec += [dict(act=n, steer=a, tag=f"steer:{a}") for n in ("R", "W", "Rx+", "Rx-")]
    gen = torch.Generator().manual_seed(seed)
    dim = blocks[0].attn.proj.out_features
    dirs = F.normalize(torch.randn(N_RAND_STEER, dim, generator=gen), dim=-1)
    for i in range(N_RAND_STEER):
        spec += [dict(act=n, steer=STEER_ALPHA, rand=i, tag="steer_rand") for n in ("R", "W")]
    rows = np.stack([acts[s["act"]] for s in spec])
    sgn = np.sign(rows[:, 6])
    comp_rows = {}
    head_rows = {}
    for j, s in enumerate(spec):
        for b, k in s.get("comps", []):
            comp_rows.setdefault((b, k), []).append(j)
        for b, h in s.get("heads", []):
            head_rows.setdefault(b, []).append((j, h))
    steer_rows = [j for j, s in enumerate(spec) if "steer" in s]
    cap: dict = {"block": -1}
    attn_heads = {}
    for b, h in picks["heads"]:
        attn_heads.setdefault(b, []).append(h)
    mass = {f"b{b}.h{h}": {} for b, h in picks["heads"]}

    def knock(b, k):
        js = comp_rows.get((b, k))
        if not js:
            return None

        def fn(_m, _a, out):
            out = out.clone()
            for j in js:
                out[j] = out[0]
            return out
        return fn

    def proj_pre(b):
        items = head_rows.get(b)
        if not items:
            return None

        def fn(_m, args):
            x = args[0].clone()
            for j, h in items:
                x[j, :, h * hd : (h + 1) * hd] = x[0, :, h * hd : (h + 1) * hd]
            return (x,) + tuple(args[1:])
        return fn

    def at_peak(_m, _a, out):
        cap["peak"] = out.float()[:, COND:].clone()
        return None

    def at_last(_m, _a, out):
        h = out.float()[:, COND:]
        size = h[:2].norm(dim=-1).mean().clamp_min(1e-9)
        cap["drift"] = ((h[0:1] - h).norm(dim=-1).mean(dim=-1) / size).cpu().numpy()
        if not steer_rows:
            return None
        hp = cap["peak"]
        out = out.clone()
        for j in steer_rows:
            s = spec[j]
            delta = hp[j] - hp[2]
            scale = h[j].norm(dim=-1, keepdim=True) / hp[j].norm(dim=-1, keepdim=True).clamp_min(1e-9)
            if "rand" in s:
                u = dirs[s["rand"]].to(h.device)
                add = float(sgn[j]) * delta.norm(dim=-1, keepdim=True) * u
            else:
                add = delta
            out[j, COND:] = (h[j] + s["steer"] * add * scale).to(out.dtype)
        return out

    hooks = []
    for b, blk in enumerate(blocks):
        hooks.append((blk, "pre_kw", lambda _m, _a, _k, b=b: cap.__setitem__("block", b)))
        for k, mod in enumerate((blk.attn, blk.mlp)):
            fn = knock(b, k)
            if fn is not None:
                hooks.append((mod, "fwd", fn))
        fn = proj_pre(b)
        if fn is not None:
            hooks.append((blk.attn.proj, "pre", fn))
    def curve(_m, _a, out):
        h = out.float()[:, COND:]
        size = h[:2].norm(dim=-1).mean().clamp_min(1e-9)
        cap.setdefault("curve", []).append(((h[0:1] - h).norm(dim=-1).mean(dim=-1) / size).cpu().numpy())
        return None

    for blk in blocks:
        hooks.append((blk, "fwd", curve))
    hooks.append((blocks[peak - 1], "fwd", at_peak))
    hooks.append((blocks[-1], "fwd", at_last))

    # Attention weights of the top heads, for the right and wrong command rows.
    fmods = []
    for blk in blocks:
        mod = sys.modules.get(type(blk.attn).__module__)
        if mod is not None and getattr(mod, "F", None) is F and mod not in fmods:
            fmods.append(mod)
    original = F.scaled_dot_product_attention

    def sdpa(q, k, v, *args, **kwargs):
        b = cap["block"]
        for h in attn_heads.get(b, []):
            logits = (q[:2, h].float() @ k[:2, h].float().transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            m = kwargs.get("attn_mask")
            if m is not None:
                logits = logits.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else logits + m.float()
            w = logits.softmax(dim=-1)[:, COND:]  # frame-token queries
            mass[f"b{b}.h{h}"] = {"action_right": float(w[0, :, 0].mean()), "action_wrong": float(w[1, :, 0].mean()),
                                  "state_right": float(w[0, :, 1].mean()), "state_wrong": float(w[1, :, 1].mean())}
        return original(q, k, v, *args, **kwargs)

    for mod in fmods:
        mod.F = _FProxy(F, sdpa)
    try:
        y = _run(model, z, state, rows, hooks)
    finally:
        for mod in fmods:
            mod.F = F
    e = (y - goal.float().unsqueeze(0)).abs().mean(dim=-1).mean(dim=-1).numpy()
    drift = cap["drift"]
    by = {}
    for j, s in enumerate(spec):
        by.setdefault(s.get("tag", s["act"]), []).append(j)

    def pref(jr, jw):
        return bool(e[jr] < e[jw])

    out = {
        "e_right": float(e[0]),
        "e_wrong": float(e[1]),
        "prefers_correct": pref(0, 1),
        "arm_pref": bool(e[0] < min(e[3], e[4])),
        "grip_last": float(drift[1]),
        "knock": {},
        "steer": {},
        "attention": mass,
    }
    layers = np.stack(cap["curve"])  # [blocks, rows]: gripper drift after each block
    out["curve"] = layers[:, 1].tolist()
    for tag in ("top_comps", "top_heads", "rand_comps", "rand_heads", "all_attn", "all_mlp"):
        js = by.get(tag, [])
        out["knock"][tag] = {"grip_last": [float(drift[j]) for j in js], "prefers_correct": [pref(0, j) for j in js],
                             "curve": layers[:, js].mean(axis=1).tolist() if js else []}
    for a in ALPHAS:
        r, w, xp, xm = by[f"steer:{a}"]
        out["steer"][str(a)] = {"prefers_correct": pref(r, w), "arm_pref": bool(e[r] < min(e[xp], e[xm])),
                                "damage": float(e[r] / e[0] - 1.0) if e[0] > 0 else float("nan")}
    js = by.get("steer_rand", [])
    out["steer_rand"] = [pref(js[i], js[i + 1]) for i in range(0, len(js), 2)]
    return out


# --------------------------------------------------------------------------- run


def _events(model: ACWorldModel, trajectories: list[Trajectory], say) -> list[tuple]:
    """Encode each clip once and keep only the frames each gripper event needs."""
    out = []
    for i, tr in enumerate(trajectories):
        evs = find_events(tr.states)
        if not evs:
            continue
        z = model.encode_frames(tr.frames)
        for before, after in evs:
            s0, s1 = tr.states[before], tr.states[after]
            sign = 1.0 if s1[6] > s0[6] else -1.0
            arm = np.clip(s1[:3] - s0[:3], -MAXNORM, MAXNORM)
            out.append((tr, before, after, z[before].clone(), z[after].clone(), s0, arm, sign))
        say(f"     encoded {i + 1}/{len(trajectories)} {tr.clip_id}: {len(evs)} gripper events")
    return out


def probe(model: ACWorldModel, trajectories: list[Trajectory], *, k_comp: int = K_COMP, k_head: int = K_HEAD, n_rand: int = N_RAND,
          seed: int = 0, say=print) -> dict:
    _check(model)
    inputs = _events(model, trajectories, say)
    events = []
    say(f"  step 1: per-part projections on {len(inputs)} events")
    for tr, before, after, z0, _z1, s0, arm, sign in inputs:
        m = measure_event(model, z0, s0, arm, sign)
        events.append({"clip_id": tr.clip_id, "episode": tr.episode, "before": before, "after": after, "closing": sign > 0,
                       "fold": fold_of(tr.episode, seed), **m})
    picks = {f: pick([e for e in events if e["fold"] == f], k_comp=k_comp, k_head=k_head, n_rand=n_rand, seed=seed + f) for f in (0, 1)}
    say("  steps 2-3: knockouts and steering with the other fold's picks")
    for i, (ev, (_tr, _b, _a, z0, z1, s0, arm, sign)) in enumerate(zip(events, inputs)):
        p = picks[1 - ev["fold"]]
        if p is None or not p["comps"]:
            continue
        t = knock_and_steer(model, z0, z1, s0, arm, sign, p, seed=seed + i)
        r_peak = ev["grip"][p["peak"]] / max(ev["arm"][p["peak"]], 1e-12)
        r_last = ev["grip"][-1] / max(ev["arm"][-1], 1e-12)
        ev["test"] = {**t, "peak": p["peak"], "r_peak": r_peak, "r_last": r_last, "arm_last": ev["arm"][-1]}
        if (i + 1) % 10 == 0:
            say(f"     tested {i + 1}/{len(events)}")
    return {"events": events, "picks": picks}


def _recovered(rs: list[dict], get) -> float:
    """Share of the peak-to-output drop in the gripper/arm ratio that a knockout brings back. The output
    baseline is the unedited wrong-command row of the same batch, so batch effects cancel."""
    if not rs:
        return float("nan")
    rp = np.mean([r["test"]["r_peak"] for r in rs])
    rl = np.mean([r["test"]["grip_last"] / max(r["test"]["arm_last"], 1e-12) for r in rs])
    rk = np.mean([get(r) / max(r["test"]["arm_last"], 1e-12) for r in rs])
    return float((rk - rl) / (rp - rl)) if rp - rl > 1e-12 else float("nan")


def _name(u, kind: str) -> str:
    b, k = u
    return f"b{b}.{KINDS[k]}" if kind == "comp" else f"b{b}.h{k}"


def summarize(raw: dict, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    events = raw["events"]
    tested = [e for e in events if "test" in e]

    def frac(get):
        return lambda rs: float(np.mean([get(r) for r in rs])) if rs else float("nan")

    knock = {}
    for kind in ("comps", "heads"):
        top = f"top_{kind}"
        rand = f"rand_{kind}"
        n_sets = min((len(e["test"]["knock"][rand]["grip_last"]) for e in tested), default=0)
        rand_vals = [_recovered(tested, lambda r, i=i: r["test"]["knock"][rand]["grip_last"][i]) for i in range(n_sets)]
        knock[kind] = {
            "recovered": _boot(tested, lambda rs: _recovered(rs, lambda r: r["test"]["knock"][top]["grip_last"][0]), n_boot=n_boot, seed=seed),
            "prefers_correct": _boot(tested, frac(lambda r: r["test"]["knock"][top]["prefers_correct"][0]), n_boot=n_boot, seed=seed),
            "random_recovered": rand_vals,
            "random_mean": float(np.mean(rand_vals)) if rand_vals else float("nan"),
            "random_max": float(np.max(rand_vals)) if rand_vals else float("nan"),
            "curve": _nanmean([r["test"]["knock"][top]["curve"] for r in tested]).tolist() if tested else [],
            "random_curve": _nanmean([r["test"]["knock"][rand]["curve"] for r in tested if r["test"]["knock"][rand]["curve"]]).tolist()
            if n_sets else [],
        }
    # Descriptive, not part of any verdict: every attention (or every MLP) output after the peak knocked out at once.
    for kind in ("all_attn", "all_mlp"):
        knock[kind] = {
            "recovered": _boot(tested, lambda rs, kind=kind: _recovered(rs, lambda r: r["test"]["knock"][kind]["grip_last"][0]), n_boot=n_boot, seed=seed),
            "prefers_correct": _boot(tested, frac(lambda r, kind=kind: r["test"]["knock"][kind]["prefers_correct"][0]), n_boot=n_boot, seed=seed),
            "curve": _nanmean([r["test"]["knock"][kind]["curve"] for r in tested]).tolist() if tested else [],
        }
    base = {
        "prefers_correct": _boot(tested, frac(lambda r: r["test"]["prefers_correct"]), n_boot=n_boot, seed=seed),
        "arm_pref": _boot(tested, frac(lambda r: r["test"]["arm_pref"]), n_boot=n_boot, seed=seed),
    }
    steer = {}
    for a in ALPHAS:
        k = str(a)
        steer[k] = {
            "prefers_correct": _boot(tested, frac(lambda r, k=k: r["test"]["steer"][k]["prefers_correct"]), n_boot=n_boot, seed=seed),
            "arm_pref": _boot(tested, frac(lambda r, k=k: r["test"]["steer"][k]["arm_pref"]), n_boot=n_boot, seed=seed),
            "damage": _boot(tested, lambda rs, k=k: float(np.nanmean([r["test"]["steer"][k]["damage"] for r in rs])) if rs else float("nan"),
                            n_boot=n_boot, seed=seed),
        }
    n_dirs = min((len(e["test"]["steer_rand"]) for e in tested), default=0)
    steer_rand = [float(np.mean([e["test"]["steer_rand"][i] for e in tested])) for i in range(n_dirs)]

    attention = {}
    for e in tested:
        for name, m in e["test"]["attention"].items():
            for key, v in m.items():
                attention.setdefault(name, {}).setdefault(key, []).append(v)
    attention = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in attention.items()}

    parts = {}
    if events:
        sg, sa = _nanmean([e["s_grip"] for e in events]), _nanmean([e["s_arm"] for e in events])
        hg, ha = _nanmean([e["sh_grip"] for e in events]), _nanmean([e["sh_arm"] for e in events])
        parts = {"s_grip": sg.tolist(), "s_arm": sa.tolist(), "sh_grip": hg.tolist(), "sh_arm": ha.tolist()}
    picks = {}
    for f, p in raw["picks"].items():
        if p is None:
            picks[str(f)] = None
            continue
        picks[str(f)] = {"peak": p["peak"], "comps": [_name(u, "comp") for u in p["comps"]], "heads": [_name(u, "head") for u in p["heads"]],
                         "comp_scores": p.get("comp_scores", []), "head_scores": p.get("head_scores", [])}
    result = {
        "n_events": len(events),
        "n_episodes": len({e["episode"] for e in events}),
        "n_tested": len(tested),
        "ratio_curve": _ratio_curve(events).tolist() if events else [],
        "parts": parts,
        "picks": picks,
        "knockout": knock,
        "baseline": base,
        "steer": steer,
        "steer_random": steer_rand,
        "curve": _nanmean([e["test"]["curve"] for e in tested]).tolist() if tested else [],
        "baseline_check": {"pass1_grip_last": float(np.mean([e["grip"][-1] for e in tested])) if tested else float("nan"),
                           "pass2_grip_last": float(np.mean([e["test"]["grip_last"] for e in tested])) if tested else float("nan")},
        "attention": attention,
    }
    result["verdict"] = verdicts(result)
    return result


def _eraser_verdict(k: dict) -> str:
    r = k["recovered"]
    if not np.isfinite(r["lo"]) or not k["random_recovered"]:
        return "insufficient_data"
    ok = r["value"] >= RECOVER and r["lo"] > 0 and r["value"] > k["random_max"] and k["random_mean"] < RANDOM_MAX
    return "erasers" if ok else "none"


def verdicts(result: dict) -> dict:
    s = result["steer"].get(str(STEER_ALPHA))
    base = result["baseline"]
    if s is None or not np.isfinite(s["prefers_correct"]["lo"]) or not result["steer_random"]:
        steer = "insufficient_data"
    else:
        ok = (s["prefers_correct"]["lo"] > FIX_LO and s["prefers_correct"]["value"] > max(result["steer_random"])
              and s["arm_pref"]["value"] >= base["arm_pref"]["value"] - ARM_DROP and s["damage"]["value"] <= DAMAGE)
        steer = "fixes_gripper" if ok else "no_fix"
    return {"components": _eraser_verdict(result["knockout"]["comps"]), "heads": _eraser_verdict(result["knockout"]["heads"]), "steer": steer}


def run_gripper_circuit(model: ACWorldModel, trajectories: list[Trajectory], out_dir: Path, *, k_comp: int = K_COMP, k_head: int = K_HEAD,
                        n_rand: int = N_RAND, seed: int = 0, n_boot: int = N_BOOT, say=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = probe(model, trajectories, k_comp=k_comp, k_head=k_head, n_rand=n_rand, seed=seed, say=say)
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "seed": seed,
        "n_boot": n_boot,
        "rules": {"k_comp": k_comp, "k_head": k_head, "n_rand": n_rand, "n_rand_steer": N_RAND_STEER, "alphas": list(ALPHAS),
                  "steer_alpha": STEER_ALPHA, "recover": RECOVER, "random_max": RANDOM_MAX, "fix_lo": FIX_LO, "arm_drop": ARM_DROP,
                  "damage": DAMAGE, "arm_step": ARM_STEP},
        **summarize(raw, n_boot=n_boot, seed=seed),
        "events": raw["events"],
    }
    (out_dir / "gripper_circuit.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
