"""``ii gripper-trace``: where does the gripper command get lost inside V-JEPA 2-AC?

``ii gripper-probe`` found the goal score blind to the gripper. This command looks inside the model
for the reason, in three steps on the same gripper events (moments where the gripper opens or closes
while the arm stays still):

1. **Encoder readout.** A logistic probe on the frozen encoder's pooled latents (mean and max over
   tokens) reads gripper open or closed on every frame. Two folds split by episode: each fold's
   probe is scored only on the other fold's frames. A probe on 16x16 pixel thumbnails is the
   baseline. If the encoder carries the gripper, the information is there for the predictor to use.
2. **Predictor trace.** For each event the predictor runs under the right and the wrong gripper
   command (arm step fixed at the real one) and under two arm steps 6 cm apart (gripper at 0). At the
   input and after each block it records how far the two runs drift apart on the action token and on
   the frame tokens, relative to the residual's size. Activation patching then copies, at one layer,
   either the action token or the frame tokens from the right-command run into the wrong-command run
   and measures how much of the final prediction gap that recovers: where the gripper information sits
   at each depth, and whether it ever reaches the frame.
3. **Prediction readout.** The step-1 probe (from the fold that did not see the event) reads the
   predicted next latent under each command. If the predictor carried the command, the probe should
   move toward "closed" under the close command about as much as it moves between the real frames
   before and after the event (``propagation``).

Verdicts are fixed before any real run, with 95% intervals from resampling whole episodes:
the encoder ``encodes_gripper`` when the probe AUROC's lower end is above ``ENCODER_AUROC`` (0.8);
the command is ``dropped`` when the upper end of ``propagation`` is below ``DROPPED`` (0.25),
``carried`` when its lower end is above ``CARRIED`` (0.5), ``partial`` otherwise. Run it again with
``--random-predictor`` for the untrained floor. Offline, on recorded video; not a rollout finding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel
from interpretableintelligence.probe import auroc, fit_logistic
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.gripper_probe import find_events
from interpretableintelligence.worldmodel.plan_probe import GRIP_MAX, MAXNORM, _boot

SCHEMA = "gripper-trace/v1"
CLOSED = 0.5  # gripper state above this counts as closed
L2 = 10.0
THUMB = 16
ARM_STEP = 0.03  # the arm contrast: +3 cm vs -3 cm along x
COND = 2  # action and state tokens ahead of each frame's tokens
ENCODER_AUROC = 0.8
DROPPED = 0.25
CARRIED = 0.5
N_BOOT = 2000


# --------------------------------------------------------------------------- step 1: encoder readout


def pool(z: torch.Tensor) -> np.ndarray:
    """``[N, P, D]`` latents to ``[N, 2D]``: mean and max over tokens."""
    z = z.float()
    return torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=-1).numpy().astype(np.float64)


def thumbnails(frames: np.ndarray, size: int = THUMB) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    return F.adaptive_avg_pool2d(x, size).flatten(1).numpy().astype(np.float64)


def fold_of(episode: int, seed: int) -> int:
    return hashlib.sha256(f"gripper-trace:{seed}:{episode}".encode()).digest()[0] % 2


class Readout:
    def __init__(self, x: np.ndarray, y: np.ndarray, l2: float = L2) -> None:
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        self.w, self.b = fit_logistic((x - self.mean) / self.std, y, l2)

    def logit(self, x: np.ndarray) -> np.ndarray:
        return ((np.atleast_2d(x) - self.mean) / self.std) @ self.w + self.b


def _auroc_boot(scores: np.ndarray, labels: np.ndarray, episodes: np.ndarray, *, n_boot: int, seed: int) -> dict:
    def stat(idx):
        y = labels[idx]
        if y.min() == y.max():
            return float("nan")
        return auroc(scores[idx][y == 1], scores[idx][y == 0])

    eps = np.unique(episodes)
    by_ep = {e: np.flatnonzero(episodes == e) for e in eps}
    value = stat(np.arange(len(labels)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(eps), size=len(eps))
        vals.append(stat(np.concatenate([by_ep[eps[k]] for k in pick])))
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
    return {"value": float(value), "lo": float(lo), "hi": float(hi)}


def encoder_readout(latents: list[torch.Tensor], trajectories: list[Trajectory], *, seed: int, n_boot: int):
    """Two-fold probes on pooled encoder latents and on pixel thumbnails. Returns the summary and the
    latent probe per fold (fold ``k``'s probe was trained on the other fold)."""
    x_enc = np.concatenate([pool(z) for z in latents])
    x_pix = np.concatenate([thumbnails(tr.frames) for tr in trajectories])
    y = np.concatenate([(tr.states[:, 6] > CLOSED).astype(np.float64) for tr in trajectories])
    ep = np.concatenate([np.full(len(tr.states), tr.episode) for tr in trajectories])
    folds = np.array([fold_of(int(e), seed) for e in ep])
    out = {"n_frames": int(len(y)), "closed_share": float(y.mean()) if len(y) else float("nan")}
    probes: dict[int, Readout | None] = {0: None, 1: None}
    for name, x in (("encoder", x_enc), ("pixels", x_pix)):
        scores = np.full(len(y), np.nan)
        for k in (0, 1):
            train, test = folds != k, folds == k
            if not test.any() or y[train].min() == y[train].max():
                continue
            r = Readout(x[train], y[train])
            scores[test] = r.logit(x[test])
            if name == "encoder":
                probes[k] = r
        ok = np.isfinite(scores)
        if ok.sum() and y[ok].min() != y[ok].max():
            out[name] = _auroc_boot(scores[ok], y[ok], ep[ok], n_boot=n_boot, seed=seed)
        else:
            out[name] = {"value": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return out, probes


# --------------------------------------------------------------------------- step 2: predictor trace


def _blocks(model: ACWorldModel):
    blocks = getattr(model.predictor, "predictor_blocks", None)
    if blocks is None:
        raise ValueError(f"{model.name} has no predictor_blocks to trace")
    return list(blocks)


@torch.inference_mode()
def trace_event(model: ACWorldModel, z: torch.Tensor, state: np.ndarray, arm: np.ndarray, sign: float) -> dict:
    """One forward pass, batched: rows 0-3 are the right command, the wrong one, arm +x and arm -x;
    then for each layer (input and every block) the wrong command with the action token patched in
    from row 0, then the same with the frame tokens patched in."""
    blocks = _blocks(model)
    n_layers = len(blocks) + 1  # the input, then each block's output
    acts = np.zeros((4, 7), dtype=np.float32)
    acts[:, :3] = arm
    acts[0, 6] = sign * GRIP_MAX
    acts[1, 6] = -sign * GRIP_MAX
    acts[2, 0] += ARM_STEP
    acts[3, 0] -= ARM_STEP
    rows = np.concatenate([acts, np.repeat(acts[1:2], 2 * n_layers, axis=0)])
    patch_layer = [-1] * 4 + list(range(n_layers)) * 2
    patch_kind = [None] * 4 + ["action"] * n_layers + ["frame"] * n_layers
    n = len(rows)
    p = model.tokens_per_frame

    drift = {k: np.zeros(n_layers) for k in ("grip_action", "grip_frame", "arm_action", "arm_frame")}

    def visit(layer: int, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        size_a = h[:2, 0].norm(dim=-1).mean().clamp_min(1e-9)
        size_f = h[:2, COND:].norm(dim=-1).mean().clamp_min(1e-9)
        drift["grip_action"][layer] = float((h[0, 0] - h[1, 0]).norm() / size_a)
        drift["grip_frame"][layer] = float((h[0, COND:] - h[1, COND:]).norm(dim=-1).mean() / size_f)
        drift["arm_action"][layer] = float((h[2, 0] - h[3, 0]).norm() / size_a)
        drift["arm_frame"][layer] = float((h[2, COND:] - h[3, COND:]).norm(dim=-1).mean() / size_f)
        x = x.clone()
        for r in range(4, n):
            if patch_layer[r] == layer:
                if patch_kind[r] == "action":
                    x[r, :COND] = x[0, :COND]
                else:
                    x[r, COND:] = x[0, COND:]
        return x

    def pre(_mod, args, kwargs):
        return (visit(0, args[0]),) + tuple(args[1:]), kwargs

    handles = [blocks[0].register_forward_pre_hook(pre, with_kwargs=True)]
    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(lambda _m, _a, out, i=i: visit(i + 1, out)))
    try:
        zd = z.to(model.device).unsqueeze(0).expand(n, -1, -1)
        sd = torch.from_numpy(np.repeat(state[None], n, axis=0)).to(model.device, torch.float32).unsqueeze(1)
        ad = torch.from_numpy(rows).to(model.device, torch.float32).unsqueeze(1)
        with model._autocast():
            y = model.predictor(zd, ad, sd)[:, -p:]
        y = F.layer_norm(y.float(), (y.shape[-1],)).cpu()
    finally:
        for h in handles:
            h.remove()
    gap = float((y[1] - y[0]).norm())
    rec = {}
    for kind, start in (("action", 4), ("frame", 4 + n_layers)):
        left = (y[start : start + n_layers] - y[0]).flatten(1).norm(dim=-1).numpy()
        rec[kind] = (1.0 - left / gap) if gap > 0 else np.full(n_layers, np.nan)
    return {
        "drift": {k: v.tolist() for k, v in drift.items()},
        "recovered_action": rec["action"].tolist(),
        "recovered_frame": rec["frame"].tolist(),
        "out_grip": float((y[0] - y[1]).norm(dim=-1).mean()),
        "out_arm": float((y[2] - y[3]).norm(dim=-1).mean()),
        "pred_right": y[0],
        "pred_wrong": y[1],
    }


# --------------------------------------------------------------------------- run


def probe(model: ACWorldModel, trajectories: list[Trajectory], *, seed: int = 0, n_boot: int = N_BOOT, say=print) -> dict:
    latents = []
    for i, tr in enumerate(trajectories):
        latents.append(model.encode_frames(tr.frames))
        say(f"     encoded {i + 1}/{len(trajectories)} {tr.clip_id}")
    readout, probes = encoder_readout(latents, trajectories, seed=seed, n_boot=n_boot)
    say(f"     encoder readout AUROC {readout['encoder']['value']:.3f}, pixels {readout['pixels']['value']:.3f}")

    events = []
    for tr, z in zip(trajectories, latents):
        for before, after in find_events(tr.states):
            s0, s1 = tr.states[before], tr.states[after]
            sign = 1.0 if s1[6] > s0[6] else -1.0
            arm = np.clip(s1[:3] - s0[:3], -MAXNORM, MAXNORM)
            t = trace_event(model, z[before], s0, arm, sign)
            rec = {"clip_id": tr.clip_id, "episode": tr.episode, "before": before, "after": after, "closing": sign > 0,
                   **{k: v for k, v in t.items() if not k.startswith("pred_")}}
            r = probes[fold_of(tr.episode, seed)]
            if r is not None:
                lz = r.logit(pool(z[[before, after]]))
                lp = r.logit(pool(torch.stack([t["pred_right"], t["pred_wrong"]])))
                rec["true_shift"] = float(sign * (lz[1] - lz[0]))
                rec["pred_shift"] = float(sign * (lp[0] - lp[1]))
            events.append(rec)
        say(f"     traced {tr.clip_id}: {len(events)} events so far")
    return {"readout": readout, "events": events}


def _curve(events: list[dict], get) -> list[float]:
    return np.nanmean(np.stack([get(e) for e in events]), axis=0).tolist() if events else []


def verdicts(readout: dict, prop: dict) -> dict:
    enc = readout["encoder"]
    encoder = "insufficient_data" if not np.isfinite(enc["lo"]) else ("encodes_gripper" if enc["lo"] > ENCODER_AUROC else "weak")
    if not np.isfinite(prop["lo"]):
        command = "insufficient_data"
    elif prop["hi"] < DROPPED:
        command = "dropped"
    elif prop["lo"] > CARRIED:
        command = "carried"
    else:
        command = "partial"
    return {"encoder": encoder, "command": command}


def summarize(result: dict, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    events = result["events"]
    read = [e for e in events if "pred_shift" in e]

    def propagation(rs):
        if not rs:
            return float("nan")
        true = np.mean([r["true_shift"] for r in rs])
        return float(np.mean([r["pred_shift"] for r in rs]) / true) if true > 0 else float("nan")

    stats = {
        "propagation": _boot(read, propagation, n_boot=n_boot, seed=seed),
        "true_shift": _boot(read, lambda rs: float(np.mean([r["true_shift"] for r in rs])) if rs else float("nan"), n_boot=n_boot, seed=seed),
        "pred_shift": _boot(read, lambda rs: float(np.mean([r["pred_shift"] for r in rs])) if rs else float("nan"), n_boot=n_boot, seed=seed),
        "out_grip_over_arm": _boot(events, lambda rs: float(np.mean([r["out_grip"] for r in rs]) / np.mean([r["out_arm"] for r in rs])) if rs else float("nan"), n_boot=n_boot, seed=seed),
    }
    layers = {
        "grip_action": _curve(events, lambda e: e["drift"]["grip_action"]),
        "grip_frame": _curve(events, lambda e: e["drift"]["grip_frame"]),
        "arm_action": _curve(events, lambda e: e["drift"]["arm_action"]),
        "arm_frame": _curve(events, lambda e: e["drift"]["arm_frame"]),
        "recovered_action": _curve(events, lambda e: e["recovered_action"]),
        "recovered_frame": _curve(events, lambda e: e["recovered_frame"]),
    }
    if events:
        layers["grip_over_arm_frame"] = (np.asarray(layers["grip_frame"]) / np.maximum(np.asarray(layers["arm_frame"]), 1e-12)).tolist()
    return {
        "n_events": len(events),
        "n_episodes": len({e["episode"] for e in events}),
        "readout": result["readout"],
        "stats": stats,
        "layers": layers,
        "verdict": verdicts(result["readout"], stats["propagation"]),
    }


def run_gripper_trace(model: ACWorldModel, trajectories: list[Trajectory], out_dir: Path, *, seed: int = 0, n_boot: int = N_BOOT, say=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = probe(model, trajectories, seed=seed, n_boot=n_boot, say=say)
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "seed": seed,
        "n_boot": n_boot,
        "rules": {"closed": CLOSED, "l2": L2, "thumb": THUMB, "arm_step": ARM_STEP, "encoder_auroc": ENCODER_AUROC,
                  "dropped": DROPPED, "carried": CARRIED},
        **summarize(raw, n_boot=n_boot, seed=seed),
        "events": raw["events"],
    }
    (out_dir / "gripper_trace.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
