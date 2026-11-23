"""``ii impossible-events``: which impossible next frames does V-JEPA 2-AC notice?

A broad screen, run before any mechanism hunt. For every transition ``t -> t + 1`` in recorded DROID
video where the arm really moves, the predictor gets the true latent ``z_t``, the pose and the real
action, and predicts ``z_{t+1}`` once. That one prediction is then compared with the real next frame
and with five impossible ones built from the same video:

``freeze``        the next frame is the current frame: the world stops while the action says the arm moved
``local_freeze``  the 3 x 3-patch square that changes most is left as it was at ``t``: the moving thing stops
``reverse``       the next frame is the previous one: time runs backwards under a forward action
``swap``          two 3 x 3-patch squares of the next frame trade places: things teleport
``splice``        the next frame comes from another episode: the scene jumps

``surprise`` is the planner's own distance, mean L1 over latent tokens, between the prediction and a
candidate next frame; local edits are scored on their edited patches only. An edit is *detected* on
a transition when its surprise is above the real frame's. Three detectors read the same frames:

``model``        V-JEPA 2-AC with the real action
``zero_action``  the same predictor with a zero action (does catching the edit need the action?)
``copy``         no model at all: the current latent as the prediction (raw visual change)

``freeze`` and ``local_freeze`` are the sharp tests: the edited frame is *closer* to the current one
than the real next frame, so ``copy`` scores them as unsurprising and only a model of the motion can
flag them. ``splice`` and ``swap`` are big visual changes that any detector should catch; they check
the pipeline and set the scale.

Verdict per edit, fixed before any real run, on the share of transitions detected with 95% intervals
from resampling whole episodes:

1. ``caught`` when the model's lower end is above ``CATCH`` (0.75) and the model beats ``copy``
   (lower end of model minus copy above 0): it notices the edit beyond raw visual change.
2. ``trivial`` when the model's lower end is above ``CATCH`` but it does not beat ``copy``.
3. ``missed`` when the model's upper end is below ``MISS`` (0.6).
4. ``mixed`` otherwise.

``--context K`` feeds the last ``K`` real frames instead of one (Meta trained the predictor
teacher-forced on 8-frame clips, so it saw at most 7 frames of context); ``--start`` fixes the first
transition so runs with different contexts score the same transitions.

The same command with ``--random-predictor`` keeps the trained encoder and swaps in an untrained
predictor: the floor for ``model``. Offline, on recorded video; not a rollout finding and never
touches a scoreboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel, poses_to_diff
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.plan_probe import _boot
from interpretableintelligence.worldmodel.regions import write_png

SCHEMA = "impossible-events/v1"
EDITS = ("freeze", "local_freeze", "reverse", "swap", "splice")
LOCAL = ("local_freeze", "swap")
DETECTORS = ("model", "zero_action", "copy")
SQUARE = 3  # patches per side of a local edit
MIN_MOVE_M = 0.01  # arm translation per model step for a transition to count
MIN_SWAP_DIFF = 12.0  # mean absolute pixel difference (0-255) the two swapped squares must have
SWAP_DRAWS = 24
CATCH = 0.75
MISS = 0.6
N_BOOT = 2000
N_TILES = 6


def resize_frames(frames: np.ndarray, crop: int) -> np.ndarray:
    """``[N, H, W, 3]`` uint8 to ``[N, crop, crop, 3]`` uint8, the same squash the model's preprocessing does."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(crop, crop), mode="bilinear", align_corners=False, antialias=True)
    return x.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def _square_slices(row: int, col: int, patch: int) -> tuple[slice, slice]:
    return slice(row * patch, (row + SQUARE) * patch), slice(col * patch, (col + SQUARE) * patch)


def _square_tokens(row: int, col: int, side: int) -> np.ndarray:
    return np.array([(row + r) * side + col + c for r in range(SQUARE) for c in range(SQUARE)], dtype=np.int64)


def local_freeze(prev: np.ndarray, nxt: np.ndarray, patch: int) -> tuple[np.ndarray, np.ndarray]:
    """``nxt`` with its most-changed ``SQUARE`` x ``SQUARE`` patch square put back as it was in ``prev``.

    Returns the edited frame and the square's ``(row, col)`` in patches.
    """
    side = nxt.shape[0] // patch
    diff = np.abs(nxt.astype(np.float32) - prev.astype(np.float32)).mean(-1)
    per_patch = diff.reshape(side, patch, side, patch).mean(axis=(1, 3))
    best, where = -1.0, (0, 0)
    for r in range(side - SQUARE + 1):
        for c in range(side - SQUARE + 1):
            v = float(per_patch[r : r + SQUARE, c : c + SQUARE].sum())
            if v > best:
                best, where = v, (r, c)
    out = nxt.copy()
    ys, xs = _square_slices(*where, patch)
    out[ys, xs] = prev[ys, xs]
    return out, np.array(where)


def swap_squares(frame: np.ndarray, patch: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """``frame`` with two non-touching patch squares traded. Of ``SWAP_DRAWS`` random pairs, the one whose
    contents differ most; ``None`` when no pair differs by ``MIN_SWAP_DIFF``. Returns the frame and
    ``[[r1, c1], [r2, c2]]``."""
    side = frame.shape[0] // patch
    hi = side - SQUARE + 1
    if hi < 1 or 2 * SQUARE + 1 > side:
        return None
    best, pair = -1.0, None
    f = frame.astype(np.float32)
    for _ in range(SWAP_DRAWS):
        a = rng.integers(0, hi, size=2)
        b = rng.integers(0, hi, size=2)
        if max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= SQUARE:
            continue  # overlapping or touching
        ya, xa = _square_slices(int(a[0]), int(a[1]), patch)
        yb, xb = _square_slices(int(b[0]), int(b[1]), patch)
        d = float(np.abs(f[ya, xa] - f[yb, xb]).mean())
        if d > best:
            best, pair = d, (a, b)
    if pair is None or best < MIN_SWAP_DIFF:
        return None
    a, b = pair
    out = frame.copy()
    ya, xa = _square_slices(int(a[0]), int(a[1]), patch)
    yb, xb = _square_slices(int(b[0]), int(b[1]), patch)
    out[ya, xa], out[yb, xb] = frame[yb, xb], frame[ya, xa]
    return out, np.stack([a, b])


def _l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``[P, D]`` pair to per-token mean absolute difference ``[P]``."""
    return (a - b).abs().mean(dim=-1)


def _heat_tile(frame: np.ndarray, gain: np.ndarray, side: int, patch: int, box: list[tuple[int, int]]) -> np.ndarray:
    """The edited frame beside a red map of per-patch surprise gain (edit minus real), edited squares outlined."""
    g = np.clip(gain / (np.abs(gain).max() + 1e-9), 0, 1).reshape(side, side)
    heat = np.kron(g, np.ones((patch, patch), np.float32))[..., None]
    base = frame.astype(np.float32) * 0.35
    red = np.clip(base + heat * np.array([255.0, 40.0, 40.0]), 0, 255)
    left = frame.copy()
    for r, c in box:
        ys, xs = _square_slices(r, c, patch)
        for img in (left, red):
            img[ys.start, xs] = img[ys.stop - 1, xs] = (255, 255, 0)
            img[ys, xs.start] = img[ys, xs.stop - 1] = (255, 255, 0)
    return np.concatenate([left, red.astype(np.uint8)], axis=1)


def screen(
    model: ACWorldModel,
    trajectories: list[Trajectory],
    *,
    seed: int = 0,
    context: int = 1,
    start: int = 1,
    heatmaps: Path | None = None,
    say=print,
) -> list[dict]:
    """One record per transition with the surprise of the real frame and of every edit, per detector.

    ``context`` real frames (``t - context + 1 .. t``, with their poses and actions) go into the
    predictor; edits only ever touch the predicted frame. Transitions start at ``start``, which must
    leave room for the context, so runs with different contexts can share one set of transitions.
    """
    if context < 1:
        raise ValueError("context must be at least 1 frame")
    start = max(start, context - 1, 1)
    rng = np.random.default_rng(seed)
    crop, patch = model.crop, model.patch
    side = crop // patch
    records: list[dict] = []
    tiles: dict[str, list[np.ndarray]] = {e: [] for e in LOCAL}
    prev_z: torch.Tensor | None = None
    for i, tr in enumerate(trajectories):
        frames = resize_frames(tr.frames, crop)
        actions = poses_to_diff(tr.states)
        moving = [t for t in range(start, len(actions)) if np.linalg.norm(actions[t, :3]) >= MIN_MOVE_M]
        z = model.encode_frames(frames)
        if not moving:
            prev_z = z
            say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: no moving transitions")
            continue
        edited, where_lf, where_sw, has_swap = [], [], [], []
        for t in moving:
            lf, w = local_freeze(frames[t], frames[t + 1], patch)
            edited.append(lf)
            where_lf.append(w)
            sw = swap_squares(frames[t + 1], patch, rng)
            has_swap.append(sw is not None)
            if sw is not None:
                edited.append(sw[0])
                where_sw.append(sw[1])
            else:
                where_sw.append(None)
        z_edit = model.encode_frames(np.stack(edited))
        win = torch.tensor([[t - context + 1 + c for c in range(context)] for t in moving])
        states = torch.from_numpy(tr.states[win.numpy()])
        act = torch.from_numpy(actions[win.numpy()])
        act0 = act.clone()
        act0[:, -1] = 0.0  # only the step being predicted loses its action
        pred = model.predict_next(z[win], states, act)
        pred0 = model.predict_next(z[win], states, act0)
        k = 0
        for j, t in enumerate(moving):
            z_lf = z_edit[k]
            k += 1
            z_sw = None
            if has_swap[j]:
                z_sw = z_edit[k]
                k += 1
            targets = {"real": z[t + 1], "freeze": z[t], "local_freeze": z_lf, "reverse": z[t - 1]}
            if z_sw is not None:
                targets["swap"] = z_sw
            if prev_z is not None:
                targets["splice"] = prev_z[min(t + 1, len(prev_z) - 1)]
            regions = {"local_freeze": _square_tokens(*where_lf[j], side)}
            if z_sw is not None:
                regions["swap"] = np.concatenate([_square_tokens(*where_sw[j][0], side), _square_tokens(*where_sw[j][1], side)])
            guesses = {"model": pred[j], "zero_action": pred0[j], "copy": z[t]}
            rec = {
                "clip_id": tr.clip_id,
                "episode": tr.episode,
                "t": int(t),
                "arm_step_m": float(np.linalg.norm(actions[t, :3])),
                "edits": [e for e in EDITS if e in targets],
            }
            for det, guess in guesses.items():
                per_tok = {name: _l1(guess, target) for name, target in targets.items()}
                for e in EDITS:
                    if e not in targets:
                        continue
                    sel = regions.get(e)
                    real = per_tok["real"] if sel is None else per_tok["real"][sel]
                    ed = per_tok[e] if sel is None else per_tok[e][sel]
                    rec[f"{det}_{e}_real"] = float(real.mean())
                    rec[f"{det}_{e}_edit"] = float(ed.mean())
                    rec[f"{det}_{e}_detected"] = bool(ed.mean() > real.mean())
            records.append(rec)
            if heatmaps is not None:
                gains = {"local_freeze": (_l1(pred[j], z_lf) - _l1(pred[j], z[t + 1])).numpy()}
                boxes = {"local_freeze": [tuple(where_lf[j])]}
                imgs = {"local_freeze": edited[k - 1 - int(z_sw is not None)]}
                if z_sw is not None:
                    gains["swap"] = (_l1(pred[j], z_sw) - _l1(pred[j], z[t + 1])).numpy()
                    boxes["swap"] = [tuple(where_sw[j][0]), tuple(where_sw[j][1])]
                    imgs["swap"] = edited[k - 1]
                for e in gains:
                    if len(tiles[e]) < N_TILES and j == len(moving) // 2:
                        tiles[e].append(_heat_tile(imgs[e], gains[e], side, patch, boxes[e]))
        prev_z = z
        say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: {len(moving)} moving transitions, {len(records)} so far")
    if heatmaps is not None:
        heatmaps.mkdir(parents=True, exist_ok=True)
        for e, ts in tiles.items():
            if ts:
                write_png(heatmaps / f"{e}.png", np.concatenate(ts, axis=0))
    return records


def verdict(model: dict, beats_copy: dict) -> str:
    if not np.isfinite(model["lo"]):
        return "insufficient_data"
    if model["lo"] > CATCH:
        return "caught" if beats_copy["lo"] > 0 else "trivial"
    if model["hi"] < MISS:
        return "missed"
    return "mixed"


def summarize(records: list[dict], *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    def frac(key):
        def f(rs):
            vals = [r[key] for r in rs if key in r]
            return float(np.mean(vals)) if vals else float("nan")

        return f

    def diff(a, b):
        return lambda rs: frac(a)(rs) - frac(b)(rs)

    def ratio(det, e):
        def f(rs):
            num = sum(r[f"{det}_{e}_edit"] for r in rs if f"{det}_{e}_edit" in r)
            den = sum(r[f"{det}_{e}_real"] for r in rs if f"{det}_{e}_real" in r)
            return num / den if den else float("nan")

        return f

    edits = {}
    for e in EDITS:
        rs = [r for r in records if e in r["edits"]]
        stats = {det: _boot(rs, frac(f"{det}_{e}_detected"), n_boot=n_boot, seed=seed) for det in DETECTORS}
        beats_copy = _boot(rs, diff(f"model_{e}_detected", f"copy_{e}_detected"), n_boot=n_boot, seed=seed)
        needs_action = _boot(rs, diff(f"model_{e}_detected", f"zero_action_{e}_detected"), n_boot=n_boot, seed=seed)
        edits[e] = {
            "n_transitions": len(rs),
            "n_episodes": len({r["episode"] for r in rs}),
            "detected": stats,
            "model_minus_copy": beats_copy,
            "model_minus_zero_action": needs_action,
            "surprise_ratio": {det: ratio(det, e)(rs) for det in DETECTORS},
            "verdict": verdict(stats["model"], beats_copy),
        }
    return {
        "n_transitions": len(records),
        "n_episodes": len({r["episode"] for r in records}),
        "edits": edits,
        "verdicts": {e: v["verdict"] for e, v in edits.items()},
    }


def run_impossible_events(
    model: ACWorldModel,
    trajectories: list[Trajectory],
    out_dir: Path,
    *,
    seed: int = 0,
    context: int = 1,
    start: int = 1,
    n_boot: int = N_BOOT,
    say=print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = screen(model, trajectories, seed=seed, context=context, start=start, heatmaps=out_dir / "heatmaps", say=say)
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "seed": seed,
        "n_boot": n_boot,
        "context": context,
        "start": max(start, context - 1, 1),
        "rules": {"square_patches": SQUARE, "min_move_m": MIN_MOVE_M, "min_swap_diff": MIN_SWAP_DIFF, "catch": CATCH, "miss": MISS},
        **summarize(records, n_boot=n_boot, seed=seed),
        "transitions": records,
    }
    (out_dir / "impossible_events.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
