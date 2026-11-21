"""``ii action-effects``: does an action-conditioned world model move only the arm, or the world too?

Hypothesis H2 of the V-JEPA 2-AC plan: the predictor moves the arm's latent patches in response to
the action but barely moves the objects the arm pushes or carries, so its "world model" is mostly an
arm model. The test runs on recorded DROID robot video with the arm's real poses.

For every transition ``t -> t + 1`` (one model step, 0.25 s at the training rate of 4 fps) the
model gets the true latent ``z_t`` of frame ``t`` and the pose ``s_t``, and predicts ``z_{t+1}``
under four actions:

``real``      the pose change that actually happened (Meta's ``poses_to_diff``)
``zero``      no motion
``shuffled``  a real action from another episode (same scale, wrong content)
``negated``   the real action reversed

Patches are labelled from pixels (``regions.py``): ``arm`` (the moving part of the arm), ``moved``
(changed, at least two patches away from the arm: a pushed object, a drawer, a door) and ``still``
background. Transitions where the segmenter missed the arm in either frame are dropped. These
region rules were fixed after 6- and 10-clip mask checks showed that patches next to the arm are
mostly arm pixels the segmenter missed, and before any full run. Every number is a mean L1 distance over a region's
tokens, the same distance the planner minimises, and is divided by the region's ``copy`` error
``|z_{t+1} - z_t|`` (how much that region really changed), summed over transitions first.

``action benefit`` of a region = (err[zero] - err[real]) / copy: the share of the region's real
change that the model predicts *because of the action*. The test compares it on ``moved`` patches
with ``arm`` patches and with a matched control: the same number of ``still`` patches drawn at random
in the same transition.

Verdict, fixed before any real run (``verdict``), on transitions with at least ``MIN_MOVED`` moved
patches, with 95% intervals from resampling whole episodes:

1. ``no_arm_effect`` when the arm's action benefit is not above 0: the model ignores the action.
2. ``object_aware`` when moved-minus-matched is above 0 and the moved benefit is at least a quarter
   of the arm's.
3. ``arm_simulator`` when the upper end of moved-minus-matched is below a quarter of the arm's.
4. ``mixed`` otherwise.

The same command with ``--random-predictor`` keeps the trained encoder and swaps in an untrained
predictor: the floor any of these numbers must clear. Offline and observational; not a rollout
finding, never touches a scoreboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.adapters.vjepa2ac.model import ACWorldModel, TRAIN_FPS, poses_to_diff, resample_indices
from interpretableintelligence.worldmodel.regions import overlay, transition_regions, write_png

SCHEMA = "action-effects/v1"
VARIANTS = ("real", "zero", "shuffled", "negated")
REGIONS = ("arm", "moved", "matched", "still", "all")
MIN_MOVED = 3
ARM_SHARE = 0.25
N_BOOT = 2000
GRIPPER_CLOSED = 0.5


@dataclass
class Trajectory:
    clip_id: str
    episode: int
    task: str
    frames: np.ndarray  # [T, H, W, 3] uint8 at the model's rate
    states: np.ndarray  # [T, 7]


def _read_video(path: Path) -> tuple[np.ndarray, float]:
    import av

    with av.open(str(path)) as src:
        stream = src.streams.video[0]
        fps = float(stream.average_rate)
        frames = [f.to_ndarray(format="rgb24") for f in src.decode(stream)]
    return np.stack(frames), fps


def load_trajectories(clips_dir: Path, *, max_clips: int | None = None, fps: float = TRAIN_FPS) -> list[Trajectory]:
    """Clips written by ``ii get-clips --with-states``, resampled to ``fps``."""
    clips_dir = Path(clips_dir)
    index = json.loads((clips_dir / "clips.json").read_text())
    out = []
    for entry in index["clips"]:
        if "states" not in entry:
            raise ValueError(f"{entry['file']} has no states: write the clips with `ii get-clips --with-states`")
        frames, source_fps = _read_video(clips_dir / entry["file"])
        states = np.load(clips_dir / entry["states"])
        n = min(len(frames), len(states))
        idx = resample_indices(n, source_fps, fps)
        if len(idx) < 2:
            continue
        out.append(
            Trajectory(
                clip_id=entry["clip_id"],
                episode=int(entry.get("episode", len(out))),
                task=entry.get("task", ""),
                frames=frames[idx],
                states=states[idx].astype(np.float32),
            )
        )
        if max_clips is not None and len(out) >= max_clips:
            break
    return out


def _l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token mean absolute difference: ``[N, P, D]`` to ``[N, P]``."""
    return (a - b).abs().mean(dim=-1)


def measure(
    model: ACWorldModel,
    trajectories: list[Trajectory],
    arm_masks,
    *,
    seed: int = 0,
    previews: Path | None = None,
    n_previews: int = 12,
    say=print,
) -> list[dict]:
    """One record per transition: region sizes and per-region copy / error / effect sums."""
    rng = np.random.default_rng(seed)
    actions = [poses_to_diff(tr.states) for tr in trajectories]
    pool = [(i, t) for i, a in enumerate(actions) for t in range(len(a))]
    records: list[dict] = []
    preview_tiles = []
    preview_from: list[int] = []
    for i, tr in enumerate(trajectories):
        z = model.encode_frames(tr.frames)
        masks = arm_masks(tr.frames)
        regions = transition_regions(tr.frames, masks, crop=model.crop, patch=model.patch)
        act = torch.from_numpy(actions[i])
        states = torch.from_numpy(tr.states[:-1])
        z0, z1 = z[:-1], z[1:]
        others = [p for p in pool if p[0] != i] or pool
        picks = rng.integers(0, len(others), size=len(act))
        shuffled = torch.from_numpy(np.stack([actions[others[k][0]][others[k][1]] for k in picks]))
        preds = {
            "real": model.predict_next(z0, states, act),
            "zero": model.predict_next(z0, states, torch.zeros_like(act)),
            "shuffled": model.predict_next(z0, states, shuffled),
            "negated": model.predict_next(z0, states, -act),
        }
        copy = _l1(z0, z1)
        err = {v: _l1(p, z1) for v, p in preds.items()}
        effect = _l1(preds["real"], preds["zero"])
        for t, reg in enumerate(regions):
            still_idx = np.flatnonzero(reg.still)
            n_moved = int(reg.moved.sum())
            matched = np.zeros_like(reg.still)
            if n_moved and len(still_idx) >= n_moved:
                matched[rng.choice(still_idx, size=n_moved, replace=False)] = True
            masks_t = {"arm": reg.arm, "moved": reg.moved, "matched": matched, "still": reg.still, "all": np.ones_like(reg.arm)}
            rec = {
                "clip_id": tr.clip_id,
                "episode": tr.episode,
                "t": t,
                "valid": bool(reg.valid),
                "gripper": float(tr.states[t, 6]),
                "action_xyz_m": float(np.linalg.norm(actions[i][t, :3])),
            }
            for name, m in masks_t.items():
                sel = torch.from_numpy(np.asarray(m, dtype=bool))
                rec[f"n_{name}"] = int(sel.sum())
                if not sel.any():
                    continue
                rec[f"copy_{name}"] = float(copy[t, sel].sum())
                for v in VARIANTS:
                    rec[f"err_{v}_{name}"] = float(err[v][t, sel].sum())
                rec[f"effect_{name}"] = float(effect[t, sel].sum())
            records.append(rec)
            # Up to two tiles per clip, from different clips, so the overlay shows many scenes.
            if (
                previews is not None
                and len(preview_tiles) < n_previews
                and n_moved >= MIN_MOVED
                and reg.valid
                and sum(1 for c in preview_from if c == i) < 2
                and t % 8 == 0
            ):
                preview_tiles.append(overlay(tr.frames[t], reg, crop=model.crop, patch=model.patch))
                preview_from.append(i)
        say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: {len(regions)} transitions")
    if previews is not None and preview_tiles:
        rows = [preview_tiles[k : k + 6] for k in range(0, len(preview_tiles), 6)]
        blank = np.zeros_like(preview_tiles[0])
        rows = [np.concatenate(r + [blank] * (6 - len(r)), axis=1) for r in rows]
        write_png(previews, np.concatenate(rows, axis=0))
    return records


def _ratio(recs: list[dict], num: list[tuple[str, float]], den: str) -> float:
    d = sum(r.get(den, 0.0) for r in recs)
    if d <= 0:
        return float("nan")
    return sum(sign * r.get(key, 0.0) for r in recs for key, sign in num) / d


def _metrics(recs: list[dict]) -> dict:
    out = {}
    for name in REGIONS:
        have = [r for r in recs if f"copy_{name}" in r]
        c = f"copy_{name}"
        out[f"benefit_{name}"] = _ratio(have, [(f"err_zero_{name}", 1.0), (f"err_real_{name}", -1.0)], c)
        out[f"specific_{name}"] = _ratio(have, [(f"err_shuffled_{name}", 1.0), (f"err_real_{name}", -1.0)], c)
        out[f"gain_{name}"] = 1.0 - _ratio(have, [(f"err_real_{name}", 1.0)], c)
        out[f"effect_{name}"] = _ratio(have, [(f"effect_{name}", 1.0)], c)
    out["moved_minus_matched"] = out["benefit_moved"] - out["benefit_matched"]
    return out


def _bootstrap(recs: list[dict], *, n_boot: int, seed: int) -> dict:
    by_ep: dict[int, list[dict]] = {}
    for r in recs:
        by_ep.setdefault(r["episode"], []).append(r)
    episodes = sorted(by_ep)
    point = _metrics(recs)
    if len(episodes) < 2:
        return {k: {"value": v, "lo": float("nan"), "hi": float("nan")} for k, v in point.items()}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(episodes), size=len(episodes))
        draws.append(_metrics([r for k in pick for r in by_ep[episodes[k]]]))
    out = {}
    for k, v in point.items():
        vals = np.array([d[k] for d in draws], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))
        out[k] = {"value": v, "lo": float(lo), "hi": float(hi)}
    return out


def verdict(stats: dict) -> str:
    """The pre-registered call (module docstring) from bootstrapped metrics on moved transitions."""
    arm = stats["benefit_arm"]
    diff = stats["moved_minus_matched"]
    moved = stats["benefit_moved"]
    if not np.isfinite(arm["lo"]) or not np.isfinite(diff["hi"]):
        return "insufficient_data"
    if arm["lo"] <= 0:
        return "no_arm_effect"
    if diff["lo"] > 0 and moved["value"] >= ARM_SHARE * arm["value"]:
        return "object_aware"
    if diff["hi"] < ARM_SHARE * arm["value"]:
        return "arm_simulator"
    return "mixed"


def summarize(records: list[dict], *, n_boot: int = N_BOOT, seed: int = 0, min_moved: int = MIN_MOVED) -> dict:
    valid = [r for r in records if r.get("valid", True)]
    moving = [r for r in valid if r["n_moved"] >= min_moved and r["n_still"] >= r["n_moved"] and r["n_arm"] > 0]
    stats = _bootstrap(moving, n_boot=n_boot, seed=seed)
    closed = [r for r in moving if r["gripper"] >= GRIPPER_CLOSED]
    opened = [r for r in moving if r["gripper"] < GRIPPER_CLOSED]
    return {
        "n_transitions": len(records),
        "n_episodes": len({r["episode"] for r in records}),
        "n_valid": len(valid),
        "moving": {
            "n_transitions": len(moving),
            "n_episodes": len({r["episode"] for r in moving}),
            "stats": stats,
            "verdict": verdict(stats),
        },
        "all_transitions": _metrics([r for r in valid if r["n_arm"] > 0]),
        "by_gripper": {
            "closed": {"n_transitions": len(closed), **_metrics(closed)} if closed else None,
            "open": {"n_transitions": len(opened), **_metrics(opened)} if opened else None,
        },
    }


def run_action_effects(
    model: ACWorldModel,
    trajectories: list[Trajectory],
    arm_masks,
    out_dir: Path,
    *,
    seed: int = 0,
    n_boot: int = N_BOOT,
    say=print,
    extra: dict | None = None,
) -> dict:
    """Measure, summarise and write ``action_effects.json``, ``transitions.parquet`` and ``regions.png``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = measure(model, trajectories, arm_masks, seed=seed, previews=out_dir / "regions.png", say=say)
    summary = summarize(records, n_boot=n_boot, seed=seed)
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "fps": TRAIN_FPS,
        "clips": [tr.clip_id for tr in trajectories],
        "seed": seed,
        "n_boot": n_boot,
        "rules": {"min_moved": MIN_MOVED, "arm_share": ARM_SHARE},
        **summary,
        **(extra or {}),
    }
    keys = sorted({k for r in records for k in r})
    table = pa.table({k: [r.get(k) for r in records] for k in keys})
    pq.write_table(table, out_dir / "transitions.parquet")
    (out_dir / "action_effects.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result
