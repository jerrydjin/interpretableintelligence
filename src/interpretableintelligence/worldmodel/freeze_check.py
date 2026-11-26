"""``ii freeze-check``: does a world model expect the world to stop?

One question, one fixed rule, any action-conditioned latent world model. For every step ``t -> t + 1``
of recorded robot video where the arm really moves, the model gets the current frame's latent, the
pose and the real action, and predicts the next latent once. Three numbers come out of that one
prediction:

``frozen_wins``       the frozen world (the current latent, unchanged) is closer to the prediction
                      than the real next latent is (mean L1 over tokens, the planners' own distance)
``change_predicted``  how far the prediction moved from now toward what really happened, projected on
                      the real change: 0 = "nothing moves", 1 = "exactly what happened"
``moving_noticed``    the most-changed 3 x 3-patch square of the real next frame, put back as it was:
                      is the prediction further from that frame than from the real one, on the square?

The same is run with the action zeroed (how much of the change the action buys) and, as the control,
on an untrained copy of the same model: the encoder kept, every other weight at a seeded random
initialisation (``reset_weights``). A model that "expects the world to stop" must do so far more than
its untrained copy does, or the answer is the architecture and the encoder, not what it learned.

Verdict, fixed before any run, on the share of moving steps where the frozen world wins, with 95%
intervals from resampling whole episodes:

1. ``yes``      lower end above ``YES`` (0.75) and at least ``MARGIN`` (0.20) above the untrained
                copy's upper end
2. ``no``       upper end below ``NO`` (0.60)
3. ``unclear``  otherwise (also when no control was run and rule 2 does not apply)

A model plugs in with two calls (``WorldModel``): ``encode_frames`` and ``predict_next``. V-JEPA 2-AC
(``adapters.vjepa2ac``) and Meta's JEPA-WMs family (``adapters.jepawm``: DINO-WM, JEPA-WM and a
retrained V-JEPA 2-AC) do. Offline, on recorded video; exploration, never a rollout finding, never
touches a scoreboard.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn

from interpretableintelligence.adapters.vjepa2ac.model import poses_to_diff
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.impossible_events import _heat_tile, _square_tokens, local_freeze, resize_frames
from interpretableintelligence.worldmodel.plan_probe import _boot
from interpretableintelligence.worldmodel.regions import write_png

SCHEMA = "freeze-check/v1"
MIN_MOVE_M = 0.01  # arm translation per model step for a step to count as moving
YES = 0.75
NO = 0.60
MARGIN = 0.20
N_BOOT = 2000
N_TILES = 6


class WorldModel(Protocol):
    """What ``freeze-check`` needs from a model. Latents are ``[P, D]`` tokens per frame, on the CPU."""

    name: str
    info: dict
    crop: int  # frames are squashed to crop x crop before encoding
    patch: int  # pixels per latent token side (crop // patch tokens per side)

    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """``[N, crop, crop, 3]`` uint8 to ``[N, P, D]``."""
        ...

    def predict_next(self, z: torch.Tensor, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``z [B, P, D]``, poses ``[B, 7]``, actions ``[B, 7]`` (pose change to the next frame) to ``[B, P, D]``."""
        ...


def reset_weights(module: nn.Module, *, seed: int = 0) -> nn.Module:
    """Put every weight of ``module`` back to a seeded random initialisation, in place.

    Layers with ``reset_parameters`` (linear, conv, norms, embeddings) use their own default init; any
    other parameter (position embeddings, mask or register tokens) gets a truncated normal with std 0.02,
    as ViTs initialise them. Used for the untrained-copy control on models we did not write.
    """
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    covered: set[int] = set()
    for m in module.modules():
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()
            covered.update(id(p) for p in m.parameters(recurse=False))
    with torch.no_grad():
        for p in module.parameters():
            if id(p) not in covered:
                noise = torch.randn(p.shape, generator=gen).clamp_(-2.0, 2.0) * 0.02
                p.copy_(noise.to(p.dtype))
    return module


def _l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``[P, D]`` pair to per-token mean absolute difference ``[P]``."""
    return (a - b).abs().mean(dim=-1)


def _change_share(pred: torch.Tensor, now: torch.Tensor, nxt: torch.Tensor) -> float:
    """Projection of the predicted change on the real change: 0 = frozen, 1 = the real next latent."""
    real = (nxt - now).double().flatten()
    den = float(real @ real)
    if den <= 0:
        return float("nan")
    return float((pred - now).double().flatten() @ real) / den


def measure(
    model: WorldModel,
    trajectories: list[Trajectory],
    *,
    heatmaps: Path | None = None,
    cache: dict | None = None,
    say=print,
) -> list[dict]:
    """One record per moving step: frozen-vs-real distances, change predicted, moving square noticed.

    ``cache`` (clip id to encoded frames) is filled on the first call and read on the next: the untrained
    copy shares the encoder, so its latents are the model's and need not be computed twice."""
    crop, patch = model.crop, model.patch
    side = crop // patch
    records: list[dict] = []
    tiles: list[np.ndarray] = []
    for i, tr in enumerate(trajectories):
        frames = resize_frames(tr.frames, crop)
        actions = poses_to_diff(tr.states)
        moving = [t for t in range(len(actions)) if np.linalg.norm(actions[t, :3]) >= MIN_MOVE_M]
        if not moving:
            say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: no moving steps")
            continue
        edited, where = zip(*(local_freeze(frames[t], frames[t + 1], patch) for t in moving))
        if cache is not None and tr.clip_id in cache:
            z, z_lf = cache[tr.clip_id]
        else:
            z, z_lf = model.encode_frames(frames), model.encode_frames(np.stack(edited))
            if cache is not None:
                cache[tr.clip_id] = (z, z_lf)
        idx = torch.tensor(moving)
        states = torch.from_numpy(tr.states[idx.numpy()])
        act = torch.from_numpy(actions[idx.numpy()])
        pred = model.predict_next(z[idx], states, act).float()
        pred0 = model.predict_next(z[idx], states, torch.zeros_like(act)).float()
        for j, t in enumerate(moving):
            now, nxt = z[t].float(), z[t + 1].float()
            d_real, d_frozen = float(_l1(pred[j], nxt).mean()), float(_l1(pred[j], now).mean())
            sq = _square_tokens(*where[j], side)
            records.append(
                {
                    "clip_id": tr.clip_id,
                    "episode": tr.episode,
                    "t": int(t),
                    "arm_step_m": float(np.linalg.norm(actions[t, :3])),
                    "d_real": d_real,
                    "d_frozen": d_frozen,
                    "frozen_wins": bool(d_frozen < d_real),
                    "frozen_wins_zero_action": bool(float(_l1(pred0[j], now).mean()) < float(_l1(pred0[j], nxt).mean())),
                    "change_predicted": _change_share(pred[j], now, nxt),
                    "change_predicted_zero_action": _change_share(pred0[j], now, nxt),
                    "moving_noticed": bool(float(_l1(pred[j], z_lf[j].float())[sq].mean()) > float(_l1(pred[j], nxt)[sq].mean())),
                }
            )
            if heatmaps is not None and len(tiles) < N_TILES and j == len(moving) // 2:
                gain = (_l1(pred[j], z_lf[j].float()) - _l1(pred[j], nxt)).numpy()
                tiles.append(_heat_tile(edited[j], gain, side, patch, [tuple(where[j])]))
        say(f"     {i + 1}/{len(trajectories)} {tr.clip_id}: {len(moving)} moving steps, {len(records)} so far")
    if heatmaps is not None and tiles:
        heatmaps.mkdir(parents=True, exist_ok=True)
        write_png(heatmaps / "moving_frozen.png", np.concatenate(tiles, axis=0))
    return records


def _mean(key: str):
    def f(rs):
        vals = [float(r[key]) for r in rs if np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return f


def _median(key: str):
    def f(rs):
        vals = [float(r[key]) for r in rs if np.isfinite(float(r[key]))]
        return float(np.median(vals)) if vals else float("nan")

    return f


def summarize(records: list[dict], *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    def gain(rs):
        return _median("change_predicted")(rs) - _median("change_predicted_zero_action")(rs)

    return {
        "n_steps": len(records),
        "n_episodes": len({r["episode"] for r in records}),
        "frozen_wins": _boot(records, _mean("frozen_wins"), n_boot=n_boot, seed=seed),
        "frozen_wins_zero_action": _boot(records, _mean("frozen_wins_zero_action"), n_boot=n_boot, seed=seed),
        "change_predicted": _boot(records, _median("change_predicted"), n_boot=n_boot, seed=seed),
        "change_predicted_zero_action": _boot(records, _median("change_predicted_zero_action"), n_boot=n_boot, seed=seed),
        "change_from_action": _boot(records, gain, n_boot=n_boot, seed=seed),
        "moving_noticed": _boot(records, _mean("moving_noticed"), n_boot=n_boot, seed=seed),
    }


def verdict(model: dict, control: dict | None) -> str:
    """The fixed rule, on ``frozen_wins`` of the model and of its untrained copy."""
    if not np.isfinite(model["lo"]):
        return "insufficient_data"
    if model["hi"] < NO:
        return "no"
    if control is not None and np.isfinite(control["hi"]) and model["lo"] > YES and model["lo"] - control["hi"] >= MARGIN:
        return "yes"
    return "unclear"


def _pct(ci: dict) -> str:
    return f"{100 * ci['value']:.0f}% [{100 * ci['lo']:.0f}, {100 * ci['hi']:.0f}]"


def _num(ci: dict) -> str:
    return f"{ci['value']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}]"


HEADLINE = {
    "yes": "Yes: it expects the world to stop.",
    "no": "No: it does not expect the world to stop.",
    "unclear": "Unclear.",
    "insufficient_data": "Not enough moving steps to tell.",
}


def write_card(result: dict, path: Path) -> None:
    """One self-contained page: the verdict, then the numbers beside the untrained copy."""
    m, c = result["model"], result.get("control")
    rows = [
        ("Frozen world closer than the real next frame", _pct(m["frozen_wins"]), _pct(c["frozen_wins"]) if c else "not run"),
        ("Share of the real change it predicts (0 = frozen, 1 = real)", _num(m["change_predicted"]), _num(c["change_predicted"]) if c else "not run"),
        ("Of that, bought by the action", _num(m["change_from_action"]), _num(c["change_from_action"]) if c else "not run"),
        ("Notices when the moving thing is frozen", _pct(m["moving_noticed"]), _pct(c["moving_noticed"]) if c else "not run"),
    ]
    body = "".join(f"<tr><td>{html.escape(a)}</td><td>{b}</td><td>{d}</td></tr>" for a, b, d in rows)
    img = '<img src="heatmaps/moving_frozen.png" alt="moving square frozen: frame and where the model noticed">' if result.get("heatmap") else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Freeze check</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1b18;--mut:#6b665d;--line:#e4dfd5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151412;--fg:#ece8e1;--mut:#9c968b;--line:#2c2a26}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:720px;margin:auto}}h1{{font-size:26px;margin:0 0 4px}}p{{color:var(--mut);margin:0 0 20px}}
table{{width:100%;border-collapse:collapse}}td,th{{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:var(--mut);font-weight:500}}img{{max-width:100%;margin-top:20px;image-rendering:pixelated}}</style></head>
<body><main><h1>{HEADLINE[result['verdict']]}</h1>
<p>{html.escape(result['model_name'])}: {m['n_steps']} moving steps from {m['n_episodes']} episodes. Untrained copy: same model, predictor weights random.</p>
<table><tr><th></th><th>Model</th><th>Untrained copy</th></tr>{body}</table>{img}</main></body></html>
"""
    path.write_text(page)


def run_freeze_check(
    model: WorldModel,
    trajectories: list[Trajectory],
    out_dir: Path,
    *,
    control: WorldModel | None = None,
    seed: int = 0,
    n_boot: int = N_BOOT,
    say=print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"  {model.name}")
    cache: dict = {}  # the untrained copy keeps the encoder: encode each clip once
    recs = measure(model, trajectories, heatmaps=out_dir / "heatmaps", cache=cache if control is not None else None, say=say)
    summary = summarize(recs, n_boot=n_boot, seed=seed)
    control_summary, control_recs = None, None
    if control is not None:
        say(f"  {control.name} (control)")
        control_recs = measure(control, trajectories, cache=cache, say=say)
        control_summary = summarize(control_recs, n_boot=n_boot, seed=seed)
    result = {
        "schema": SCHEMA,
        "model_name": model.name,
        "model_info": model.info,
        "control_name": control.name if control is not None else None,
        "seed": seed,
        "n_boot": n_boot,
        "rules": {"min_move_m": MIN_MOVE_M, "yes": YES, "no": NO, "margin": MARGIN},
        "verdict": verdict(summary["frozen_wins"], control_summary["frozen_wins"] if control_summary else None),
        "model": summary,
        "control": control_summary,
        "heatmap": (out_dir / "heatmaps" / "moving_frozen.png").is_file(),
        "steps": recs,
        "control_steps": control_recs,
    }
    (out_dir / "freeze_check.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    write_card(result, out_dir / "index.html")
    return result
