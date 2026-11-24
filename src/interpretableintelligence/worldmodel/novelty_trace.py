"""``ii novelty-trace``: where does V-JEPA 2-AC lose what is new in the next frame?

Earlier screens found the 2-AC prediction always closer to a frozen world than to the real next frame
(``ii impossible-events``), and that turning the action up only makes it worse (``ii action-amp``). This
command splits the next frame into the part that is **predictable** from the current frame and the
part that is **new**, then follows the new part through the model, layer by layer.

For every transition ``t -> t + 1`` where the arm really moves, the change of each latent token is
``delta = z_{t+1} - z_t`` (``z`` is the encoder's layer-normed latent). The predictor gets ``z_t``, the
pose and the real action, as in Meta's planner. Ridge readouts, one linear map shared by all tokens,
then split and follow the change:

``input``    reads ``delta`` from ``z_t`` itself. What it explains is the **predictable** share; what it
             misses, ``new = delta - input(z_t)``, is the **new** part of the next frame.
``L0..Ln``   read ``new`` from the predictor's frame tokens after its input layer and after each block
             (layer-normed): how much of the new part the predictor holds at that depth
``output``   reads ``new`` from the prediction
``native``   no readout: the prediction's own error, ``|z_t + prediction change - z_{t+1}|``, against
             the size of the new part (above 0 means the prediction gets some of the new part right)

Readouts are cross-fitted: two folds split by episode, each fold scored with the maps fitted on the
other. The ridge penalty is chosen inside the training fold (two halves of its episodes, fit on one
and scored on the other) from ``RIDGES``, so the scored fold never picks it. Scores are shares of
variance explained (R squared), on the moving tokens (the ``MOVING_SHARE`` of patches whose pixels
change most) and on all tokens. ``gain`` at a layer is its share of the new part.

Verdict on the moving tokens, fixed before any real run, with 95% intervals from resampling whole
episodes (``verdict``), at the layer with the highest R squared (``peak``):

1. ``never_added`` when the peak gain's upper end is below ``GAIN_MIN`` (0.02), or, with a
   ``--control`` run (the same command with ``--random-predictor``), the trained peak gain does not
   beat the untrained one (lower end of the paired difference not above 0). The predictor never holds
   the new information, so the loss is upstream: the encoder, the action or the data.
2. ``lost_inside`` when the share of the peak gain gone by the last block has its lower end above
   ``LOST`` (0.5): the predictor works the change out, then throws most of it away.
3. ``lost_at_output`` when the last block keeps it (upper end of the lost share below ``LOST``) but the
   prediction's own change falls short of what a readout of the prediction gets by more than
   ``LOST`` times the peak gain (lower end): the information reaches the end but is not expressed.
4. ``kept`` when the last block keeps it and that shortfall's upper end is below ``LOST`` times the
   peak gain.
5. ``mixed`` otherwise.

The encoder is checked on the same transitions, without any fitting: at each encoder block (and on the
pixels), how much a token changes between the two frames (mean L1 of the layer-normed tokens, the
planner's distance; ``1 - cos`` of the centred tokens is reported beside it), on moving minus still
patches: the motion signal. ``encoder`` is ``damps`` when the final latent
keeps under ``LOST`` of the encoder's peak motion signal (upper end), ``keeps`` when it keeps over
``KEEPS`` (lower end), ``mixed`` otherwise. Offline, on recorded video; not a rollout finding and
never touches a scoreboard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence.adapters.vjepa2ac.model import (
    ACWorldModel,
    poses_to_diff,
)
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.impossible_events import resize_frames

SCHEMA = "novelty-trace/v1"
MIN_MOVE_M = 0.01  # arm translation per model step for a transition to count
MOVING_SHARE = 0.2  # patches with the largest pixel change
STILL_SHARE = 0.5  # patches with the smallest pixel change
RIDGES = (
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
)  # ridge penalties tried, relative to the mean feature variance
GAIN_MIN = 0.02
LOST = 0.5
KEEPS = 0.8
N_BOOT = 2000
BATCH = 16
REGIONS = ("moving", "all")


def fold_of(episode: int, seed: int) -> int:
    return hashlib.sha256(f"novelty-trace:{seed}:{episode}".encode()).digest()[0] % 2


def halves(episodes, seed: int) -> dict[int, int]:
    """The inner split of each fold's episodes, used only to choose the ridge: alternate episodes in
    hash order, so each fold with two episodes or more has both halves."""
    key = lambda e: hashlib.sha256(
        f"novelty-trace-half:{seed}:{e}".encode()
    ).hexdigest()
    out = {}
    for f in (0, 1):
        for i, e in enumerate(
            sorted({int(e) for e in episodes if fold_of(int(e), seed) == f}, key=key)
        ):
            out[e] = i % 2
    return out


def patch_change(prev: np.ndarray, nxt: np.ndarray, patch: int) -> np.ndarray:
    """Mean absolute pixel change of each patch, ``[P]``, between two ``[crop, crop, 3]`` uint8 frames."""
    side = prev.shape[0] // patch
    d = np.abs(nxt.astype(np.float32) - prev.astype(np.float32)).mean(-1)
    return d.reshape(side, patch, side, patch).mean(axis=(1, 3)).reshape(-1)


def region_masks(change: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``moving`` (top ``MOVING_SHARE``) and ``still`` (bottom ``STILL_SHARE``) boolean masks by pixel change."""
    p = len(change)
    order = np.argsort(-change, kind="stable")
    moving = np.zeros(p, dtype=bool)
    moving[order[: max(1, round(MOVING_SHARE * p))]] = True
    still = np.zeros(p, dtype=bool)
    still[order[p - max(1, round(STILL_SHARE * p)) :]] = True
    return moving, still


# --------------------------------------------------------------------------- readouts


class RidgeStats:
    """Running sums for a ridge regression ``y ~ x W + b``, accumulated in float64 on the device."""

    def __init__(self, d_in: int, d_out: int, device: torch.device) -> None:
        f64 = torch.float64
        self.xtx = torch.zeros(d_in, d_in, dtype=f64, device=device)
        self.xty = torch.zeros(d_in, d_out, dtype=f64, device=device)
        self.sx = torch.zeros(d_in, dtype=f64, device=device)
        self.sy = torch.zeros(d_out, dtype=f64, device=device)
        self.yy = 0.0
        self.n = 0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x, y = x.float(), y.float()
        self.xtx += (x.T @ x).double()
        self.xty += (x.T @ y).double()
        self.sx += x.sum(0).double()
        self.sy += y.sum(0).double()
        self.yy += float((y.double() ** 2).sum())
        self.n += len(x)

    def merged(self, other: RidgeStats) -> RidgeStats:
        out = RidgeStats.__new__(RidgeStats)
        for k in ("xtx", "xty", "sx", "sy", "yy", "n"):
            setattr(out, k, getattr(self, k) + getattr(other, k))
        return out

    def sse(self, w: torch.Tensor, b: torch.Tensor) -> float:
        """Sum of squared errors of ``x W + b`` on the rows summed here, from the sums alone."""
        w, b = w.double(), b.double()
        quad = float(torch.sum(w * (self.xtx @ w)))
        cross = float(torch.sum(w * self.xty))
        return (
            self.yy
            + quad
            + self.n * float(b @ b)
            - 2 * cross
            - 2 * float(b @ self.sy)
            + 2 * float(b @ (self.sx @ w))
        )

    def sst(self, mean: torch.Tensor) -> float:
        m = mean.double()
        return self.yy - 2 * float(m @ self.sy) + self.n * float(m @ m)

    def solve(
        self, ridge: float = 1e-3
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``W [d_in, d_out]``, ``b [d_out]`` and the target mean ``[d_out]``, all float32."""
        if self.n < 2:
            raise ValueError("not enough rows to fit a readout")
        mx, my = self.sx / self.n, self.sy / self.n
        cxx = self.xtx / self.n - torch.outer(mx, mx)
        cxy = self.xty / self.n - torch.outer(mx, my)
        lam = ridge * float(torch.diagonal(cxx).mean().clamp_min(1e-12))
        eye = torch.eye(len(cxx), dtype=cxx.dtype, device=cxx.device)
        w = torch.linalg.solve(cxx + lam * eye, cxy)
        b = my - mx @ w
        return w.float(), b.float(), my.float()


def fit_inner_cv(
    a: RidgeStats, b: RidgeStats, ridges=RIDGES
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], float]:
    """Pick the ridge on two halves of the training episodes (fit on one, score on the other, both
    ways), then refit on both halves. Returns the fit and the chosen ridge."""
    best, pick = float("inf"), ridges[0]
    for r in ridges:
        err = 0.0
        for fit, held in ((a, b), (b, a)):
            w, bias, _ = fit.solve(r)
            err += held.sse(w, bias)
        if err < best:
            best, pick = err, r
    return a.merged(b).solve(pick), pick


def _blocks(model: ACWorldModel) -> list:
    blocks = getattr(model.predictor, "predictor_blocks", None)
    if blocks is None:
        raise ValueError(f"{model.name} has no predictor_blocks to trace")
    return list(blocks)


def source_names(n_blocks: int) -> list[str]:
    return ["input"] + [f"L{i}" for i in range(n_blocks + 1)] + ["output"]


@torch.inference_mode()
def forward_layers(
    model: ACWorldModel, z: torch.Tensor, states: torch.Tensor, actions: torch.Tensor
) -> dict[str, torch.Tensor]:
    """One predictor pass for ``z [B, P, D]``; every readout source as ``[B, P, d]`` float32 on the device.

    ``L0`` is the frame tokens going into the first block and ``Lk`` those coming out of block ``k``,
    each layer-normed (no parameters) so massive residual dimensions do not swamp the readout.
    """
    blocks = _blocks(model)
    p = model.tokens_per_frame
    hidden: list[torch.Tensor | None] = [None] * (len(blocks) + 1)

    def keep(i: int, x: torch.Tensor) -> None:
        h = x[:, -p:].float()
        hidden[i] = F.layer_norm(h, (h.shape[-1],))

    def pre(_mod, args, kwargs):
        keep(0, args[0])

    handles = [blocks[0].register_forward_pre_hook(pre, with_kwargs=True)]
    for i, blk in enumerate(blocks):
        handles.append(
            blk.register_forward_hook(lambda _m, _a, out, i=i: keep(i + 1, out))
        )
    try:
        zd = z.to(model.device)
        sd = states.to(model.device, torch.float32).unsqueeze(1)
        ad = actions.to(model.device, torch.float32).unsqueeze(1)
        with model._autocast():
            y = model.predictor(zd, ad, sd)[:, -p:]
        y = F.layer_norm(y.float(), (y.shape[-1],))
    finally:
        for h in handles:
            h.remove()
    out = {"input": zd.float()}
    for i, h in enumerate(hidden):
        out[f"L{i}"] = h
    out["output"] = y
    return out


# --------------------------------------------------------------------------- encoder


def _centred(x: torch.Tensor) -> torch.Tensor:
    """``[N, P, D]``: layer-norm each token, then remove each frame's mean token."""
    x = F.layer_norm(x.float(), (x.shape[-1],))
    return x - x.mean(dim=1, keepdim=True)


def _change(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-token ``1 - cos`` between two ``[P, D]`` centred frames."""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def _pixel_tokens(frames: np.ndarray, patch: int) -> torch.Tensor:
    """``[N, crop, crop, 3]`` uint8 to ``[N, P, patch * patch * 3]`` float patches."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0
    n, h, _, c = x.shape
    s = h // patch
    return (
        x.reshape(n, s, patch, s, patch, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(n, s * s, -1)
    )


def encode_with_trace(
    model: ACWorldModel,
    frames: np.ndarray,
    moving_t: list[int],
    masks: dict[int, tuple],
) -> tuple[torch.Tensor, dict[int, np.ndarray]]:
    """Encode ``frames`` (already resized) and, for each moving transition ``t``, how much the tokens
    change at the pixels, every encoder block and the final latent: ``[4, n_layers]`` rows of
    (moving, still) mean L1 between layer-normed tokens (the planner's distance), then (moving, still)
    ``1 - cos`` of the centred tokens. Without encoder ``blocks`` (the tiny stand-in) only the pixels
    and the latent are traced."""
    blocks = list(getattr(model.encoder, "blocks", []) or [])
    want = set(moving_t)
    p = model.tokens_per_frame
    per_layer: list[torch.Tensor | None] = [None] * len(blocks)
    handles = [
        blk.register_forward_hook(
            lambda _m, _a, out, i=i: per_layer.__setitem__(i, out[:, -p:].float())
        )
        for i, blk in enumerate(blocks)
    ]
    zs, carry, out = [], None, {}
    chunk = max(1, model.encode_batch)
    pix = _pixel_tokens(frames, model.patch)
    try:
        for s in range(0, len(frames), chunk):
            z = model.encode_frames(frames[s : s + chunk])
            zs.append(z)
            raw = (
                [pix[s : s + chunk].to(model.device)]
                + [h for h in per_layer]
                + [z.to(model.device)]
            )
            cur = [
                (F.layer_norm(h, (h.shape[-1],)) if k else h, _centred(h))
                for k, h in enumerate(raw)
            ]
            for j in range(len(z)):
                t = s + j - 1  # transition t -> t + 1 ends at this frame
                if t not in want:
                    continue
                prev = carry if j == 0 else [(ln[j - 1], c[j - 1]) for ln, c in cur]
                if prev is None:
                    continue
                mov, still = (torch.from_numpy(m).to(model.device) for m in masks[t])
                rows = np.zeros((4, len(cur)), dtype=np.float64)
                for k, (ln, c) in enumerate(cur):
                    l1 = (ln[j] - prev[k][0]).abs().mean(-1)
                    cs = _change(prev[k][1], c[j])
                    rows[:, k] = [
                        float(l1[mov].mean()),
                        float(l1[still].mean()),
                        float(cs[mov].mean()),
                        float(cs[still].mean()),
                    ]
                out[t] = rows
            carry = [(ln[-1], c[-1]) for ln, c in cur]
    finally:
        for h in handles:
            h.remove()
    return torch.cat(zs), out


# --------------------------------------------------------------------------- the trace


def _moving_transitions(tr: Trajectory) -> tuple[np.ndarray, list[int]]:
    actions = poses_to_diff(tr.states)
    moving = [
        t for t in range(len(actions)) if np.linalg.norm(actions[t, :3]) >= MIN_MOVE_M
    ]
    return actions, moving


def trace(
    model: ACWorldModel,
    trajectories: list[Trajectory],
    *,
    seed: int = 0,
    batch: int = BATCH,
    say=print,
) -> dict:
    """Three passes over the moving transitions. (1) Fit, per fold, the ``input`` readout of the change
    from ``z_t``; what it misses is the *new* part ``n``. (2) Fit, per fold, readouts of ``n`` from
    every predictor layer and the prediction. (3) Score each transition with the other fold's readouts.
    Returns per-transition arrays (the ``.npz`` written beside the summary)."""
    n_blocks = len(_blocks(model))
    hidden_names = source_names(n_blocks)[1:]  # L0..Ln, output
    crop, patch = model.crop, model.patch
    clips = []  # (trajectory, latents, actions, moving, masks)
    enc_rows: list[np.ndarray] = []
    for i, tr in enumerate(trajectories):
        frames = resize_frames(tr.frames, crop)
        actions, moving = _moving_transitions(tr)
        masks = {
            t: region_masks(patch_change(frames[t], frames[t + 1], patch))
            for t in moving
        }
        z, enc = encode_with_trace(model, frames, moving, masks)
        clips.append((tr, z.half(), actions, moving, masks))
        enc_rows.extend(enc[t] for t in moving)
        say(
            f"     encoded {i + 1}/{len(trajectories)} {tr.clip_id}: {len(moving)} moving transitions"
        )
    half = halves([c[0].episode for c in clips if c[3]], seed)
    groups = [(f, h) for f in (0, 1) for h in (0, 1)]

    def batches():
        for tr, z, actions, moving, masks in clips:
            for s in range(0, len(moving), batch):
                ts = moving[s : s + batch]
                idx = torch.tensor(ts)
                zt = z[idx].float().to(model.device)
                delta = z[idx + 1].float().to(model.device) - zt
                g = (fold_of(tr.episode, seed), half[int(tr.episode)])
                yield (
                    tr,
                    g,
                    ts,
                    zt,
                    delta,
                    torch.from_numpy(tr.states[ts]),
                    torch.from_numpy(actions[ts]),
                    masks,
                )

    def flat(x):
        return x.reshape(-1, x.shape[-1])

    def fit_all(stats):
        if any(not stats[g] for g in groups):
            raise ValueError(
                "each episode fold needs at least two episodes with moving transitions: use more clips"
            )
        fits, ridges = {0: {}, 1: {}}, {}
        for f in (0, 1):
            for name in stats[(f, 0)]:
                fits[f][name], r = fit_inner_cv(
                    stats[(f, 0)][name], stats[(f, 1)][name]
                )
                ridges.setdefault(name, []).append(r)
        return fits, ridges

    def novel(fit, zt, delta):
        w, b, _ = fit
        return delta - (zt @ w + b)

    # (1) the predictable part: the change read linearly from the current frame
    stats: dict = {g: {} for g in groups}
    for tr, g, ts, zt, delta, st, ac, _ in batches():
        stats[g].setdefault(
            "input", RidgeStats(zt.shape[-1], delta.shape[-1], model.device)
        ).add(flat(zt), flat(delta))
    in_fits, ridges = fit_all(stats)
    # (2) the new part, read from every predictor layer
    stats = {g: {} for g in groups}
    for tr, g, ts, zt, delta, st, ac, _ in batches():
        n = flat(novel(in_fits[g[0]]["input"], zt, delta))
        src = forward_layers(model, zt, st, ac)
        for name in hidden_names:
            x = flat(src[name])
            stats[g].setdefault(
                name, RidgeStats(x.shape[-1], n.shape[-1], model.device)
            ).add(x, n)
    fits, more = fit_all(stats)
    ridges.update(more)
    del stats
    say(f"     fitted {len(hidden_names) + 1} readouts per fold")

    # (3) score with the other fold's readouts
    cols = ["input"] + hidden_names + ["native"]
    rec = {"clip_id": [], "episode": [], "t": [], "fold": [], "arm_step_m": []}
    sse = {r: [] for r in REGIONS}
    sst = {f"{k}_{r}": [] for k in ("delta", "novel") for r in REGIONS}
    for tr, g, ts, zt, delta, st, ac, masks in batches():
        other = 1 - g[0]
        src = forward_layers(model, zt, st, ac)
        n = novel(in_fits[other]["input"], zt, delta)
        mov = torch.from_numpy(np.stack([masks[t][0] for t in ts])).to(model.device)
        sel = {"moving": mov.float(), "all": torch.ones_like(mov, dtype=torch.float32)}
        errs = [
            (n**2).sum(-1)
        ]  # the input readout's error on the change is the new part itself
        for name in hidden_names:
            w, b, _ = fits[other][name]
            errs.append(((src[name] @ w + b - n) ** 2).sum(-1))
        errs.append(
            ((src["output"] - zt - delta) ** 2).sum(-1)
        )  # the prediction's own change
        err = torch.stack(errs, dim=1)  # [B, S, P]
        tot = {
            "delta": ((delta - in_fits[other]["input"][2]) ** 2).sum(-1),
            "novel": (n**2).sum(-1),
        }
        for r in REGIONS:
            sse[r].append((err * sel[r].unsqueeze(1)).sum(-1).double().cpu().numpy())
            for k, v in tot.items():
                sst[f"{k}_{r}"].append((v * sel[r]).sum(-1).double().cpu().numpy())
        for j, t in enumerate(ts):
            rec["clip_id"].append(tr.clip_id)
            rec["episode"].append(tr.episode)
            rec["t"].append(int(t))
            rec["fold"].append(g[0])
            rec["arm_step_m"].append(float(np.linalg.norm(ac[j, :3].numpy())))
    say(f"     scored {len(rec['t'])} transitions")
    out = {k: np.asarray(v) for k, v in rec.items()}
    for r in REGIONS:
        out[f"sse_{r}"] = np.concatenate(sse[r])
    for k, v in sst.items():
        out[f"sst_{k}"] = np.concatenate(v)
    out["encoder"] = np.stack(enc_rows) if enc_rows else np.zeros((0, 4, 0))
    out["columns"] = np.asarray(cols)
    out["ridge"] = np.asarray(
        [ridges[c] for c in ["input"] + hidden_names]
    )  # [sources, folds]
    return out


# --------------------------------------------------------------------------- summary


def _episode_sums(
    episodes: np.ndarray, arrays: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    eps = np.unique(episodes)
    index = np.searchsorted(eps, episodes)
    sums = {}
    for k, a in arrays.items():
        s = np.zeros((len(eps),) + a.shape[1:], dtype=np.float64)
        np.add.at(s, index, a)
        sums[k] = s
    return eps, sums


def _boot(sums: dict[str, np.ndarray], stat, *, n_boot: int, seed: int) -> dict:
    """``stat`` of the episode sums, with a 95% interval from resampling whole episodes."""
    n = len(next(iter(sums.values())))
    value = float(stat({k: v.sum(0) for k, v in sums.items()}))
    if n < 2:
        return {"value": value, "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        vals.append(stat({k: np.tensordot(w, v, axes=1) for k, v in sums.items()}))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
    return {"value": value, "lo": float(lo), "hi": float(hi)}


def verdict(gain: dict, lost: dict, shortfall: dict, beats_control: dict | None) -> str:
    if not np.isfinite(gain["hi"]):
        return "insufficient_data"
    if gain["hi"] < GAIN_MIN or (
        beats_control is not None and not beats_control["lo"] > 0
    ):
        return "never_added"
    if not np.isfinite(lost["lo"]):
        return "insufficient_data"
    if lost["lo"] > LOST:
        return "lost_inside"
    if lost["hi"] < LOST:
        scale = LOST * gain["value"]
        if shortfall["lo"] > scale:
            return "lost_at_output"
        if shortfall["hi"] < scale:
            return "kept"
    return "mixed"


def encoder_verdict(kept: dict) -> str:
    if not np.isfinite(kept["lo"]):
        return "insufficient_data"
    if kept["hi"] < LOST:
        return "damps"
    if kept["lo"] > KEEPS:
        return "keeps"
    return "mixed"


def summarize(
    data: dict, *, control: dict | None = None, n_boot: int = N_BOOT, seed: int = 0
) -> dict:
    cols = [str(c) for c in data["columns"]]
    col = {c: i for i, c in enumerate(cols)}
    layers = [c for c in cols if c.startswith("L")]
    keys = [f"sse_{r}" for r in REGIONS] + [
        f"sst_{k}_{r}" for k in ("delta", "novel") for r in REGIONS
    ]
    arrays = {k: data[k] for k in keys}
    if control is not None:
        key = lambda d: list(
            zip([str(c) for c in d["clip_id"]], [int(t) for t in d["t"]])
        )
        if key(control) != key(data):
            raise ValueError(
                "the control run scored different transitions: run both with the same clips"
            )
        arrays.update({f"c_{k}": control[k] for k in keys})
    eps, sums = _episode_sums(np.asarray(data["episode"]), arrays)
    total = {k: v.sum(0) for k, v in sums.items()}

    def r2(c, region="moving", pre=""):
        """``input``: share of the change predictable from the current frame; every other column:
        share of the new part read (``native``: the prediction's own error against the new part)."""
        base = "delta" if c == "input" else "novel"
        return lambda s: (
            1.0 - s[f"{pre}sse_{region}"][col[c]] / s[f"{pre}sst_{base}_{region}"]
        )

    curves = {r: {c: float(r2(c, r)(total)) for c in cols} for r in REGIONS}
    peak = max(layers, key=lambda c: curves["moving"][c])
    last = layers[-1]

    def lost_share(s):
        g = r2(peak)(s)
        return (g - r2(last)(s)) / g if g > 0 else float("nan")

    stats = {
        "predictable_share": _boot(sums, r2("input"), n_boot=n_boot, seed=seed),
        "peak_gain": _boot(sums, r2(peak), n_boot=n_boot, seed=seed),
        "last_block": _boot(sums, r2(last), n_boot=n_boot, seed=seed),
        "output_readout": _boot(sums, r2("output"), n_boot=n_boot, seed=seed),
        "native": _boot(sums, r2("native"), n_boot=n_boot, seed=seed),
        "lost_share": _boot(sums, lost_share, n_boot=n_boot, seed=seed),
        "output_shortfall": _boot(
            sums, lambda s: r2("output")(s) - r2("native")(s), n_boot=n_boot, seed=seed
        ),
        "native_all": _boot(sums, r2("native", "all"), n_boot=n_boot, seed=seed),
        "predictable_share_all": _boot(
            sums, r2("input", "all"), n_boot=n_boot, seed=seed
        ),
    }
    beats, control_info = None, None
    if control is not None:
        ctot = {k[2:]: v for k, v in total.items() if k.startswith("c_")}
        c_curves = {r: {c: float(r2(c, r)(ctot)) for c in cols} for r in REGIONS}
        c_peak = max(layers, key=lambda c: c_curves["moving"][c])
        beats = _boot(
            sums,
            lambda s: r2(peak)(s) - r2(c_peak, pre="c_")(s),
            n_boot=n_boot,
            seed=seed,
        )
        stats["peak_gain_minus_control"] = beats
        control_info = {
            "model": str(control.get("model", "control")),
            "peak": c_peak,
            "curves": c_curves,
        }

    enc = np.asarray(data["encoder"], dtype=np.float64)
    encoder = None
    if enc.size:
        signal = enc[:, 0] - enc[:, 1]  # [N, layers]: moving minus still L1 change
        _, e_sums = _episode_sums(np.asarray(data["episode"]), {"signal": signal})
        curve = (e_sums["signal"].sum(0) / len(signal)).tolist()
        names = (
            ["pixels"] + [f"E{i}" for i in range(1, signal.shape[1] - 1)] + ["latent"]
        )
        inner = list(
            range(1, signal.shape[1])
        )  # encoder blocks and the latent, not the pixels
        e_peak = max(inner, key=lambda k: curve[k])
        kept = _boot(
            e_sums,
            lambda s: (
                s["signal"][-1] / s["signal"][e_peak]
                if s["signal"][e_peak] > 0
                else float("nan")
            ),
            n_boot=n_boot,
            seed=seed,
        )
        m = enc.mean(0)
        encoder = {
            "layers": names,
            "l1_moving": m[0].tolist(),
            "l1_still": m[1].tolist(),
            "motion_signal": curve,
            "cos_moving": m[2].tolist(),
            "cos_still": m[3].tolist(),
            "peak": names[e_peak],
            "latent_kept_share": kept,
            "verdict": encoder_verdict(kept),
        }

    ridge_cols = ["input"] + layers + ["output"]
    return {
        "n_transitions": len(data["t"]),
        "n_episodes": len(eps),
        "n_per_fold": [int((np.asarray(data["fold"]) == f).sum()) for f in (0, 1)],
        "peak": peak,
        "last": last,
        "curves": curves,
        "stats": stats,
        "control": control_info,
        "encoder": encoder,
        "ridge": {
            c: [float(x) for x in r]
            for c, r in zip(ridge_cols, np.asarray(data["ridge"]))
        }
        if "ridge" in data
        else None,
        "verdict": verdict(
            stats["peak_gain"], stats["lost_share"], stats["output_shortfall"], beats
        ),
        "encoder_verdict": encoder["verdict"] if encoder else "not_traced",
    }


def load_transitions(path: Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    meta = Path(path).with_name("novelty_trace.json")
    if meta.exists():
        data["model"] = json.loads(meta.read_text()).get("model", "control")
    return data


def run_novelty_trace(
    model: ACWorldModel,
    trajectories: list[Trajectory],
    out_dir: Path,
    *,
    control_dir: Path | None = None,
    seed: int = 0,
    n_boot: int = N_BOOT,
    say=print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    control = (
        load_transitions(Path(control_dir) / "transitions.npz")
        if control_dir is not None
        else None
    )
    data = trace(model, trajectories, seed=seed, say=say)
    np.savez_compressed(out_dir / "transitions.npz", **data)
    result = {
        "schema": SCHEMA,
        "model": model.name,
        "model_info": model.info,
        "seed": seed,
        "n_boot": n_boot,
        "rules": {
            "min_move_m": MIN_MOVE_M,
            "moving_share": MOVING_SHARE,
            "still_share": STILL_SHARE,
            "ridges": list(RIDGES),
            "gain_min": GAIN_MIN,
            "lost": LOST,
            "keeps": KEEPS,
        },
        **summarize(data, control=control, n_boot=n_boot, seed=seed),
    }
    (out_dir / "novelty_trace.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n"
    )
    return result
