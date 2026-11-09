"""SAE and feature scores. Gates live in ``physint.thresholds``."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from physint.adapters.base import Adapter, Clip
from physint.sae import SAE
from physint.stats import pearson, percentile_le
from physint.thresholds import (
    COSINE_EPS,
    DEAD_ACT_EPS,
    N_RANDOM_FEATURES,
    PATTERN_ABSENT_MAX_CORR,
    TOP_K_EXAMPLES,
)


@dataclass
class Token:
    clip_index: int
    clip_id: str
    window: int
    t: int
    h: int
    w: int
    split: str
    masked: bool


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity along the last dimension.

    A zero vector against a non-zero vector counts as distance 1 (the output
    collapsed or appeared). Two zero vectors count as distance 0.
    """
    an = a.norm(dim=-1)
    bn = b.norm(dim=-1)
    dot = (a * b).sum(dim=-1)
    denom = (an * bn).clamp(min=COSINE_EPS)
    cos = (dot / denom).clamp(-1.0, 1.0)
    dist = 1.0 - cos
    both = (an < COSINE_EPS) & (bn < COSINE_EPS)
    one = (an < COSINE_EPS) ^ (bn < COSINE_EPS)
    dist = torch.where(both, torch.zeros_like(dist), dist)
    dist = torch.where(one, torch.ones_like(dist), dist)
    return dist


def explained_variance(x: torch.Tensor, xhat: torch.Tensor) -> float:
    mu = x.mean(dim=0, keepdim=True)
    sst = ((x - mu) ** 2).sum()
    if float(sst) <= 1e-12:
        return 0.0
    sse = ((x - xhat) ** 2).sum()
    return float((1.0 - sse / sst).detach().cpu())


def dead_fraction(features: torch.Tensor) -> float:
    alive = (features > DEAD_ACT_EPS).any(dim=0)
    return float((~alive).float().mean())


def capture_activations(adapter: Adapter, clips: list[Clip], membership: dict[str, str]) -> tuple[list[Token], torch.Tensor]:
    tokens: list[Token] = []
    chunks: list[torch.Tensor] = []
    for clip_index, clip in enumerate(clips):
        split = membership[clip.clip_id]
        for window in adapter.windows(clip):
            with adapter.hook(adapter.layer, mode="capture") as handle:
                adapter.encode(clip, window)
            if len(handle.captured) != 1:
                raise RuntimeError(f"expected one captured tensor for {clip.clip_id}, got {len(handle.captured)}")
            acts = handle.captured[0]
            if acts.ndim != 4:
                raise RuntimeError(f"activations must be [t, h, w, d], got {tuple(acts.shape)}")
            t, h, w, _d = acts.shape
            mask = adapter.mask(window).detach().cpu()
            if tuple(mask.shape) != (t, h, w):
                raise RuntimeError(f"mask shape {tuple(mask.shape)} != activation grid {(t, h, w)}")
            flat_cpu = acts.detach().to(dtype=torch.float32, device="cpu").reshape(t * h * w, -1)
            chunks.append(flat_cpu)
            for ti in range(t):
                for hi in range(h):
                    for wi in range(w):
                        tokens.append(
                            Token(
                                clip_index=clip_index,
                                clip_id=clip.clip_id,
                                window=window.index,
                                t=ti,
                                h=hi,
                                w=wi,
                                split=split,
                                masked=bool(mask[ti, hi, wi]),
                            )
                        )
    if not chunks:
        raise RuntimeError("no activations captured")
    return tokens, torch.cat(chunks, dim=0)


def splice_edit(sae: SAE):
    def edit(acts: torch.Tensor) -> torch.Tensor:
        flat = acts.reshape(-1, acts.shape[-1])
        return sae.reconstruct(flat).reshape_as(acts)

    return edit


def ablate_edit(sae: SAE, feature_id: int):
    column = sae.W_dec[:, feature_id]

    def edit(acts: torch.Tensor) -> torch.Tensor:
        flat = acts.reshape(-1, acts.shape[-1])
        features = sae.encode(flat)
        contrib = features[:, feature_id : feature_id + 1] * column
        return (flat - contrib).reshape_as(acts)

    return edit


def mean_cosine_shift(adapter: Adapter, clips: list[Clip], edit, clean: dict[tuple[str, int], torch.Tensor] | None = None) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for clip in clips:
            for window in adapter.windows(clip):
                key = (clip.clip_id, window.index)
                baseline = None if clean is None else clean.get(key)
                if baseline is None:
                    baseline, _target = adapter.predict_masked(clip, window)
                edited, _target = adapter.predict_masked(clip, window, edit=edit)
                dist = cosine_distance(baseline, edited)
                total += float(dist.sum())
                count += int(dist.numel())
    if count == 0:
        return 0.0
    return total / count


def clean_predictions(adapter: Adapter, clips: list[Clip]) -> dict[tuple[str, int], torch.Tensor]:
    out: dict[tuple[str, int], torch.Tensor] = {}
    with torch.no_grad():
        for clip in clips:
            for window in adapter.windows(clip):
                pred, _target = adapter.predict_masked(clip, window)
                out[(clip.clip_id, window.index)] = pred
    return out


def mean_predictor_loss(adapter: Adapter, clips: list[Clip], edit=None, zero: bool = False) -> float:
    sse = 0.0
    count = 0
    with torch.no_grad():
        for clip in clips:
            for window in adapter.windows(clip):
                if zero:
                    with adapter.hook(adapter.layer, mode="zero"):
                        pred, target = adapter.predict_masked(clip, window)
                else:
                    pred, target = adapter.predict_masked(clip, window, edit=edit)
                sse += float(((pred - target) ** 2).sum())
                count += int(pred.shape[0])
    if count == 0:
        raise RuntimeError("no masked tokens to score")
    return sse / count


def estimate_effects(adapter: Adapter, sae: SAE, clip: Clip, window) -> torch.Tensor:
    """Activation × ∂(mean predictor loss)/∂(feature) at every grid cell.

    Straight-through: the forward value is the true activation, and the
    gradient flows through the SAE decoder. Returns [t, h, w, n_features].
    """
    with torch.enable_grad():
        raw = adapter.raw_activations(clip, window)
        t, h, w, d = raw.shape
        flat = raw.reshape(-1, d).detach()
        features = sae.encode(flat).detach().requires_grad_(True)

        def edit(acts: torch.Tensor) -> torch.Tensor:
            recon = sae.decode(features).reshape_as(acts)
            recon_det = sae.decode(features.detach()).reshape_as(acts)
            return acts + recon - recon_det

        pred, target = adapter.predict_masked(clip, window, edit=edit)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        if features.grad is None:
            raise RuntimeError("predictor loss did not depend on SAE features")
        effect = (features.detach() * features.grad).reshape(t, h, w, -1)
        return effect.detach()


def top_examples(values: np.ndarray, tokens: list[Token], k: int = TOP_K_EXAMPLES) -> tuple[list[dict], int]:
    positive = np.flatnonzero(values > DEAD_ACT_EPS)
    if positive.size == 0:
        return [], 0
    order = positive[np.argsort(-values[positive])][:k]
    examples = []
    clips: set[str] = set()
    for index in order:
        token = tokens[int(index)]
        examples.append(
            {
                "clip_id": token.clip_id,
                "window": int(token.window),
                "t": int(token.t),
                "value": float(values[int(index)]),
            }
        )
        clips.add(token.clip_id)
    return examples, len(clips)


def random_percentile(
    *,
    adapter: Adapter,
    sae: SAE,
    feature_id: int,
    real_effect: float,
    tokens: list[Token],
    features: np.ndarray,
    heldout_clips: list[Clip],
    clean: dict[tuple[str, int], torch.Tensor],
    seed: int,
    n_random: int = N_RANDOM_FEATURES,
) -> float:
    """Percent of matched random features with ablation effect <= this feature's.

    A random feature keeps the same number of held-out firings and the same
    activation values (so decoder-norm × mean activation matches) but uses a
    random decoder direction and random held-out positions.
    """
    held_rows = [i for i, token in enumerate(tokens) if token.split == "heldout"]
    column = features[:, feature_id]
    fire_rows = [i for i in held_rows if column[i] > DEAD_ACT_EPS]
    values = column[np.array(fire_rows, dtype=np.int64)] if fire_rows else np.zeros(0, dtype=np.float64)
    if len(fire_rows) == 0:
        samples = np.zeros(n_random, dtype=np.float64)
        return percentile_le(real_effect, samples)

    decoder = sae.W_dec[:, feature_id].detach()
    dec_norm = float(decoder.norm().clamp(min=1e-8))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(feature_id), 17]))
    pool = [tokens[i] for i in held_rows]
    samples = np.empty(n_random, dtype=np.float64)
    d_model = int(decoder.shape[0])
    device = decoder.device

    for trial in range(n_random):
        choice = rng.choice(len(pool), size=len(fire_rows), replace=False)
        shuffled = rng.permutation(values)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
        direction = torch.randn(d_model, generator=gen, dtype=torch.float32)
        direction = direction / direction.norm().clamp(min=1e-8) * dec_norm
        direction = direction.to(device)
        by_clip: dict[str, list[tuple[int, int, int, float]]] = defaultdict(list)
        for slot, value in zip(choice, shuffled, strict=True):
            token = pool[int(slot)]
            by_clip[token.clip_id].append((token.t, token.h, token.w, float(value)))

        total = 0.0
        count = 0
        with torch.no_grad():
            for clip in heldout_clips:
                positions = by_clip.get(clip.clip_id, [])

                def clip_edit(
                    acts: torch.Tensor,
                    positions: list[tuple[int, int, int, float]] = positions,
                    direc: torch.Tensor = direction,
                ) -> torch.Tensor:
                    out = acts.clone()
                    for t, h, w, value in positions:
                        out[t, h, w] = out[t, h, w] - value * direc
                    return out

                for window in adapter.windows(clip):
                    edited, _target = adapter.predict_masked(clip, window, edit=clip_edit)
                    baseline = clean[(clip.clip_id, window.index)]
                    dist = cosine_distance(baseline, edited)
                    total += float(dist.sum())
                    count += int(dist.numel())
        samples[trial] = 0.0 if count == 0 else total / count
    return percentile_le(real_effect, samples)


def absent_in_random_init(pattern: np.ndarray, random_patterns: np.ndarray) -> bool:
    """True when no random-init SAE feature reproduces this firing pattern.

    ``random_patterns`` has shape ``[n_tokens, n_features]`` and the same row
    order as ``pattern``. A flat pattern matches other dead features, so it is
    not treated as absent.
    """
    pattern = np.asarray(pattern, dtype=np.float64).ravel()
    others = np.asarray(random_patterns, dtype=np.float64)
    if others.ndim != 2 or others.shape[0] != pattern.shape[0]:
        raise ValueError(
            f"random-init patterns must be [n_tokens, n_features], got {others.shape} vs {pattern.shape}"
        )
    if float(np.std(pattern)) <= DEAD_ACT_EPS:
        return False
    best = 0.0
    for index in range(others.shape[1]):
        other = others[:, index]
        if float(np.std(other)) <= DEAD_ACT_EPS:
            continue
        best = max(best, pearson(pattern, other))
    return bool(best < PATTERN_ABSENT_MAX_CORR)
