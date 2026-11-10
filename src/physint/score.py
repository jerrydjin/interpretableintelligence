"""SAE and feature scores. Gates live in ``physint.thresholds``."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from physint.adapters.base import Adapter, Clip
from physint.sae import SAE
from physint.stats import monte_carlo_pvalue, pearson, percentile_le
from physint.thresholds import (
    COSINE_EPS,
    DEAD_ACT_EPS,
    EFFECT_ALPHA,
    EFFECT_KIND_GROUP_LOSS_VIDEO,
    EFFECT_KIND_GROUP_LOSS_WINDOW,
    EFFECT_KIND_LOSS_VIDEO,
    EFFECT_KIND_LOSS_WINDOW,
    EFFECT_POS_EPS,
    GROUP_ABLATION_SIZE,
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


def mean_visible_baseline_loss(adapter: Adapter, clips: list[Clip]) -> float:
    """MSE of guessing every masked token as the mean of the visible patches.

    Same reduction as ``mean_predictor_loss``: sum of squares over the feature
    axis, averaged over masked tokens.
    """
    sse = 0.0
    count = 0
    with torch.no_grad():
        for clip in clips:
            for window in adapter.windows(clip):
                baseline = adapter.visible_mean_baseline(clip, window)
                _pred, target = adapter.predict_masked(clip, window)
                if tuple(baseline.shape) != tuple(target.shape):
                    raise RuntimeError(
                        f"visible-mean baseline shape {tuple(baseline.shape)} != target {tuple(target.shape)}"
                    )
                sse += float(((baseline - target) ** 2).sum())
                count += int(target.shape[0])
    if count == 0:
        raise RuntimeError("no masked tokens to score")
    return sse / count


def predictor_sse(adapter: Adapter, clips: list[Clip], edit=None, zero: bool = False) -> tuple[float, int]:
    """Sum of squared masked errors and the number of masked tokens."""
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
    return sse, count


def mean_predictor_loss(adapter: Adapter, clips: list[Clip], edit=None, zero: bool = False) -> float:
    sse, count = predictor_sse(adapter, clips, edit=edit, zero=zero)
    if count == 0:
        raise RuntimeError("no masked tokens to score")
    return sse / count


def estimate_effects(adapter: Adapter, sae: SAE, clip: Clip, window, objective: str = "prediction_loss") -> torch.Tensor:
    """Activation × ∂(objective)/∂(feature) at every grid cell.

    ``objective`` is an adapter scalar: ``prediction_loss`` (mean masked MSE)
    or ``action_loglik`` (mean log-likelihood of the taken action).
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

        scalar = adapter.attribution_objective(clip, window, objective, edit=edit)
        if not isinstance(scalar, torch.Tensor) or scalar.ndim != 0:
            raise RuntimeError(f"{objective} objective must return a 0-dim tensor")
        scalar.backward()
        if features.grad is None:
            raise RuntimeError(f"{objective} did not depend on SAE features")
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


@dataclass
class EffectScore:
    """Loss change from ablating one feature or a co-active group.

    ``effect`` is edited masked prediction loss minus the clean loss, over
    every in-scope window (the whole video, unless ``effect_kind`` says
    window). Positive means the ablation made the model's own predictions
    worse. ``real`` requires that increase to clear dust and to beat matched
    random controls of the same size and activation mass at ``EFFECT_ALPHA``.
    Ties count against the candidate, so a flat zero does not pass.
    """

    feature_ids: tuple[int, ...]
    effect: float
    random_percentile: float
    p_value: float
    n_random: int
    real: bool
    control_effects: np.ndarray
    effect_kind: str

    def controls_are_real(self) -> np.ndarray:
        """True where a matched control's loss change reaches this candidate."""
        baseline = _snap_effect(self.effect)
        snapped = np.array([_snap_effect(float(value)) for value in self.control_effects])
        return snapped >= baseline


def _snap_effect(effect: float) -> float:
    if not math.isfinite(effect):
        return float(effect)
    if abs(effect) < EFFECT_POS_EPS:
        return 0.0
    return float(effect)


def loss_effect_kind(grouped: bool, scope: str) -> str:
    if scope not in {"video", "window"}:
        raise ValueError(f"scope must be video or window, got {scope!r}")
    if grouped and scope == "video":
        return EFFECT_KIND_GROUP_LOSS_VIDEO
    if grouped and scope == "window":
        return EFFECT_KIND_GROUP_LOSS_WINDOW
    if scope == "video":
        return EFFECT_KIND_LOSS_VIDEO
    return EFFECT_KIND_LOSS_WINDOW


def ablate_features_edit(sae: SAE, feature_ids: list[int] | tuple[int, ...]):
    """Subtract ``f_i * decoder_i`` for every id. Single-feature ``ablate_edit`` is the k=1 case."""
    ids = [int(feature_id) for feature_id in feature_ids]
    if not ids:
        raise ValueError("need at least one feature to ablate")
    columns = sae.W_dec[:, ids]

    def edit(acts: torch.Tensor, columns: torch.Tensor = columns, ids: list[int] = ids) -> torch.Tensor:
        flat = acts.reshape(-1, acts.shape[-1])
        features = sae.encode(flat)
        contrib = features[:, ids] @ columns.to(device=flat.device, dtype=flat.dtype).T
        return (flat - contrib).reshape_as(acts)

    return edit


def coactive_group(
    features: np.ndarray,
    k: int = GROUP_ABLATION_SIZE,
    threshold: float = DEAD_ACT_EPS,
) -> list[int]:
    """k features that fire on the same tokens.

    Starts from the pair with the most overlapping firings, then adds the
    feature that co-fires most with the set already chosen. Mass breaks ties
    only when nothing fires.
    """
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError(f"features must be [n_tokens, n_features], got {matrix.shape}")
    n_features = int(matrix.shape[1])
    if n_features == 0 or k <= 0:
        return []
    k = min(int(k), n_features)
    if k == 1:
        return [int(np.argmax(matrix.sum(axis=0)))]
    binary = matrix > threshold
    if not bool(binary.any()):
        order = np.argsort(-matrix.sum(axis=0))
        return [int(index) for index in order[:k]]
    co = binary.T.astype(np.float64) @ binary
    off = co.copy()
    np.fill_diagonal(off, -np.inf)
    left, right = np.unravel_index(int(np.argmax(off)), off.shape)
    chosen = [int(left), int(right)]
    while len(chosen) < k:
        scores = co[chosen].sum(axis=0)
        scores[chosen] = -np.inf
        chosen.append(int(np.argmax(scores)))
    return chosen


def _iter_scope(adapter: Adapter, clips: list[Clip], window_index: int | None):
    for clip in clips:
        for window in adapter.windows(clip):
            if window_index is None or window.index == window_index:
                yield clip, window


def _masked_loss(adapter: Adapter, clips: list[Clip], edit, window_index: int | None) -> float:
    """Mean masked prediction loss. Same reduction as ``mean_predictor_loss``."""
    sse = 0.0
    count = 0
    with torch.no_grad():
        for clip, window in _iter_scope(adapter, clips, window_index):
            pred, target = adapter.predict_masked(clip, window, edit=edit)
            sse += float(((pred - target) ** 2).sum())
            count += int(pred.shape[0])
    if count == 0:
        raise RuntimeError("no masked tokens to score")
    return sse / count


def _scope_rows(tokens: list[Token], clips: list[Clip], window_index: int | None) -> list[int]:
    clip_ids = {clip.clip_id for clip in clips}
    rows = []
    for index, token in enumerate(tokens):
        if token.clip_id not in clip_ids:
            continue
        if window_index is not None and token.window != window_index:
            continue
        rows.append(index)
    return rows


def _control_loss(
    adapter: Adapter,
    clips: list[Clip],
    scope_tokens: list[Token],
    values_per_feature: list[np.ndarray],
    norms: list[float],
    rng: np.random.Generator,
    window_index: int | None,
    d_model: int,
    device: torch.device,
) -> float:
    deltas: dict[tuple[str, int], torch.Tensor] = {}
    spatial: dict[tuple[str, int], tuple[int, int]] = {}
    for clip, window in _iter_scope(adapter, clips, window_index):
        shape = tuple(int(size) for size in adapter.mask(window).shape)
        key = (clip.clip_id, window.index)
        deltas[key] = torch.zeros(*shape, d_model, dtype=torch.float32, device=device)
        spatial[key] = (shape[1], shape[2])
    n_scope = len(scope_tokens)
    clip_ids = np.array([token.clip_id for token in scope_tokens], dtype=object)
    windows = np.array([token.window for token in scope_tokens], dtype=np.int64)
    ts = np.array([token.t for token in scope_tokens], dtype=np.int64)
    hs = np.array([token.h for token in scope_tokens], dtype=np.int64)
    ws = np.array([token.w for token in scope_tokens], dtype=np.int64)
    for values, dec_norm in zip(values_per_feature, norms, strict=True):
        if values.size == 0 or n_scope == 0:
            continue
        if values.size > n_scope:
            raise ValueError(f"feature fires on {values.size} tokens but the scope has {n_scope}")
        choice = rng.choice(n_scope, size=values.size, replace=False)
        shuffled = np.asarray(rng.permutation(values), dtype=np.float32)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
        direction = torch.randn(d_model, generator=gen, dtype=torch.float32)
        direction = direction / direction.norm().clamp(min=1e-8) * float(dec_norm)
        direction = direction.to(device)
        chosen_clip = clip_ids[choice]
        chosen_window = windows[choice]
        for key, delta in deltas.items():
            selected = (chosen_clip == key[0]) & (chosen_window == key[1])
            if not np.any(selected):
                continue
            height, width = spatial[key]
            flat_index = (
                ts[choice][selected] * (height * width) + hs[choice][selected] * width + ws[choice][selected]
            )
            index = torch.as_tensor(np.ascontiguousarray(flat_index), dtype=torch.long, device=device)
            weight = torch.as_tensor(np.ascontiguousarray(shuffled[selected]), dtype=torch.float32, device=device)
            delta.reshape(-1, d_model).index_add_(0, index, -(weight[:, None] * direction))

    sse = 0.0
    count = 0
    with torch.no_grad():
        for clip, window in _iter_scope(adapter, clips, window_index):
            delta = deltas[(clip.clip_id, window.index)]

            def edit(acts: torch.Tensor, delta: torch.Tensor = delta) -> torch.Tensor:
                return acts + delta.to(device=acts.device, dtype=acts.dtype)

            pred, target = adapter.predict_masked(clip, window, edit=edit)
            sse += float(((pred - target) ** 2).sum())
            count += int(pred.shape[0])
    if count == 0:
        raise RuntimeError("no masked tokens to score")
    return sse / count


def score_loss_ablation(
    *,
    adapter: Adapter,
    sae: SAE,
    feature_ids: list[int] | tuple[int, ...],
    tokens: list[Token],
    features: np.ndarray,
    clips: list[Clip],
    seed: int,
    n_random: int = N_RANDOM_FEATURES,
    scope: str = "video",
    window_index: int | None = None,
    alpha: float = EFFECT_ALPHA,
) -> EffectScore:
    """Ablate features over the full video, or one window, and compare to matched controls.

    Controls keep each feature's decoder norm and its multiset of positive
    activations, then use a random direction and random in-scope positions.
    The effect is the change in the model's own masked prediction loss.
    """
    ids = [int(feature_id) for feature_id in feature_ids]
    if not ids:
        raise ValueError("score_loss_ablation requires at least one feature")
    if scope not in {"video", "window"}:
        raise ValueError(f"scope must be video or window, got {scope!r}")
    if scope == "window" and window_index is None:
        raise ValueError("window scope requires window_index")
    if scope == "video":
        window_index = None
    if n_random < 1:
        raise ValueError(f"n_random must be positive, got {n_random}")
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] != len(tokens):
        raise ValueError(
            f"features must be [n_tokens, n_features] aligned with tokens, got {matrix.shape} vs {len(tokens)}"
        )

    scoped = None if scope == "video" else window_index
    clean = _masked_loss(adapter, clips, edit=None, window_index=scoped)
    edited = _masked_loss(adapter, clips, edit=ablate_features_edit(sae, ids), window_index=scoped)
    effect = float(edited - clean)

    scope_rows = _scope_rows(tokens, clips, scoped)
    scope_tokens = [tokens[index] for index in scope_rows]
    values_per_feature: list[np.ndarray] = []
    norms: list[float] = []
    decoder = sae.W_dec[:, ids[0]].detach()
    d_model = int(decoder.shape[0])
    device = decoder.device
    for feature_id in ids:
        column = matrix[np.array(scope_rows, dtype=np.int64), feature_id] if scope_rows else np.zeros(0)
        values_per_feature.append(np.asarray(column[column > DEAD_ACT_EPS], dtype=np.float64))
        weight = sae.W_dec[:, feature_id].detach()
        norms.append(float(weight.norm().clamp(min=1e-8)))

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), *[int(feature_id) for feature_id in ids], 29]))
    controls = np.empty(n_random, dtype=np.float64)
    for trial in range(n_random):
        edited_control = _control_loss(
            adapter,
            clips,
            scope_tokens,
            values_per_feature,
            norms,
            rng,
            scoped,
            d_model,
            device,
        )
        controls[trial] = float(edited_control - clean)

    snapped_effect = _snap_effect(effect)
    snapped_controls = np.array([_snap_effect(float(value)) for value in controls], dtype=np.float64)
    p_value = monte_carlo_pvalue(snapped_effect, snapped_controls)
    real = bool(snapped_effect > 0.0 and p_value <= alpha)
    return EffectScore(
        feature_ids=tuple(ids),
        effect=effect,
        random_percentile=percentile_le(snapped_effect, snapped_controls),
        p_value=p_value,
        n_random=n_random,
        real=real,
        control_effects=controls,
        effect_kind=loss_effect_kind(grouped=len(ids) > 1, scope=scope),
    )


def loss_frame_rows(
    adapter: Adapter,
    clips: list[Clip],
    edit,
    feature_ids: list[int] | tuple[int, ...],
    effect_kind: str,
) -> dict[str, list]:
    """Per-frame masked loss increase. The video score is the token-weighted mean."""
    rows: dict[str, list] = {key: [] for key in ("feature_id", "clip_id", "window", "t", "effect", "effect_kind")}
    ids = [int(feature_id) for feature_id in feature_ids]
    with torch.no_grad():
        for clip in clips:
            for window in adapter.windows(clip):
                clean_pred, clean_target = adapter.predict_masked(clip, window)
                edited_pred, edited_target = adapter.predict_masked(clip, window, edit=edit)
                coords = adapter.mask(window).detach().cpu().nonzero(as_tuple=False)
                clean_tok = ((clean_pred - clean_target) ** 2).sum(dim=-1).detach().cpu()
                edit_tok = ((edited_pred - edited_target) ** 2).sum(dim=-1).detach().cpu()
                delta = edit_tok - clean_tok
                n_t = int(adapter.mask(window).shape[0])
                for t in range(n_t):
                    selected = coords[:, 0] == t
                    value = float(delta[selected].mean()) if int(selected.sum()) else 0.0
                    for feature_id in ids:
                        rows["feature_id"].append(feature_id)
                        rows["clip_id"].append(clip.clip_id)
                        rows["window"].append(int(window.index))
                        rows["t"].append(t)
                        rows["effect"].append(value)
                        rows["effect_kind"].append(effect_kind)
    return rows


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


EXTRA_EFFECT_MODES = ("loss", "group")


def check_extra_effects(extra_effects: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Reject anything other than ``loss`` and ``group`` before a run starts."""
    modes = tuple(str(part) for part in extra_effects)
    unknown = set(modes) - set(EXTRA_EFFECT_MODES)
    if unknown:
        raise ValueError(
            f"extra_effects must be a subset of {sorted(EXTRA_EFFECT_MODES)}, got {sorted(unknown)}"
        )
    return modes


def extra_effect_rows(
    adapter: Adapter,
    sae: SAE,
    clips: list[Clip],
    features: np.ndarray,
    ablated: set[tuple[str, int]],
    modes: tuple[str, ...],
    log,
) -> dict[str, list]:
    """Full-video loss rows. The cosine scoreboard path does not read these."""
    rows: dict[str, list] = {key: [] for key in ("feature_id", "clip_id", "window", "t", "effect", "effect_kind")}
    if "loss" in modes:
        log("full-video single-feature loss")
        by_feature: dict[int, list] = {}
        for clip in clips:
            for feature_id in range(features.shape[1]):
                if (clip.clip_id, feature_id) in ablated:
                    by_feature.setdefault(feature_id, []).append(clip)
        for feature_id, targets in by_feature.items():
            edit = ablate_edit(sae, feature_id)
            part = loss_frame_rows(adapter, targets, edit, [feature_id], loss_effect_kind(False, "video"))
            for key in rows:
                rows[key].extend(part[key])
    if "group" in modes:
        log("full-video group loss")
        group = coactive_group(features, k=min(GROUP_ABLATION_SIZE, features.shape[1]))
        if group:
            edit = ablate_features_edit(sae, group)
            part = loss_frame_rows(adapter, clips, edit, group, loss_effect_kind(True, "video"))
            for key in rows:
                rows[key].extend(part[key])
    return rows
