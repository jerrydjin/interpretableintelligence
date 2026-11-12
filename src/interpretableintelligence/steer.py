"""Feature steering for the viewer's playground.

Pick SAE features, edit them inside the instrumented layer, and see how the
model's masked prediction moves. This is exploration, not evidence: nothing
here writes to the run folder or the scoreboard, and a steering result never
makes a feature ``real``. The scoreboard gates stay the only route to that.

The edit keeps everything the chosen features do not explain, including the
SAE's reconstruction error::

    f        = sae.encode(x)                   # current activations of every feature
    f'_i     = new value for each chosen i     # ablate: 0, scale: f_i * v, clamp: v, add: f_i + v
    x_edited = x + sum_i (f'_i - f_i) * W_dec[:, i]

``ablate`` matches the scorer's single-feature ablation (``score.ablate_edit``).
``clamp`` and ``add`` write the feature on every token of the window, including
tokens where it was silent, which is what steering means.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from interpretableintelligence.adapters import get_adapter
from interpretableintelligence.adapters.base import Clip, Window
from interpretableintelligence.clips import load_clip
from interpretableintelligence.device import pick_device
from interpretableintelligence.explain import ExplainError, load_run
from interpretableintelligence.sae import SAE, load_sae
from interpretableintelligence.score import cosine_distance
from interpretableintelligence.thresholds import DEAD_ACT_EPS

KINDS = ("ablate", "scale", "clamp", "add")
MAX_FEATURES = 8
MAX_ABS_VALUE = 1e4


class SteerError(ExplainError):
    """The steering request cannot be run."""


@dataclass(frozen=True)
class SteerRequest:
    clip_id: str
    window: int
    feature_ids: tuple[int, ...]
    kind: str
    value: float

    @classmethod
    def parse(cls, payload: object, n_features: int) -> "SteerRequest":
        if not isinstance(payload, dict):
            raise SteerError("request body must be a JSON object")
        clip_id = payload.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise SteerError("clip_id must be a non-empty string")
        window = payload.get("window", 0)
        if not _is_int(window) or window < 0:
            raise SteerError("window must be a non-negative integer")
        raw = payload.get("features")
        if _is_int(raw):
            raw = [raw]
        if not isinstance(raw, list) or not raw or not all(_is_int(item) for item in raw):
            raise SteerError("features must be a feature id or a non-empty list of feature ids")
        ids = tuple(sorted(set(int(item) for item in raw)))
        if len(ids) > MAX_FEATURES:
            raise SteerError(f"steer at most {MAX_FEATURES} features at once")
        bad = [item for item in ids if item < 0 or item >= n_features]
        if bad:
            raise SteerError(f"unknown feature ids {bad}; this SAE has features 0..{n_features - 1}")
        kind = payload.get("kind", "ablate")
        if kind not in KINDS:
            raise SteerError(f"kind must be one of {', '.join(KINDS)}")
        value = payload.get("value", 0.0)
        if kind == "ablate":
            value = 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SteerError("value must be a finite number")
        if abs(float(value)) > MAX_ABS_VALUE:
            raise SteerError(f"value must be within ±{MAX_ABS_VALUE:g}")
        return cls(clip_id=clip_id, window=int(window), feature_ids=ids, kind=kind, value=float(value))


def steer_edit(sae: SAE, feature_ids: tuple[int, ...] | list[int], kind: str, value: float):
    """Edit function for ``adapter.predict_masked(..., edit=...)``. See the module docstring."""
    if kind not in KINDS:
        raise SteerError(f"kind must be one of {', '.join(KINDS)}")
    index = torch.tensor(list(feature_ids), dtype=torch.long, device=sae.W_dec.device)
    columns = sae.W_dec.index_select(1, index)  # [d_model, k]

    def edit(acts: torch.Tensor) -> torch.Tensor:
        flat = acts.reshape(-1, acts.shape[-1])
        current = sae.encode(flat.to(columns.dtype)).index_select(1, index)  # [n, k]
        if kind == "ablate":
            target = torch.zeros_like(current)
        elif kind == "scale":
            target = current * value
        elif kind == "clamp":
            target = torch.full_like(current, value)
        else:
            target = current + value
        delta = (target - current) @ columns.T
        return (flat + delta.to(flat.dtype)).reshape_as(acts)

    return edit


class Steerer:
    """Loads a run's adapter and SAE once and answers steering requests for its clips."""

    def __init__(self, run_dir: Path, clips_dir: Path | None = None) -> None:
        from interpretableintelligence.view import _resolve_pixels  # noqa: PLC0415 (view stays adapter-free)

        self.ctx = load_run(Path(run_dir))
        ctx = self.ctx
        self.device = pick_device()
        try:
            self.sae, self.config = load_sae(ctx.run_dir / "sae", self.device)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            raise SteerError(f"SAE for run {ctx.run_id} could not be loaded ({exc})", run_id=ctx.run_id) from exc
        try:
            adapter_cls = get_adapter(ctx.adapter_name)
        except ValueError as exc:
            raise SteerError(str(exc), run_id=ctx.run_id) from exc
        self.adapter = adapter_cls(layer=ctx.layer, device=self.device)
        self.adapter.load(random_init=False)
        loaded_sha = self.adapter.model_sha()
        if loaded_sha != ctx.model_sha:
            raise SteerError(
                f"loaded {ctx.adapter_name} weights sha {loaded_sha} does not match run {ctx.run_id} model sha {ctx.model_sha}",
                run_id=ctx.run_id,
            )
        sae_block = ctx.scoreboard.get("sae")
        self.sae_usable = isinstance(sae_block, dict) and sae_block.get("usable") is True
        self._meta = {str(clip["clip_id"]): clip for clip in ctx.manifest.get("clips") or [] if isinstance(clip, dict)}
        self._pixels = _resolve_pixels(ctx.manifest, ctx.run_dir, None if clips_dir is None else Path(clips_dir))
        self._clips: dict[str, Clip] = {}
        self._lock = threading.Lock()

    def steer(self, payload: object) -> dict:
        request = SteerRequest.parse(payload, self.config.n_features)
        with self._lock:
            clip = self._clip(request.clip_id)
            windows = self.adapter.windows(clip)
            if request.window >= len(windows):
                raise SteerError(f"clip {request.clip_id} has {len(windows)} window(s); window {request.window} does not exist")
            return self._run(request, clip, windows[request.window])

    def _clip(self, clip_id: str) -> Clip:
        cached = self._clips.get(clip_id)
        if cached is not None:
            return cached
        meta = self._meta.get(clip_id)
        if meta is None:
            raise SteerError(f"run {self.ctx.run_id} has no clip {clip_id}")
        loaded = self._pixels.get(clip_id)
        if loaded is None or loaded.status != "ok" or loaded.path is None:
            message = loaded.message if loaded is not None and loaded.message else "Clip file not found."
            raise SteerError(f"clip {clip_id} cannot be steered: {message}")
        clip = load_clip({**meta, "path": str(loaded.path), "fps": meta.get("fps", 8.0), "sha256": meta.get("sha256", "")})
        self._clips = {clip_id: clip}  # keep one decoded clip in memory
        return clip

    def _run(self, request: SteerRequest, clip: Clip, window: Window) -> dict:
        adapter, sae = self.adapter, self.sae
        grid = adapter.grid()
        edit = steer_edit(sae, request.feature_ids, request.kind, request.value)
        with torch.no_grad():
            with adapter.hook(self.ctx.layer, mode="capture") as handle:
                adapter.encode(clip, window)
            acts = handle.captured[0].detach().to(dtype=torch.float32, device=sae.W_enc.device)
            features = sae.encode(acts.reshape(-1, sae.d_model)).reshape(grid.t, grid.h, grid.w, sae.n_features)
            index = torch.tensor(list(request.feature_ids), dtype=torch.long, device=features.device)
            chosen = features.index_select(3, index).sum(dim=3).cpu()
            baseline, target = adapter.predict_masked(clip, window)
            edited, _target = adapter.predict_masked(clip, window, edit=edit)
            dist = cosine_distance(baseline, edited).detach().cpu()
            norm = baseline.norm(dim=-1).clamp(min=1e-12)
            rel = ((edited - baseline).norm(dim=-1) / norm).detach().cpu()
            loss_clean = float(((baseline - target) ** 2).mean())
            loss_edited = float(((edited - target) ** 2).mean())
        coords = adapter.mask(window).detach().cpu().nonzero(as_tuple=False).tolist()
        shift = [[[None for _w in range(grid.w)] for _h in range(grid.h)] for _t in range(grid.t)]
        for (t, h, w), value in zip(coords, dist.tolist()):
            shift[t][h][w] = _finite(value)
        activation = [[[_finite(chosen[t, h, w].item()) for w in range(grid.w)] for h in range(grid.h)] for t in range(grid.t)]
        fired = int((chosen > DEAD_ACT_EPS).sum())
        return {
            "run_id": self.ctx.run_id,
            "model_sha": self.ctx.model_sha,
            "layer": self.ctx.layer,
            "clip_id": request.clip_id,
            "window": request.window,
            "features": list(request.feature_ids),
            "kind": request.kind,
            "value": request.value,
            "grid": {"t": grid.t, "h": grid.h, "w": grid.w},
            "shift": shift,
            "activation": activation,
            "summary": {
                "mean_shift": _finite(float(dist.mean())) if dist.numel() else 0.0,
                "max_shift": _finite(float(dist.max())) if dist.numel() else 0.0,
                "mean_rel_change": _finite(float(rel.mean())) if rel.numel() else 0.0,
                "loss_clean": _finite(loss_clean),
                "loss_edited": _finite(loss_edited),
                "loss_change": _finite(loss_edited - loss_clean),
                "tokens_where_feature_fired": fired,
                "tokens_total": grid.t * grid.h * grid.w,
            },
            "sae_usable": self.sae_usable,
            "evidence": False,
            "note": "Exploration only. Steering does not change the scoreboard; only the scored gates mark a feature real.",
        }


def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
