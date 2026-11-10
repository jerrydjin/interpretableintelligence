"""Deterministic synthetic world model with planted causal features and decoys.

Three cues (stripe, pulse, corner) are written into the activations along
directions the predictor actually reads. Three decoys (flicker, band, speckle)
fire just as clearly and just as often, on directions the predictor ignores.

``load(random_init=True)`` scrambles tokens before the random encoder, so the
same cues do not line up with the same positions and an SAE trained on that
model does not reproduce the planted firing patterns.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np
import torch

from physint.adapters.base import Clip, Grid, HookHandle, Window

CAUSAL_CUES: tuple[str, ...] = ("stripe", "pulse", "corner")
DECOY_CUES: tuple[str, ...] = ("flicker", "band", "speckle")
CUE_NAMES: tuple[str, ...] = CAUSAL_CUES + DECOY_CUES

N_FRAMES = 16
HEIGHT = 16
WIDTH = 16
N_CHANNELS = 8
TUBELET = 2
PATCH_PX = 4
D_MODEL = 64
PRED_DIM = 8
SITES_PER_CUE = 4
READOUT_SCALE = 0.85
ACT_NOISE = 0.01
# Constant head bias, orthogonal to the cue subspace, so cosine distance
# measures a real rotation of the predictor output instead of a collapse to 0.
PRED_BIAS = 4.0
FPS = 8.0

# Geometry of one clip. grid t,h,w = frames/tubelet, height/patch, width/patch.
GRID = Grid(
    t=N_FRAMES // TUBELET,
    h=HEIGHT // PATCH_PX,
    w=WIDTH // PATCH_PX,
    tubelet=TUBELET,
    patch_px=PATCH_PX,
)


def _generator(seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *[int(p) for p in parts]]))


def unravel(flat: int, t: int, h: int, w: int) -> tuple[int, int, int]:
    wi = int(flat % w)
    hi = int((flat // w) % h)
    ti = int(flat // (w * h))
    return ti, hi, wi


def site_values(clip_index: int) -> list[float]:
    """One peak per clip, then weaker sites on the same cue. Peaks dominate top-20."""
    peak = 1.0 + 0.01 * (clip_index % 17)
    return [peak, 0.45, 0.40, 0.35]


def cue_sites(clip_index: int, cue_index: int, seed: int, n_tokens: int) -> list[tuple[int, float]]:
    rng = _generator(seed, clip_index, cue_index, 11)
    positions = rng.choice(n_tokens, size=SITES_PER_CUE, replace=False)
    values = site_values(clip_index)
    return [(int(p), float(v)) for p, v in zip(positions, values, strict=True)]


def _paint_cues(frames: np.ndarray, clip_index: int, seed: int, cues: tuple[str, ...]) -> None:
    unknown = [name for name in cues if name not in CUE_NAMES]
    if unknown:
        known = ", ".join(CUE_NAMES)
        raise ValueError(f"unknown cue(s) {unknown}. Known cues: {known}")
    wanted = set(cues)
    n_tokens = GRID.t * GRID.h * GRID.w
    for cue_index, name in enumerate(CUE_NAMES):
        if name not in wanted:
            continue
        for flat, value in cue_sites(clip_index, cue_index, seed, n_tokens):
            ti, hi, wi = unravel(flat, GRID.t, GRID.h, GRID.w)
            frames[
                ti * TUBELET : (ti + 1) * TUBELET,
                hi * PATCH_PX : (hi + 1) * PATCH_PX,
                wi * PATCH_PX : (wi + 1) * PATCH_PX,
                cue_index,
            ] = value


def render_clip(clip_index: int, seed: int) -> np.ndarray:
    """Paint each cue into its own channel so the encoder can read it back exactly."""
    frames = np.zeros((N_FRAMES, HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)
    rng = _generator(seed, clip_index, 99)
    frames[..., N_CUES:] = rng.normal(0.0, 0.02, size=frames[..., N_CUES:].shape).astype(np.float32)
    _paint_cues(frames, clip_index, seed, CUE_NAMES)
    return frames


def render_cues(clip_index: int, seed: int, cues: tuple[str, ...] | list[str]) -> np.ndarray:
    """Paint only the named cues, using the same sites as ``render_clip``.

    A one-cue clip puts that cue's direction into the layer and leaves the
    others at zero. Distractor channels stay zero; the encoder does not read them.
    """
    frames = np.zeros((N_FRAMES, HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)
    _paint_cues(frames, clip_index, seed, tuple(cues))
    return frames


def pool_frames(frames: np.ndarray) -> np.ndarray:
    """Average-pool tubelet × patch. Returns [t, h, w, C]."""
    if frames.shape != (N_FRAMES, HEIGHT, WIDTH, N_CHANNELS):
        raise ValueError(
            f"fake clips must have shape {(N_FRAMES, HEIGHT, WIDTH, N_CHANNELS)}, got {frames.shape}"
        )
    view = np.ascontiguousarray(frames).reshape(
        GRID.t, TUBELET, GRID.h, PATCH_PX, GRID.w, PATCH_PX, N_CHANNELS
    )
    return view.mean(axis=(1, 3, 5))


def cue_volume(frames: np.ndarray) -> np.ndarray:
    """Cue strength on the grid, shape [n_cues, t, h, w], in CUE_NAMES order."""
    pooled = pool_frames(frames)
    return np.moveaxis(pooled[..., :N_CUES], -1, 0).astype(np.float32, copy=False)


N_CUES = len(CUE_NAMES)
N_CAUSAL = len(CAUSAL_CUES)


def generate_clips(out_dir, n: int, seed: int) -> None:
    """Write ``clip-XXX.npy`` arrays and a ``clips.json`` index."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if n < 1:
        raise ValueError("n must be positive")
    width = max(3, len(str(n - 1)))
    entries = []
    for index in range(n):
        clip_id = f"clip-{index:0{width}d}"
        path = out / f"{clip_id}.npy"
        np.save(path, render_clip(index, seed))
        entries.append({"clip_id": clip_id, "file": path.name, "fps": FPS})
    payload = {
        "fps": FPS,
        "seed": seed,
        "n_frames": N_FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "channels": N_CHANNELS,
        "clips": entries,
    }
    (out / "clips.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class FakeAdapter:
    """Tiny position-wise predictor. One instrumented layer, id chosen by the run."""

    name = "fake"

    def __init__(self, layer: int, device: torch.device | None = None) -> None:
        if layer < 0:
            raise ValueError(f"layer must be >= 0, got {layer}")
        self.layer = int(layer)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self._random_init = False
        self._loaded = False
        self._hooks: list[HookHandle] = []
        self._cache: dict[tuple[str, int], torch.Tensor] = {}
        self.dirs: torch.Tensor | None = None
        self.pred_basis: torch.Tensor | None = None
        self.pred_bias: torch.Tensor | None = None
        self.perm: torch.Tensor | None = None
        self.rand_proj: torch.Tensor | None = None
        self.rand_readout: torch.Tensor | None = None

    @property
    def d_model(self) -> int:
        return D_MODEL

    @property
    def random_init(self) -> bool:
        return self._random_init

    def grid(self) -> Grid:
        return GRID

    def load(self, random_init: bool = False) -> None:
        self._random_init = bool(random_init)
        self._cache.clear()
        self.dirs = _orthonormal_rows(N_CUES, D_MODEL, seed=0).to(self.device)
        self.pred_basis = _cue_basis().to(self.device)
        self.pred_bias = _bias_vector().to(self.device)
        n_tokens = GRID.t * GRID.h * GRID.w
        gen = torch.Generator(device="cpu")
        gen.manual_seed(7)
        self.perm = torch.randperm(n_tokens, generator=gen)
        self.rand_proj = _orthonormal_rows(N_CUES, D_MODEL, seed=8).to(self.device)
        self.rand_readout = _orthonormal_rows(N_CAUSAL, D_MODEL, seed=9).to(self.device)
        self._loaded = True

    def model_sha(self) -> str:
        self._ensure_loaded()
        assert self.dirs is not None and self.pred_basis is not None and self.pred_bias is not None
        digest = hashlib.sha256()
        digest.update(b"fake-v1")
        digest.update(self.grid().as_dict().__repr__().encode())
        digest.update(str(READOUT_SCALE).encode())
        digest.update(np.float32(ACT_NOISE).tobytes())
        digest.update(np.float32(PRED_BIAS).tobytes())
        for name, tensor in (
            ("dirs", self.dirs),
            ("pred_basis", self.pred_basis),
            ("pred_bias", self.pred_bias),
        ):
            digest.update(name.encode())
            array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def windows(self, clip: Clip) -> list[Window]:
        self._check_frames(clip.frames)
        return [Window(index=0, start_frame=0, frames=clip.frames)]

    def mask(self, window: Window) -> torch.Tensor:
        del window
        t = torch.arange(GRID.t, device=self.device)
        h = torch.arange(GRID.h, device=self.device)
        w = torch.arange(GRID.w, device=self.device)
        return (t[:, None, None] + h[None, :, None] + w[None, None, :]) % 2 == 0

    @contextmanager
    def hook(self, layer: int, mode: str = "capture", fn=None):
        if mode not in {"capture", "edit", "zero"}:
            raise ValueError(f"hook mode must be capture|edit|zero, got {mode!r}")
        if mode == "edit" and fn is None:
            raise ValueError("edit hook requires fn")
        if layer != self.layer:
            raise ValueError(f"fake adapter is bound to layer {self.layer}, got hook on {layer}")
        handle = HookHandle(layer=layer, mode=mode, fn=fn, captured=[])  # type: ignore[arg-type]
        self._hooks.append(handle)
        try:
            yield handle
        finally:
            self._hooks.remove(handle)

    def encode(self, clip: Clip, window: Window) -> torch.Tensor:
        x = self.raw_activations(clip, window)
        return self._apply_hooks(x)

    def predict_masked(self, clip: Clip, window: Window, edit=None):
        x = self.raw_activations(clip, window)
        if edit is not None:
            x = edit(x)
        else:
            x = self._apply_hooks(x)
        pred = self._readout(x)
        target = self._target(window)
        m = self.mask(window)
        return pred[m], target[m]

    def raw_activations(self, clip: Clip, window: Window) -> torch.Tensor:
        self._ensure_loaded()
        self._check_frames(window.frames)
        key = (clip.clip_id, window.index)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._compute_raw(clip, window)
            self._cache[key] = cached
        return cached.clone()

    def _compute_raw(self, clip: Clip, window: Window) -> torch.Tensor:
        assert self.dirs is not None and self.perm is not None and self.rand_proj is not None
        frames = torch.as_tensor(window.frames, dtype=torch.float32, device=self.device)
        pooled = _pool_torch(frames)
        cues = pooled[..., :N_CUES]
        flat = cues.reshape(-1, N_CUES)
        if self._random_init:
            scrambled = flat[self.perm.to(self.device)]
            acts = scrambled @ self.rand_proj
            noise = self._noise(clip.clip_id, acts.shape)
            return (acts + noise).reshape(GRID.t, GRID.h, GRID.w, D_MODEL)
        acts = flat @ self.dirs
        noise = self._noise(clip.clip_id, acts.shape)
        return (acts + noise).reshape(GRID.t, GRID.h, GRID.w, D_MODEL)

    def _noise(self, clip_id: str, shape: torch.Size) -> torch.Tensor:
        digest = hashlib.sha256(f"act-noise:{self._random_init}:{clip_id}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        noise = torch.randn(shape, generator=gen, dtype=torch.float32) * ACT_NOISE
        return noise.to(self.device)

    def _readout(self, acts: torch.Tensor) -> torch.Tensor:
        assert (
            self.dirs is not None
            and self.pred_basis is not None
            and self.pred_bias is not None
            and self.rand_readout is not None
        )
        flat = acts.reshape(-1, D_MODEL)
        if self._random_init:
            coeffs = flat @ self.rand_readout.T
        else:
            coeffs = flat @ self.dirs[:N_CAUSAL].T
        pred = self.pred_bias + READOUT_SCALE * (coeffs @ self.pred_basis)
        return pred.reshape(GRID.t, GRID.h, GRID.w, PRED_DIM)

    def _target(self, window: Window) -> torch.Tensor:
        assert self.pred_basis is not None and self.pred_bias is not None
        frames = torch.as_tensor(window.frames, dtype=torch.float32, device=self.device)
        pooled = _pool_torch(frames)
        causal = pooled[..., :N_CAUSAL].reshape(-1, N_CAUSAL)
        target = self.pred_bias + causal @ self.pred_basis
        return target.reshape(GRID.t, GRID.h, GRID.w, PRED_DIM)

    def _apply_hooks(self, x: torch.Tensor) -> torch.Tensor:
        for handle in self._hooks:
            if handle.layer != self.layer:
                continue
            if handle.mode == "capture":
                handle.captured.append(x.detach().clone())
            elif handle.mode == "zero":
                x = torch.zeros_like(x)
            elif handle.mode == "edit":
                assert handle.fn is not None
                x = handle.fn(x)
        return x

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("call load() before using the fake adapter")

    @staticmethod
    def _check_frames(frames: np.ndarray) -> None:
        expected = (N_FRAMES, HEIGHT, WIDTH, N_CHANNELS)
        if frames.shape != expected:
            raise ValueError(f"fake adapter expected frames {expected}, got {frames.shape}")


def _cue_basis() -> torch.Tensor:
    """Orthonormal cue rows in every pred dimension except the bias axis."""
    rows = _orthonormal_rows(N_CAUSAL, PRED_DIM - 1, seed=1)
    basis = torch.zeros(N_CAUSAL, PRED_DIM)
    basis[:, 1:] = rows
    return basis.contiguous()


def _bias_vector() -> torch.Tensor:
    bias = torch.zeros(PRED_DIM)
    bias[0] = PRED_BIAS
    return bias


def _orthonormal_rows(rows: int, cols: int, seed: int) -> torch.Tensor:
    if rows > cols:
        raise ValueError(f"need cols >= rows to orthonormalize, got {rows} x {cols}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    raw = torch.randn(cols, rows, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw)
    return q[:, :rows].T.contiguous()


def _pool_torch(frames: torch.Tensor) -> torch.Tensor:
    view = frames.reshape(GRID.t, TUBELET, GRID.h, PATCH_PX, GRID.w, PATCH_PX, N_CHANNELS)
    return view.mean(dim=(1, 3, 5))
