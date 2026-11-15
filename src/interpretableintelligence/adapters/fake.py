"""Deterministic synthetic world model with planted causal features and decoys.

Three cues (stripe, pulse, corner) are written into the activations along
directions the predictor actually reads. Three decoys (flicker, band, speckle)
fire just as clearly and just as often, on directions the predictor ignores.

``load(random_init=True)`` scrambles tokens before the random encoder, so the
same cues do not line up with the same positions and an SAE trained on that
model does not reproduce the planted firing patterns.

``plant`` selects a fixture without changing the default cue world:

- ``cues``: the acceptance model above.
- ``group``: three co-active features that are redundant for the readout.
  Each one alone barely moves prediction loss. Zeroing all three does.
  Two temporal windows, so a full-video score has to read every window.
- ``null``: features fire, but the predictor does not read the layer.
  Ablation changes nothing. This is the false-flag calibration fixture.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np
import torch

from interpretableintelligence.adapters.base import Clip, Grid, HookHandle, Window
from interpretableintelligence.thresholds import GROUP_ABLATION_SIZE

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

# Planted policy case. A loud direction dominates activation magnitude and is
# invisible to the action head. A quiet direction is what the log-likelihood
# of the taken action actually depends on. Both are orthogonal to the cue
# readout, so prediction_loss is unchanged.
POLICY_LOUD_SCALE = 4.0
POLICY_QUIET_SCALE = 0.2
POLICY_ACTION_GAIN = 5.0
POLICY_ACTION_TARGET = 1.5
POLICY_ACTION_SIGMA = 1.0

# Redundant group fixture. Scales differ slightly so each feature has a small
# marginal loss change; the readout uses max(c0, c1, c2), so only the joint
# ablation removes the signal. No activation noise: firings stay sparse.
PLANT_MODES = ("cues", "group", "null")
GROUP_SCALES = (1.0, 1.02, 1.04)
GROUP_SITES = 6
GROUP_DECOY_FEATURES = 3
NULL_SITES = 8
NULL_FEATURES = GROUP_ABLATION_SIZE

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


def _clip_index(clip_id: str) -> int:
    digits = "".join(ch for ch in clip_id if ch.isdigit())
    return int(digits) if digits else 0


def _choice(clip_index: int, window_index: int, n_tokens: int, n_sites: int, salt: int) -> np.ndarray:
    rng = _generator(clip_index, window_index, salt)
    return rng.choice(n_tokens, size=n_sites, replace=False)


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


FAILURE_LABEL_MODES = ("planted", "null")
FAILURE_LABELS_FILE = "failures.json"
# The planted failure: an episode fails when the stripe cue's strongest site in the
# first half of the clip is above FAILURE_LEVEL, i.e. its peak or its second site
# (0.45) lands there, not only the weaker 0.40 / 0.35 sites. Onset is the end of the
# clip, so a probe that sees only tokens at least 1 s before onset sees exactly that
# first half. The level sits between two stripe site values, so pooling has to
# happen along the stripe direction: max-pooling raw coordinates or random
# directions mixes in the other cues' peaks.
FAILURE_CUE = "stripe"
FAILURE_LEVEL = 0.425


def failure_labels(n: int, seed: int, mode: str) -> dict:
    """Seeded episode failure labels for ``generate_clips(n, seed)``, as ``ii probe --labels`` reads them.

    ``planted``: failure iff the largest ``FAILURE_CUE`` site in a time bin before
    ``GRID.t // 2`` is above ``FAILURE_LEVEL``. Only the stripe direction carries
    the signal. ``null``: a seeded fair coin per clip, independent of every pixel.
    Onset is the clip end (``N_FRAMES / FPS`` seconds) for failures.
    """
    if mode not in FAILURE_LABEL_MODES:
        raise ValueError(f"failure label mode must be one of {FAILURE_LABEL_MODES}, got {mode!r}")
    if n < 1:
        raise ValueError("n must be positive")
    width = max(3, len(str(n - 1)))
    n_tokens = GRID.t * GRID.h * GRID.w
    cue_index = CUE_NAMES.index(FAILURE_CUE)
    onset = N_FRAMES / FPS
    episodes = []
    for index in range(n):
        if mode == "planted":
            early = [
                value
                for flat, value in cue_sites(index, cue_index, seed, n_tokens)
                if unravel(flat, GRID.t, GRID.h, GRID.w)[0] < GRID.t // 2
            ]
            failure = max(early, default=0.0) > FAILURE_LEVEL
        else:
            failure = bool(_generator(seed, index, 4242).random() < 0.5)
        episodes.append({"clip_id": f"clip-{index:0{width}d}", "failure": bool(failure), "onset_s": onset if failure else None})
    source = f"fake-{mode}-v1" + (f":{FAILURE_CUE}-first-half-above-{FAILURE_LEVEL:g}" if mode == "planted" else ":coin")
    return {"source": source, "seed": int(seed), "episodes": episodes}


class FakeAdapter:
    """Tiny position-wise predictor. One instrumented layer, id chosen by the run."""

    name = "fake"

    def __init__(
        self,
        layer: int,
        device: torch.device | None = None,
        *,
        tiny: bool = False,
        plant: str = "cues",
    ) -> None:
        del tiny
        if layer < 0:
            raise ValueError(f"layer must be >= 0, got {layer}")
        if plant not in PLANT_MODES:
            raise ValueError(f"plant must be one of {PLANT_MODES}, got {plant!r}")
        self.layer = int(layer)
        self.plant = plant
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
        self._policy_case = False
        self.loud_dir: torch.Tensor | None = None
        self.quiet_dir: torch.Tensor | None = None
        self.action_dir: torch.Tensor | None = None
        self.loud_scale = POLICY_LOUD_SCALE
        self.quiet_scale = POLICY_QUIET_SCALE
        self.action_gain = POLICY_ACTION_GAIN
        self.action_target = 0.0
        self.action_sigma = POLICY_ACTION_SIGMA
        # Directions of the group/null fixture, row-normalized. Index 0..k-1 are
        # the co-active group; anything after that is a decoy the readout ignores.
        self.feature_dirs: torch.Tensor | None = None
        self.group_basis: torch.Tensor | None = None

    @property
    def d_model(self) -> int:
        return D_MODEL

    @property
    def random_init(self) -> bool:
        return self._random_init

    def grid(self) -> Grid:
        if self.plant != "group":
            return GRID
        t, h, w = self._group_grid()
        return Grid(t=t, h=h, w=w, tubelet=TUBELET, patch_px=PATCH_PX)

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
        loud, quiet = _policy_directions(self.dirs.detach().cpu())
        self.loud_dir = loud.to(self.device)
        self.quiet_dir = quiet.to(self.device)
        self.action_dir = self.quiet_dir
        self._policy_case = False
        self.action_target = 0.0
        self._loaded = True
        if self.plant != "cues":
            self._init_plant()

    def plant_policy_case(self) -> None:
        """Add a loud non-causal direction and a quiet action-causal one.

        Magnitude ranking prefers the loud direction. ``action_loglik``
        attribution prefers the quiet one, because the taken action is only
        explained by that direction. Call ``load`` first. Prediction loss is
        unchanged: both directions are orthogonal to the cue readout.
        """
        self._ensure_loaded()
        self._policy_case = True
        self.loud_scale = POLICY_LOUD_SCALE
        self.quiet_scale = POLICY_QUIET_SCALE
        self.action_gain = POLICY_ACTION_GAIN
        self.action_target = POLICY_ACTION_TARGET
        self.action_sigma = POLICY_ACTION_SIGMA
        self.action_dir = self.quiet_dir
        self._cache.clear()

    def model_sha(self) -> str:
        self._ensure_loaded()
        assert self.dirs is not None and self.pred_basis is not None and self.pred_bias is not None
        digest = hashlib.sha256()
        digest.update(b"fake-v1")
        if self.plant != "cues":
            digest.update(self.plant.encode())
            assert self.feature_dirs is not None
            planted = self.feature_dirs.detach().cpu().numpy().astype(np.float32, copy=False)
            digest.update(np.ascontiguousarray(planted).tobytes())
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

    def cache_spec(self, clip: Clip) -> dict:
        """What besides weights, layer and frames decides this adapter's tensors (``act_cache``).

        The activation noise is seeded by the clip id, so the id is part of the key.
        """
        return {"adapter": "fake-v1", "plant": self.plant, "policy_case": bool(self._policy_case), "clip_id": clip.clip_id}

    def windows(self, clip: Clip) -> list[Window]:
        self._check_frames(clip.frames)
        if self.plant != "group":
            return [Window(index=0, start_frame=0, frames=clip.frames)]
        mid = N_FRAMES // 2
        return [
            Window(index=0, start_frame=0, frames=clip.frames[:mid]),
            Window(index=1, start_frame=mid, frames=clip.frames[mid:]),
        ]

    def mask(self, window: Window) -> torch.Tensor:
        if self.plant != "group":
            del window
            t = torch.arange(GRID.t, device=self.device)
            h = torch.arange(GRID.h, device=self.device)
            w = torch.arange(GRID.w, device=self.device)
            return (t[:, None, None] + h[None, :, None] + w[None, None, :]) % 2 == 0
        gt, gh, gw = self._frame_grid(window.frames)
        return self._checker_mask(gt, gh, gw)

    def visible_mean_baseline(self, clip: Clip, window: Window) -> torch.Tensor:
        """Mean of the unmasked target vectors, repeated on the masked cells."""
        self._ensure_loaded()
        target = self._target(clip, window)
        keep = self.mask(window)
        visible = target[~keep]
        if visible.shape[0] == 0:
            raise RuntimeError("mask leaves no visible patches")
        mean = visible.mean(dim=0)
        return mean.expand_as(target[keep]).clone()

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
        target = self._target(clip, window)
        m = self.mask(window)
        return pred[m], target[m]

    def attribution_objective(self, clip: Clip, window: Window, objective: str, edit=None) -> torch.Tensor:
        """``prediction_loss`` (masked MSE) or ``action_loglik`` (taken-action log density)."""
        if objective == "prediction_loss":
            pred, target = self.predict_masked(clip, window, edit=edit)
            return ((pred - target) ** 2).mean()
        if objective == "action_loglik":
            return self._action_loglik(clip, window, edit=edit)
        raise ValueError(
            f"unknown attribution objective {objective!r}. Expected prediction_loss or action_loglik."
        )

    def raw_activations(self, clip: Clip, window: Window) -> torch.Tensor:
        self._ensure_loaded()
        if self.plant == "group":
            self._check_group_frames(window.frames)
        else:
            self._check_frames(window.frames)
        key = (clip.clip_id, window.index)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._compute_raw(clip, window)
            self._cache[key] = cached
        return cached.clone()

    def _compute_raw(self, clip: Clip, window: Window) -> torch.Tensor:
        if self.plant in {"group", "null"}:
            return self._plant_activations(clip, window)
        assert self.dirs is not None and self.perm is not None and self.rand_proj is not None
        frames = torch.as_tensor(window.frames, dtype=torch.float32, device=self.device)
        pooled = _pool_torch(frames)
        cues = pooled[..., :N_CUES]
        flat = cues.reshape(-1, N_CUES)
        if self._random_init:
            scrambled = flat[self.perm.to(self.device)]
            acts = scrambled @ self.rand_proj
        else:
            acts = flat @ self.dirs
        noise = self._noise(clip.clip_id, acts.shape)
        acts = acts + noise
        if self._policy_case:
            assert self.loud_dir is not None and self.quiet_dir is not None
            acts = acts + self.loud_scale * self.loud_dir + self.quiet_scale * self.quiet_dir
        return acts.reshape(GRID.t, GRID.h, GRID.w, D_MODEL)

    def _action_loglik(self, clip: Clip, window: Window, edit=None) -> torch.Tensor:
        """Mean Gaussian log-density of the taken action. Higher is better.

        The mean is ``action_gain * (activation · action_dir)``. In the planted
        policy case that direction is the quiet one, and the recorded action
        sits off the clean mean so the gradient does not vanish.
        """
        assert self.action_dir is not None
        acts = self.raw_activations(clip, window)
        if edit is not None:
            acts = edit(acts)
        else:
            acts = self._apply_hooks(acts)
        mu = self.action_gain * (acts * self.action_dir).sum(dim=-1)
        resid = (self.action_target - mu) / self.action_sigma
        return (-0.5 * resid.square()).mean()

    def _noise(self, clip_id: str, shape: torch.Size) -> torch.Tensor:
        digest = hashlib.sha256(f"act-noise:{self._random_init}:{clip_id}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        noise = torch.randn(shape, generator=gen, dtype=torch.float32) * ACT_NOISE
        return noise.to(self.device)

    def _readout(self, acts: torch.Tensor) -> torch.Tensor:
        if self.plant == "null":
            return self._bias_field(acts.shape[0], acts.shape[1], acts.shape[2])
        if self.plant == "group":
            return self._group_readout(acts)
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

    def _target(self, clip: Clip, window: Window) -> torch.Tensor:
        if self.plant == "null":
            t, h, w = self._frame_grid(window.frames)
            return self._bias_field(t, h, w)
        if self.plant == "group":
            return self._group_target(clip, window)
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

    def _init_plant(self) -> None:
        """Orthonormal feature directions for the group and null fixtures."""
        if self.plant == "group":
            n_feat = GROUP_ABLATION_SIZE + GROUP_DECOY_FEATURES
            seed = 21
        else:
            n_feat = NULL_FEATURES
            seed = 22
        dirs = _orthonormal_rows(n_feat, D_MODEL, seed=seed).to(self.device)
        self.feature_dirs = dirs
        basis = torch.zeros(PRED_DIM)
        basis[1] = 1.0
        self.group_basis = basis.to(self.device)

    def _plant_activations(self, clip: Clip, window: Window) -> torch.Tensor:
        assert self.feature_dirs is not None
        t, h, w = self._frame_grid(window.frames)
        planted = self._planted_coeffs(clip, window, t, h, w)
        flat = planted.reshape(-1, planted.shape[-1]) @ self.feature_dirs
        acts = flat.reshape(t, h, w, D_MODEL)
        if self._random_init:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(50_000 + _clip_index(clip.clip_id) * 10 + int(window.index))
            perm = torch.randperm(acts.shape[0] * acts.shape[1] * acts.shape[2], generator=gen)
            acts = acts.reshape(-1, D_MODEL)[perm].reshape_as(acts)
        return acts

    def _planted_coeffs(self, clip: Clip, window: Window, t: int, h: int, w: int) -> torch.Tensor:
        assert self.feature_dirs is not None
        n_feat = int(self.feature_dirs.shape[0])
        coeff = torch.zeros(t, h, w, n_feat, device=self.device)
        n_tokens = t * h * w
        index = _clip_index(clip.clip_id)
        n_group = GROUP_ABLATION_SIZE if self.plant == "group" else n_feat
        scales = GROUP_SCALES if self.plant == "group" else tuple(1.0 for _ in range(n_group))
        n_sites = GROUP_SITES if self.plant == "group" else NULL_SITES
        shared = _choice(index, int(window.index), n_tokens, n_sites, salt=11)
        for flat in shared:
            ti, hi, wi = unravel(int(flat), t, h, w)
            for feature, scale in enumerate(scales):
                coeff[ti, hi, wi, feature] = float(scale)
        if self.plant == "group":
            for decoy in range(GROUP_DECOY_FEATURES):
                positions = _choice(index, int(window.index), n_tokens, GROUP_SITES, salt=40 + decoy)
                for flat in positions:
                    ti, hi, wi = unravel(int(flat), t, h, w)
                    coeff[ti, hi, wi, n_group + decoy] = 1.0
        return coeff

    def _group_readout(self, acts: torch.Tensor) -> torch.Tensor:
        assert self.feature_dirs is not None and self.group_basis is not None and self.pred_bias is not None
        flat = acts.reshape(-1, D_MODEL)
        coeff = flat @ self.feature_dirs[:GROUP_ABLATION_SIZE].T
        shared = coeff.max(dim=-1).values
        pred = self.pred_bias + READOUT_SCALE * shared[:, None] * self.group_basis
        return pred.reshape(acts.shape[0], acts.shape[1], acts.shape[2], PRED_DIM)

    def _group_target(self, clip: Clip, window: Window) -> torch.Tensor:
        assert self.group_basis is not None and self.pred_bias is not None
        t, h, w = self._frame_grid(window.frames)
        planted = self._planted_coeffs(clip, window, t, h, w)
        shared = planted[..., :GROUP_ABLATION_SIZE].max(dim=-1).values.reshape(-1)
        pred = self.pred_bias + shared[:, None] * self.group_basis
        return pred.reshape(t, h, w, PRED_DIM)

    def _bias_field(self, t: int, h: int, w: int) -> torch.Tensor:
        assert self.pred_bias is not None
        return self.pred_bias.view(1, 1, 1, -1).expand(t, h, w, PRED_DIM).contiguous()

    def _checker_mask(self, t: int, h: int, w: int) -> torch.Tensor:
        tt = torch.arange(t, device=self.device)
        hh = torch.arange(h, device=self.device)
        ww = torch.arange(w, device=self.device)
        return (tt[:, None, None] + hh[None, :, None] + ww[None, None, :]) % 2 == 0

    @staticmethod
    def _group_grid() -> tuple[int, int, int]:
        return (N_FRAMES // 2) // TUBELET, HEIGHT // PATCH_PX, WIDTH // PATCH_PX

    @staticmethod
    def _frame_grid(frames: np.ndarray) -> tuple[int, int, int]:
        return (
            int(frames.shape[0] // TUBELET),
            int(frames.shape[1] // PATCH_PX),
            int(frames.shape[2] // PATCH_PX),
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("call load() before using the fake adapter")

    @staticmethod
    def _check_frames(frames: np.ndarray) -> None:
        expected = (N_FRAMES, HEIGHT, WIDTH, N_CHANNELS)
        if frames.shape != expected:
            raise ValueError(f"fake adapter expected frames {expected}, got {frames.shape}")

    @staticmethod
    def _check_group_frames(frames: np.ndarray) -> None:
        expected = (N_FRAMES // 2, HEIGHT, WIDTH, N_CHANNELS)
        if frames.shape != expected:
            raise ValueError(f"group window expected frames {expected}, got {frames.shape}")


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


def _policy_directions(cue_dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Loud and quiet unit vectors, orthogonal to each other and to the cues."""
    cue = cue_dirs.detach().to(dtype=torch.float32, device="cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(19)
    raw = torch.randn(cue.shape[1], 2, generator=gen, dtype=torch.float32)
    raw = raw - cue.T @ (cue @ raw)
    q, _ = torch.linalg.qr(raw)
    return q[:, 0].contiguous(), q[:, 1].contiguous()


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
