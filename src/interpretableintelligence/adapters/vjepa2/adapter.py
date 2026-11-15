"""V-JEPA 2 through the interpretableintelligence adapter protocol.

The checkpoint is ``facebook/vjepa2-vitl-fpc64-256`` via Hugging Face
transformers. Downloads use ``allow_patterns`` so the 5.1 GB
``original/model.pth`` is never fetched.

Two training details that the transformers forward does not reproduce:

* The context encoder sees visible tokens only. ``VJEPA2Model.forward`` runs
  the encoder on the whole clip, then gathers the context, so masked patches
  leak into every visible token.
* The training target is ``F.layer_norm`` of the target encoder output, with
  no affine. ``predictor_output.target_hidden_state`` stops at the encoder's
  own layernorm. Without the extra norm the predictor is no better than the
  mean of the visible patches.

Capture runs under ``torch.inference_mode``. Edits that the scorer
differentiates (the straight-through SAE estimate) are applied after that
boundary, so the prefix of the encoder is not on the autograd graph.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence.adapters.base import Clip, Grid, HookHandle, Window
from interpretableintelligence.device import ensure_mps_fallback

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
# Safetensors only. ``original/model.pth`` is the 5.1 GB raw Meta checkpoint.
WEIGHT_ALLOW_PATTERNS = ["*.json", "model.safetensors", "README.md"]
CONFIG_ALLOW_PATTERNS = ["config.json"]
# Edited grids per batched predictor forward in ``predict_masked_many``.
PREDICT_BATCH = 32

TUBELET = 2
PATCH = 16
FULL_FRAMES = 16
FULL_CROP = 256
TINY_FRAMES = 4
TINY_CROP = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Set before any MPS op. ``setdefault`` leaves an explicit user value alone.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def fetch_checkpoint(
    model_id: str = MODEL_ID,
    *,
    allow_patterns: list[str] | None = None,
) -> str:
    """Download a V-JEPA 2 snapshot, skipping ``original/*.pth`` by default."""
    from huggingface_hub import snapshot_download

    patterns = list(WEIGHT_ALLOW_PATTERNS if allow_patterns is None else allow_patterns)
    return snapshot_download(model_id, allow_patterns=patterns)


def tiny_config():
    """Randomly initialised V-JEPA 2 small enough to score on CPU."""
    from transformers import VJEPA2Config

    config = VJEPA2Config(
        crop_size=TINY_CROP,
        patch_size=PATCH,
        tubelet_size=TUBELET,
        frames_per_clip=TINY_FRAMES,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        mlp_ratio=1.0,
        pred_hidden_size=32,
        pred_num_attention_heads=4,
        pred_num_hidden_layers=1,
        pred_num_mask_tokens=2,
        pred_mlp_ratio=1.0,
        drop_path_rate=0.0,
        attention_probs_dropout_prob=0.0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return config


def generate_tiny_clips(out_dir, n: int, seed: int = 3) -> None:
    """Write ``n`` float clips shaped for ``VJEPA2Adapter(tiny=True)``.

    The default seed is a draw on which the tiny random predictor's zeroed loss
    stays above its clean loss for the default clip split. ``loss_recovered``
    needs that gap. Another seed can land on the other side for an untrained net.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    width = max(3, len(str(n - 1)))
    entries = []
    for index in range(n):
        clip_id = f"clip-{index:0{width}d}"
        frames = rng.random((TINY_FRAMES, TINY_CROP, TINY_CROP, 3), dtype=np.float32)
        path = out / f"{clip_id}.npy"
        np.save(path, frames)
        entries.append({"clip_id": clip_id, "file": path.name, "fps": 8.0})
    payload = {
        "fps": 8.0,
        "seed": seed,
        "n_frames": TINY_FRAMES,
        "height": TINY_CROP,
        "width": TINY_CROP,
        "channels": 3,
        "clips": entries,
    }
    (out / "clips.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class VJEPA2Adapter:
    """Instrumented V-JEPA 2. ``layer`` is 1-indexed and matches ``hidden_states[layer]``."""

    name = "vjepa2"

    def __init__(self, layer: int, device: torch.device | None = None, *, tiny: bool = False) -> None:
        if layer < 1:
            raise ValueError(f"layer must be >= 1 (hidden_states[layer]), got {layer}")
        self.layer = int(layer)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.tiny = bool(tiny)
        self.n_frames = TINY_FRAMES if self.tiny else FULL_FRAMES
        self.crop = TINY_CROP if self.tiny else FULL_CROP
        self._random_init = False
        self._loaded = False
        self._hooks: list[HookHandle] = []
        self.model = None
        self.model_name = "vjepa2-tiny" if self.tiny else MODEL_ID
        self._ctx: torch.Tensor | None = None
        self._tgt: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self._n_tokens = 0
        self._pixels_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._visible_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._layer_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._encoder_cache: dict[tuple[str, int], torch.Tensor] = {}
        ensure_mps_fallback(self.device)

    @property
    def d_model(self) -> int:
        self._ensure()
        return int(self.model.config.hidden_size)

    @property
    def random_init(self) -> bool:
        return self._random_init

    def grid(self) -> Grid:
        return Grid(
            t=self.n_frames // TUBELET,
            h=self.crop // PATCH,
            w=self.crop // PATCH,
            tubelet=TUBELET,
            patch_px=PATCH,
        )

    def load(self, random_init: bool = False) -> None:
        from transformers import VJEPA2Config, VJEPA2Model

        ensure_mps_fallback(self.device)
        self._random_init = bool(random_init)
        self._clear_caches()
        if self.tiny:
            config = tiny_config()
            # Distinct from load(random_init=True) so the control is not a copy.
            torch.manual_seed(1 if random_init else 0)
            model = VJEPA2Model(config)
        elif random_init:
            folder = fetch_checkpoint(self.model_name, allow_patterns=list(CONFIG_ALLOW_PATTERNS))
            config = VJEPA2Config.from_pretrained(folder)
            config._attn_implementation = "sdpa"
            torch.manual_seed(0)
            model = VJEPA2Model(config)
        else:
            folder = fetch_checkpoint(self.model_name)
            model = VJEPA2Model.from_pretrained(folder, attn_implementation="sdpa")
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model.to(self.device)
        n_layers = len(self.model.encoder.layer)
        if not 1 <= self.layer <= n_layers:
            kind = "tiny " if self.tiny else ""
            raise ValueError(
                f"layer must be between 1 and {n_layers} inclusive for this {kind}V-JEPA 2, got {self.layer}"
            )
        self._bind_mask()
        self._loaded = True

    def model_sha(self) -> str:
        self._ensure()
        digest = hashlib.sha256()
        digest.update(b"vjepa2-v1")
        digest.update(self.model_name.encode())
        digest.update(b"tiny" if self.tiny else b"full")
        digest.update(b"random" if self._random_init else b"weights")
        for name, tensor in self.model.state_dict().items():
            digest.update(name.encode())
            array = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def windows(self, clip: Clip) -> list[Window]:
        frames = np.asarray(clip.frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"vjepa2 clips must be [frames, height, width, 3], got {frames.shape}")
        if frames.shape[0] < 1:
            raise ValueError(f"{clip.clip_id} has no frames")
        index = _sample_indices(int(frames.shape[0]), self.n_frames)
        sampled = np.ascontiguousarray(frames[index])
        return [Window(index=0, start_frame=int(index[0]), frames=sampled)]

    def mask(self, window: Window) -> torch.Tensor:
        del window
        self._ensure()
        assert self._mask is not None
        return self._mask

    def visible_mean_baseline(self, clip: Clip, window: Window) -> torch.Tensor:
        """Predict every masked token as the layer-normed mean of the visible patches.

        This is the floor a trained V-JEPA predictor has to beat. The mean is
        taken in the target encoder's space, then passed through the same
        parameter-free layer norm as the training target.
        """
        self._ensure()
        assert self._ctx is not None and self._tgt is not None
        tokens = self._cached_encoder_tokens(clip, window)
        gathered = torch.gather(tokens, 1, _expand_ids(self._ctx, tokens.shape[-1]))
        mean = F.layer_norm(gathered.mean(dim=1), (tokens.shape[-1],)).squeeze(0)
        return mean.expand(int(self._tgt.shape[1]), tokens.shape[-1]).detach()

    @contextmanager
    def hook(self, layer: int, mode: str = "capture", fn=None):
        if mode not in {"capture", "edit", "zero"}:
            raise ValueError(f"hook mode must be capture|edit|zero, got {mode!r}")
        if mode == "edit" and fn is None:
            raise ValueError("edit hook requires fn")
        if layer != self.layer:
            raise ValueError(f"vjepa2 adapter is bound to layer {self.layer}, got hook on {layer}")
        handle = HookHandle(layer=layer, mode=mode, fn=fn, captured=[])  # type: ignore[arg-type]
        self._hooks.append(handle)
        try:
            yield handle
        finally:
            self._hooks.remove(handle)

    def encode(self, clip: Clip, window: Window) -> torch.Tensor:
        return self._apply_hooks(self.raw_activations(clip, window))

    def raw_activations(self, clip: Clip, window: Window) -> torch.Tensor:
        """Full-video residual at ``layer``, shaped ``[t, h, w, d]``. Hooks are not applied."""
        self._ensure()
        tokens = self._cached_layer_tokens(clip, window)
        grid = self.grid()
        return tokens[0].reshape(grid.t, grid.h, grid.w, -1).clone()

    def predict_masked(self, clip: Clip, window: Window, edit=None):
        self._ensure()
        assert self._ctx is not None and self._tgt is not None
        visible = self._cached_visible(clip, window)
        grid = _scatter_grid(visible, self._ctx, self.grid(), self._n_tokens)
        if edit is not None:
            grid = edit(grid)
        else:
            grid = self._apply_hooks(grid)
        expected = (self.grid().t, self.grid().h, self.grid().w, self.d_model)
        if tuple(grid.shape) != expected:
            raise ValueError(f"edit must return {expected}, got {tuple(grid.shape)}")
        context = _gather_ids(grid.reshape(1, self._n_tokens, grid.shape[-1]), self._ctx)
        hidden = self._finish_visible(context, self._ctx)
        pred = self._predictor(hidden, self._ctx, self._tgt)[0]
        target = self._training_target(self._masked_encoder_hidden(clip, window))
        if not pred.requires_grad:
            pred = pred.detach()
        return pred, target.detach()

    def predict_masked_many(self, clip: Clip, window: Window, edits) -> list[torch.Tensor]:
        """``predict_masked(clip, window, edit=e)[0]`` for every ``e`` in ``edits``, without grad.

        The edited grids go through the remaining encoder blocks and the
        predictor ``PREDICT_BATCH`` at a time instead of one forward each.
        """
        self._ensure()
        assert self._ctx is not None and self._tgt is not None
        visible = self._cached_visible(clip, window)
        grid = _scatter_grid(visible, self._ctx, self.grid(), self._n_tokens)
        expected = (self.grid().t, self.grid().h, self.grid().w, self.d_model)
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(edits), PREDICT_BATCH):
                chunk = edits[start : start + PREDICT_BATCH]
                edited = []
                for edit in chunk:
                    out = edit(grid)
                    if tuple(out.shape) != expected:
                        raise ValueError(f"edit must return {expected}, got {tuple(out.shape)}")
                    edited.append(out)
                n = len(edited)
                stacked = torch.stack(edited).reshape(n, self._n_tokens, -1)
                ctx = self._ctx.expand(n, -1)
                tgt = self._tgt.expand(n, -1)
                hidden = self._finish_visible(_gather_ids(stacked, ctx), ctx)
                preds.extend(pred.detach() for pred in self._predictor(hidden, ctx, tgt))
        return preds

    def attribution_objective(self, clip: Clip, window: Window, objective: str, edit=None) -> torch.Tensor:
        """Masked predictor MSE. V-JEPA has no action log-likelihood."""
        if objective != "prediction_loss":
            raise ValueError(
                f"vjepa2 attribution objective must be prediction_loss, got {objective!r}"
            )
        pred, target = self.predict_masked(clip, window, edit=edit)
        return ((pred - target) ** 2).mean()

    def encoder_output(self, clip: Clip, window: Window) -> torch.Tensor:
        """Target-encoder hidden states after the encoder layernorm, ``[t, h, w, d]``.

        This is the tensor transformers exposes as ``target_hidden_state`` (no
        extra layer norm). ``predict_masked``'s target is the training norm of it.
        """
        self._ensure()
        tokens = self._cached_encoder_tokens(clip, window)
        grid = self.grid()
        return tokens[0].reshape(grid.t, grid.h, grid.w, -1)

    def pixel_values(self, clip: Clip, window: Window) -> torch.Tensor:
        self._ensure()
        key = (clip.clip_id, window.index)
        cached = self._pixels_cache.get(key)
        if cached is None:
            cached = self._preprocess(window.frames).to(self.device)
            self._pixels_cache[key] = cached
        return cached

    def cache_spec(self, clip: Clip) -> dict:
        """What besides weights, layer and frames decides this adapter's tensors (``act_cache``)."""
        del clip
        return {
            "adapter": "vjepa2-v1",
            "frames": int(self.n_frames),
            "crop": int(self.crop),
            "tubelet": TUBELET,
            "patch": PATCH,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "context_tokens": None if self._ctx is None else int(self._ctx.shape[-1]),
        }

    def clean_state(self, clip: Clip, window: Window) -> dict[str, torch.Tensor]:
        """Full-video layer tokens, visible-only layer tokens and target-encoder tokens for one window."""
        self._ensure()
        return {
            "layer": self._cached_layer_tokens(clip, window),
            "visible": self._cached_visible(clip, window),
            "encoder": self._cached_encoder_tokens(clip, window),
        }

    def load_clean_state(self, clip: Clip, window: Window, state: dict[str, torch.Tensor]) -> None:
        """Put :meth:`clean_state` back, so encode and every prediction skip the encoder up to ``layer``."""
        self._ensure()
        assert self._ctx is not None
        width = self.d_model
        expected = {
            "layer": (1, self._n_tokens, width),
            "visible": (1, int(self._ctx.shape[-1]), width),
            "encoder": (1, self._n_tokens, width),
        }
        loaded = {}
        for name, shape in expected.items():
            tensor = state[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(f"cached {name} tokens {tuple(tensor.shape)} != {shape}")
            loaded[name] = tensor.to(device=self.device, dtype=torch.float32)
        key = (clip.clip_id, window.index)
        self._layer_cache[key] = loaded["layer"]
        self._visible_cache[key] = loaded["visible"]
        self._encoder_cache[key] = loaded["encoder"]

    def _training_target(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """No-affine layer norm applied on top of the target encoder.

        transformers' ``target_hidden_state`` omits this. V-JEPA 2 trains the
        predictor against the normalised tensor; without it the predictor does
        not beat the mean of the visible patches.
        """
        return F.layer_norm(encoder_hidden, (encoder_hidden.shape[-1],))

    def _visible_hidden(self, pixel_values: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Encoder hidden state at ``layer`` for visible tokens only.

        The gather happens before the blocks, and RoPE uses the original token
        ids. Running the blocks on the full clip (the transformers default)
        lets masked patches leak into the context.
        """
        with torch.inference_mode():
            embedded = self.model.encoder.embeddings(pixel_values)
            visible = torch.gather(embedded, 1, _expand_ids(ctx, embedded.shape[-1]))
        return self._run_blocks(visible, ctx, self.layer)

    def _masked_encoder_hidden(self, clip: Clip, window: Window) -> torch.Tensor:
        assert self._tgt is not None
        tokens = self._cached_encoder_tokens(clip, window)
        return torch.gather(tokens, 1, _expand_ids(self._tgt, tokens.shape[-1]))[0]

    def _cached_visible(self, clip: Clip, window: Window) -> torch.Tensor:
        assert self._ctx is not None
        key = (clip.clip_id, window.index)
        cached = self._visible_cache.get(key)
        if cached is None:
            cached = self._visible_hidden(self.pixel_values(clip, window), self._ctx)
            self._visible_cache[key] = cached
        return cached

    def _cached_layer_tokens(self, clip: Clip, window: Window) -> torch.Tensor:
        key = (clip.clip_id, window.index)
        cached = self._layer_cache.get(key)
        if cached is None:
            pixels = self.pixel_values(clip, window)
            with torch.inference_mode():
                embedded = self.model.encoder.embeddings(pixels)
            cached = self._run_blocks(embedded, None, self.layer)
            self._layer_cache[key] = cached
        return cached

    def _cached_encoder_tokens(self, clip: Clip, window: Window) -> torch.Tensor:
        key = (clip.clip_id, window.index)
        cached = self._encoder_cache.get(key)
        if cached is None:
            cached = self._encoder_output_tokens(self.pixel_values(clip, window))
            self._encoder_cache[key] = cached
        return cached

    def _encoder_output_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Full-clip target encoder, including its final layernorm. Sees every token."""
        with torch.inference_mode():
            encoder = self.model.encoder
            hidden = encoder.embeddings(pixel_values)
            for block in encoder.layer:
                hidden = block(hidden, None)[0]
            hidden = encoder.layernorm(hidden)
        return hidden.clone()

    def _run_blocks(self, hidden: torch.Tensor, position_ids: torch.Tensor | None, until_layer: int) -> torch.Tensor:
        """Run encoder blocks ``1..until_layer`` under inference mode and return a normal tensor."""
        with torch.inference_mode():
            hidden_out = hidden
            for index, block in enumerate(self.model.encoder.layer, start=1):
                hidden_out = block(hidden_out, position_ids)[0]
                if index == until_layer:
                    break
            else:
                raise RuntimeError(
                    f"layer {until_layer} is outside 1..{len(self.model.encoder.layer)}"
                )
        return hidden_out.clone()

    def _finish_visible(self, hidden: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Blocks after the edited layer, then the encoder layernorm. May carry grad."""
        encoder = self.model.encoder
        hidden_out = hidden
        for index, block in enumerate(encoder.layer, start=1):
            if index <= self.layer:
                continue
            hidden_out = block(hidden_out, ctx)[0]
        return encoder.layernorm(hidden_out)

    def _predictor(self, context_out: torch.Tensor, ctx: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        width = context_out.shape[-1]
        full = context_out.new_zeros(context_out.shape[0], self._n_tokens, width)
        full = full.scatter(1, _expand_ids(ctx, width), context_out)
        return self.model.predictor(
            encoder_hidden_states=full,
            context_mask=[ctx],
            target_mask=[tgt],
        ).last_hidden_state

    def _apply_hooks(self, acts: torch.Tensor) -> torch.Tensor:
        edited = acts
        for handle in self._hooks:
            if handle.layer != self.layer:
                continue
            if handle.mode == "capture":
                handle.captured.append(edited.detach().clone())
            elif handle.mode == "zero":
                edited = torch.zeros_like(edited)
            elif handle.mode == "edit":
                assert handle.fn is not None
                edited = handle.fn(edited)
        return edited

    def view_frames(self, frames: np.ndarray) -> np.ndarray:
        """Frames as the model sees them (the centre crop), in [0, 1], for the concept labeller."""
        video = torch.from_numpy(np.ascontiguousarray(frames)).float()
        if float(video.max()) > 1.5:
            video = video / 255.0
        video = _resize_center_crop(video.permute(0, 3, 1, 2).contiguous(), self.crop)
        return video.permute(0, 2, 3, 1).contiguous().numpy()

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        video = torch.from_numpy(np.ascontiguousarray(frames))
        if video.dtype != torch.float32:
            video = video.float()
        if video.numel() == 0:
            raise ValueError("window has no pixels")
        if float(video.max()) > 1.5:
            video = video / 255.0
        video = video.permute(0, 3, 1, 2).contiguous()
        video = _resize_center_crop(video, self.crop)
        mean = video.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = video.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        video = (video - mean) / std
        if video.shape[0] != self.n_frames or video.shape[-2:] != (self.crop, self.crop):
            raise ValueError(
                f"preprocessed video must be [{self.n_frames}, 3, {self.crop}, {self.crop}], got {tuple(video.shape)}"
            )
        return video.unsqueeze(0)

    def _bind_mask(self) -> None:
        mask = _tube_mask(self.grid())
        flat = mask.reshape(-1)
        ctx = torch.from_numpy(np.flatnonzero(~flat)).to(device=self.device, dtype=torch.long)
        tgt = torch.from_numpy(np.flatnonzero(flat)).to(device=self.device, dtype=torch.long)
        self._ctx = ctx.unsqueeze(0)
        self._tgt = tgt.unsqueeze(0)
        self._mask = torch.from_numpy(mask).to(self.device)
        self._n_tokens = int(flat.shape[0])

    def _clear_caches(self) -> None:
        self._pixels_cache.clear()
        self._visible_cache.clear()
        self._layer_cache.clear()
        self._encoder_cache.clear()

    def _ensure(self) -> None:
        if not self._loaded or self.model is None:
            raise RuntimeError("call load() before using the vjepa2 adapter")


def _tube_mask(grid: Grid) -> np.ndarray:
    """Same spatial block in every tubelet. Centre block on the full 16x16 grid."""
    mask = np.zeros((grid.t, grid.h, grid.w), dtype=bool)
    block_h = max(1, grid.h // 2)
    block_w = max(1, grid.w // 2)
    if block_h == grid.h and block_w == grid.w:
        block_w = grid.w - 1
    y0 = (grid.h - block_h) // 2
    x0 = (grid.w - block_w) // 2
    mask[:, y0 : y0 + block_h, x0 : x0 + block_w] = True
    if not mask.any() or bool(mask.all()):
        raise RuntimeError("tube mask must leave both visible and masked patches")
    return mask


def _sample_indices(n_frames: int, n_out: int) -> np.ndarray:
    """Centred window. Stride 2 when the clip is long enough, else stride 1, else pad."""
    stride = 2 if n_frames >= n_out * 2 else 1
    span = n_out * stride
    if n_frames >= span:
        start = (n_frames - span) // 2
        return start + np.arange(n_out) * stride
    return np.minimum(np.arange(n_out), n_frames - 1)


def _resize_center_crop(video: torch.Tensor, crop: int) -> torch.Tensor:
    """Match ``VJEPA2VideoProcessor``: shortest edge to ``crop * 256/224``, then centre crop.

    Frames that are already ``crop`` by ``crop`` are returned unchanged so a
    patch edit cannot bleed across the tubelet convolution.
    """
    if video.shape[-2] == crop and video.shape[-1] == crop:
        return video
    height, width = int(video.shape[-2]), int(video.shape[-1])
    target_short = max(crop, int(crop * 256 / 224))
    scale = target_short / max(1, min(height, width))
    resized_h = max(crop, int(round(height * scale)))
    resized_w = max(crop, int(round(width * scale)))
    video = F.interpolate(video, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    top = max(0, (resized_h - crop) // 2)
    left = max(0, (resized_w - crop) // 2)
    return video[:, :, top : top + crop, left : left + crop]


def _expand_ids(ids: torch.Tensor, width: int) -> torch.Tensor:
    return ids.unsqueeze(-1).expand(-1, -1, width)


def _scatter_grid(visible: torch.Tensor, ctx: torch.Tensor, grid: Grid, n_tokens: int) -> torch.Tensor:
    width = visible.shape[-1]
    full = visible.new_zeros(1, n_tokens, width)
    full = full.scatter(1, _expand_ids(ctx, width), visible)
    return full[0].reshape(grid.t, grid.h, grid.w, width)


def _gather_ids(tokens: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(tokens, 1, _expand_ids(ids, tokens.shape[-1]))
