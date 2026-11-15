"""Generic Hugging Face vision adapter: bring any ViT-style image or video model by name.

The adapter name carries the model, so a run folder can rebuild it later::

    hf:<model id or local directory>[#<dotted path to the block list>]

for example ``hf:facebook/dinov2-small`` or ``hf:MCG-NJU/videomae-base``.
``ii run hf --model <id>`` writes the same name.

What it hooks
    The output of transformer block ``layer`` (1-indexed, the same tensor
    transformers returns as ``hidden_states[layer]``). The block list is found
    automatically: the ``nn.ModuleList`` whose length is
    ``config.num_hidden_layers``, preferring an encoder over a predictor or
    decoder. Pass ``#<path>`` to choose it by hand.

Token grid
    Prefix tokens (CLS, registers) are left alone and only the patch tokens
    form the ``[t, h, w, d]`` grid. Image models see each sampled frame as one
    batch item, so ``t`` is the number of frames and the tubelet is 1. Video
    models (a ``num_frames`` or ``frames_per_clip`` config) see the whole
    window at once and ``t`` is frames over tubelet.

What "prediction" means here
    A generic encoder has no masked-prediction head, so the readout is the
    model's own final patch tokens (``last_hidden_state``). Every token is
    "masked" (scored), the target is the clean final output, and the
    visible-mean baseline replaces every token with the average token. So:

    - ablation effect: how far removing a feature moves the model's output;
    - loss recovered: how much of the output survives splicing the SAE in,
      against zeroing the layer;
    - the usable-SAE check asks the splice to beat "every token is the mean".

Frames are resized to the model's square input with no crop, so the token
grid covers the whole frame and the viewer overlay lines up with the clip.

What else it accepts
    - Image-text models and VLMs (CLIP, SigLIP, SmolVLM, PaliGemma): only the
      vision tower is hooked, so ``--model openai/clip-vit-base-patch32``
      just works.
    - Convolutional models (ResNet, ConvNeXt): a "block" is a stage, its
      ``[batch, channels, h, w]`` map is the token grid, and the readout is
      the final map resized to that grid.
    - LeRobot policies with a frozen VLM vision encoder (SmolVLA): the
      encoder is the backbone's, so the backbone's vision tower is used.

Anything else fails early with a message saying why and what to try.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence.adapters.base import Clip, Grid, HookHandle, Window
from interpretableintelligence.device import ensure_mps_fallback

PREFIX = "hf:"
IMAGE_FRAMES = 8
# Batch items (edits x frames) per forward in ``predict_masked_many``.
MANY_BATCH_ITEMS = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_SKIP_LISTS = ("predictor", "decoder", "head")


class _StopForward(Exception):
    """Raised inside the block hook once the layer is captured; nothing later is needed."""


def parse_spec(spec: str) -> tuple[str, str | None]:
    """``hf:<model>[#<blocks>]`` to ``(model, blocks)``."""
    if not spec.startswith(PREFIX):
        raise ValueError(f"hf adapter names start with {PREFIX!r}, got {spec!r}")
    body = spec[len(PREFIX) :]
    model, _sep, blocks = body.partition("#")
    if not model:
        raise ValueError("hf adapter needs a model, e.g. hf:facebook/dinov2-small")
    return model, (blocks or None)


def make_spec(model: str, blocks: str | None = None) -> str:
    if not model:
        raise ValueError("--model is required for the hf adapter")
    if "#" in model:
        raise ValueError("model ids and paths cannot contain '#'")
    return f"{PREFIX}{model}" + (f"#{blocks}" if blocks else "")


def middle_block(spec: str) -> int:
    """A sensible default layer, 1-indexed.

    Transformers: the middle block (mid-depth features tend to be the most
    legible). Convolutional models: the second-to-last stage, the deepest map
    that still keeps a usable spatial grid (14x14 at 224 pixels).
    """
    model, _blocks = parse_spec(spec)
    vision = _vision_config(load_config(model)[0])
    depth = _depth(vision)
    if _is_conv(vision):
        return max(1, depth - 1)
    return max(1, depth // 2)


def adapter_class(spec: str) -> type:
    """A class with the pipeline's ``(layer, device, tiny=)`` constructor, bound to one model."""
    model, blocks = parse_spec(spec)

    class BoundHFAdapter(HFAdapter):
        def __init__(self, layer: int, device: torch.device | None = None, *, tiny: bool = False) -> None:
            if tiny:
                raise ValueError("--tiny is a V-JEPA 2 option; point hf at a small model instead")
            super().__init__(model, layer, device, blocks=blocks)

    BoundHFAdapter.__name__ = BoundHFAdapter.__qualname__ = "HFAdapter"
    return BoundHFAdapter


class HFAdapter:
    # predict_masked scores against the model's own clean output, so prediction_loss
    # and its gradient are zero at the clean point: rank by output_sensitivity instead.
    attribution_default = "output_sensitivity"

    def __init__(
        self,
        model: str,
        layer: int,
        device: torch.device | None = None,
        *,
        blocks: str | None = None,
        frames: int = IMAGE_FRAMES,
    ) -> None:
        if layer < 1:
            raise ValueError(f"hf layers are 1-indexed block outputs, got {layer}")
        self.model_name = model
        self.blocks_path = blocks
        self.name = make_spec(model, blocks)
        self.layer = int(layer)
        self.device = device or torch.device("cpu")
        ensure_mps_fallback(self.device)
        self.image_frames = int(frames)
        self.model: torch.nn.Module | None = None
        self._random_init = False
        self._hooks: list[HookHandle] = []
        self._cache: dict = {}
        self._transform = None
        self._stop = False

    # ------------------------------------------------------------------ setup

    @property
    def d_model(self) -> int:
        return int(self._d_model)

    @property
    def random_init(self) -> bool:
        return self._random_init

    def grid(self) -> Grid:
        return Grid(t=self._t, h=self._h, w=self._w, tubelet=self._tubelet, patch_px=self._patch)

    def load(self, random_init: bool = False) -> None:
        from transformers import AutoModel

        config, source = load_config(self.model_name)
        if random_init:
            torch.manual_seed(0)
            model = _with_fast_attention(lambda attn: AutoModel.from_config(config, attn_implementation=attn))
        else:
            model = _with_fast_attention(
                lambda attn: AutoModel.from_pretrained(source, attn_implementation=attn, dtype=torch.float32)
            )
        vision = _vision_config(config)
        if vision is not config:
            model = _vision_tower(model, source)
        model = model.to(device=self.device, dtype=torch.float32).eval()
        model.requires_grad_(False)
        self.model = model
        self._source = source
        self._random_init = bool(random_init)
        self._cache.clear()
        self._bind(vision)

    def _bind(self, config) -> None:
        assert self.model is not None
        self._blocks = _find_blocks(self.model, config, self.blocks_path)
        if self.layer > len(self._blocks):
            raise ValueError(f"{self.model_name} has {len(self._blocks)} blocks; layer {self.layer} does not exist")
        self._conv = _is_conv(config)
        preprocessor = _preprocessor_config(self._source)
        native = _pair(getattr(config, "image_size", None) or getattr(config, "crop_size", None))
        preferred = _input_size(preprocessor)
        if native is None and self._conv:
            native = (preferred or 224, preferred or 224)
        patch = (1, 1) if self._conv else _pair(getattr(config, "patch_size", None))  # a stage's map sets its own grid
        if native is None or patch is None:
            raise ValueError(
                f"{self.model_name} ({type(config).__name__}) has no image_size and patch_size, so it is not a "
                "ViT-style or convolutional vision model; ii's hf adapter cannot read it"
            )
        if native[0] != native[1] or (not self._conv and patch[0] != patch[1]):
            raise ValueError(f"{self.model_name} needs square images and patches, got {native} and {patch}")
        self._patch = 1 if self._conv else int(patch[0])
        params = inspect.signature(self.model.forward).parameters
        self._input_key = "pixel_values_videos" if "pixel_values_videos" in params else "pixel_values"
        if self._input_key not in params:
            raise ValueError(f"{type(self.model).__name__}.forward takes no pixel_values; use a vision model")
        self._extra = {"skip_predictor": True} if "skip_predictor" in params else {}
        video_frames = getattr(config, "num_frames", None) or getattr(config, "frames_per_clip", None)
        self.is_video = self._input_key == "pixel_values_videos" or bool(video_frames and getattr(config, "tubelet_size", None))
        if self.is_video:
            self.n_frames = int(video_frames)
            self._tubelet = int(getattr(config, "tubelet_size", 1) or 1)
            self._t = self.n_frames // self._tubelet
        else:
            self.n_frames = self.image_frames
            self._tubelet = 1
            self._t = self.n_frames
        self._mean, self._std = _norm_stats(preprocessor)
        native_px = int(native[0])
        # The model card's preprocessor size is what the model was trained and is
        # meant to be fed at (DINOv2: config 518, preprocessor 224). Use it when
        # the model accepts it; otherwise fall back to the config size.
        candidates = [preferred] if preferred and preferred != native_px and preferred % self._patch == 0 else []
        if candidates and "interpolate_pos_encoding" in params:
            self._extra = {**self._extra, "interpolate_pos_encoding": True}
        for size_px in [*candidates, native_px]:
            self._set_size(size_px)
            try:
                self._probe()
                break
            except Exception:  # noqa: BLE001 (a model that cannot take this size falls back to its config size)
                if size_px == native_px:
                    raise
                self._extra.pop("interpolate_pos_encoding", None)

    def _set_size(self, size_px: int) -> None:
        self._size = int(size_px)
        self._h = self._w = self._size // self._patch

    def _probe(self) -> None:
        """One dummy forward to learn the token count, prefix length, and width."""
        pixels = torch.zeros(self._pixel_shape(), device=self.device)
        seen: dict = {}

        def grab(hidden: torch.Tensor):
            seen["shape"] = tuple(hidden.shape)
            raise _StopForward

        with torch.no_grad():
            self._forward(pixels, raw=grab)
        if "shape" not in seen:
            raise RuntimeError(f"{self.model_name}: block {self.layer} never ran; pass hf:<model>#<blocks path>")
        if len(seen["shape"]) == 4:
            # Convolutional stage: [batch, channels, h, w]. The map is the grid.
            batch, width, height, across = seen["shape"]
            if height != across or self._size % height:
                raise RuntimeError(f"{self.model_name} stage {self.layer} map {seen['shape']} is not a square grid of {self._size}px")
            self._layout = "map"
            self._h = self._w = int(height)
            self._patch = self._size // int(height)
            self._prefix = 0
            self._d_model = int(width)
            return
        if len(seen["shape"]) != 3:
            raise RuntimeError(f"{self.model_name} block output {seen['shape']} is neither tokens nor a feature map")
        batch, n_tokens, width = seen["shape"]
        per_item = (self._t * self._h * self._w) // batch
        prefix = n_tokens - per_item
        if prefix < 0 or batch * per_item != self._t * self._h * self._w:
            raise RuntimeError(
                f"{self.model_name} block output {seen['shape']} does not fit a {self._t}x{self._h}x{self._w} patch grid"
            )
        self._layout = "tokens"
        self._prefix = int(prefix)
        self._d_model = int(width)

    def model_sha(self) -> str:
        self._ensure()
        assert self.model is not None
        digest = hashlib.sha256()
        digest.update(b"hf-v1")
        digest.update(self.name.encode())
        digest.update(b"random" if self._random_init else b"weights")
        for name, tensor in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(tensor.detach().to("cpu", torch.float32).numpy()).tobytes())
        return digest.hexdigest()

    # --------------------------------------------------------------- protocol

    def windows(self, clip: Clip) -> list[Window]:
        frames = np.asarray(clip.frames)
        if frames.ndim != 4 or frames.shape[-1] not in (1, 3):
            raise ValueError(f"hf clips must be [frames, height, width, 3], got {frames.shape}")
        if frames.shape[0] < 1:
            raise ValueError(f"{clip.clip_id} has no frames")
        index = _sample_indices(int(frames.shape[0]), self.n_frames)
        return [Window(index=0, start_frame=int(index[0]), frames=np.ascontiguousarray(frames[index]))]

    def mask(self, window: Window) -> torch.Tensor:
        del window
        return torch.ones((self._t, self._h, self._w), dtype=torch.bool, device=self.device)

    @contextmanager
    def hook(self, layer: int, mode: str = "capture", fn=None):
        if mode not in {"capture", "edit", "zero"}:
            raise ValueError(f"hook mode must be capture|edit|zero, got {mode!r}")
        if mode == "edit" and fn is None:
            raise ValueError("edit hook requires fn")
        if layer != self.layer:
            raise ValueError(f"hf adapter is bound to layer {self.layer}, got hook on {layer}")
        handle = HookHandle(layer=layer, mode=mode, fn=fn, captured=[])  # type: ignore[arg-type]
        self._hooks.append(handle)
        try:
            yield handle
        finally:
            self._hooks.remove(handle)

    def encode(self, clip: Clip, window: Window) -> torch.Tensor:
        return self._apply_hooks(self.raw_activations(clip, window))

    def raw_activations(self, clip: Clip, window: Window) -> torch.Tensor:
        """Patch tokens at ``layer``, ``[t, h, w, d]``. Hooks are not applied."""
        self._ensure()
        key = ("raw", clip.clip_id, window.index)
        cached = self._cache.get(key)
        if cached is None:
            seen: dict = {}

            def grab(hidden: torch.Tensor):
                seen["full"] = hidden.detach().clone()
                raise _StopForward

            with torch.no_grad():
                self._forward(self.pixel_values(clip, window), raw=grab)
            self._cache[("full", clip.clip_id, window.index)] = seen["full"]
            cached = self._to_grid(seen["full"])
            self._cache[key] = cached
        return cached.clone()

    def predict_masked(self, clip: Clip, window: Window, edit=None):
        """Final patch tokens with the layer edited, against the clean final tokens. Every token is scored."""
        self._ensure()
        target = self._clean_output(clip, window)
        if edit is None and not self._hooks:
            return target.clone(), target
        transform = edit if edit is not None else self._apply_hooks
        pred = self._final_tokens(self._forward(self.pixel_values(clip, window), transform=transform))
        return pred, target

    def predict_masked_many(self, clip: Clip, window: Window, edits) -> list[torch.Tensor]:
        """``predict_masked(clip, window, edit=e)[0]`` for each edit, batched.

        The clean block output is cached, so only the blocks after ``layer``
        run, on every edited copy at once. The blocks up to ``layer`` are
        swapped for pass-throughs for the duration of the call.
        """
        self._ensure()
        edits = list(edits)
        if not edits:
            return []
        self.raw_activations(clip, window)
        full = self._cache[("full", clip.clip_id, window.index)]
        pixels = self.pixel_values(clip, window)
        per = max(1, MANY_BATCH_ITEMS // int(full.shape[0]))
        out: list[torch.Tensor] = []
        for start in range(0, len(edits), per):
            chunk = edits[start : start + per]
            spliced = []
            for edit in chunk:
                grid = self._to_grid(full)
                edited = edit(grid)
                if tuple(edited.shape) != tuple(grid.shape):
                    raise ValueError(f"edit must return {tuple(grid.shape)}, got {tuple(edited.shape)}")
                spliced.append(self._from_grid(full, edited))
            hidden = torch.cat(spliced, dim=0)
            repeat = [len(chunk)] + [1] * (pixels.dim() - 1)
            outputs = self._forward_after(pixels.repeat(*repeat), hidden)
            final = outputs.last_hidden_state
            for index in range(len(chunk)):
                out.append(self._final_tokens(final[index * full.shape[0] : (index + 1) * full.shape[0]]))
        return out

    def visible_mean_baseline(self, clip: Clip, window: Window) -> torch.Tensor:
        """Every token predicted as the model's average output token for this window."""
        target = self._clean_output(clip, window)
        return target.mean(dim=0, keepdim=True).expand_as(target).clone()

    def attribution_objective(self, clip: Clip, window: Window, objective: str, edit=None) -> torch.Tensor:
        if objective != "prediction_loss":
            raise ValueError(f"hf attribution objective must be prediction_loss, got {objective!r}")
        pred, target = self.predict_masked(clip, window, edit=edit)
        return ((pred - target) ** 2).mean()

    def pixel_values(self, clip: Clip, window: Window) -> torch.Tensor:
        self._ensure()
        key = ("pixels", clip.clip_id, window.index)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._preprocess(window.frames)
            self._cache[key] = cached
        return cached

    # ------------------------------------------------------ activation cache

    def cache_spec(self, clip: Clip) -> dict:
        """What besides weights, layer and frames decides this adapter's tensors (``act_cache``)."""
        del clip
        self._ensure()
        return {
            "adapter": "hf-v1",
            "input_key": self._input_key,
            "size": int(self._size),
            "patch": int(self._patch),
            "prefix": int(self._prefix),
            "frames": int(self.n_frames),
            "tubelet": int(self._tubelet),
            "video": bool(self.is_video),
            "mean": [float(x) for x in self._mean],
            "std": [float(x) for x in self._std],
            "extra": sorted(str(key) for key in self._extra),
        }

    def clean_state(self, clip: Clip, window: Window) -> dict[str, torch.Tensor]:
        """Unedited block output (prefix included) and final tokens for one window."""
        self.raw_activations(clip, window)
        target = self._clean_output(clip, window)
        return {"full": self._cache[("full", clip.clip_id, window.index)], "target": target}

    def load_clean_state(self, clip: Clip, window: Window, state: dict[str, torch.Tensor]) -> None:
        """Put :meth:`clean_state` back, so encode and the clean prediction skip their forwards."""
        self._ensure()
        full = state["full"].to(device=self.device, dtype=torch.float32)
        target = state["target"].to(device=self.device, dtype=torch.float32)
        tokens = self._t * self._h * self._w
        if full.dim() != 3 or full.shape[0] * (full.shape[1] - self._prefix) != tokens or full.shape[-1] != self._d_model:
            raise ValueError(f"cached block output {tuple(full.shape)} does not fit this model")
        if target.dim() != 2 or target.shape[0] != tokens:
            raise ValueError(f"cached final tokens {tuple(target.shape)} do not fit this model")
        self._cache[("full", clip.clip_id, window.index)] = full
        self._cache[("raw", clip.clip_id, window.index)] = self._to_grid(full)
        self._cache[("target", clip.clip_id, window.index)] = target

    # -------------------------------------------------------------- internals

    def _clean_output(self, clip: Clip, window: Window) -> torch.Tensor:
        key = ("target", clip.clip_id, window.index)
        cached = self._cache.get(key)
        if cached is None:
            with torch.no_grad():
                cached = self._final_tokens(self._forward(self.pixel_values(clip, window))).detach()
            self._cache[key] = cached
        return cached

    def _forward(self, pixels: torch.Tensor, transform=None, raw=None):
        """Run the model with the block output passed through ``transform`` (grid in, grid out).

        ``raw`` sees the block output as transformers returns it and may raise
        ``_StopForward`` to skip the rest of the network.
        """
        assert self.model is not None

        def block_hook(_module, _args, output):
            self._block_returns_tuple = isinstance(output, tuple)
            hidden = output[0] if isinstance(output, tuple) else output
            if raw is not None:
                raw(hidden)
            if transform is None:
                return None
            grid = self._to_grid(hidden)
            edited = transform(grid)
            if tuple(edited.shape) != tuple(grid.shape):
                raise ValueError(f"edit must return {tuple(grid.shape)}, got {tuple(edited.shape)}")
            new = self._from_grid(hidden, edited)
            return (new, *output[1:]) if isinstance(output, tuple) else new

        handle = self._blocks[self.layer - 1].register_forward_hook(block_hook)
        try:
            return self.model(**{self._input_key: pixels}, **self._extra)
        except _StopForward:
            return None
        finally:
            handle.remove()

    def _forward_after(self, pixels: torch.Tensor, hidden: torch.Tensor):
        """Forward where blocks ``1..layer`` are skipped and block ``layer`` emits ``hidden``."""
        assert self.model is not None
        blocks = self._blocks
        tuple_out = self._block_returns_tuple
        originals = [blocks[index] for index in range(self.layer)]
        for index in range(self.layer - 1):
            blocks[index] = _PassThrough(tuple_out)
        blocks[self.layer - 1] = _Emit(hidden, tuple_out)
        try:
            return self.model(**{self._input_key: pixels}, **self._extra)
        finally:
            for index, module in enumerate(originals):
                blocks[index] = module

    def _final_tokens(self, outputs) -> torch.Tensor:
        hidden = outputs if isinstance(outputs, torch.Tensor) else getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError(f"{type(self.model).__name__} returned no last_hidden_state")
        if hidden.dim() == 4:
            # Final feature map, resized to the hooked stage's grid so token i of
            # the readout sits over token i of the layer.
            if tuple(hidden.shape[-2:]) != (self._h, self._w):
                hidden = F.interpolate(hidden, size=(self._h, self._w), mode="bilinear", align_corners=False)
            return hidden.permute(0, 2, 3, 1).reshape(-1, hidden.shape[1])
        patches = hidden[:, self._prefix :]
        if patches.shape[0] * patches.shape[1] != self._t * self._h * self._w:
            raise RuntimeError(f"final output {tuple(hidden.shape)} does not match the patch grid")
        return patches.reshape(-1, hidden.shape[-1])

    def _to_grid(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 4:
            return hidden.permute(0, 2, 3, 1).reshape(self._t, self._h, self._w, hidden.shape[1])
        return hidden[:, self._prefix :].reshape(self._t, self._h, self._w, hidden.shape[-1])

    def _from_grid(self, hidden: torch.Tensor, edited: torch.Tensor) -> torch.Tensor:
        """``hidden`` with its patch tokens (or map) replaced by the ``[t, h, w, d]`` grid ``edited``."""
        if hidden.dim() == 4:
            return edited.reshape(hidden.shape[0], self._h, self._w, -1).permute(0, 3, 1, 2).contiguous().to(hidden.dtype)
        patches = edited.reshape(hidden.shape[0], -1, hidden.shape[-1]).to(hidden.dtype)
        return torch.cat([hidden[:, : self._prefix], patches], dim=1)

    def _apply_hooks(self, acts: torch.Tensor) -> torch.Tensor:
        edited = acts
        for handle in self._hooks:
            if handle.mode == "capture":
                handle.captured.append(edited.detach().clone())
            elif handle.mode == "zero":
                edited = torch.zeros_like(edited)
            else:
                assert handle.fn is not None
                edited = handle.fn(edited)
        return edited

    def _pixel_shape(self) -> tuple[int, ...]:
        if self.is_video:
            return (1, self.n_frames, 3, self._size, self._size)
        return (self.n_frames, 3, self._size, self._size)

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        video = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device).float()
        if video.numel() == 0:
            raise ValueError("window has no pixels")
        if float(video.max()) > 1.5:
            video = video / 255.0
        video = video.permute(0, 3, 1, 2)
        if video.shape[1] == 1:
            video = video.expand(-1, 3, -1, -1)
        if video.shape[-2:] != (self._size, self._size):
            video = F.interpolate(video, size=(self._size, self._size), mode="bilinear", align_corners=False, antialias=True)
        mean = video.new_tensor(self._mean).view(1, 3, 1, 1)
        std = video.new_tensor(self._std).view(1, 3, 1, 1)
        video = ((video - mean) / std).contiguous()
        return video.unsqueeze(0) if self.is_video else video

    def _ensure(self) -> None:
        if self.model is None:
            raise RuntimeError("call load() before using the hf adapter")


class _PassThrough(torch.nn.Module):
    """Stands in for a skipped block: returns its hidden-state input unchanged."""

    def __init__(self, tuple_out: bool) -> None:
        super().__init__()
        self.tuple_out = tuple_out

    def forward(self, *args, **kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        return (hidden,) if self.tuple_out else hidden


class _Emit(torch.nn.Module):
    """Stands in for the hooked block: ignores its input and returns the edited states."""

    def __init__(self, hidden: torch.Tensor, tuple_out: bool) -> None:
        super().__init__()
        self.hidden = hidden
        self.tuple_out = tuple_out

    def forward(self, *args, **kwargs):
        del args, kwargs
        return (self.hidden,) if self.tuple_out else self.hidden


_NOTED: set[str] = set()
_TOWER_NAMES = ("vision_model", "vision_tower", "vision_encoder", "visual", "image_encoder")


def load_config(model: str):
    """``(config, source)`` for a model id or directory, with errors a newcomer can act on.

    ``source`` is where weights load from. It differs from ``model`` only for a
    LeRobot policy whose frozen vision encoder is its VLM backbone's (SmolVLA).
    """
    from transformers import AutoConfig

    local = Path(model).expanduser()
    if not local.exists() and model.count("/") > 1:
        raise ValueError(f"{model!r} is not a folder on this machine or a Hugging Face model id (owner/name)")
    try:
        config = AutoConfig.from_pretrained(model)
    except OSError as exc:
        text = str(exc)
        if "gated" in text.lower() or "401" in text or "403" in text:
            raise ValueError(
                f"{model} is gated or private on Hugging Face. Accept its licence on huggingface.co, "
                "then run `hf auth login` (or set HF_TOKEN) and try again."
            ) from exc
        raise ValueError(
            f"could not find {model!r}: check the id on huggingface.co (owner/name) or the folder path. "
            "If the id is right, this machine may not reach huggingface.co."
        ) from exc
    except ValueError as exc:
        policy = _raw_config(model)
        backbone = policy.get("vlm_model_name")
        if backbone and policy.get("freeze_vision_encoder", False):
            if model not in _NOTED:
                _NOTED.add(model)
                print(f"{model} is a LeRobot policy with a frozen vision encoder; reading it from its backbone {backbone}", file=sys.stderr)
            return load_config(backbone)[0], backbone
        if backbone:
            raise ValueError(
                f"{model} is a LeRobot policy that fine-tunes its vision encoder, which ii cannot load yet. "
                f"Try its backbone, --model {backbone}, to see the pretrained features."
            ) from exc
        if policy.get("type") or "lerobot" in model.lower():
            raise ValueError(
                f"{model} is a LeRobot policy, not a transformers model. ii runs GR00T policies with "
                "`ii run groot`; for others, point --model at the policy's vision backbone."
            ) from exc
        raise ValueError(f"{model} has no transformers model type, so ii cannot load it ({exc})") from exc
    vision = _vision_config(config)
    if not _is_vision(vision):
        raise ValueError(
            f"{model} is a {type(config).__name__.removesuffix('Config')} model with no vision tower. "
            "The hf adapter reads image and video models (ViT, DINOv2, VideoMAE, ResNet, ConvNeXt), "
            "image-text models (CLIP, SigLIP) and VLMs through their vision tower."
        )
    return config, model


def _raw_config(model: str) -> dict:
    """config.json as a plain dict, or ``{}``."""
    try:
        local = Path(model).expanduser() / "config.json"
        if local.is_file():
            path = local
        else:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(model, "config.json"))
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 (no readable config means nothing to suggest)
        return {}


def _vision_config(config):
    """The vision part of a composite config (CLIP, SigLIP, VLMs), else the config itself."""
    vision = getattr(config, "vision_config", None)
    return vision if vision is not None else config


def _is_conv(config) -> bool:
    return getattr(config, "patch_size", None) is None and bool(getattr(config, "depths", None) or getattr(config, "num_stages", None)) or (
        type(config).__name__.startswith("ConvNext")
    )


def _is_vision(config) -> bool:
    has_pixels = any(getattr(config, key, None) is not None for key in ("num_channels", "image_size", "crop_size"))
    return has_pixels and (getattr(config, "patch_size", None) is not None or _is_conv(config))


def _depth(config) -> int:
    for value in (getattr(config, "num_hidden_layers", None), getattr(config, "num_stages", None)):
        if isinstance(value, int) and value > 0:
            return value
    depths = getattr(config, "depths", None)
    return len(depths) if depths else 2


def _vision_tower(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """The vision tower of an image-text model or VLM; the rest is dropped."""
    found = [
        (name.count("."), name, module)
        for name, module in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in _TOWER_NAMES and name.count(".") < 3
    ]
    if not found:
        raise ValueError(f"{model_name} has a vision config but no vision tower ii can find; pass hf:<model>#<blocks path>")
    return min(found, key=lambda item: (item[0], _TOWER_NAMES.index(item[1].rsplit(".", 1)[-1])))[2]


def _with_fast_attention(build):
    """Prefer PyTorch SDPA kernels; fall back to eager for models without them."""
    try:
        return build("sdpa")
    except (ValueError, ImportError):
        return build("eager")


def _find_blocks(model: torch.nn.Module, config, path: str | None) -> torch.nn.ModuleList:
    if path:
        try:
            found = model.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f"{type(model).__name__} has no submodule {path!r}") from exc
        if not isinstance(found, torch.nn.ModuleList) or len(found) == 0:
            raise ValueError(f"{path!r} is not a non-empty nn.ModuleList of blocks")
        return found
    lists = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.ModuleList) and len(module)]
    if not lists:
        raise ValueError(f"{type(model).__name__} has no block list; pass hf:<model>#<blocks path>")
    depth = _depth(config)
    ranked = sorted(
        lists,
        key=lambda item: (
            len(item[1]) != depth,
            any(word in item[0] for word in _SKIP_LISTS),
            "encoder" not in item[0],
            -len(item[1]),
        ),
    )
    return ranked[0][1]


def _norm_stats(preprocessor: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean and std from the model's preprocessor config, else ImageNet."""
    mean, std = preprocessor.get("image_mean"), preprocessor.get("image_std")
    if isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) == 3:
        return tuple(float(x) for x in mean), tuple(float(x) for x in std)
    return IMAGENET_MEAN, IMAGENET_STD


def _input_size(preprocessor: dict) -> int | None:
    """Square input side the preprocessor feeds the model: crop size, else a fixed resize."""
    for key in ("crop_size", "size"):
        value = preprocessor.get(key)
        if isinstance(value, int):
            return value if key == "crop_size" else None
        if isinstance(value, dict) and "height" in value and "width" in value and value["height"] == value["width"]:
            return int(value["height"])
        if isinstance(value, dict) and set(value) == {"shortest_edge"} and key == "size":
            return int(value["shortest_edge"])  # frames are resized square with no crop, so this is the side
    return None


def _preprocessor_config(model: str) -> dict:
    """The model's preprocessor config (image or video), or ``{}``."""
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        try:
            local = Path(model) / name
            if local.is_file():
                path = local
            elif Path(model).is_dir():
                continue
            else:
                from huggingface_hub import hf_hub_download

                path = Path(hf_hub_download(model, name))
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001 (any lookup failure means no preprocessor config)
            continue
    return {}


def _pair(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return (int(value[0]), int(value[-1]))
    return (int(value), int(value))


def _sample_indices(n_frames: int, n_out: int) -> np.ndarray:
    """Centred window. Stride 2 when the clip is long enough, else stride 1, else pad."""
    stride = 2 if n_frames >= n_out * 2 else 1
    span = n_out * stride
    if n_frames >= span:
        start = (n_frames - span) // 2
        return start + np.arange(n_out) * stride
    return np.minimum(np.arange(n_out), n_frames - 1)
