"""GR00T N1.7 on the ``ii`` pipeline: capture, SAE, scoring, and the playground.

The adapter name carries the checkpoint, so a run folder can rebuild it::

    groot:<LeRobot GR00T checkpoint id or local directory>
    groot:tiny

``groot:nvidia/gr00t17-lerobot-libero_10-640`` is the LIBERO checkpoint the
task 1028 work used. ``groot:tiny`` is a GR00T-shaped network with random
weights (same token layout, same flow-matching loop, a few hundred thousand
parameters). It exists so tests and CI exercise the adapter without the
12.6 GB download. It is not GR00T, and nothing learned on it says anything
about GR00T.

What it hooks
    The output of the action head's DiT block ``layer``
    (``_groot_model.action_head.model.transformer_blocks.{layer}``, 0-indexed),
    on the final flow-matching step of each prediction. That is the same
    tensor the v8/v9 Colab capture and the ablation runs read, so SAEs and
    feature ids line up with that work. The default is block 5.

Token grid
    The DiT sequence is one state token followed by one token per action step
    (``action_horizon`` of them, 40 for N1.7). The grid is ``t=1, h=1,
    w=1+action_horizon``. A window is one observation, and its ``tubelet`` is
    the number of steps the robot executes from one prediction (16 for the
    LIBERO checkpoint), so the viewer shows each window over the frames its
    chunk covers.

What "prediction" means here
    The action chunk. ``predict_masked`` returns one row per action step
    (the first ``action_dim`` dims, in the policy's normalised units), in the
    order of the action tokens, so the cell for action token ``k`` holds how
    far action step ``k`` moved. The state token is visible (not scored). The
    target is the clean chunk under the same flow-matching noise, so an edit
    is the only thing that changes between the two. The visible-mean baseline
    predicts every step as the chunk's average action.

Inputs
    GR00T needs two cameras, the robot state, and an instruction. A clip's
    frames are the main (agent-view) camera. The rest comes from a sidecar
    next to the clip file, ``<stem>.groot.json``::

        {"task": "put both moka pots on the stove",
         "state": [[...8 floats...], ...],      # one row per frame, or a single row
         "wrist": "<stem>.wrist.npy"}           # optional, frames aligned with the clip

    Without a sidecar the wrist camera repeats the main frame, the state is
    zeros, and the instruction is empty. The model still runs, but its
    actions are off-distribution; the adapter warns once.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from interpretableintelligence.adapters.base import Clip, Grid, HookHandle, Window
from interpretableintelligence.device import ensure_mps_fallback

PREFIX = "groot:"
TINY = "tiny"
DEFAULT_CHECKPOINT = "nvidia/gr00t17-lerobot-libero_10-640"
DEFAULT_LAYER = 5
SIDECAR_SUFFIX = ".groot.json"
# The text and image processor for the Cosmos-Reason2-2B backbone. That repo is gated; its
# processor files are byte-identical to this ungated one (see v8/inproc_v8.py).
VLM_PROCESSOR_REPO = os.environ.get("GROOT_VLM_PROCESSOR_REPO", "Qwen/Qwen3-VL-2B-Instruct")
# Edited copies per DiT batch in ``predict_masked_many``.
MANY_BATCH = 32


def parse_spec(spec: str) -> str:
    """``groot:<checkpoint>`` to the checkpoint."""
    if not spec.startswith(PREFIX):
        raise ValueError(f"groot adapter names start with {PREFIX!r}, got {spec!r}")
    checkpoint = spec[len(PREFIX) :]
    if not checkpoint:
        raise ValueError(f"groot adapter needs a checkpoint, e.g. {PREFIX}{DEFAULT_CHECKPOINT}")
    return checkpoint


def make_spec(checkpoint: str | None = None, *, tiny: bool = False) -> str:
    if tiny:
        if checkpoint not in (None, TINY):
            raise ValueError("--tiny replaces the checkpoint; pass one or the other")
        return PREFIX + TINY
    return PREFIX + (checkpoint or DEFAULT_CHECKPOINT)


def adapter_class(spec: str) -> type:
    """A class with the pipeline's ``(layer, device, tiny=)`` constructor, bound to one checkpoint."""
    checkpoint = parse_spec(spec)

    class BoundGrootAdapter(GrootAdapter):
        def __init__(self, layer: int, device: torch.device | None = None, *, tiny: bool = False) -> None:
            if tiny and checkpoint != TINY:
                raise ValueError(f"--tiny runs {PREFIX}{TINY}; this run is bound to {spec}")
            super().__init__(checkpoint, layer, device)

    BoundGrootAdapter.__name__ = BoundGrootAdapter.__qualname__ = "GrootAdapter"
    return BoundGrootAdapter


@dataclass
class Observation:
    """One policy input: two HWC uint8 images, the robot state, and the instruction."""

    image: np.ndarray
    wrist: np.ndarray
    state: np.ndarray
    task: str


class Backend(Protocol):
    """The model behind the adapter. ``LeRobotGroot`` is GR00T; ``TinyGroot`` is the random stand-in."""

    chunk: int  # steps executed per prediction
    action_horizon: int  # action tokens in the DiT sequence
    action_dim: int  # action dims the robot uses (the rest is padding)
    noise_dim: int  # width of the flow-matching sample
    state_dim: int
    n_denoise: int
    d_model: int
    blocks: torch.nn.ModuleList

    def load(self, random_init: bool, device: torch.device) -> None: ...

    def context(self, obs: Observation) -> dict: ...

    def actions(self, context: dict, noise: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """``[B, action_horizon, action_dim]`` from ``noise`` ``[B, action_horizon, noise_dim]``."""
        ...

    def fingerprint(self) -> bytes: ...


class GrootAdapter:
    def __init__(
        self,
        checkpoint: str,
        layer: int,
        device: torch.device | None = None,
        *,
        backend: Backend | None = None,
    ) -> None:
        if layer < 0:
            raise ValueError(f"groot layers are 0-indexed DiT blocks, got {layer}")
        self.checkpoint = checkpoint
        self.name = make_spec(checkpoint)
        self.layer = int(layer)
        self.device = device or torch.device("cpu")
        ensure_mps_fallback(self.device)
        if backend is None:
            backend = TinyGroot() if checkpoint == TINY else LeRobotGroot(checkpoint)
        self.backend = backend
        self._random_init = False
        self._loaded = False
        self._hooks: list[HookHandle] = []
        self._cache: dict = {}
        self._sidecars: dict[str, dict] = {}
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ setup

    @property
    def d_model(self) -> int:
        return int(self.backend.d_model)

    @property
    def random_init(self) -> bool:
        return self._random_init

    def load(self, random_init: bool = False) -> None:
        self.backend.load(random_init=bool(random_init), device=self.device)
        n_blocks = len(self.backend.blocks)
        if self.layer >= n_blocks:
            raise ValueError(f"{self.name} has {n_blocks} DiT blocks (0..{n_blocks - 1}); layer {self.layer} does not exist")
        self._random_init = bool(random_init)
        self._loaded = True
        self._cache.clear()

    def grid(self) -> Grid:
        return Grid(t=1, h=1, w=1 + int(self.backend.action_horizon), tubelet=int(self.backend.chunk), patch_px=1)

    def model_sha(self) -> str:
        self._ensure()
        digest = hashlib.sha256()
        digest.update(b"groot-v1")
        digest.update(self.name.encode())
        digest.update(b"random" if self._random_init else b"weights")
        digest.update(self.backend.fingerprint())
        return digest.hexdigest()

    # --------------------------------------------------------------- protocol

    def windows(self, clip: Clip) -> list[Window]:
        """One window per action chunk: frames ``[k * chunk, (k + 1) * chunk)``, observed at its first frame."""
        frames = np.asarray(clip.frames)
        if frames.ndim != 4 or frames.shape[-1] not in (1, 3):
            raise ValueError(f"groot clips must be [frames, height, width, 3], got {frames.shape}")
        if frames.shape[0] < 1:
            raise ValueError(f"{clip.clip_id} has no frames")
        chunk = int(self.backend.chunk)
        return [
            Window(index=index, start_frame=start, frames=np.ascontiguousarray(frames[start : start + chunk]))
            for index, start in enumerate(range(0, int(frames.shape[0]), chunk))
        ]

    def mask(self, window: Window) -> torch.Tensor:
        del window
        mask = torch.ones((1, 1, 1 + int(self.backend.action_horizon)), dtype=torch.bool, device=self.device)
        mask[..., 0] = False  # the state token is an input, not a prediction
        return mask

    @contextmanager
    def hook(self, layer: int, mode: str = "capture", fn=None):
        if mode not in {"capture", "edit", "zero"}:
            raise ValueError(f"hook mode must be capture|edit|zero, got {mode!r}")
        if mode == "edit" and fn is None:
            raise ValueError("edit hook requires fn")
        if layer != self.layer:
            raise ValueError(f"groot adapter is bound to layer {self.layer}, got hook on {layer}")
        handle = HookHandle(layer=layer, mode=mode, fn=fn, captured=[])  # type: ignore[arg-type]
        self._hooks.append(handle)
        try:
            yield handle
        finally:
            self._hooks.remove(handle)

    def encode(self, clip: Clip, window: Window) -> torch.Tensor:
        return self._apply_hooks(self.raw_activations(clip, window))

    def raw_activations(self, clip: Clip, window: Window) -> torch.Tensor:
        """DiT block ``layer`` at the final denoise step, ``[1, 1, 1 + action_horizon, d]``. Hooks are not applied."""
        self._ensure()
        key = ("raw", clip.clip_id, window.index)
        cached = self._cache.get(key)
        if cached is None:
            with torch.no_grad():
                actions, captured = self._run(clip, window, [None], capture=True)
            cached = captured[0].reshape(1, 1, -1, captured.shape[-1])
            self._cache[key] = cached
            self._cache.setdefault(("target", clip.clip_id, window.index), actions[0].detach())
        return cached.clone()

    def predict_masked(self, clip: Clip, window: Window, edit=None):
        """The action chunk with the layer edited, against the clean chunk. One row per action step."""
        self._ensure()
        target = self._clean_output(clip, window)
        if edit is None and not self._hooks:
            return target.clone(), target
        transform = edit if edit is not None else self._apply_hooks
        actions, _ = self._run(clip, window, [transform], grad=torch.is_grad_enabled())
        return actions[0], target

    def predict_masked_many(self, clip: Clip, window: Window, edits) -> list[torch.Tensor]:
        """``predict_masked(clip, window, edit=e)[0]`` for each edit, as batched action-head calls.

        The backbone runs once per window (cached); only the action head runs per edit.
        """
        self._ensure()
        edits = list(edits)
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(edits), MANY_BATCH):
                actions, _ = self._run(clip, window, edits[start : start + MANY_BATCH])
                out.extend(actions[index] for index in range(actions.shape[0]))
        return out

    def visible_mean_baseline(self, clip: Clip, window: Window) -> torch.Tensor:
        """Every action step predicted as the chunk's average action."""
        target = self._clean_output(clip, window)
        return target.mean(dim=0, keepdim=True).expand_as(target).clone()

    def attribution_objective(self, clip: Clip, window: Window, objective: str, edit=None) -> torch.Tensor:
        if objective != "prediction_loss":
            raise ValueError(f"groot attribution objective must be prediction_loss, got {objective!r}")
        pred, target = self.predict_masked(clip, window, edit=edit)
        return ((pred - target) ** 2).mean()

    def observation(self, clip: Clip, window: Window) -> Observation:
        """The policy input for a window: its first frame plus the clip's sidecar at that frame."""
        side = self._sidecar(clip)
        frame = int(window.start_frame)
        image = _to_uint8(window.frames[0])
        wrist_frames = side.get("wrist")
        wrist = _to_uint8(wrist_frames[min(frame, len(wrist_frames) - 1)]) if wrist_frames is not None else image
        state = side.get("state")
        if state is None:
            vector = np.zeros(int(self.backend.state_dim), dtype=np.float32)
        else:
            vector = np.asarray(state[min(frame, len(state) - 1)], dtype=np.float32).reshape(-1)
        return Observation(image=image, wrist=wrist, state=vector, task=str(side.get("task", "")))

    # -------------------------------------------------------------- internals

    def _run(self, clip: Clip, window: Window, transforms: list, *, capture: bool = False, grad: bool = False):
        """Run the action head with ``transforms[b]`` applied to batch row ``b`` at the final denoise step.

        Every row starts from the same noise, so rows differ only by their edit.
        Returns ``(actions [B, horizon, action_dim] float32, captured block output or None)``.
        """
        backend = self.backend
        context = self._context(clip, window)
        noise = self._noise(clip, window).expand(len(transforms), -1, -1)
        state = {"calls": 0, "captured": None}

        def block_hook(_module, _args, output):
            state["calls"] += 1
            if state["calls"] != backend.n_denoise:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            if capture:
                state["captured"] = hidden.detach().to(torch.float32).clone()
            if all(fn is None for fn in transforms):
                return None
            rows = []
            for index, fn in enumerate(transforms):
                row = hidden[index : index + 1]
                if fn is not None:
                    grid = row.reshape(1, 1, row.shape[1], row.shape[2]).to(torch.float32)
                    edited = fn(grid)
                    if tuple(edited.shape) != tuple(grid.shape):
                        raise ValueError(f"edit must return {tuple(grid.shape)}, got {tuple(edited.shape)}")
                    row = edited.reshape(row.shape).to(hidden.dtype)
                rows.append(row)
            new = torch.cat(rows, dim=0)
            return (new, *output[1:]) if isinstance(output, tuple) else new

        handle = backend.blocks[self.layer].register_forward_hook(block_hook)
        try:
            actions = backend.actions(context, noise, grad=grad)
        finally:
            handle.remove()
        if state["calls"] != backend.n_denoise:
            raise RuntimeError(f"DiT block {self.layer} ran {state['calls']} times, expected {backend.n_denoise} denoise steps")
        return actions.to(torch.float32), state["captured"]

    def _clean_output(self, clip: Clip, window: Window) -> torch.Tensor:
        key = ("target", clip.clip_id, window.index)
        cached = self._cache.get(key)
        if cached is None:
            with torch.no_grad():
                actions, _ = self._run(clip, window, [None])
            cached = actions[0].detach()
            self._cache[key] = cached
        return cached

    def _context(self, clip: Clip, window: Window) -> dict:
        key = ("context", clip.clip_id, window.index)
        cached = self._cache.get(key)
        if cached is None:
            with torch.no_grad():
                cached = self.backend.context(self.observation(clip, window))
            self._cache[key] = cached
        return cached

    def _noise(self, clip: Clip, window: Window) -> torch.Tensor:
        """The flow-matching start, fixed per (clip, window) so clean and edited runs share it."""
        seed = int.from_bytes(hashlib.sha256(f"{clip.sha256}:{clip.clip_id}:{window.index}".encode()).digest()[:8], "little")
        generator = torch.Generator().manual_seed(seed % (2**63))
        shape = (1, int(self.backend.action_horizon), int(self.backend.noise_dim))
        return torch.randn(shape, generator=generator).to(self.device)

    def _sidecar(self, clip: Clip) -> dict:
        cached = self._sidecars.get(clip.clip_id)
        if cached is not None:
            return cached
        path = Path(clip.path)
        sidecar = path.with_name(path.stem + SIDECAR_SUFFIX)
        loaded: dict = {}
        if sidecar.is_file():
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"{sidecar} must hold a JSON object")
            loaded["task"] = str(raw.get("task", ""))
            if raw.get("state") is not None:
                state = np.asarray(raw["state"], dtype=np.float32)
                loaded["state"] = state.reshape(1, -1) if state.ndim == 1 else state
            if raw.get("wrist"):
                loaded["wrist"] = np.load(sidecar.parent / str(raw["wrist"]), mmap_mode="r")
        elif clip.clip_id not in self._warned:
            self._warned.add(clip.clip_id)
            warnings.warn(
                f"no {sidecar.name} for clip {clip.clip_id}: GR00T sees the main frame as its wrist camera, "
                "a zero robot state and an empty instruction, so its actions are off-distribution",
                stacklevel=2,
            )
        self._sidecars[clip.clip_id] = loaded
        return loaded

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

    def _ensure(self) -> None:
        if not self._loaded:
            raise RuntimeError("call load() before using the groot adapter")


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3:
        raise ValueError(f"frames must be [height, width, channels], got {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size and float(image.max()) <= 1.5:
            image = image * 255.0
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """HWC uint8 to ``[1, 3, H, W]`` float in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(image)).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0


# ---------------------------------------------------------------------- GR00T


class LeRobotGroot:
    """NVIDIA GR00T N1.7 through LeRobot 0.7 (``pip install 'lerobot[groot]==0.7.0'``).

    Weights stream in with ``low_ram_policy_load`` (no fp32 copy of the
    12.6 GB checkpoint). The observation goes through the checkpoint's own
    preprocessor, the same path as ``lerobot_eval`` and the v8/v9 runners.
    The backbone runs once per window; edits only re-run the action head.
    """

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.policy = None

    def load(self, random_init: bool, device: torch.device) -> None:
        _shim_av_option()
        try:
            from lerobot.policies import make_pre_post_processors
        except ImportError as exc:
            raise RuntimeError(
                "the groot adapter needs LeRobot's GR00T port: pip install 'lerobot[groot]==0.7.0' "
                "(or try the random stand-in with --tiny)"
            ) from exc
        from interpretableintelligence.adapters.groot.low_ram_policy_load import load_groot_low_ram

        dtype = _groot_dtype(device)
        policy, _stats = load_groot_low_ram(self.checkpoint, device=str(device), dtype=dtype, log=_quiet)
        policy.config.use_bf16 = False  # weights are already in ``dtype``; no autocast on top
        model = policy._groot_model
        head = model.action_head
        if random_init:
            # The control the random-init gate needs: same backbone, an untrained action head.
            torch.manual_seed(0)
            for module in head.modules():
                reset = getattr(module, "reset_parameters", None)
                if callable(reset) and not isinstance(module, torch.nn.ModuleList):
                    reset()
            head.to(device=device, dtype=dtype)
        overrides = {
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": {}},
            "groot_n1_7_vlm_encode_v1": {"device": str(device), "model_name": VLM_PROCESSOR_REPO},
        }
        self.pre, _post = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=self.checkpoint, preprocessor_overrides=overrides
        )
        self.policy, self.model, self.head = policy, model, head
        self.device, self.dtype = device, dtype
        self.blocks = head.model.transformer_blocks
        self.action_horizon = int(head.action_horizon)
        self.noise_dim = int(head.action_dim)
        self.n_denoise = int(head.num_inference_timesteps)
        self.chunk = int(policy._action_queue_steps)
        features = policy.config.output_features or {}
        self.action_dim = int(features["action"].shape[0]) if "action" in features else self.noise_dim
        state_features = policy.config.input_features or {}
        state = state_features.get("observation.state")
        self.state_dim = int(state.shape[0]) if state is not None else int(head.config.max_state_dim)
        cfg = head.model.config
        self.d_model = int(cfg.num_attention_heads) * int(cfg.attention_head_dim)
        self.camera_keys = [key for key in state_features if key.startswith("observation.images.")]
        self._fingerprint = _checkpoint_fingerprint(self.checkpoint)

    def context(self, obs: Observation) -> dict:
        batch = {"observation.state": torch.from_numpy(obs.state).reshape(1, -1).to(self.device), "task": [obs.task]}
        images = [obs.image, obs.wrist]
        for index, key in enumerate(self.camera_keys):
            batch[key] = _image_tensor(images[min(index, len(images) - 1)], self.device)
        batch = self.pre(batch)
        inputs = self.policy._filter_groot_inputs(batch, include_action=False)
        backbone_in, action_in = self.model.prepare_input(inputs)
        backbone_out = self.model.backbone(backbone_in)
        features = self.head._encode_features(backbone_out, action_in)
        return {
            "backbone_features": features.backbone_features,
            "state_features": features.state_features,
            "embodiment_id": action_in.embodiment_id,
            "backbone_output": backbone_out,
            "action_input": action_in,
        }

    def actions(self, context: dict, noise: torch.Tensor, grad: bool = False) -> torch.Tensor:
        batch = int(noise.shape[0])
        # ``get_action_with_features`` is wrapped in no_grad; attribution needs the graph.
        step = getattr(type(self.head).get_action_with_features, "__wrapped__", type(self.head).get_action_with_features)
        with torch.set_grad_enabled(grad):
            out = step(
                self.head,
                backbone_features=_rows(context["backbone_features"], batch),
                state_features=_rows(context["state_features"], batch),
                embodiment_id=_rows(context["embodiment_id"], batch),
                backbone_output=_batch_rows(context["backbone_output"], batch),
                action_input=_batch_rows(context["action_input"], batch),
                noise=noise.to(device=self.device, dtype=self.dtype),
            )
        return out["action_pred"][:, : self.action_horizon, : self.action_dim]

    def fingerprint(self) -> bytes:
        return self._fingerprint


def _shim_av_option() -> None:
    """Let LeRobot 0.7 import next to PyAV 18+.

    LeRobot's dataset video module annotates a helper with ``av.option.Option``
    at import time. PyAV 18 dropped ``av.option`` (ii needs PyAV 18+), so the
    import fails before any GR00T code runs. The adapter never touches that
    video code; a stand-in type is enough for the annotation.
    """
    try:
        import av
    except ImportError:
        return
    if hasattr(av, "option"):
        return
    import sys
    import types

    module = types.ModuleType("av.option")
    module.Option = type("Option", (), {})  # type: ignore[attr-defined]
    av.option = module  # type: ignore[attr-defined]
    sys.modules.setdefault("av.option", module)


def _groot_dtype(device: torch.device) -> torch.dtype:
    """fp16 on CUDA (as the Colab runs), bf16 elsewhere; ``II_GROOT_DTYPE`` overrides."""
    name = os.environ.get("II_GROOT_DTYPE", "").strip().lower()
    table = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp32": torch.float32, "float32": torch.float32}
    if name:
        if name not in table:
            raise ValueError(f"II_GROOT_DTYPE must be one of {', '.join(sorted(table))}, got {name!r}")
        return table[name]
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def _rows(tensor, batch: int):
    if isinstance(tensor, torch.Tensor) and tensor.shape[:1] == (1,) and batch != 1:
        return tensor.expand(batch, *tensor.shape[1:])
    return tensor


def _batch_rows(feature, batch: int):
    """Copy of a BatchFeature (or dict) with every batch-1 tensor expanded to ``batch`` rows."""
    if batch == 1:
        return feature
    out = copy.copy(feature)
    for key in list(feature.keys()):
        out[key] = _rows(feature[key], batch)
    return out


def _quiet(_message: str) -> None:
    return None


def _checkpoint_fingerprint(checkpoint: str) -> bytes:
    """Identity of the weights file without hashing 12.6 GB: the Hub's sha256 of it, else size and name."""
    local = Path(checkpoint) / "model.safetensors"
    if local.is_file():
        stat = local.stat()
        return f"local:{local.resolve()}:{stat.st_size}:{int(stat.st_mtime)}".encode()
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(checkpoint, files_metadata=True)
        for sibling in info.siblings or []:
            if sibling.rfilename == "model.safetensors" and sibling.lfs is not None:
                return f"hub:{checkpoint}:{sibling.lfs.sha256}".encode()
        return f"hub:{checkpoint}:{info.sha}".encode()
    except Exception:  # noqa: BLE001 (offline: fall back to the cached file's blob name, which is its sha256)
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(checkpoint, "model.safetensors")
        if isinstance(cached, str):
            return f"hub:{checkpoint}:{Path(cached).resolve().name}".encode()
        raise


# ----------------------------------------------------------------------- tiny


class TinyGroot:
    """A GR00T-shaped network with random weights, for tests and ``--tiny``. Not GR00T.

    Same interface and token layout as N1.7: two camera images and an
    instruction become conditioning tokens, the state becomes one token, and
    a DiT over ``[state] + action tokens`` (cross-attention on even blocks,
    self-attention on odd ones) is integrated with 4 Euler flow-matching steps.
    """

    chunk = 4
    action_horizon = 8
    action_dim = 7
    noise_dim = 8
    state_dim = 8
    n_denoise = 4
    d_model = 32
    n_blocks = 8
    image_size = 16
    patch = 4

    def __init__(self) -> None:
        self.net: _TinyNet | None = None

    def load(self, random_init: bool, device: torch.device) -> None:
        torch.manual_seed(1 if random_init else 0)
        self.net = _TinyNet(self).to(device).eval()
        self.net.requires_grad_(False)
        self.device = device
        self.blocks = self.net.blocks

    def context(self, obs: Observation) -> dict:
        assert self.net is not None
        images = torch.cat([_image_tensor(obs.image, self.device), _image_tensor(obs.wrist, self.device)], dim=0)
        images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False, antialias=True)
        state = torch.from_numpy(np.resize(obs.state, self.state_dim).astype(np.float32)).to(self.device).reshape(1, 1, -1)
        task = int.from_bytes(hashlib.sha256(obs.task.encode()).digest()[:2], "little") % self.net.task_embed.num_embeddings
        return self.net.encode(images, state, task)

    def actions(self, context: dict, noise: torch.Tensor, grad: bool = False) -> torch.Tensor:
        assert self.net is not None
        with torch.set_grad_enabled(grad):
            return self.net.denoise(context, noise.to(self.device))[..., : self.action_dim]

    def fingerprint(self) -> bytes:
        assert self.net is not None
        digest = hashlib.sha256()
        for name, tensor in self.net.state_dict().items():
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(tensor.detach().cpu().float().numpy()).tobytes())
        return digest.digest()


class _TinyBlock(torch.nn.Module):
    def __init__(self, d: int, cross: bool) -> None:
        super().__init__()
        self.cross = cross
        self.norm1 = torch.nn.LayerNorm(d)
        self.attn = torch.nn.MultiheadAttention(d, 2, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(d)
        self.ff = torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.GELU(), torch.nn.Linear(2 * d, d))
        self.time = torch.nn.Linear(d, d)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.norm1(hidden) + self.time(temb)[:, None]
        kv = cond if self.cross else x
        hidden = hidden + self.attn(x, kv, kv, need_weights=False)[0]
        return hidden + self.ff(self.norm2(hidden))


class _TinyNet(torch.nn.Module):
    def __init__(self, spec: TinyGroot) -> None:
        super().__init__()
        d = spec.d_model
        self.spec = spec
        self.patch_embed = torch.nn.Conv2d(3, d, kernel_size=spec.patch, stride=spec.patch)
        self.camera_embed = torch.nn.Parameter(torch.randn(2, 1, d) * 0.02)
        self.task_embed = torch.nn.Embedding(64, d)
        self.state_enc = torch.nn.Sequential(torch.nn.Linear(spec.state_dim, d), torch.nn.GELU(), torch.nn.Linear(d, d))
        self.action_enc = torch.nn.Linear(spec.noise_dim, d)
        self.pos = torch.nn.Parameter(torch.randn(1, spec.action_horizon, d) * 0.02)
        self.time_enc = torch.nn.Sequential(torch.nn.Linear(1, d), torch.nn.SiLU(), torch.nn.Linear(d, d))
        self.blocks = torch.nn.ModuleList(_TinyBlock(d, cross=index % 2 == 0) for index in range(spec.n_blocks))
        self.decoder = torch.nn.Linear(d, spec.noise_dim)

    def encode(self, images: torch.Tensor, state: torch.Tensor, task: int) -> dict:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2) + self.camera_embed  # [2, P, d]
        text = self.task_embed.weight[task].reshape(1, 1, -1)
        cond = torch.cat([patches.reshape(1, -1, patches.shape[-1]), text], dim=1)
        return {"cond": cond, "state": self.state_enc(state)}

    def denoise(self, context: dict, noise: torch.Tensor) -> torch.Tensor:
        batch = noise.shape[0]
        cond = context["cond"].expand(batch, -1, -1)
        state = context["state"].expand(batch, -1, -1)
        actions = noise
        steps = self.spec.n_denoise
        for step in range(steps):
            t = torch.full((batch, 1), step / steps, device=noise.device)
            temb = self.time_enc(t)
            hidden = torch.cat([state, self.action_enc(actions) + self.pos], dim=1)
            for block in self.blocks:
                hidden = block(hidden, cond, temb)
            velocity = self.decoder(hidden[:, 1:])
            actions = actions + velocity / steps
        return actions
