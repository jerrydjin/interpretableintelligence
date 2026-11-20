"""V-JEPA 2-AC, Meta's action-conditioned video world model, behind one small interface.

V-JEPA 2-AC keeps the frozen V-JEPA 2 ViT-g encoder (1.0B parameters) and adds a 305M-parameter
predictor (24 blocks, width 1024) trained on DROID robot video at 4 frames per second. The predictor
reads the latent tokens of the current frame plus two extra tokens per frame, one for the arm's
7-D pose (``x y z`` in metres, extrinsic ``xyz`` Euler angles, gripper 0 open to 1 closed) and one
for the action, which is the pose change to the next frame. It returns the predicted latent tokens
of the next frame. Its planner searches for actions whose predicted latent is closest (mean L1) to
the latent of a goal image.

The weights and the model code are Meta's: ``load_vjepa2ac`` imports the code from a checkout of
github.com/facebookresearch/vjepa2 (``VJEPA2_COMMIT``) and downloads ``vjepa2-ac-vitg.pt``
(several GB) the first time. Neither is a dependency of this package. ``tiny_world_model`` is a random
stand-in with the same interface, for tests on CPU.

Frame handling follows Meta's training and energy-landscape code: the whole camera frame is
resized to 256 x 256 (DROID frames are 16:9 and training resized them with scale 1.777, so the
frame is squashed, not cropped), each frame is repeated to fill one 2-frame tubelet, and both the
encoder output and the predictor output go through a parameter-free layer norm.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
VJEPA2_REPO = "https://github.com/facebookresearch/vjepa2"
VJEPA2_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
CROP = 256
PATCH = 16
TUBELET = 2
STATE_DIM = 7
TRAIN_FPS = 4.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- pose math
# scipy's Rotation.from_euler("xyz") (lowercase: extrinsic x, then y, then z), written out so the
# package does not need scipy. R = Rz(c) @ Ry(b) @ Rx(a).


def euler_xyz_to_matrix(angles: np.ndarray) -> np.ndarray:
    """``[..., 3]`` extrinsic xyz Euler angles (radians) to ``[..., 3, 3]`` rotation matrices."""
    a, b, c = (np.asarray(angles, dtype=np.float64)[..., i] for i in range(3))
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    m = np.empty(a.shape + (3, 3))
    m[..., 0, 0] = cc * cb
    m[..., 0, 1] = cc * sb * sa - sc * ca
    m[..., 0, 2] = cc * sb * ca + sc * sa
    m[..., 1, 0] = sc * cb
    m[..., 1, 1] = sc * sb * sa + cc * ca
    m[..., 1, 2] = sc * sb * ca - cc * sa
    m[..., 2, 0] = -sb
    m[..., 2, 1] = cb * sa
    m[..., 2, 2] = cb * ca
    return m


def matrix_to_euler_xyz(m: np.ndarray) -> np.ndarray:
    """Inverse of ``euler_xyz_to_matrix`` (pitch in ``[-pi/2, pi/2]``)."""
    m = np.asarray(m, dtype=np.float64)
    b = -np.arcsin(np.clip(m[..., 2, 0], -1.0, 1.0))
    a = np.arctan2(m[..., 2, 1], m[..., 2, 2])
    c = np.arctan2(m[..., 1, 0], m[..., 0, 0])
    return np.stack([a, b, c], axis=-1)


def poses_to_diff(states: np.ndarray) -> np.ndarray:
    """Actions between consecutive poses, exactly as Meta's DROID loader builds them.

    ``states`` is ``[T, 7]``; the result is ``[T - 1, 7]``: the xyz difference, the Euler angles
    of ``R[t + 1] @ R[t].T`` and the gripper difference.
    """
    states = np.asarray(states, dtype=np.float64)
    rot = euler_xyz_to_matrix(states[:, 3:6])
    rel = rot[1:] @ np.swapaxes(rot[:-1], -1, -2)
    return np.concatenate(
        [states[1:, :3] - states[:-1, :3], matrix_to_euler_xyz(rel), states[1:, 6:7] - states[:-1, 6:7]],
        axis=1,
    ).astype(np.float32)


# --------------------------------------------------------------------------- frames


def preprocess(frames: np.ndarray, crop: int = CROP) -> torch.Tensor:
    """``[N, H, W, 3]`` uint8 frames to ``[N, 3, TUBELET, crop, crop]`` normalised video tensors."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(crop, crop), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(2).repeat(1, 1, TUBELET, 1, 1)


# --------------------------------------------------------------------------- the model


@dataclass
class ACWorldModel:
    """Encoder plus action-conditioned predictor. Latents are ``[P, D]`` per frame, layer-normed."""

    encoder: nn.Module
    predictor: nn.Module
    device: torch.device
    crop: int = CROP
    patch: int = PATCH
    dtype: torch.dtype = torch.float32
    name: str = "vjepa2-ac"
    info: dict = field(default_factory=dict)
    encode_batch: int = 8
    predict_batch: int = 16

    @property
    def grid(self) -> tuple[int, int]:
        return (self.crop // self.patch, self.crop // self.patch)

    @property
    def tokens_per_frame(self) -> int:
        h, w = self.grid
        return h * w

    def _autocast(self):
        if self.device.type == "cuda" and self.dtype != torch.float32:
            return torch.autocast("cuda", dtype=self.dtype)
        return torch.autocast("cpu", enabled=False)

    @torch.inference_mode()
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """``[N, H, W, 3]`` uint8 frames to ``[N, P, D]`` float32 latents on the CPU."""
        out = []
        for i in range(0, len(frames), self.encode_batch):
            x = preprocess(frames[i : i + self.encode_batch], self.crop).to(self.device)
            with self._autocast():
                h = self.encoder(x)
            h = F.layer_norm(h.float(), (h.shape[-1],))
            out.append(h.cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def predict_next(self, z: torch.Tensor, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """One step: ``z [B, P, D]``, ``states [B, 7]``, ``actions [B, 7]`` to ``[B, P, D]`` on the CPU."""
        out = []
        p = self.tokens_per_frame
        for i in range(0, len(z), self.predict_batch):
            zb = z[i : i + self.predict_batch].to(self.device)
            sb = states[i : i + self.predict_batch].to(self.device, torch.float32).unsqueeze(1)
            ab = actions[i : i + self.predict_batch].to(self.device, torch.float32).unsqueeze(1)
            with self._autocast():
                y = self.predictor(zb, ab, sb)[:, -p:]
            y = F.layer_norm(y.float(), (y.shape[-1],))
            out.append(y.cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def energy(
        self,
        z: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
        actions: torch.Tensor,
        *,
        per_token: bool = False,
    ) -> torch.Tensor:
        """The planner's cost for each of ``actions [N, 7]`` from one frame: mean L1 to ``goal``.

        ``z`` and ``goal`` are ``[P, D]``, ``state`` is ``[7]``. Returns ``[N]`` on the CPU, or
        ``[N, P]`` per-token costs with ``per_token``. Everything stays on the device until the end.
        """
        p = self.tokens_per_frame
        zd = z.to(self.device).unsqueeze(0)
        gd = goal.to(self.device).float().unsqueeze(0)
        sd = state.to(self.device, torch.float32).view(1, 1, -1)
        out = []
        for i in range(0, len(actions), self.predict_batch):
            ab = actions[i : i + self.predict_batch].to(self.device, torch.float32).unsqueeze(1)
            n = len(ab)
            with self._autocast():
                y = self.predictor(zd.expand(n, -1, -1), ab, sd.expand(n, -1, -1))[:, -p:]
            y = F.layer_norm(y.float(), (y.shape[-1],))
            cost = (y - gd).abs().mean(dim=-1)
            out.append((cost if per_token else cost.mean(dim=-1)).cpu())
        return torch.cat(out)


def load_vjepa2ac(
    device: torch.device | str,
    *,
    code_dir: Path | str,
    checkpoint: Path | str | None = None,
    random_predictor: bool = False,
    seed: int = 0,
    dtype: torch.dtype | None = None,
) -> ACWorldModel:
    """Meta's V-JEPA 2-AC from a checkout of the vjepa2 repo at ``code_dir``.

    ``checkpoint`` is a local ``vjepa2-ac-vitg.pt``; without one it is downloaded to the torch hub
    cache. ``random_predictor`` keeps the trained encoder and leaves the predictor at its seeded
    initialisation: the random-weights control for anything that reads the predictor.
    """
    device = torch.device(device)
    code_dir = str(Path(code_dir).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from src.models.ac_predictor import vit_ac_predictor  # type: ignore[import-not-found]
    from src.models.vision_transformer import vit_giant_xformers  # type: ignore[import-not-found]

    encoder = vit_giant_xformers(
        patch_size=PATCH,
        img_size=(CROP, CROP),
        num_frames=64,
        tubelet_size=TUBELET,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
    )
    torch.manual_seed(seed)
    predictor = vit_ac_predictor(
        img_size=(CROP, CROP),
        patch_size=PATCH,
        num_frames=64,
        tubelet_size=TUBELET,
        embed_dim=encoder.embed_dim,
    )
    if checkpoint is None:
        state = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location="cpu")
    else:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)

    def clean(sd: dict) -> dict:
        return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}

    missing, unexpected = encoder.load_state_dict(clean(state["encoder"]), strict=False)
    info = {"encoder_missing": len(missing), "encoder_unexpected": len(unexpected), "random_predictor": random_predictor}
    if not random_predictor:
        predictor.load_state_dict(clean(state["predictor"]), strict=True)
    del state
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Half-precision weights on the GPU: 1.3B parameters in fp32 plus the segmenter do not fit a 16 GB T4.
    weight_dtype = dtype if device.type == "cuda" else torch.float32
    encoder.to(device=device, dtype=weight_dtype).eval()
    predictor.to(device=device, dtype=weight_dtype).eval()
    name = "vjepa2-ac-random-predictor" if random_predictor else "vjepa2-ac"
    return ACWorldModel(encoder=encoder, predictor=predictor, device=device, dtype=dtype, name=name, info=info)


# --------------------------------------------------------------------------- tiny stand-in


class _TinyEncoder(nn.Module):
    def __init__(self, dim: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, 3, T, H, W]
        return self.proj(x[:, :, 0]).flatten(2).transpose(1, 2)


class _TinyPredictor(nn.Module):
    """Copies the frame and adds an action-driven change on the tokens in ``action_mask`` only.

    A deliberately "arm-only" world model, so tests know the answer ``ii action-effects`` must give.
    """

    def __init__(self, dim: int, action_mask: torch.Tensor) -> None:
        super().__init__()
        self.act = nn.Linear(2 * STATE_DIM, dim)
        self.register_buffer("gate", action_mask.float().view(1, -1, 1))

    def forward(self, z: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        p = self.gate.shape[1]
        drive = torch.tanh(self.act(torch.cat([actions[:, -1], states[:, -1]], dim=-1) * 20.0))
        return z[:, -p:] + 2.0 * self.gate * drive.unsqueeze(1)


def tiny_world_model(*, seed: int = 0, crop: int = 128, patch: int = 16, dim: int = 16, action_tokens=None) -> ACWorldModel:
    """Random CPU stand-in with V-JEPA 2-AC's interface. Actions move only ``action_tokens``."""
    torch.manual_seed(seed)
    side = crop // patch
    mask = torch.zeros(side * side, dtype=torch.bool)
    if action_tokens is None:
        action_tokens = range(side)  # the top row of the grid
    mask[list(action_tokens)] = True
    model = ACWorldModel(
        encoder=_TinyEncoder(dim, patch).eval(),
        predictor=_TinyPredictor(dim, mask).eval(),
        device=torch.device("cpu"),
        crop=crop,
        patch=patch,
        name="tiny",
        info={"action_tokens": [int(i) for i in torch.nonzero(mask).flatten()]},
    )
    return model


def frames_per_step(source_fps: float, fps: float = TRAIN_FPS) -> float:
    """Source frames per model step; DROID video is 15 fps and V-JEPA 2-AC was trained at 4."""
    if source_fps <= 0 or fps <= 0:
        raise ValueError("fps must be positive")
    return source_fps / fps


def resample_indices(n_frames: int, source_fps: float, fps: float = TRAIN_FPS) -> list[int]:
    """Indices of the source frames nearest to each ``1 / fps`` tick, starting at frame 0."""
    step = frames_per_step(source_fps, fps)
    count = int(math.floor((n_frames - 1) / step)) + 1
    return [min(n_frames - 1, int(round(i * step))) for i in range(count)]
