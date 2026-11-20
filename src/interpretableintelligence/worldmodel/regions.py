"""Which latent patches show the arm, which show something else that moved, which show still background.

The world model sees each frame as a ``gh x gw`` grid of patches (16 x 16 for V-JEPA 2-AC). For each
transition ``t -> t + 1`` (one model step) we label patches from pixels, never from the model:

``arm``
    At least ``arm_frac`` of the patch is arm, in frame ``t`` or ``t + 1``, and at least ``move_frac``
    of its pixels changed: the part of the arm that is moving. The arm mask comes from a
    text-prompted segmenter (``segment.py``) or, in tests, from a stub.
``moved``
    Not within ``ring`` patches of the arm (in either frame; default 2), and at least ``move_frac`` of its pixels
    changed by more than ``diff_thresh`` grey levels between the two frames. DROID's exterior
    cameras are fixed, so on a robot episode this is something other than the arm that moved: a
    pushed object, a drawer or door, something falling. Objects held in the gripper sit next to the
    arm and are left out on purpose: at 16-pixel patches they cannot be told apart from the gripper,
    and arm pixels the segmenter misses (often the gripper tip) sit next to the arm too.
``still``
    Not within ``ring`` patches of the arm, not ``moved``, and under ``still_frac`` changed pixels.

A transition is ``valid`` only when the segmenter found the arm in both frames (at least
``min_arm_px`` of the frame), so a missed detection cannot turn the arm into a "moved" region.

All masks are computed on the frame resized to the model's square input, the same squash the model
sees, so a patch here is a token there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

DIFF_THRESH = 25.0
ARM_FRAC = 0.2
MOVE_FRAC = 0.1
STILL_FRAC = 0.02
DILATE_PX = 6
MIN_ARM_PX = 0.005
RING = 2


@dataclass
class TransitionRegions:
    arm: np.ndarray  # [P] bool
    moved: np.ndarray  # [P] bool
    still: np.ndarray  # [P] bool
    valid: bool = True


def _resize(x: np.ndarray, crop: int) -> torch.Tensor:
    """``[N, H, W]`` float array to ``[N, 1, crop, crop]``."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(1)
    return F.interpolate(t, size=(crop, crop), mode="bilinear", align_corners=False, antialias=True)


def _gray(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def patch_fractions(mask: torch.Tensor, patch: int) -> torch.Tensor:
    """``[N, 1, crop, crop]`` 0/1 masks to ``[N, P]`` fractions per patch, row-major like the tokens."""
    return F.avg_pool2d(mask, kernel_size=patch, stride=patch).flatten(1)


def transition_regions(
    frames: np.ndarray,
    arm_masks: np.ndarray,
    *,
    crop: int,
    patch: int,
    diff_thresh: float = DIFF_THRESH,
    arm_frac: float = ARM_FRAC,
    move_frac: float = MOVE_FRAC,
    still_frac: float = STILL_FRAC,
    dilate_px: int = DILATE_PX,
    min_arm_px: float = MIN_ARM_PX,
    ring: int = RING,
) -> list[TransitionRegions]:
    """Patch labels for every consecutive pair of ``frames [T, H, W, 3]`` with ``arm_masks [T, H, W]``."""
    if len(frames) != len(arm_masks):
        raise ValueError("one arm mask per frame")
    side = crop // patch
    gray = _resize(_gray(frames), crop)
    gray = F.avg_pool2d(gray, 3, stride=1, padding=1)  # soften compression noise
    arm = (_resize(arm_masks.astype(np.float32), crop) > 0.5).float()
    out = []
    for t in range(len(frames) - 1):
        arm_px = torch.maximum(arm[t], arm[t + 1]).unsqueeze(0)
        grown = F.max_pool2d(arm_px, 2 * dilate_px + 1, stride=1, padding=dilate_px) if dilate_px > 0 else arm_px
        diff = ((gray[t + 1] - gray[t]).abs() > diff_thresh).float().unsqueeze(0)
        moved_px = diff * (1.0 - grown)
        changed = patch_fractions(moved_px, patch)[0]
        arm_any = patch_fractions(arm_px, patch)[0] >= arm_frac
        arm_p = arm_any & (patch_fractions(diff, patch)[0] >= move_frac)
        near = F.max_pool2d(arm_any.float().view(1, 1, side, side), 2 * ring + 1, stride=1, padding=ring).flatten() > 0
        moved_p = ~near & (changed >= move_frac)
        still_p = ~near & ~moved_p & (patch_fractions(diff, patch)[0] < still_frac)
        valid = bool(arm[t].mean() >= min_arm_px and arm[t + 1].mean() >= min_arm_px)
        out.append(TransitionRegions(arm=arm_p.numpy(), moved=moved_p.numpy(), still=still_p.numpy(), valid=valid))
    return out


def overlay(frame: np.ndarray, regions: TransitionRegions, *, crop: int, patch: int) -> np.ndarray:
    """The squashed frame with arm patches tinted red, moved patches green, still patches dimmed."""
    side = crop // patch
    img = _resize(frame.astype(np.float32).transpose(2, 0, 1), crop)[:, 0].numpy().transpose(1, 2, 0)

    def up(mask: np.ndarray) -> np.ndarray:
        return np.kron(mask.reshape(side, side).astype(np.float32), np.ones((patch, patch), np.float32))[..., None]

    tint = img.copy()
    tint = tint * (1 - 0.5 * up(regions.arm)) + up(regions.arm) * 0.5 * np.array([255, 0, 0], np.float32)
    tint = tint * (1 - 0.5 * up(regions.moved)) + up(regions.moved) * 0.5 * np.array([0, 255, 0], np.float32)
    tint = tint * (1 - 0.4 * up(regions.still))
    return np.clip(tint, 0, 255).astype(np.uint8)


def write_png(path, image: np.ndarray) -> None:
    """Minimal RGB PNG writer, so previews need no imaging library."""
    import struct
    import zlib

    image = np.ascontiguousarray(image, dtype=np.uint8)
    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
