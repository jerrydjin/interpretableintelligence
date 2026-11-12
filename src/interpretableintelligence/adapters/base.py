"""Adapter protocol shared by every later model lane."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

import numpy as np
import torch

HookMode = Literal["capture", "edit", "zero"]
EditFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class Grid:
    t: int
    h: int
    w: int
    tubelet: int
    patch_px: int

    def as_dict(self) -> dict[str, int]:
        return {
            "t": self.t,
            "h": self.h,
            "w": self.w,
            "tubelet": self.tubelet,
            "patch_px": self.patch_px,
        }


@dataclass
class Clip:
    clip_id: str
    path: Path
    frames: np.ndarray
    fps: float
    sha256: str

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])


@dataclass
class Window:
    index: int
    start_frame: int
    frames: np.ndarray


@dataclass
class HookHandle:
    layer: int
    mode: HookMode
    fn: EditFn | None
    captured: list[torch.Tensor]


class Adapter(Protocol):
    """Minimal surface the scoring core needs from a video world model or policy.

    ``predict_masked`` returns ``(pred, target)`` for the masked region only,
    in row-major ``(t, h, w)`` order of the True entries of ``mask(window)``.

    An adapter may also define ``predict_masked_many(clip, window, edits)``,
    returning ``predict_masked(clip, window, edit=e)[0]`` for each edit, to
    batch the random-feature controls. Without it the scorer loops.
    """

    name: str
    layer: int

    def load(self, random_init: bool = False) -> None: ...

    def windows(self, clip: Clip) -> list[Window]: ...

    def hook(
        self,
        layer: int,
        mode: HookMode = "capture",
        fn: EditFn | None = None,
    ) -> AbstractContextManager[HookHandle]: ...

    def encode(self, clip: Clip, window: Window) -> torch.Tensor: ...

    def predict_masked(
        self,
        clip: Clip,
        window: Window,
        edit: EditFn | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def attribution_objective(
        self,
        clip: Clip,
        window: Window,
        objective: str,
        edit: EditFn | None = None,
    ) -> torch.Tensor:
        """Differentiable scalar the attribution prefilter backprops into.

        ``prediction_loss`` is the mean masked predictor MSE (lower is better).
        ``action_loglik`` is the mean log-likelihood of the action the policy
        took (higher is better), a stand-in for success when success itself
        is not differentiable. The returned value is a 0-dim tensor.
        """

    def grid(self) -> Grid: ...

    def mask(self, window: Window) -> torch.Tensor: ...

    def raw_activations(self, clip: Clip, window: Window) -> torch.Tensor:
        """Layer residual before hooks, ``[t, h, w, d_model]``.

        The straight-through feature estimate reads this tensor.
        """
        ...

    def visible_mean_baseline(self, clip: Clip, window: Window) -> torch.Tensor:
        """Masked-token prediction equal to the mean of the visible patches.

        Same shape and token order as ``predict_masked``'s target. The scorer
        records the loss of this guess and refuses a usable SAE that does not
        beat it.
        """
        ...

    def model_sha(self) -> str: ...

    @property
    def d_model(self) -> int: ...

    @property
    def random_init(self) -> bool: ...
