"""Policy + simulator protocol for causal rollouts.

A pair exposes ``reset``, ``step``, and ``act``. ``act`` runs the instrumented
layer through an SAE and applies the active intervention to the latents
before the controller reads them. ``intervene`` is the hook that installs
that intervention for the body of an episode.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

InterventionKind = Literal["ablate", "scale", "clamp"]
KINDS: tuple[str, ...] = ("ablate", "scale", "clamp")


@dataclass(frozen=True)
class Intervention:
    """Edit SAE latents at one layer.

    ``ablate`` sets each listed feature to 0 (``value`` is recorded as 0).
    ``scale`` multiplies each listed feature by ``value``.
    ``clamp`` sets each listed feature to the constant ``value``.
    """

    feature_ids: tuple[int, ...]
    layer: int
    kind: str
    value: float

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"intervention kind must be ablate|scale|clamp, got {self.kind!r}")
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError(f"layer must be a non-negative integer, got {self.layer!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or not math.isfinite(float(self.value)):
            raise ValueError(f"value must be finite, got {self.value!r}")
        ids: list[int] = []
        for item in self.feature_ids:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"feature ids must be non-negative integers, got {item!r}")
            ids.append(int(item))
        if not ids:
            raise ValueError("an intervention needs at least one feature id")
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate feature id in {ids}")
        object.__setattr__(self, "feature_ids", tuple(sorted(ids)))
        object.__setattr__(self, "value", float(self.value))


def apply_latent_intervention(latents: np.ndarray, intervention: Intervention | None) -> np.ndarray:
    """Return a copy of ``latents`` with the intervention applied on the last axis."""
    out = np.array(latents, dtype=np.float64, copy=True)
    if intervention is None:
        return out
    index = np.asarray(intervention.feature_ids, dtype=np.int64)
    if int(index.max()) >= out.shape[-1]:
        raise ValueError(
            f"feature id {int(index.max())} is outside latent width {out.shape[-1]}"
        )
    if intervention.kind == "ablate":
        out[..., index] = 0.0
    elif intervention.kind == "scale":
        out[..., index] *= intervention.value
    elif intervention.kind == "clamp":
        out[..., index] = intervention.value
    else:
        raise ValueError(f"unknown intervention kind {intervention.kind!r}")
    return out


class PolicyEnv(Protocol):
    """Closed loop the rollout runner drives.

    ``reset`` starts an episode and returns the first observation.
    ``act`` is the policy. It encodes the instrumented layer with an SAE,
    applies the intervention installed by ``intervene``, decodes, and
    returns an action.
    ``step`` is the simulator. It returns ``(observation, reward, done, info)``
    and puts ``info["success"]`` on the terminal step.
    """

    name: str
    layer: int
    n_features: int
    horizon: int

    def reset(self, *, task_id: str, seed: int) -> np.ndarray: ...

    def act(self, observation: np.ndarray) -> np.ndarray: ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...

    def intervene(self, intervention: Intervention | None) -> AbstractContextManager[None]: ...
