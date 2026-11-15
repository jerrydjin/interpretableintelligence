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


TRAINED_DICTIONARY = "trained"
_RANDOM_PREFIX = "random:"


def random_dictionary(seed: int) -> str:
    """Name of the untrained SAE drawn with ``seed``."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"random dictionary seed must be a non-negative integer, got {seed!r}")
    return f"{_RANDOM_PREFIX}{int(seed)}"


def dictionary_seed(name: str) -> int | None:
    """``None`` for the trained SAE, else the seed of a ``random:<seed>`` dictionary."""
    if name == TRAINED_DICTIONARY:
        return None
    if isinstance(name, str) and name.startswith(_RANDOM_PREFIX):
        digits = name[len(_RANDOM_PREFIX) :]
        if digits.isdigit():
            return int(digits)
    raise ValueError(f"dictionary must be {TRAINED_DICTIONARY!r} or 'random:<seed>', got {name!r}")


@dataclass(frozen=True)
class Intervention:
    """Edit SAE latents at one layer.

    ``ablate`` sets each listed feature to 0 (``value`` is recorded as 0).
    ``scale`` multiplies each listed feature by ``value``.
    ``clamp`` sets each listed feature to the constant ``value``.

    ``dictionary`` names the SAE whose latents are edited: ``trained`` (the
    policy's own SAE) or ``random:<seed>``, an untrained SAE of the same shape
    (:func:`interpretableintelligence.sae.untrained_sae`). That one is the
    random-weights SAE control.
    """

    feature_ids: tuple[int, ...]
    layer: int
    kind: str
    value: float
    dictionary: str = TRAINED_DICTIONARY

    def __post_init__(self) -> None:
        dictionary_seed(self.dictionary)  # validates the name
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


def edit_through_dictionary(activation: np.ndarray, sae, intervention: Intervention) -> np.ndarray:
    """Apply ``intervention`` to ``sae``'s latents of ``activation`` and keep the SAE's error.

    ``x' = x + W_dec (z' - z)`` with ``z = encode(x)``: only the edited features'
    decoder contributions change, and an edit that leaves the latents alone returns
    ``x`` exactly. GR00T's v9 hook and the picker's offline prune edit this way. An
    untrained SAE reconstructs badly, so replacing ``x`` with its reconstruction
    would measure the reconstruction, not the edit.
    """
    import torch

    x = np.asarray(activation, dtype=np.float64)
    with torch.no_grad():
        latents = sae.encode(torch.as_tensor(x, dtype=torch.float32)).double().numpy()
    edited = apply_latent_intervention(latents, intervention)
    decoder = sae.W_dec.detach().double().numpy()  # [d_model, n_features]
    return x + (edited - latents) @ decoder.T


class PolicyEnv(Protocol):
    """Closed loop the rollout runner drives.

    ``reset`` starts an episode and returns the first observation.
    ``act`` is the policy. It encodes the instrumented layer with an SAE,
    applies the intervention installed by ``intervene``, decodes, and
    returns an action.
    ``step`` is the simulator. It returns ``(observation, reward, done, info)``
    and puts ``info["success"]`` on the terminal step.

    Optional capabilities, found with ``getattr``:

    - ``random_dictionaries = True``: ``intervene`` accepts
      ``Intervention.dictionary = "random:<seed>"`` and edits an untrained SAE of
      the trained one's shape. ``ii rollouts --random-sae`` needs it.
    - ``pick_features(dictionary, n, *, kind, value, seed) -> list[int]``: the
      policy's own candidate picker. Without it ``ii rollouts --pick`` screens
      every feature offline (:mod:`interpretableintelligence.rollout_pick`).
    """

    name: str
    layer: int
    n_features: int
    horizon: int

    def reset(self, *, task_id: str, seed: int) -> np.ndarray: ...

    def act(self, observation: np.ndarray) -> np.ndarray: ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...

    def intervene(self, intervention: Intervention | None) -> AbstractContextManager[None]: ...
