"""Fake policy and reacher with one planted causal SAE feature.

Ablating feature ``PLANTED_FEATURE_ID`` drops terminal success from about
0.80 to about 0.30. Ablating any other feature leaves the trajectory, and
therefore the success bit, unchanged.

The loop is a point-mass reacher. Perception writes the target into one SAE
coordinate at reset. The controller servos to that coordinate. On a hard
episode (probability 0.2) a late gust knocks a gripper that has already
reached the target off it, so an informed policy succeeds on the other 0.8.
With the coordinate removed the gripper stays at the origin and only the
closest targets count, which is 0.3 of episodes after the same gust rule.

``plant=False`` (policy name ``fake-null``) turns that effect off. The
controller ignores every latent, and the terminal bit is a fair coin of the
post-intervention latents. Every arm then has the same success probability,
but zeroing a coordinate changes the draw, so the arms are exchangeable
without being identical.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np

from interpretableintelligence.policies.base import (
    Intervention,
    apply_latent_intervention,
    dictionary_seed,
    edit_through_dictionary,
)

PLANTED_FEATURE_ID = 3
N_FEATURES = 32
LAYER = 0
HORIZON = 6
# |target| <= this is an accidental success for a policy that never moves.
# Combined with the hard-episode gust below, the uninformed success rate is
# SUCCESS_RADIUS * (1 - HARD_EPISODE) = 0.375 * 0.8 = 0.30.
SUCCESS_RADIUS = 0.375
HARD_EPISODE = 0.80
TASK_ID = "reach"
_SEED_SALT = 0x5A1E


class FakePolicyEnv:
    """Identity SAE. Latent ``PLANTED_FEATURE_ID`` is the target location.

    ``plant=True`` is the causal reacher. ``plant=False`` is the null policy:
    no feature is causal, and success is a fair coin of the steered latents.
    """

    layer = LAYER
    n_features = N_FEATURES
    horizon = HORIZON
    planted_feature_id = PLANTED_FEATURE_ID
    # Untrained SAEs of the identity SAE's shape (32 x 32), for ``--random-sae``.
    random_dictionaries = True

    def __init__(self, *, plant: bool = True) -> None:
        self.plant = bool(plant)
        self.name = "fake" if self.plant else "fake-null"
        self._intervention: Intervention | None = None
        self._latents: np.ndarray | None = None
        self._steered: np.ndarray | None = None
        self._episode_seed = 0
        self._pos = 0.0
        self._target = 0.0
        self._u = 0.0
        self._steps = 0
        self._task_id = TASK_ID
        self._random_saes: dict[int, object] = {}

    def reset(self, *, task_id: str, seed: int) -> np.ndarray:
        if task_id != TASK_ID:
            raise ValueError(f"fake policy only implements task {TASK_ID!r}, got {task_id!r}")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), _SEED_SALT]))
        self._u = float(rng.random())
        self._target = float(rng.uniform(-1.0, 1.0))
        latents = rng.normal(0.0, 1.0, size=self.n_features)
        latents[PLANTED_FEATURE_ID] = self._target
        self._latents = np.asarray(latents, dtype=np.float64)
        self._steered = None
        self._episode_seed = int(seed)
        self._pos = 0.0
        self._steps = 0
        self._task_id = task_id
        return self._observe()

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self._latents is None:
            raise RuntimeError("call reset() before act()")
        activation = self.layer_activation(observation)
        seed = None if self._intervention is None else dictionary_seed(self._intervention.dictionary)
        if seed is None:
            latents = self.encode_latents(activation)
            # Hook: SAE latents at `self.layer`, after encode and before decode.
            latents = apply_latent_intervention(latents, self._intervention)
            steered = self.decode_latents(latents)
        else:
            steered = edit_through_dictionary(activation, self.random_sae(seed), self._intervention)
        self._steered = steered
        return self.controller(steered, observation)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self._steps += 1
        self._pos = float(np.clip(self._pos + float(action[0]), -2.0, 2.0))
        if (
            self._steps == self.horizon
            and self._u >= HARD_EPISODE
            and abs(self._pos - self._target) <= SUCCESS_RADIUS
        ):
            # Late disturbance. It only throws a gripper that is already on the target.
            self._pos = float(self._pos + (1.0 if self._target >= 0.0 else -1.0))
        done = self._steps >= self.horizon
        if self.plant:
            success = bool(done and abs(self._pos - self._target) <= SUCCESS_RADIUS)
        else:
            success = bool(done and self._null_success())
        reward = 1.0 if success else 0.0
        return self._observe(), reward, done, {"success": success}

    def _null_success(self) -> bool:
        """Fair bit of the steered latents. The bias does not depend on which feature was edited."""
        if self._steered is None:
            raise RuntimeError("call act() before reading the null success bit")
        digest = hashlib.sha256()
        digest.update(b"interpretableintelligence-fake-null-v1")
        digest.update(int(self._episode_seed).to_bytes(8, "little", signed=True))
        digest.update(np.ascontiguousarray(self._steered, dtype=np.float64).tobytes())
        word = int.from_bytes(digest.digest()[:8], "little")
        return (word & 1) == 0

    @contextmanager
    def intervene(self, intervention: Intervention | None):
        if intervention is not None and intervention.layer != self.layer:
            raise ValueError(
                f"fake policy is instrumented at layer {self.layer}, got intervention on {intervention.layer}"
            )
        previous = self._intervention
        self._intervention = intervention
        try:
            yield None
        finally:
            self._intervention = previous

    def random_sae(self, seed: int):
        """The untrained SAE ``random:<seed>``: training's own initialisation, never stepped.

        ``b_dec`` is 0, the mean of this layer's activations (Gaussian codes and a
        target uniform on [-1, 1]).
        """
        cached = self._random_saes.get(int(seed))
        if cached is None:
            import torch

            from interpretableintelligence.sae import SAEConfig, untrained_sae

            config = SAEConfig(d_model=N_FEATURES, n_features=N_FEATURES, seed=int(seed))
            cached = untrained_sae(config, torch.zeros(N_FEATURES), torch.device("cpu")).eval().requires_grad_(False)
            self._random_saes[int(seed)] = cached
        return cached

    def layer_activation(self, observation: np.ndarray) -> np.ndarray:
        """Perception code for this episode. Cached at reset; the position is separate."""
        del observation
        if self._latents is None:
            raise RuntimeError("call reset() before reading the layer")
        return self._latents.copy()

    def encode_latents(self, activation: np.ndarray) -> np.ndarray:
        """Identity SAE encoder. A real adapter would apply the run's ``W_enc``."""
        return np.array(activation, dtype=np.float64, copy=True)

    def decode_latents(self, latents: np.ndarray) -> np.ndarray:
        """Identity SAE decoder."""
        return np.array(latents, dtype=np.float64, copy=True)

    def controller(self, latents: np.ndarray, observation: np.ndarray) -> np.ndarray:
        if not self.plant:
            return np.array([0.0], dtype=np.float64)
        signal = float(latents[PLANTED_FEATURE_ID])
        pos = float(observation[0])
        if abs(signal) < 1e-8:
            action = 0.0
        else:
            action = float(np.clip(signal - pos, -1.0, 1.0))
        return np.array([action], dtype=np.float64)

    def _observe(self) -> np.ndarray:
        return np.array([self._pos], dtype=np.float64)
