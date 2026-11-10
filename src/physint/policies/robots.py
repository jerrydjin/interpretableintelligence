"""Stubs for LIBERO, OpenVLA, and pi0.5.

The optional ``robots`` extra marks this lane. It depends on nothing: the
default install and the extra both stay free of simulators and GPU packages.
Constructing one of these policies raises until a real adapter is wired up.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from physint.policies.base import Intervention


ROBOT_POLICY_NAMES: tuple[str, ...] = ("libero", "openvla", "pi05", "pi0.5")


def unavailable_policy(name: str) -> None:
    raise NotImplementedError(
        f"policy {name!r} is a stub. The optional 'robots' extra reserves the "
        "LIBERO / OpenVLA / pi0.5 lane and intentionally pulls no simulator or "
        "GPU dependency. There is no real adapter in this build."
    )


class RobotPolicyStub:
    """Protocol-shaped placeholder. Every call raises."""

    def __init__(self, name: str) -> None:
        if name not in ROBOT_POLICY_NAMES:
            raise ValueError(f"not a robot-policy stub: {name!r}")
        self.name = name
        self.layer = 0
        self.n_features = 0
        self.horizon = 0

    def reset(self, *, task_id: str, seed: int) -> np.ndarray:
        del task_id, seed
        unavailable_policy(self.name)
        raise AssertionError

    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        unavailable_policy(self.name)
        raise AssertionError

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        del action
        unavailable_policy(self.name)
        raise AssertionError

    @contextmanager
    def intervene(self, intervention: Intervention | None):
        del intervention
        unavailable_policy(self.name)
        yield None
