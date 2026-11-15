"""GR00T rollout policy stubs and metadata for ``ii rollouts``.

``groot-mock`` replays the frozen verification-slice fixture in CI.
``groot-1028`` runs (or imports) the v9 LIBERO-plus ablation runner on GPU.
"""

from __future__ import annotations

from contextlib import contextmanager

from interpretableintelligence.policies.base import Intervention

GROOT_POLICIES: frozenset[str] = frozenset({"groot-1028", "groot-mock"})
GROOT_LAYER = 5
GROOT_TASK_ID = "1028"
N_FEATURES = 8192
HORIZON = 600


def is_groot_policy(name: str) -> bool:
    return name in GROOT_POLICIES


class GrootRolloutPolicyStub:
    """Metadata and no-op loop; execution goes through ``rollout_groot.execute_groot_rollouts``."""

    layer = GROOT_LAYER
    n_features = N_FEATURES
    horizon = HORIZON

    def __init__(self, name: str) -> None:
        if not is_groot_policy(name):
            raise ValueError(f"not a groot rollout policy: {name!r}")
        self.name = name

    def reset(self, *, task_id: str, seed: int):
        del task_id, seed
        raise NotImplementedError(
            f"policy {self.name!r} episodes are driven by the v9 ablation runner; use ii rollouts"
        )

    def act(self, observation):
        del observation
        raise NotImplementedError(f"policy {self.name!r} is not a step-wise simulator")

    def step(self, action):
        del action
        raise NotImplementedError(f"policy {self.name!r} is not a step-wise simulator")

    @contextmanager
    def intervene(self, intervention: Intervention | None):
        del intervention
        yield None


def groot_policy(name: str) -> GrootRolloutPolicyStub:
    return GrootRolloutPolicyStub(name)