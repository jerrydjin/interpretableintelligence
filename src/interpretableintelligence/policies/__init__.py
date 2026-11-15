"""Policy + simulator pairs. The default install ships ``fake`` and ``fake-null``."""

from __future__ import annotations

from interpretableintelligence.policies.base import Intervention, PolicyEnv, apply_latent_intervention
from interpretableintelligence.policies.fake import PLANTED_FEATURE_ID, FakePolicyEnv
from interpretableintelligence.policies.groot import groot_policy
from interpretableintelligence.policies.robots import ROBOT_POLICY_NAMES, unavailable_policy
from interpretableintelligence.policies.groot import GROOT_POLICIES

__all__ = [
    "PLANTED_FEATURE_ID",
    "ROBOT_POLICY_NAMES",
    "FakePolicyEnv",
    "Intervention",
    "PolicyEnv",
    "apply_latent_intervention",
    "get_policy",
]


def get_policy(name: str) -> PolicyEnv:
    if name == "fake":
        return FakePolicyEnv(plant=True)
    if name == "fake-null":
        return FakePolicyEnv(plant=False)
    if name in GROOT_POLICIES:
        return groot_policy(name)
    if name in ROBOT_POLICY_NAMES:
        unavailable_policy(name)
    known = ", ".join(("fake", "fake-null", *sorted(GROOT_POLICIES), *ROBOT_POLICY_NAMES))
    raise ValueError(f"unknown policy {name!r}. Known policies: {known}.")
