"""V-JEPA 2-AC: Meta's action-conditioned world model for robot arms (``ii action-effects``)."""

from interpretableintelligence.adapters.vjepa2ac.model import (
    ACWorldModel,
    load_vjepa2ac,
    poses_to_diff,
    tiny_world_model,
)

__all__ = ["ACWorldModel", "load_vjepa2ac", "poses_to_diff", "tiny_world_model"]
