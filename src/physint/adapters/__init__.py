"""Model adapters. ``fake`` is the scoring acceptance test; ``vjepa2`` is V-JEPA 2."""

from physint.adapters.base import Adapter, Clip, Grid, Window
from physint.adapters.fake import FakeAdapter

ADAPTERS: dict[str, type] = {"fake": FakeAdapter}


def get_adapter(name: str) -> type:
    if name == "vjepa2" and name not in ADAPTERS:
        from physint.adapters.vjepa2 import VJEPA2Adapter

        ADAPTERS[name] = VJEPA2Adapter
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(set(ADAPTERS) | {"vjepa2"}))
        raise ValueError(f"unknown adapter {name!r}. Known adapters: {known}.") from exc


__all__ = ["Adapter", "Clip", "FakeAdapter", "Grid", "Window", "get_adapter"]
