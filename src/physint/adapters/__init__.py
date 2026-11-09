"""Model adapters. This lane ships the fake adapter only."""

from physint.adapters.base import Adapter, Clip, Grid, Window
from physint.adapters.fake import FakeAdapter

ADAPTERS: dict[str, type] = {"fake": FakeAdapter}


def get_adapter(name: str) -> type:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(
            f"unknown adapter {name!r}. This lane ships only: {known}. "
            "V-JEPA 2 and pi0.5 adapters are a later lane."
        ) from exc


__all__ = ["Adapter", "Clip", "FakeAdapter", "Grid", "Window", "get_adapter"]
