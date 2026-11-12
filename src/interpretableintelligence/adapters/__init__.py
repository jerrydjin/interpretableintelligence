"""Model adapters. ``fake`` is the scoring acceptance test; ``vjepa2`` is V-JEPA 2.

``hf:<model>`` is the generic Hugging Face vision adapter (see ``adapters/hf.py``).
"""

from interpretableintelligence.adapters.base import Adapter, Clip, Grid, Window
from interpretableintelligence.adapters.fake import FakeAdapter

ADAPTERS: dict[str, type] = {"fake": FakeAdapter}


def get_adapter(name: str) -> type:
    if name.startswith("hf:"):
        from interpretableintelligence.adapters.hf import adapter_class

        return adapter_class(name)
    if name == "vjepa2" and name not in ADAPTERS:
        from interpretableintelligence.adapters.vjepa2 import VJEPA2Adapter

        ADAPTERS[name] = VJEPA2Adapter
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(set(ADAPTERS) | {"vjepa2", "hf:<model>"}))
        raise ValueError(f"unknown adapter {name!r}. Known adapters: {known}.") from exc


__all__ = ["Adapter", "Clip", "FakeAdapter", "Grid", "Window", "get_adapter"]
