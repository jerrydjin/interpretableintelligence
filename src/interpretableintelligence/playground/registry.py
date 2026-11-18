"""The packs a backend serves: the ones it shipped with, plus packs built from models picked on the page.

Shipped packs load when the server starts and stay loaded. Built packs (``playground.builds``)
load the first time someone uses them, and only the ``max_built`` most recently used stay in
memory, so a backend that has built many models does not hold them all on its GPU.

A ``PackRegistry`` is a read-only mapping from pack id to ``PackModel``, which is what
``playground.http.create_app`` takes, so the routing does not change.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import torch

from interpretableintelligence.playground.compute import PackLoadError, PackModel

DEFAULT_MAX_BUILT = 3


class PackRegistry(Mapping):
    def __init__(
        self,
        shipped: Mapping[str, PackModel],
        built_dir: Path | None = None,
        device: torch.device | None = None,
        max_built: int = DEFAULT_MAX_BUILT,
        before_load: Callable[[Path], None] = lambda _dir: None,
        sync: Callable[[], None] = lambda: None,
    ) -> None:
        self.shipped = dict(shipped)
        self.built_dir = Path(built_dir) if built_dir is not None else None
        self.device = device
        self.max_built = max(1, int(max_built))
        self._before_load = before_load
        self._sync = sync
        self._loaded: OrderedDict[str, PackModel] = OrderedDict()
        self._lock = threading.Lock()

    def _built_ids(self) -> list[str]:
        if self.built_dir is None or not self.built_dir.is_dir():
            return []
        return sorted(p.parent.name for p in self.built_dir.glob("*/pack.json") if not p.parent.name.startswith("."))

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(set(self.shipped) | set(self._built_ids())))

    def __len__(self) -> int:
        return len(set(self.shipped) | set(self._built_ids()))

    def __contains__(self, pack_id: object) -> bool:
        return isinstance(pack_id, str) and (pack_id in self.shipped or self._built_path(pack_id) is not None)

    def _built_path(self, pack_id: str) -> Path | None:
        if self.built_dir is None or not pack_id or "/" in pack_id or pack_id.startswith("."):
            return None
        path = self.built_dir / pack_id
        if not (path / "pack.json").is_file():
            self._sync()
        return path if (path / "pack.json").is_file() else None

    def __getitem__(self, pack_id: str) -> PackModel:
        if pack_id in self.shipped:
            return self.shipped[pack_id]
        with self._lock:
            model = self._loaded.get(pack_id)
            if model is not None:
                self._loaded.move_to_end(pack_id)
                return model
            path = self._built_path(pack_id)
            if path is None:
                raise KeyError(pack_id)
            self._before_load(path)
            try:
                model = PackModel(path, device=self.device)
            except (OSError, ValueError, RuntimeError) as exc:
                raise PackLoadError(f"pack {pack_id} could not be loaded ({exc})") from exc
            self._loaded[pack_id] = model
            while len(self._loaded) > self.max_built:
                _old_id, old = self._loaded.popitem(last=False)
                del old
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return model

    def get(self, pack_id: str, default=None):  # type: ignore[override]
        try:
            return self[pack_id]
        except KeyError:
            return default

    def summaries(self) -> list[dict]:
        """``GET /v1/packs`` without loading every built model: shipped ones from memory, built ones from pack.json."""
        out = {pack_id: model.info() for pack_id, model in self.shipped.items()}
        for pack_id in self._built_ids():
            if pack_id in out:
                continue
            if pack_id in self._loaded:
                out[pack_id] = self._loaded[pack_id].info()
                continue
            try:
                out[pack_id] = describe_pack(self.built_dir / pack_id, self.device)
            except (OSError, ValueError, KeyError):
                continue
        return [out[k] for k in sorted(out)]


def describe_pack(pack_dir: Path, device: torch.device | None) -> dict:
    """``PackModel.info()`` from pack.json alone."""
    pack = json.loads((Path(pack_dir) / "pack.json").read_text(encoding="utf-8"))
    model = pack["model"]
    return {
        "id": Path(pack_dir).name,
        "title": pack.get("title"),
        "model": model["name"],
        "layer": int(model["layer"]),
        "image_size": int(model["image_size"]),
        "grid": {"h": int(model["grid"]["h"]), "w": int(model["grid"]["w"])},
        "n_features": int(pack["sae"]["n_features"]),
        "sae": {"activation": pack["sae"]["activation"], "k": pack["sae"].get("k")},
        "device": (device.type if device is not None else "cpu"),
        "loaded": False,
    }
