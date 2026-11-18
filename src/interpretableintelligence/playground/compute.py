"""The playground's model side: one live pack loaded in PyTorch, answering read and steer.

This module touches model tensors, so it is Python (AGENTS.md, "Language"). It knows nothing
about HTTP: ``playground.http`` maps the versioned API (docs/PLAYGROUND-API.md) onto
``PackModel.read`` and ``PackModel.steer``, so a gateway in another language can sit in front of
the same contract later without changing this file.

The maths is the in-browser engine's (demo-site/live/engine.js) and ``steer.steer_edit``:

- read: pixels -> hidden state at the SAE's layer -> the SAE's features on every patch token.
- steer: x' = x + sum_i (f'_i - f_i) W_dec[:, i] on the patch tokens, the rest of the network,
  then the per-patch cosine distance between the clean and edited final tokens.

The SAE comes from the pack's ``sae.safetensors``, the same fp16 numbers as its ONNX files, so
this backend and the browser engine read one dictionary. Exploration only: nothing here can
change a feature's badge.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.steer import KINDS, MAX_ABS_VALUE, steer_edit

PACK_FORMAT = "ii-live-pack"
PACK_VERSION = 1
MAX_FEATURES_PER_EDIT = 16
DEFAULT_CACHE_IMAGES = 64


class PlaygroundError(Exception):
    """An error the API reports to the client. ``code`` is part of the contract."""

    status = 400
    code = "bad_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadRequest(PlaygroundError):
    pass


class PackNotFound(PlaygroundError):
    status = 404
    code = "pack_not_found"


class UnknownImage(PlaygroundError):
    """The image is not in this server's cache (evicted, or another replica read it): read it again."""

    status = 404
    code = "unknown_image"


class PackLoadError(RuntimeError):
    pass


@dataclass
class _Read:
    hidden: torch.Tensor  # [1, prefix + n_tokens, d]
    base: torch.Tensor | None = None  # clean final patch tokens [n_tokens, d]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class PackModel:
    """A pack's model and SAE, loaded once. Thread-safe: calls run one at a time."""

    def __init__(self, pack_dir: Path, device: torch.device | None = None, cache_images: int = DEFAULT_CACHE_IMAGES) -> None:
        from interpretableintelligence.adapters import get_adapter
        from interpretableintelligence.device import pick_device
        from interpretableintelligence.live import SAE_WEIGHTS, split_model
        from interpretableintelligence.sae import SAE

        from safetensors.torch import load_file

        self.dir = Path(pack_dir)
        try:
            pack = json.loads((self.dir / "pack.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PackLoadError(f"{self.dir} has no readable pack.json ({exc})") from exc
        if pack.get("format") != PACK_FORMAT or pack.get("version") != PACK_VERSION:
            raise PackLoadError(f"{self.dir}/pack.json is not an {PACK_FORMAT} v{PACK_VERSION}")
        weights = self.dir / (pack.get("files") or {}).get("sae_weights", SAE_WEIGHTS)
        if not weights.is_file():
            raise PackLoadError(f"{self.dir} has no {weights.name}; re-export it with ii export-live")
        self.pack = pack
        self.id = self.dir.name
        model = pack["model"]
        sae_info = pack["sae"]
        self.device = device or pick_device()
        started = time.perf_counter()

        adapter = get_adapter(model["adapter"])(layer=int(model["layer"]), device=self.device)
        adapter.load(random_init=False)
        sha = adapter.model_sha()
        if model.get("sha") and sha != model["sha"]:
            raise PackLoadError(f"{model['name']} loaded with weights sha {sha}, but the pack was made from {model['sha']}")
        self.front, self.back = split_model(adapter)
        self.size = int(model["image_size"])
        self.prefix = int(model["prefix_tokens"])
        self.grid_h, self.grid_w = int(model["grid"]["h"]), int(model["grid"]["w"])
        self.n_tokens = self.grid_h * self.grid_w
        self.d_model = int(model["d_model"])

        tensors = {name: tensor.float() for name, tensor in load_file(str(weights), device="cpu").items()}
        sae = SAE(self.d_model, int(sae_info["n_features"]), sae_info["activation"], sae_info.get("k"), sae_info.get("threshold"))
        sae.load_state_dict(tensors)
        self.sae = sae.to(self.device).eval().requires_grad_(False)
        self.n_features = int(sae_info["n_features"])

        self.mean = torch.tensor(model["mean"], dtype=torch.float32, device=self.device).view(3, 1, 1)
        self.std = torch.tensor(model["std"], dtype=torch.float32, device=self.device).view(3, 1, 1)
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, _Read] = OrderedDict()
        self._cache_images = max(1, int(cache_images))
        self.load_ms = _ms(started)

    # The model moves between devices after loading, so a CPU memory snapshot can be restored
    # onto a GPU (remote/playground_app.py).
    def to(self, device: torch.device) -> "PackModel":
        with self._lock:
            self.device = torch.device(device)
            self.front.to(self.device)
            self.back.to(self.device)
            self.sae.to(self.device)
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)
            self._cache.clear()
        return self

    def info(self) -> dict:
        model = self.pack["model"]
        return {
            "id": self.id,
            "title": self.pack.get("title"),
            "model": model["name"],
            "layer": int(model["layer"]),
            "image_size": self.size,
            "grid": {"h": self.grid_h, "w": self.grid_w},
            "n_features": self.n_features,
            "sae": {"activation": self.pack["sae"]["activation"], "k": self.pack["sae"].get("k")},
            "device": self.device.type,
        }

    def image_id(self, rgb: bytes) -> str:
        return hashlib.sha256(self.id.encode() + b"\0" + rgb).hexdigest()[:32]

    def read(self, rgb: bytes) -> dict:
        """``rgb`` is the picture already resized to image_size x image_size, 8-bit RGB, row-major."""
        want = self.size * self.size * 3
        if len(rgb) != want:
            raise BadRequest(f"expected {want} bytes ({self.size}x{self.size} RGB, 8 bits each), got {len(rgb)}")
        started = time.perf_counter()
        image_id = self.image_id(rgb)
        pixels = torch.frombuffer(bytearray(rgb), dtype=torch.uint8).view(self.size, self.size, 3)
        with self._lock, torch.inference_mode():
            x = pixels.to(self.device).permute(2, 0, 1).float().div_(255).sub_(self.mean).div_(self.std)
            hidden = self.front(x.unsqueeze(0))
            acts = self.sae.encode(hidden[0, self.prefix :])
            self._remember(image_id, _Read(hidden=hidden))
            acts = acts.float().cpu().numpy()
        model_ms = _ms(started)
        offsets, ids, values = _sparse_rows(acts)
        return {
            "image_id": image_id,
            "grid": {"h": self.grid_h, "w": self.grid_w},
            "n_features": self.n_features,
            "patches": {"offsets": offsets, "ids": ids, "values": values},
            "timing_ms": {"model": model_ms, "total": _ms(started)},
        }

    def steer(self, image_id: str, features: list[int], kind: str, value: float) -> dict:
        features, kind, value = self._check_edit(features, kind, value)
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            entry = self._cache.get(image_id)
            if entry is None:
                raise UnknownImage(f"image {image_id} is not loaded here; read it again")
            self._cache.move_to_end(image_id)
            if entry.base is None:
                entry.base = self.back(entry.hidden)[0, self.prefix :]
            edit = steer_edit(self.sae, features, kind, value)
            edited = entry.hidden.clone()
            edited[:, self.prefix :] = edit(entry.hidden[:, self.prefix :])
            final = self.back(edited)[0, self.prefix :]
            base = entry.base
            shift = 1 - torch.nn.functional.cosine_similarity(base, final, dim=-1, eps=1e-12)
            loss_change = float((base - final).pow(2).mean())
            shift = shift.float().cpu().numpy()
        return {
            "image_id": image_id,
            "shift": [round(float(v), 7) for v in shift],
            "mean": float(shift.mean()),
            "max": float(shift.max()),
            "loss_change": loss_change,
            "timing_ms": {"total": _ms(started)},
        }

    def _check_edit(self, features: object, kind: object, value: object) -> tuple[list[int], str, float]:
        if kind not in KINDS:
            raise BadRequest(f"kind must be one of {', '.join(KINDS)}")
        if not isinstance(features, list) or not features or len(features) > MAX_FEATURES_PER_EDIT:
            raise BadRequest(f"features must be a list of 1 to {MAX_FEATURES_PER_EDIT} feature ids")
        if any(isinstance(f, bool) or not isinstance(f, int) or not 0 <= f < self.n_features for f in features):
            raise BadRequest(f"feature ids must be integers from 0 to {self.n_features - 1}")
        if len(set(features)) != len(features):
            raise BadRequest("feature ids must not repeat")
        if kind == "ablate":
            value = 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or abs(value) > MAX_ABS_VALUE:
            raise BadRequest(f"value must be a finite number within ±{MAX_ABS_VALUE:g}")
        return list(features), str(kind), float(value)

    def _remember(self, image_id: str, entry: _Read) -> None:
        self._cache[image_id] = entry
        self._cache.move_to_end(image_id)
        while len(self._cache) > self._cache_images:
            self._cache.popitem(last=False)


def _sparse_rows(acts: np.ndarray) -> tuple[list[int], list[int], list[float]]:
    """Every patch's firing features, strongest first, as compressed rows (CSR)."""
    offsets = [0]
    ids: list[int] = []
    values: list[float] = []
    for row in acts:
        nz = np.flatnonzero(row > 0)
        order = nz[np.argsort(-row[nz], kind="stable")]
        ids.extend(int(i) for i in order)
        values.extend(round(float(v), 6) for v in row[order])
        offsets.append(len(ids))
    return offsets, ids, values


def load_packs(packs_dir: Path, device: torch.device | None = None, only: list[str] | None = None) -> dict[str, PackModel]:
    """Every pack under ``packs_dir`` (directories with a pack.json), keyed by directory name."""
    packs_dir = Path(packs_dir)
    found = sorted(p.parent for p in packs_dir.glob("*/pack.json"))
    if only:
        found = [p for p in found if p.name in only]
        missing = set(only) - {p.name for p in found}
        if missing:
            raise PackLoadError(f"no pack named {', '.join(sorted(missing))} under {packs_dir}")
    if not found:
        raise PackLoadError(f"no packs (*/pack.json) under {packs_dir}")
    return {p.name: PackModel(p, device=device) for p in found}
