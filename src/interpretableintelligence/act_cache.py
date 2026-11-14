"""Content-addressed activation cache shared by every run on this machine.

A forward pass is the expensive part of a rerun. The cache keeps the tensors a
window produces, keyed by everything that decides them:

* the adapter name, the loaded weights' ``model_sha``, and whether they are the random-init control,
* the instrumented layer and the device type (cpu, cuda and mps kernels differ in the last bits),
* the clip's sha256 and the window (index, start frame, shape, and a sha256 of the
  frames the adapter feeds the model),
* the adapter's preprocessing (``adapter.cache_spec(clip)``: input size, normalisation,
  frame count, and anything else the adapter's tensors depend on),
* the cache format and the tool version.

The key is the sha256 of that payload as canonical JSON. One window is one
safetensors file at ``<root>/<kind>/<key[:2]>/<key>.safetensors``; the payload
is stored in the file's metadata and checked on read, so a collision or a
truncated file is a miss, never a wrong tensor. Writes go to a temporary file
and are renamed into place, so concurrent runs at worst compute the same window
twice.

Two kinds of entry:

``layer``
    The captured layer, ``[t, h, w, d]`` float32. ``ii run`` reads it for the
    capture stage and for the random-init control capture.
``clean``
    The adapter's unedited intermediate tensors for one window
    (``adapter.clean_state``), which ``adapter.load_clean_state`` puts back.
    The playground uses it so a cold ``ii view --steer`` or ``ii export``
    answers its first edit without the clean forward passes.

An adapter without ``cache_spec`` is never cached. GR00T has none yet: its
activations also depend on per-clip sidecars and the denoising noise, which the
key does not cover.

The cache lives outside run folders and is not part of the run contract
(``SCHEMA.md``). It holds no results, only model activations, and deleting it
only costs time. Location: ``$II_ACTS_CACHE`` when set (``off`` disables it),
otherwise ``$XDG_CACHE_HOME/interpretableintelligence/acts`` or
``~/.cache/interpretableintelligence/acts``. ``ii cache`` reports its size and
``ii cache --clear`` empties it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from interpretableintelligence import __version__
from interpretableintelligence.adapters.base import Clip, Window

CACHE_ENV = "II_ACTS_CACHE"
CACHE_FORMAT = 1
KINDS = ("layer", "clean")
_OFF = {"0", "off", "false", "no", "none"}
_SUFFIX = ".safetensors"


def default_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "interpretableintelligence" / "acts"


def resolve_root(root: Path | str | None = None) -> Path | None:
    """The cache directory to use, or ``None`` when caching is off.

    An explicit ``root`` wins, then ``$II_ACTS_CACHE``, then the default location.
    """
    if root is not None:
        return Path(root).expanduser()
    raw = os.environ.get(CACHE_ENV)
    if raw is None or not raw.strip():
        return default_root()
    if raw.strip().lower() in _OFF:
        return None
    return Path(raw.strip()).expanduser()


@dataclass(frozen=True)
class CacheKey:
    kind: str
    digest: str
    payload: str  # canonical JSON the digest was taken over


@dataclass
class ActivationCache:
    """Disk cache of per-window tensors. ``root=None`` is a cache that never hits or stores."""

    root: Path | None
    hits: int = 0
    misses: int = 0
    stored: int = 0
    bytes_read: int = 0
    bytes_written: int = 0

    @classmethod
    def open(cls, root: Path | str | None = None, *, enabled: bool = True) -> "ActivationCache":
        return cls(resolve_root(root) if enabled else None)

    @property
    def enabled(self) -> bool:
        return self.root is not None

    # ------------------------------------------------------------------ keys

    def key(self, kind: str, adapter, model_sha: str, clip: Clip, window: Window) -> CacheKey | None:
        """The key for one window, or ``None`` when this window cannot be cached."""
        if kind not in KINDS:
            raise ValueError(f"cache kind must be one of {KINDS}, got {kind!r}")
        if self.root is None or not model_sha:
            return None
        spec_fn = getattr(adapter, "cache_spec", None)
        if spec_fn is None or not clip.sha256:
            return None
        spec = spec_fn(clip)
        if spec is None:
            return None
        payload = {
            "format": CACHE_FORMAT,
            "kind": kind,
            "tool_version": __version__,
            "adapter": str(adapter.name),
            "model_sha": str(model_sha),
            "random_init": bool(getattr(adapter, "random_init", False)),
            "layer": int(adapter.layer),
            "device": _device_type(adapter),
            "clip_sha256": str(clip.sha256),
            "window": {
                "index": int(window.index),
                "start_frame": int(window.start_frame),
                "shape": [int(dim) for dim in np.shape(window.frames)],
                "dtype": str(np.asarray(window.frames).dtype),
                "frames_sha256": _frames_sha(window.frames),
            },
            "preprocessing": spec,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return CacheKey(kind=kind, digest=hashlib.sha256(text.encode("utf-8")).hexdigest(), payload=text)

    def path(self, key: CacheKey) -> Path:
        assert self.root is not None
        return self.root / key.kind / key.digest[:2] / f"{key.digest}{_SUFFIX}"

    # ------------------------------------------------------------ read/write

    def load(self, key: CacheKey | None) -> dict[str, torch.Tensor] | None:
        """The stored tensors (CPU), or ``None`` on a miss. A damaged entry is removed and counts as a miss."""
        if key is None or self.root is None:
            return None
        path = self.path(key)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if (handle.metadata() or {}).get("key") != key.payload:
                    raise ValueError("cache entry key does not match")
                tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        except Exception:  # noqa: BLE001 (any unreadable entry is a miss; the caller recomputes it)
            _unlink(path)
            self.misses += 1
            return None
        self.hits += 1
        self.bytes_read += path.stat().st_size
        return tensors

    def store(self, key: CacheKey | None, tensors: dict[str, torch.Tensor]) -> None:
        if key is None or self.root is None:
            return
        path = self.path(key)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        ready = {name: tensor.detach().to("cpu").contiguous().clone() for name, tensor in tensors.items()}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(ready, str(tmp), metadata={"key": key.payload})
            os.replace(tmp, path)
        except OSError:
            # A full or read-only disk must not fail the run; the next run recomputes.
            _unlink(tmp)
            return
        self.stored += 1
        self.bytes_written += path.stat().st_size

    def summary(self) -> str:
        if self.root is None:
            return "activation cache off"
        return (
            f"activation cache {self.root}: {self.hits} hit(s), {self.misses} miss(es), "
            f"{self.stored} stored, {self.bytes_read / 1e6:.1f} MB read, {self.bytes_written / 1e6:.1f} MB written"
        )


# ---------------------------------------------------------------- helpers


def captured_layer(adapter, clip: Clip, window: Window, *, cache: ActivationCache, model_sha: str) -> torch.Tensor:
    """The layer ``adapter`` captures on this window, ``[t, h, w, d]`` float32 on CPU.

    Read from the cache when the key matches; otherwise one capture forward,
    whose result is stored.
    """
    key = cache.key("layer", adapter, model_sha, clip, window)
    hit = cache.load(key)
    if hit is not None and "acts" in hit:
        return hit["acts"]
    with adapter.hook(adapter.layer, mode="capture") as handle:
        adapter.encode(clip, window)
    if len(handle.captured) != 1:
        raise RuntimeError(f"expected one captured tensor for {clip.clip_id}")
    acts = handle.captured[0].detach().to(dtype=torch.float32, device="cpu")
    cache.store(key, {"acts": acts})
    return acts


def prime_clean(adapter, clip: Clip, window: Window, *, cache: ActivationCache, model_sha: str) -> bool | None:
    """Load the adapter's unedited tensors for this window from the cache, or compute and store them.

    Returns ``True`` on a hit, ``False`` on a miss that was stored, and ``None``
    when the adapter or window cannot be cached (nothing is computed then).
    """
    if not (hasattr(adapter, "clean_state") and hasattr(adapter, "load_clean_state")):
        return None
    key = cache.key("clean", adapter, model_sha, clip, window)
    if key is None:
        return None
    state = cache.load(key)
    if state is not None:
        try:
            adapter.load_clean_state(clip, window, state)
            return True
        except (KeyError, ValueError, RuntimeError):
            _unlink(cache.path(key))  # written by an older adapter layout; recompute below
    cache.store(key, adapter.clean_state(clip, window))
    return False


def _device_type(adapter) -> str:
    device = getattr(adapter, "device", None)
    return "cpu" if device is None else torch.device(device).type


def _frames_sha(frames) -> str:
    array = np.ascontiguousarray(frames)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def cache_usage(root: Path | None) -> dict:
    """Entries and bytes per kind under ``root``."""
    usage = {"root": None if root is None else str(root), "kinds": {}, "entries": 0, "bytes": 0}
    if root is None or not root.is_dir():
        return usage
    for kind in KINDS:
        entries = 0
        size = 0
        for path in (root / kind).glob(f"*/*{_SUFFIX}"):
            try:
                size += path.stat().st_size
            except OSError:
                continue
            entries += 1
        usage["kinds"][kind] = {"entries": entries, "bytes": size}
        usage["entries"] += entries
        usage["bytes"] += size
    return usage


def clear_cache(root: Path | None) -> int:
    """Delete every entry under ``root``. Returns the number of entries removed."""
    usage = cache_usage(root)
    if root is None:
        return 0
    for kind in KINDS:
        shutil.rmtree(root / kind, ignore_errors=True)
    return int(usage["entries"])


def _unlink(path) -> None:
    if path is None:
        return
    try:
        Path(path).unlink()
    except OSError:
        pass
