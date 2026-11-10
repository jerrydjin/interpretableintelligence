"""Load a directory of ``.npy`` clips."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from physint.adapters.base import Clip


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_clip_meta(path: Path) -> list[dict]:
    """Clip id, path, fps, sha256, and shape. Does not read pixel values into memory."""
    if not path.is_dir():
        raise FileNotFoundError(f"clips directory not found: {path}")
    index_path = path / "clips.json"
    default_fps = 8.0
    if index_path.exists():
        meta = json.loads(index_path.read_text(encoding="utf-8"))
        default_fps = float(meta.get("fps", default_fps))
        entries = meta.get("clips")
    else:
        entries = None
    if not entries:
        files = sorted(path.glob("*.npy"))
        entries = [{"clip_id": file.stem, "file": file.name, "fps": default_fps} for file in files]
    found: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        clip_id = str(entry["clip_id"])
        if clip_id in seen:
            raise ValueError(f"duplicate clip id {clip_id}")
        seen.add(clip_id)
        file_path = path / str(entry["file"])
        if not file_path.is_file():
            raise FileNotFoundError(f"missing clip file {file_path}")
        if file_path.suffix.lower() != ".npy":
            raise ValueError(f"this lane reads .npy clips, got {file_path.name}")
        shape = _npy_shape(file_path)
        if len(shape) < 1 or int(shape[0]) <= 0:
            raise ValueError(f"{file_path.name} has no frames")
        fps = float(entry.get("fps", default_fps))
        if not fps > 0:
            raise ValueError(f"{clip_id} fps must be positive")
        found.append(
            {
                "clip_id": clip_id,
                "path": file_path,
                "fps": fps,
                "sha256": file_sha256(file_path),
                "n_frames": int(shape[0]),
                "shape": tuple(int(dim) for dim in shape),
            }
        )
    found.sort(key=lambda item: item["clip_id"])
    if not found:
        raise ValueError(f"no clips in {path}")
    return found


def load_clip(meta: dict) -> Clip:
    """Load one clip's frames. The caller drops the clip when the window is written."""
    frames = np.ascontiguousarray(np.load(meta["path"]), dtype=np.float32)
    return Clip(
        clip_id=str(meta["clip_id"]),
        path=meta["path"],
        frames=frames,
        fps=float(meta["fps"]),
        sha256=str(meta["sha256"]),
    )


def load_clips(path: Path) -> list[Clip]:
    return [load_clip(meta) for meta in list_clip_meta(path)]


def _npy_shape(path: Path) -> tuple[int, ...]:
    array = np.load(path, mmap_mode="r")
    try:
        return tuple(int(dim) for dim in array.shape)
    finally:
        mapped = getattr(array, "_mmap", None)
        del array
        if mapped is not None:
            mapped.close()
