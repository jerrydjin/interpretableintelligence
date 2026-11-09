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


def load_clips(path: Path) -> list[Clip]:
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
    clips: list[Clip] = []
    seen: set[str] = set()
    for entry in entries:
        clip_id = str(entry["clip_id"])
        if clip_id in seen:
            raise ValueError(f"duplicate clip id {clip_id}")
        seen.add(clip_id)
        file_path = path / str(entry["file"])
        if not file_path.is_file():
            raise FileNotFoundError(f"missing clip file {file_path}")
        frames = np.load(file_path)
        frames = np.ascontiguousarray(frames, dtype=np.float32)
        clips.append(
            Clip(
                clip_id=clip_id,
                path=file_path,
                frames=frames,
                fps=float(entry.get("fps", default_fps)),
                sha256=file_sha256(file_path),
            )
        )
    clips.sort(key=lambda clip: clip.clip_id)
    if not clips:
        raise ValueError(f"no clips in {path}")
    return clips
