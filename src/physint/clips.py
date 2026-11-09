"""Load a directory of ``.npy`` clips or video files.

Video files are decoded the way Jeff's ``vjepa_interp.decode_video`` decodes
them for V-JEPA 2: every frame, short edge resampled to the processor size
(292, ``int(256 * 256 / 224)``) at decode time, display rotation applied, and
the container frame rate kept. The adapter then takes its window from that
stream (stride 2 when the clip is long enough). A 4K file is never expanded
to raw frames first.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from physint.adapters.base import Clip

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
# ``VJEPA2VideoProcessor`` shortest edge for the 256 crop, and Jeff's default.
VJEPA_SHORT_EDGE = int(256 * 256 / 224)


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
        files = _clip_files(path)
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
        if file_path.suffix.lower() in VIDEO_SUFFIXES:
            decoded, video_fps = decode_video(file_path, short_edge=VJEPA_SHORT_EDGE)
            frames = np.ascontiguousarray(decoded, dtype=np.float32)
            fps = float(video_fps)
        else:
            frames = np.ascontiguousarray(np.load(file_path), dtype=np.float32)
            fps = float(entry.get("fps", default_fps))
        clips.append(
            Clip(
                clip_id=clip_id,
                path=file_path,
                frames=frames,
                fps=fps,
                sha256=file_sha256(file_path),
            )
        )
    clips.sort(key=lambda clip: clip.clip_id)
    if not clips:
        raise ValueError(f"no clips in {path}")
    return clips


def _clip_files(path: Path) -> list[Path]:
    files = [
        item
        for item in path.iterdir()
        if item.is_file() and (item.suffix.lower() == ".npy" or item.suffix.lower() in VIDEO_SUFFIXES)
    ]
    return sorted(files, key=lambda item: item.name)


def decode_video(path: Path, short_edge: int = VJEPA_SHORT_EDGE) -> tuple[np.ndarray, float]:
    """Decode a video to uint8 ``[T, H, W, 3]`` at the model's short-edge size.

    Matches ``vjepa_interp.decode_video``: scale the short edge to
    ``short_edge`` while decoding (so a 4K phone clip is not held raw), apply
    display rotation, and return the container frame rate unchanged. ``.mp4``,
    ``.webm``, and ``.mov`` all go through this path.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("reading video clips requires the 'av' package") from exc
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate or 30
        fps = float(rate)
        for frame in container.decode(stream):
            scale = short_edge / min(frame.width, frame.height)
            image = frame.to_ndarray(
                format="rgb24",
                width=max(1, round(frame.width * scale)),
                height=max(1, round(frame.height * scale)),
            )
            rotation = getattr(frame, "rotation", 0) or 0
            if rotation:
                image = np.rot90(image, k=int(round(rotation / 90)) % 4).copy()
            frames.append(image)
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames), fps
