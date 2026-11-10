"""Local clip-first viewer over a finished run folder.

Reads ``manifest.json``. ``features/feature_acts.parquet`` is queried per clip and
per feature through DuckDB (the file is not loaded into memory).
``features/clip_effects.parquet``, ``features/features.jsonl``,
``features/candidates.json``, and ``scoreboard.json`` are used when present
and skipped when absent. A scoreboard feature's string ``status`` is shown
on the badge unless ``real`` is true. ``heldout_loss.json``, when present,
is the hardest-moments list. Clip pixels
come from a path recorded on the manifest, or from a directory passed as
``--clips``. A clip's ``name`` (the exact filename, such as
``droid_ep000.mp4``) is tried first, then the filename stem, then sha256.

Frame mapping
-------------
A feature token at ``(window, t)`` lands on one source frame:

    frame = window_start + t * tubelet * stride

``tubelet`` is ``manifest.grid.tubelet``. ``t`` counts tubelets inside the
window. ``stride`` is the temporal stride of the capture: how many source
frames separate consecutive model frames. The frames that share one tubelet
step (``tubelet * stride`` source frames) show the same features.

``stride`` and ``window_start`` are chosen in this order:

1. An explicit per-clip ``window_starts`` (or ``windows[].start_frame`` /
   ``start``, or ``window_start``), and an explicit per-clip ``stride``.
   The same start keys on ``manifest.grid`` count as explicit. With no
   recorded stride this is ``window_start + t * tubelet``.
2. Otherwise, when ``manifest.extra`` records ``frames``, ``stride``, or
   ``windows``, starts and stride come from Jeff's placement: ``span =
   frames * stride`` source frames, ``k`` windows spread across the clip, a
   single window centered. A clip shorter than ``span`` is resampled at
   stride 1. Per-clip ``n_frames`` and ``n_windows`` are the inputs; ``frames``
   and the requested stride come from ``manifest.extra``.
3. Otherwise windows tile the clip from frame 0 and ``stride`` is 1. The
   distance between those tiled windows is ``grid.window_stride`` when that
   is set, and ``grid.t * tubelet`` otherwise.

The heatmap is the model's patch grid. V-JEPA centre-crops the displayed
frame to a square before it sees it, so the grid is drawn on that square
(side ``min(H, W)``, inset by ``(W - side) / 2`` or ``(H - side) / 2``)
and the leftover side bands are dimmed. ``H, W`` are the clip's ``hw``.
``manifest.extra.crop`` replaces that guess.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import subprocess
import tempfile
import threading
import webbrowser
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from physint.artifacts import read_json

STATIC_DIR = Path(__file__).resolve().parent / "static"

_REQUIRED = (
    "manifest.json",
    "features/feature_acts.parquet",
)
# Stem matches prefer a browser-native video over a transcode, and a transcode over npy.
_MEDIA_SUFFIXES = (".mp4", ".webm", ".m4v", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".npy")
_NATIVE_VIDEO = {".mp4": "video/mp4", ".webm": "video/webm", ".m4v": "video/mp4"}
_TRANSCODE_VIDEO = {".mov", ".avi", ".mkv", ".mpeg", ".mpg"}
_SCORE_FIELDS = (
    "distinct_clips_top20",
    "label_holds_heldout",
    "ablation_effect",
    "random_percentile",
    "n_random",
    "absent_in_random_init",
    "real",
)
_KIND_ORDER = {"ablation": 0, "estimate": 1}
_TOP_FEATURES = 40
_TOP_CLIPS = 20
_SHA_EAGER_LIMIT = 8 * 1024 * 1024
_CLIP_ID = re.compile(r"^/api/clips/(?P<clip>[^/]+)$")
_FRAME_PNG = re.compile(r"^/api/clips/(?P<clip>[^/]+)/frames/(?P<frame>[0-9]+)\.png$")
_MEDIA = re.compile(r"^/api/clips/(?P<clip>[^/]+)/media$")
_HEAT = re.compile(r"^/api/clips/(?P<clip>[^/]+)/heatmap$")
_FEATURE = re.compile(r"^/api/features/(?P<feature>[0-9]+)$")
_FEATURE_VIZ = re.compile(r"^/api/features/(?P<feature>[0-9]+)/viz$")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".json": "application/json",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".m4v": "video/mp4",
}
# Distinct colors for a channel-wise composite. Not a cue legend: the viewer
# does not know adapter channel names.
_PALETTE = np.array(
    [
        [230, 92, 48],
        [42, 128, 168],
        [46, 150, 96],
        [214, 168, 48],
        [132, 96, 186],
        [186, 72, 112],
        [96, 112, 124],
        [64, 140, 136],
    ],
    dtype=np.float32,
)


def frame_index(frame: int, grid: dict, starts: list[int] | None = None, sample_stride: int = 1) -> tuple[int, int]:
    """Map a source frame to ``(window, t)`` using :func:`window_starts`."""
    if starts is None:
        span = window_span(grid) * max(int(sample_stride), 1)
        starts = [index * window_stride(grid, span) for index in range((int(frame) // span) + 1)]
    window, tubelet_index = frame_to_token(int(frame), starts, grid, sample_stride)
    return window, 0 if tubelet_index is None else tubelet_index


def window_span(grid: dict) -> int:
    """Frames in one window: ``grid.t * grid.tubelet`` (16 for V-JEPA)."""
    return max(int(grid.get("t") or 1), 1) * tubelet_of(grid)


def tubelet_of(grid: dict) -> int:
    return max(int(grid.get("tubelet") or 1), 1)


def window_stride(grid: dict, span: int | None = None) -> int:
    if _is_int(grid.get("window_stride")):
        return max(int(grid["window_stride"]), 1)
    return window_span(grid) if span is None else span


def sampled_window_starts(n_frames: int, n_windows: int, frames: int = 16, stride: int = 2) -> tuple[list[int], int]:
    """Jeff's window placement. Returns ``(starts, stride)``.

    The body is his function: ``span = frames * stride`` source frames, fewer
    windows than requested when the clip cannot hold them, one window centered,
    and stride 1 when the clip is shorter than ``span``.
    """
    span = frames * stride
    if n_frames < span and stride > 1:
        stride, span = 1, frames
    k = max(1, min(n_windows, n_frames // span))
    if k == 1:
        return [max(0, (n_frames - span) // 2)], stride
    gap = (n_frames - k * span) / (k - 1)
    return [int(round(i * (span + gap))) for i in range(k)], stride


def window_starts(clip: dict, grid: dict, n_windows: int, extra: dict | None = None) -> list[int]:
    """Frame index where each window begins. See the module docstring."""
    return clip_sampling(clip, grid, n_windows, extra)[0]


def clip_sampling(clip: dict, grid: dict, n_windows: int, extra: dict | None = None) -> tuple[list[int], int]:
    """``(window_starts, sample_stride)`` for one clip.

    Precedence is explicit per-clip starts and stride, then
    :func:`sampled_window_starts` from ``manifest.extra``, then tiling from
    frame 0 at stride 1. ``n_windows`` is the highest window index the acts
    need a start for.
    """
    clip = clip if isinstance(clip, dict) else {}
    explicit = _explicit_window_starts(clip, grid)
    explicit_stride = int(clip["stride"]) if _is_int(clip.get("stride")) else None
    if explicit is not None:
        stride = explicit_stride if explicit_stride is not None else 1
        return _extend_starts(explicit, n_windows, window_stride(grid)), max(stride, 1)

    spec = _sampling_spec(extra)
    n_frames = int(clip["n_frames"]) if _is_int(clip.get("n_frames")) else None
    if spec is not None and n_frames is not None:
        frames, requested, extra_windows = spec
        if explicit_stride is not None:
            requested = explicit_stride
        if _is_int(clip.get("n_windows")):
            count = int(clip["n_windows"])
        elif extra_windows is not None:
            count = extra_windows
        else:
            count = n_windows + 1
        starts, stride = sampled_window_starts(n_frames, max(count, 1), max(frames, 1), max(requested, 1))
        return _extend_starts(starts, n_windows, max(frames * max(stride, 1), 1)), max(stride, 1)

    stride = explicit_stride if explicit_stride is not None else 1
    return _extend_starts([], n_windows, window_stride(grid)), max(stride, 1)


def _sampling_spec(extra: dict | None) -> tuple[int, int, int | None] | None:
    if not isinstance(extra, dict):
        return None
    if not any(key in extra for key in ("frames", "stride", "windows")):
        return None
    frames = int(extra["frames"]) if _is_int(extra.get("frames")) else 16
    stride = int(extra["stride"]) if _is_int(extra.get("stride")) else 2
    windows = int(extra["windows"]) if _is_int(extra.get("windows")) else None
    return frames, stride, windows


def _extend_starts(starts: list[int], n_windows: int, step: int) -> list[int]:
    extended = [int(item) for item in starts]
    step = max(int(step), 1)
    while len(extended) <= n_windows:
        extended.append(extended[-1] + step if extended else 0)
    return extended


def _starts_covering(clip: dict, grid: dict, extra: dict | None, frame: int) -> tuple[list[int], int]:
    """Window starts that reach ``frame``.

    Recorded windows (Jeff's placement, or an explicit list) are kept as they
    are. A clip that tiles from frame 0 is extended one window at a time until
    ``frame`` falls inside a window, which is what a single recorded start at 0
    needs.
    """
    hint = int(clip["n_windows"]) - 1 if _is_int(clip.get("n_windows")) else 0
    starts, stride = clip_sampling(clip, grid, max(hint, 0), extra)
    if _sampling_spec(extra) is not None or _explicit_window_starts(clip, grid) is not None:
        return starts, stride
    step = tubelet_of(grid) * max(int(stride), 1)
    span = max(int(grid.get("t") or 1), 1) * step
    guard = 0
    while starts and starts[-1] + span <= frame and guard < frame + 1:
        starts = [*starts, starts[-1] + span]
        guard += 1
    return starts, stride


def frame_to_token(frame: int, starts: list[int], grid: dict, sample_stride: int = 1) -> tuple[int, int | None]:
    """Return ``(window, t)``. ``t`` is ``None`` when ``frame`` sits in no window.

    Inside a window, ``t = (frame - window_start) // (tubelet * stride)``, so
    every source frame in that token's step shares the token. A frame in the
    gap takes the nearer edge token: the previous window's last ``t``, or the
    next window's ``t = 0``. The displayed frame does not move. A window that
    starts at 55 still reports frame 65 as ``t = (65 - 55) // step`` (anchor
    63 when the step is 4), not as frame 55.
    """
    if not starts:
        starts = [0]
    step = tubelet_of(grid) * max(int(sample_stride), 1)
    n_tokens = max(int(grid.get("t") or 1), 1)
    span = n_tokens * step
    if frame < starts[0]:
        return 0, None
    window = 0
    for index, start in enumerate(starts):
        if start <= frame:
            window = index
        else:
            break
    offset = frame - starts[window]
    if offset < span:
        return window, offset // step
    next_start = starts[window + 1] if window + 1 < len(starts) else None
    if next_start is None:
        return window, None
    if offset - span <= next_start - frame:
        return window, n_tokens - 1
    return window + 1, 0


def token_to_frame(window: int, t: int, starts: list[int], grid: dict, sample_stride: int = 1) -> int:
    """Inverse of :func:`frame_to_token`: ``window_start + t * tubelet * stride``."""
    step = tubelet_of(grid) * max(int(sample_stride), 1)
    if 0 <= window < len(starts):
        return starts[window] + int(t) * step
    base = starts[-1] if starts else 0
    beyond = window - (len(starts) - 1) if starts else window
    return base + beyond * window_stride(grid) * max(int(sample_stride), 1) + int(t) * step


def _explicit_window_starts(clip: dict, grid: dict) -> list[int] | None:
    for source in (clip, grid):
        if not isinstance(source, dict):
            continue
        raw = source.get("window_starts")
        if isinstance(raw, list) and raw and all(_is_int(item) for item in raw):
            return [int(item) for item in raw]
    if isinstance(clip, dict):
        windows = clip.get("windows")
        parsed = _parse_windows(windows, grid)
        if parsed:
            return parsed
        if _is_int(clip.get("window_start")):
            return [int(clip["window_start"])]
    if isinstance(grid, dict) and _is_int(grid.get("window_start")):
        return [int(grid["window_start"])]
    return None


def _parse_windows(windows: object, grid: dict) -> list[int] | None:
    if not isinstance(windows, list) or not windows:
        return None
    pairs: list[tuple[int, int]] = []
    for index, item in enumerate(windows):
        if isinstance(item, dict):
            start = item.get("start_frame", item.get("start", item.get("frame")))
            window = item.get("index", item.get("window", index))
            if _is_int(start) and _is_int(window):
                pairs.append((int(window), int(start)))
        elif _is_int(item):
            pairs.append((index, int(item)))
    if not pairs:
        return None
    pairs.sort()
    starts: list[int] = []
    stride = window_stride(grid)
    for window, start in pairs:
        while len(starts) < window:
            starts.append(starts[-1] + stride if starts else 0)
        if len(starts) == window:
            starts.append(start)
        else:
            starts[window] = start
    return starts


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def overlay_rect(hw: list[float] | None, extra: dict | None, adapter: object) -> dict[str, float] | None:
    """Model-input rectangle on the displayed frame, in pixels.

    ``hw`` is ``[height, width]`` of the frame the viewer shows. ``manifest.extra.crop``
    wins when it is present: a dict with ``x``/``left``, ``y``/``top``, and
    ``width``/``w`` plus ``height``/``h`` (or a single ``size``), a list
    ``[x, y, width, height]``, or ``"center"``. Otherwise a ``vjepa2`` adapter
    uses the centre square, ``side = min(H, W)``, offset ``(W - side) / 2``
    horizontally or ``(H - side) / 2`` vertically. Any other adapter leaves
    the overlay on the full frame.
    """
    explicit = extra.get("crop") if isinstance(extra, dict) and "crop" in extra else None
    if explicit is not None:
        parsed = _parse_crop(explicit, hw)
        if parsed is not None:
            return parsed
    if adapter == "vjepa2" and hw is not None:
        return _center_square(hw[0], hw[1])
    return None


def _center_square(height: float, width: float) -> dict[str, float]:
    side = min(height, width)
    return {"x": (width - side) / 2, "y": (height - side) / 2, "width": side, "height": side}


def _parse_crop(raw: object, hw: list[float] | None) -> dict[str, float] | None:
    if isinstance(raw, str) and raw.strip().lower() in {"center", "centre", "center_square", "centre_square"}:
        return _center_square(hw[0], hw[1]) if hw is not None else None
    if isinstance(raw, dict):
        mode = raw.get("mode") if isinstance(raw.get("mode"), str) else raw.get("type")
        if isinstance(mode, str) and mode.strip().lower() in {"center", "centre", "center_square", "centre_square"}:
            return _center_square(hw[0], hw[1]) if hw is not None else None
        x = _plain_num(raw.get("x", raw.get("left")))
        y = _plain_num(raw.get("y", raw.get("top")))
        width = _plain_num(raw.get("width", raw.get("w", raw.get("size"))))
        height = _plain_num(raw.get("height", raw.get("h", raw.get("size"))))
        if x is None or y is None or width is None or height is None or width <= 0 or height <= 0:
            return None
        return {"x": x, "y": y, "width": width, "height": height}
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        x, y, width, height = (_plain_num(item) for item in raw)
        if x is None or y is None or width is None or height is None or width <= 0 or height <= 0:
            return None
        return {"x": x, "y": y, "width": width, "height": height}
    return None


def _plain_num(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    return _num(value)


def _displayed_hw(clip: dict, loaded: LoadedClip) -> list[float] | None:
    """Displayed frame size ``[height, width]``. Manifest ``hw`` wins over the file."""
    raw = clip.get("hw")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        height, width = _plain_num(raw[0]), _plain_num(raw[1])
        if height is not None and width is not None and height > 0 and width > 0:
            return [height, width]
    if loaded.height and loaded.width and loaded.height > 0 and loaded.width > 0:
        return [float(loaded.height), float(loaded.width)]
    return None


def badge_for(real: object, status: object = None) -> str:
    """Only a JSON true is real. A status string is its own badge; otherwise unverified."""
    if real is True:
        return "real"
    return "status" if status_text(status) else "unverified"


def status_text(status: object) -> str | None:
    """Badge copy for a non-real feature. The word ``real`` is never a status."""
    if not isinstance(status, str):
        return None
    text = status.strip()
    if not text or text.casefold() == "real":
        return None
    return text


def ablation_passed(status: object) -> bool:
    return isinstance(status, str) and "ablation-passed" in status.casefold()


def serve_run(
    run_dir: Path,
    clips_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = True,
) -> None:
    """Load ``run_dir``, serve the viewer, and optionally open a browser."""
    model = load_run(Path(run_dir), None if clips_dir is None else Path(clips_dir))
    httpd = bind_server(model, host, port)
    bound_host, bound_port = httpd.server_address[:2]
    url = f"http://{bound_host}:{bound_port}/"
    print(f"physint view  {url}", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except OSError:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def bind_server(model: RunModel, host: str, port: int) -> ThreadingHTTPServer:
    httpd = _Server((host, port), _handler(model))
    return httpd


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@dataclass
class RunModel:
    run_payload: dict
    clips: dict[str, dict]
    loaded: dict[str, LoadedClip] = field(default_factory=dict)
    frames: dict[str, np.ndarray] = field(default_factory=dict)
    peaks: dict[str, float] = field(default_factory=dict)
    meta: dict[str, dict] = field(default_factory=dict)
    manifest_clips: dict[str, dict] = field(default_factory=dict)
    grid: dict = field(default_factory=dict)
    extra: dict | None = None
    features_by_id: dict[int, dict] = field(default_factory=dict)
    viz: dict[int, Path] = field(default_factory=dict)
    run_dir: Path = field(default_factory=Path)
    adapter: str | None = None
    acts: "_Acts | None" = None
    _png: dict[tuple[str, int], bytes] = field(default_factory=dict)
    _media: dict[str, tuple[Path, str]] = field(default_factory=dict)
    _viz_stat: dict[int, Path | None] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def frame_png(self, clip_id: str, frame: int) -> bytes | None:
        clip = self.meta.get(clip_id)
        if clip is None or clip["pixels"] != "ok":
            return None
        if frame < 0 or frame >= int(clip["n_frames"]):
            return None
        key = (clip_id, frame)
        with self._lock:
            cached = self._png.get(key)
            if cached is None:
                cached = self._render_png(clip_id, frame)
                if cached is not None:
                    self._png[key] = cached
            return cached

    def media_file(self, clip_id: str) -> tuple[Path, str] | None:
        """Playable video path and content type. Transcodes avi/mov/mkv once."""
        clip = self.meta.get(clip_id)
        loaded = self.loaded.get(clip_id)
        if clip is None or loaded is None or clip.get("media") != "video" or loaded.path is None:
            return None
        with self._lock:
            cached = self._media.get(clip_id)
            if cached is None:
                cached = _playable_video(loaded.path)
                if cached is not None:
                    self._media[clip_id] = cached
            return cached

    def _render_png(self, clip_id: str, frame: int) -> bytes | None:
        array = self.frames.get(clip_id)
        if array is not None:
            if frame >= int(array.shape[0]):
                return None
            image = render_frame(array[frame], self.peaks.get(clip_id, 1.0))
            scale = _display_scale(image.shape[0], image.shape[1])
            return encode_png(_upsample(image, scale))
        loaded = self.loaded.get(clip_id)
        clip = self.meta.get(clip_id) or {}
        if loaded is None or loaded.path is None:
            return None
        fps = float(clip.get("fps") or 0.0) or 8.0
        return _video_frame_png(loaded.path, frame, fps)

    def feature(self, feature_id: int) -> dict | None:
        """One feature, with its top clips queried on demand."""
        base = self.features_by_id.get(feature_id)
        tops = self._top_for(feature_id) if self.acts is not None else []
        if base is None and not tops:
            return None
        detail = dict(base or _blank_feature(feature_id))
        detail["top_clips"] = tops
        detail["viz"] = self.viz_file(feature_id) is not None
        return detail

    def viz_file(self, feature_id: int) -> Path | None:
        if feature_id in self._viz_stat:
            return self._viz_stat[feature_id]
        found = self.viz.get(feature_id)
        if found is None:
            for name in (f"feature_{feature_id:05d}.png", f"feature_{feature_id}.png"):
                candidate = self.run_dir / "features" / "viz" / name
                if candidate.is_file():
                    found = candidate
                    break
        self._viz_stat[feature_id] = found
        return found

    def heatmap(self, clip_id: str, frame: int, feature_id: int) -> list[list[float]] | None:
        """Patch grid for one token. Empty cells are 0."""
        clip = self.manifest_clips.get(clip_id)
        if clip is None or self.acts is None:
            return None
        n_frames = int(clip["n_frames"]) if _is_int(clip.get("n_frames")) else 0
        if frame < 0 or frame >= n_frames:
            return None
        height, width = int(self.grid["h"]), int(self.grid["w"])
        grid = [[0.0 for _ in range(width)] for _ in range(height)]
        starts, stride = _starts_covering(clip, self.grid, self.extra, frame)
        _window, tubelet_index = frame_to_token(frame, starts, self.grid, stride)
        if tubelet_index is None:
            return grid
        for h, w, value in self.acts.cells(clip_id, _window, tubelet_index, feature_id):
            if 0 <= h < height and 0 <= w < width:
                grid[h][w] = value
        return grid

    def _top_for(self, feature_id: int) -> list[dict]:
        rows = []
        for clip_id, window, tubelet_index, h, w, value in self.acts.top_clips(feature_id):
            clip = self.manifest_clips.get(str(clip_id), {})
            starts, stride = clip_sampling(clip, self.grid, int(window), self.extra)
            rows.append(
                {
                    "clip_id": str(clip_id),
                    "split": clip.get("split"),
                    "window": int(window),
                    "t": int(tubelet_index),
                    "h": int(h),
                    "w": int(w),
                    "value": float(value),
                    "frame": token_to_frame(int(window), int(tubelet_index), starts, self.grid, stride),
                }
            )
        return rows

    def _build_clip(self, clip_id: str) -> dict | None:
        clip = self.manifest_clips.get(clip_id)
        if clip is None or self.acts is None:
            return None
        loaded = self.loaded.get(clip_id) or LoadedClip(
            "missing",
            "Clip file not found. Pass --clips, or record a path in the manifest.",
        )
        if loaded.kind == "video" and loaded.width is None and loaded.path is not None and loaded.status == "ok":
            loaded.width, loaded.height = _probe_video(loaded.path)
        return _clip_payload(
            clip,
            self.grid,
            loaded,
            self.acts.timeline(clip_id),
            self.acts.effects(clip_id),
            self.features_by_id,
            self.extra,
            self.adapter,
        )


def load_run(run_dir: Path, clips_dir: Path | None = None) -> RunModel:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run folder not found: {run_dir}")
    missing = [rel for rel in _REQUIRED if not (run_dir / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing {', '.join(missing)}")
    if clips_dir is not None and not Path(clips_dir).is_dir():
        raise FileNotFoundError(f"clips directory not found: {clips_dir}")

    manifest = read_json(run_dir / "manifest.json")
    _require_keys(manifest, ("run_id", "model", "layer", "grid", "clips"), "manifest")
    _require_keys(manifest["grid"], ("t", "h", "w"), "manifest.grid")
    if not isinstance(manifest["model"], dict):
        raise ValueError("manifest.model must be an object")
    if not isinstance(manifest["clips"], list):
        raise ValueError("manifest.clips must be a list")

    scoreboard = _read_optional_json(run_dir / "scoreboard.json")
    labels = _stream_labels(run_dir / "features" / "features.jsonl")
    candidates = _read_candidates(run_dir / "features" / "candidates.json")
    acts_path = run_dir / "features" / "feature_acts.parquet"
    _require_columns(pq.read_schema(acts_path), ("feature_id", "clip_id", "window", "t", "h", "w", "value"), "feature_acts.parquet")
    effects_path = run_dir / "features" / "clip_effects.parquet"
    if effects_path.is_file():
        _require_columns(
            pq.read_schema(effects_path),
            ("feature_id", "clip_id", "window", "t", "effect", "effect_kind"),
            "clip_effects.parquet",
        )
    acts = _Acts(acts_path, effects_path if effects_path.is_file() else None)

    grid = _grid_dict(manifest["grid"])
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else None
    features, viz = _feature_index(scoreboard, labels, candidates, run_dir)
    moments = _read_moments(run_dir / "heldout_loss.json")
    pixels = _resolve_pixels(manifest, run_dir, None if clips_dir is None else Path(clips_dir))
    manifest_clips: dict[str, dict] = {}
    loaded: dict[str, LoadedClip] = {}
    frames: dict[str, np.ndarray] = {}
    peaks: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for clip in manifest["clips"]:
        if not isinstance(clip, dict) or "clip_id" not in clip:
            continue
        clip_id = str(clip["clip_id"])
        manifest_clips[clip_id] = clip
        file_info = pixels.get(clip_id) or LoadedClip(
            "missing",
            "Clip file not found. Pass --clips, or record a path in the manifest.",
        )
        loaded[clip_id] = file_info
        if file_info.array is not None and file_info.status == "ok":
            frames[clip_id] = file_info.array
            peaks[clip_id] = file_info.peak
        if file_info.kind == "video" and file_info.status == "ok":
            media = "video"
        elif file_info.kind == "array" and file_info.status == "ok":
            media = "frames"
        else:
            media = "none"
        meta[clip_id] = {
            "clip_id": clip_id,
            "fps": float(clip.get("fps") or 0.0),
            "n_frames": int(clip["n_frames"]) if _is_int(clip.get("n_frames")) else 0,
            "split": clip.get("split"),
            "pixels": file_info.status,
            "media": media,
        }

    model_info = manifest["model"]
    run_payload = {
        "run_id": manifest["run_id"],
        "model": {
            "adapter": model_info.get("adapter"),
            "name": model_info.get("name"),
            "sha": model_info.get("sha"),
        },
        "layer": int(manifest["layer"]),
        "grid": grid,
        "scoreboard": scoreboard is not None,
        "split": _split_payload(scoreboard, manifest),
        "sae": _sae_payload(scoreboard),
        "clips": list(meta.values()),
        "features": features,
        "moments": moments,
    }
    model = RunModel(
        run_payload=run_payload,
        clips={},
        loaded=loaded,
        frames=frames,
        peaks=peaks,
        meta=meta,
        manifest_clips=manifest_clips,
        grid=grid,
        extra=extra,
        features_by_id={feature["id"]: feature for feature in features},
        viz=viz,
        run_dir=run_dir,
        adapter=model_info.get("adapter") if isinstance(model_info.get("adapter"), str) else None,
        acts=acts,
    )
    model.clips = _LazyClips(model._build_clip)
    return model


def render_frame(frame: np.ndarray, peak: float) -> np.ndarray:
    """Return a uint8 RGB image of one clip frame.

    One or three channels are shown directly. Any other channel count is drawn
    as a dark field with the strongest channel tinted, so a sparse multi-channel
    array still reads as an image.
    """
    array = np.nan_to_num(np.asarray(frame, dtype=np.float32))
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"frame must be HxW or HxWxC, got {array.shape}")
    channels = int(array.shape[-1])
    if channels <= 0:
        raise ValueError("frame has no channels")
    if channels == 1:
        tone = _unit_interval(array[..., 0])
        rgb = np.repeat(tone[..., None], 3, axis=2)
    elif channels in (3, 4):
        rgb = _unit_interval(array[..., :3])
    else:
        rgb = _composite(array, peak)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def encode_png(rgb: np.ndarray) -> bytes:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG encoder expected uint8 HxWx3")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    raw = b"".join(b"\x00" + np.ascontiguousarray(rgb[row]).tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")


def _handler(model: RunModel):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            del fmt, args

        protocol_version = "HTTP/1.1"

        def do_HEAD(self) -> None:  # noqa: N802
            self.close_connection = True
            try:
                path, query = self._target()
                self._route(path, query, head=True)
            except BrokenPipeError:
                return
            except Exception as exc:  # noqa: BLE001
                self._send(json.dumps({"error": str(exc)}).encode(), "application/json", status=500, head=True)

        def do_GET(self) -> None:  # noqa: N802
            self.close_connection = True
            try:
                path, query = self._target()
                self._route(path, query, head=False)
            except BrokenPipeError:
                return
            except Exception as exc:  # noqa: BLE001
                body = json.dumps({"error": str(exc)}).encode()
                self._send(body, "application/json", status=500)

        def _target(self) -> tuple[str, dict[str, list[str]]]:
            parsed = urlparse(self.path)
            return unquote(parsed.path), parse_qs(parsed.query)

        def _route(self, path: str, query: dict[str, list[str]], head: bool) -> None:
            if path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", head=head)
                return
            if path.startswith("/static/"):
                self._send_static(path[len("/static/") :], head=head)
                return
            if path == "/api/run":
                self._send_json(model.run_payload, head=head)
                return
            viz_match = _FEATURE_VIZ.match(path)
            if viz_match:
                viz = model.viz_file(int(viz_match.group("feature")))
                if viz is None:
                    self._send_json({"error": "no viz"}, status=404, head=head)
                    return
                self._send(viz.read_bytes(), "image/png", cache="private, max-age=3600", head=head)
                return
            feature_match = _FEATURE.match(path)
            if feature_match:
                detail = model.feature(int(feature_match.group("feature")))
                if detail is None:
                    self._send_json({"error": "unknown feature"}, status=404, head=head)
                    return
                self._send_json(detail, head=head)
                return
            heat_match = _HEAT.match(path)
            if heat_match:
                frame = _query_int(query, "frame")
                feature_id = _query_int(query, "feature")
                if frame is None or feature_id is None:
                    self._send_json({"error": "frame and feature are required"}, status=400, head=head)
                    return
                heat = model.heatmap(heat_match.group("clip"), frame, feature_id)
                if heat is None:
                    self._send_json({"error": "frame not available"}, status=404, head=head)
                    return
                self._send_json({"heatmap": heat}, head=head)
                return
            media_match = _MEDIA.match(path)
            if media_match:
                played = model.media_file(media_match.group("clip"))
                if played is None:
                    self._send_json({"error": "video not available"}, status=404, head=head)
                    return
                media_path, content_type = played
                self._send_range(media_path, content_type, head=head)
                return
            frame_match = _FRAME_PNG.match(path)
            if frame_match:
                clip_id = frame_match.group("clip")
                frame = int(frame_match.group("frame"))
                png = model.frame_png(clip_id, frame)
                if png is None:
                    self._send_json({"error": "frame not available"}, status=404, head=head)
                    return
                self._send(png, "image/png", cache="private, max-age=3600", head=head)
                return
            clip_match = _CLIP_ID.match(path)
            if clip_match:
                payload = model.clips.get(clip_match.group("clip"))
                if payload is None:
                    self._send_json({"error": "unknown clip"}, status=404, head=head)
                    return
                self._send_json(payload, head=head)
                return
            self._send_json({"error": "not found"}, status=404, head=head)

        def _send_static(self, relative: str, head: bool) -> None:
            candidate = _safe_static(relative)
            if candidate is None:
                self._send_json({"error": "not found"}, status=404, head=head)
                return
            self._send_file(candidate, head=head)

        def _send_file(self, path: Path, head: bool) -> None:
            if not path.is_file():
                self._send_json({"error": "not found"}, status=404, head=head)
                return
            content_type = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
            cache = "public, max-age=86400" if path.suffix.lower() == ".woff2" else "no-cache"
            self._send(path.read_bytes(), content_type, cache=cache, head=head)

        def _send_json(self, payload: dict, status: int = 200, head: bool = False) -> None:
            body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
            self._send(body, "application/json", status=status, head=head)

        def _send(self, body: bytes, content_type: str, status: int = 200, cache: str = "no-cache", head: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("Connection", "close")
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def _send_range(self, path: Path, content_type: str, head: bool) -> None:
            size = path.stat().st_size
            start, end = 0, max(size - 1, 0)
            status = 200
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes=") and size:
                spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
                start_text, _, end_text = spec.partition("-")
                try:
                    if start_text == "":
                        suffix = int(end_text or "0")
                        start = max(0, size - suffix)
                        end = size - 1
                    else:
                        start = int(start_text)
                        end = int(end_text) if end_text else size - 1
                except ValueError:
                    start, end = 0, size - 1
                else:
                    if start < 0 or start >= size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        return
                    end = min(end, size - 1)
                    status = 206
            length = 0 if size == 0 else end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "private, max-age=3600")
            self.send_header("Connection", "close")
            self.end_headers()
            if head or length == 0:
                return
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def _safe_static(relative: str) -> Path | None:
    if not relative or relative.startswith(("/", "\\")) or ".." in Path(relative).parts:
        return None
    root = STATIC_DIR.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _read_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _grid_dict(grid: dict) -> dict:
    parsed = {key: int(grid[key]) for key in ("t", "h", "w") if key in grid}
    parsed["tubelet"] = int(grid["tubelet"]) if _is_int(grid.get("tubelet")) else tubelet_of(grid)
    parsed["patch_px"] = int(grid["patch_px"]) if _is_int(grid.get("patch_px")) else 1
    if _is_int(grid.get("window_stride")):
        parsed["window_stride"] = int(grid["window_stride"])
    if _is_int(grid.get("window_start")):
        parsed["window_start"] = int(grid["window_start"])
    raw_starts = grid.get("window_starts")
    if isinstance(raw_starts, list) and raw_starts and all(_is_int(item) for item in raw_starts):
        parsed["window_starts"] = [int(item) for item in raw_starts]
    return parsed


class _LazyClips(dict):
    """Builds a clip payload the first time it is requested."""

    def __init__(self, build):
        super().__init__()
        self._build = build

    def get(self, key, default=None):
        if key in self:
            return super().get(key, default)
        built = self._build(str(key))
        if built is None:
            return default
        self[str(key)] = built
        return built

    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value


class _LRU:
    def __init__(self, cap: int):
        self.cap = cap
        self._data: dict = {}
        self._order: list = []

    def get(self, key):
        if key not in self._data:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._data[key]

    def put(self, key, value) -> None:
        if key in self._data:
            self._order.remove(key)
        self._data[key] = value
        self._order.append(key)
        while len(self._order) > self.cap:
            self._data.pop(self._order.pop(0), None)


class _Acts:
    """DuckDB queries over feature_acts, with predicate pushdown and a small cache.

    The parquet is never materialized. Timeline rows sum ``value`` over ``h, w``.
    Heatmaps are one token. Top clips are the per-clip peak row.
    """

    def __init__(self, acts_path: Path, effects_path: Path | None):
        self.acts_path = str(acts_path)
        schema = pq.read_schema(acts_path)
        self.clip_is_int = pa.types.is_integer(schema.field("clip_id").type)
        self.effects_path = str(effects_path) if effects_path is not None else None
        self.effects_clip_is_int = False
        if effects_path is not None:
            effect_schema = pq.read_schema(effects_path)
            self.effects_clip_is_int = pa.types.is_integer(effect_schema.field("clip_id").type)
        self._con = duckdb.connect()
        self._con.execute("SET memory_limit='768MB'")
        self._con.execute("SET threads=2")
        self._con.execute("SET preserve_insertion_order=false")
        self._lock = threading.Lock()
        self._timeline = _LRU(8)
        self._effects = _LRU(8)
        self._cells = _LRU(64)
        self._tops = _LRU(32)

    def timeline(self, clip_id: str) -> list[tuple]:
        key = str(clip_id)
        with self._lock:
            cached = self._timeline.get(key)
            if cached is not None:
                return cached
            rows = self._con.execute(
                """
                WITH summed AS (
                    SELECT "window", "t", feature_id, sum(value)::DOUBLE AS activation
                    FROM read_parquet(?)
                    WHERE clip_id = ?
                    GROUP BY "window", "t", feature_id
                ),
                ranked AS (
                    SELECT "window", "t", feature_id, activation,
                           row_number() OVER (
                               PARTITION BY "window", "t"
                               ORDER BY activation DESC, feature_id
                           ) AS rn
                    FROM summed
                )
                SELECT "window", "t", feature_id, activation
                FROM ranked
                WHERE rn <= ?
                """,
                [self.acts_path, self._clip_param(key, self.clip_is_int), _TOP_FEATURES],
            ).fetchall()
            self._timeline.put(key, rows)
            return rows

    def effects(self, clip_id: str) -> list[tuple]:
        key = str(clip_id)
        with self._lock:
            cached = self._effects.get(key)
            if cached is not None:
                return cached
            if self.effects_path is None:
                self._effects.put(key, [])
                return []
            rows = self._con.execute(
                """
                SELECT feature_id, "window", "t", effect_kind, avg(effect)::DOUBLE
                FROM read_parquet(?)
                WHERE clip_id = ?
                GROUP BY feature_id, "window", "t", effect_kind
                """,
                [self.effects_path, self._clip_param(key, self.effects_clip_is_int)],
            ).fetchall()
            self._effects.put(key, rows)
            return rows

    def cells(self, clip_id: str, window: int, tubelet_index: int, feature_id: int) -> list[tuple]:
        key = (str(clip_id), int(window), int(tubelet_index), int(feature_id))
        with self._lock:
            cached = self._cells.get(key)
            if cached is not None:
                return cached
            rows = self._con.execute(
                """
                SELECT h, w, max(value)::DOUBLE
                FROM read_parquet(?)
                WHERE clip_id = ? AND "window" = ? AND "t" = ? AND feature_id = ?
                GROUP BY h, w
                """,
                [
                    self.acts_path,
                    self._clip_param(key[0], self.clip_is_int),
                    int(window),
                    int(tubelet_index),
                    int(feature_id),
                ],
            ).fetchall()
            self._cells.put(key, rows)
            return rows

    def top_clips(self, feature_id: int) -> list[tuple]:
        key = int(feature_id)
        with self._lock:
            cached = self._tops.get(key)
            if cached is not None:
                return cached
            rows = self._con.execute(
                """
                WITH ranked AS (
                    SELECT clip_id, "window", "t", h, w, value::DOUBLE AS value,
                           row_number() OVER (
                               PARTITION BY clip_id
                               ORDER BY value DESC, "window", "t", h, w
                           ) AS rn
                    FROM read_parquet(?)
                    WHERE feature_id = ?
                )
                SELECT clip_id, "window", "t", h, w, value
                FROM ranked
                WHERE rn = 1
                ORDER BY value DESC, clip_id
                LIMIT ?
                """,
                [self.acts_path, key, _TOP_CLIPS],
            ).fetchall()
            self._tops.put(key, rows)
            return rows

    def _clip_param(self, clip_id: str, integer: bool):
        if integer:
            try:
                return int(clip_id)
            except ValueError:
                return clip_id
        return str(clip_id)


def _stream_labels(path: Path) -> dict[int, dict]:
    """One pass over features.jsonl. Only id, label, and label_model are kept."""
    found: dict[int, dict] = {}
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} has a line that is not JSON") from exc
            if not isinstance(row, dict) or not _is_int(row.get("id")):
                continue
            found[int(row["id"])] = {
                "label": row["label"] if isinstance(row.get("label"), str) else "",
                "label_model": row["label_model"] if isinstance(row.get("label_model"), str) else None,
            }
    return found


def _read_candidates(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        ranked = payload.get("ranked")
        if isinstance(ranked, list) and ranked and isinstance(ranked[0], dict):
            payload = ranked
        else:
            payload = payload.get("candidates") or payload.get("features") or []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict) and _is_int(row.get("feature_id"))]


def _blank_feature(feature_id: int) -> dict:
    record = {
        "id": feature_id,
        "label": "",
        "label_model": None,
        "badge": "unverified",
        "status": None,
        "scored": False,
        "candidate": False,
    }
    for key in _SCORE_FIELDS:
        record[key] = None
    return record


def _feature_index(scoreboard: dict | None, labels: dict[int, dict], candidates: list[dict], run_dir: Path) -> tuple[list[dict], dict[int, Path]]:
    """Scoreboard and jsonl features, with candidates.json entries first.

    Nothing here reads feature_acts. Top clips are filled in when a feature is opened.
    """
    scored: dict[int, dict] = {}
    rows = scoreboard.get("features") if isinstance(scoreboard, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and "id" in row and _is_int(row["id"]):
                scored[int(row["id"])] = row
    ordered: list[int] = []
    seen: set[int] = set()
    viz: dict[int, Path] = {}
    for row in candidates:
        feature_id = int(row["feature_id"])
        if feature_id in seen:
            continue
        seen.add(feature_id)
        ordered.append(feature_id)
        picture = _resolve_viz(run_dir, feature_id, row.get("viz"))
        if picture is not None:
            viz[feature_id] = picture
    for feature_id in sorted((set(scored) | set(labels)) - seen):
        ordered.append(feature_id)
    records = []
    for feature_id in ordered:
        row = scored.get(feature_id)
        meta = labels.get(feature_id, {})
        jsonl_label = meta.get("label") if isinstance(meta.get("label"), str) else ""
        if row is None:
            record = _blank_feature(feature_id)
            record["label"] = jsonl_label
            record["label_model"] = meta.get("label_model")
        else:
            label = row.get("label") if isinstance(row.get("label"), str) else jsonl_label
            status = status_text(row.get("status"))
            record = {
                "id": feature_id,
                "label": label or "",
                "label_model": meta.get("label_model"),
                "badge": badge_for(row.get("real"), status),
                "status": status,
                "scored": True,
                "candidate": feature_id in seen,
            }
            for key in _SCORE_FIELDS:
                record[key] = _json_value(row[key]) if key in row else None
        record["candidate"] = feature_id in seen
        records.append(record)
    return records, viz


def _resolve_viz(run_dir: Path, feature_id: int, raw: object) -> Path | None:
    if isinstance(raw, str) and raw and ".." not in Path(raw).parts and not Path(raw).is_absolute():
        candidate = (run_dir / raw).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError:
            candidate = None
        if candidate is not None and candidate.is_file():
            return candidate
    for name in (f"feature_{feature_id:05d}.png", f"feature_{feature_id}.png"):
        candidate = run_dir / "features" / "viz" / name
        if candidate.is_file():
            return candidate
    return None


def _frame_feature_key(row: dict) -> tuple:
    """Higher activation first. Ablation-passed wins inside the same tenth."""
    activation = float(row["activation"])
    passed = 0 if ablation_passed(row.get("status")) else 1
    return (-round(activation, 1), passed, -activation, int(row["id"]))


def _read_moments(path: Path) -> list[dict]:
    """``heldout_loss.json`` rows, highest loss first. Links use ``clip_id``."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, dict):
        payload = payload.get("moments") or payload.get("losses") or payload.get("windows") or []
    if not isinstance(payload, list):
        return []
    rows = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not _is_int(item.get("clip_id")):
            continue
        if not _is_int(item.get("frame_start")) or not _is_int(item.get("frame_end")):
            continue
        loss = _num(item.get("loss"))
        if loss is None:
            continue
        start, end = int(item["frame_start"]), int(item["frame_end"])
        if end < start:
            start, end = end, start
        task = item.get("task_text")
        if not isinstance(task, str) or not task.strip():
            task = None
        else:
            task = task.strip()
        name = item.get("clip_name")
        clip_id = str(int(item["clip_id"]))
        rows.append(
            {
                "clip_id": clip_id,
                "clip_name": name.strip() if isinstance(name, str) and name.strip() else clip_id,
                "window": int(item["window"]) if _is_int(item.get("window")) else None,
                "frame_start": start,
                "frame_end": end,
                "frame": (start + end) // 2,
                "loss": loss,
                "task_text": task,
                "loss_source": item.get("loss_source") if isinstance(item.get("loss_source"), str) else None,
                "_order": index,
            }
        )
    rows.sort(key=lambda row: (-row["loss"], row["_order"]))
    for row in rows:
        del row["_order"]
    return rows


def _query_int(query: dict[str, list[str]], name: str) -> int | None:
    raw = (query.get(name) or [None])[0]
    if raw is None or not str(raw).isdigit():
        return None
    return int(raw)


def _split_payload(scoreboard: dict | None, manifest: dict) -> dict:
    run = scoreboard.get("run") if isinstance(scoreboard, dict) else None
    if isinstance(run, dict) and _is_int(run.get("n_train_clips")) and _is_int(run.get("n_heldout_clips")):
        return {"n_train": int(run["n_train_clips"]), "n_heldout": int(run["n_heldout_clips"])}
    clips = manifest.get("clips") or []
    return {
        "n_train": sum(1 for clip in clips if isinstance(clip, dict) and clip.get("split") == "train"),
        "n_heldout": sum(1 for clip in clips if isinstance(clip, dict) and clip.get("split") == "heldout"),
    }


def _sae_payload(scoreboard: dict | None) -> dict | None:
    sae = scoreboard.get("sae") if isinstance(scoreboard, dict) else None
    if not isinstance(sae, dict):
        return None
    if not any(key in sae for key in ("usable", "ev_heldout", "loss_recovered")):
        return None
    return {
        "usable": sae.get("usable") is True,
        "ev_heldout": _num(sae["ev_heldout"]) if "ev_heldout" in sae else None,
        "loss_recovered": _num(sae["loss_recovered"]) if "loss_recovered" in sae else None,
    }


def _clip_payload(clip, grid, loaded, timeline, effect_rows, meta_by_id, extra, adapter=None):
    """Per-frame top features for one clip. Activations are sums over h, w.

    Heatmaps are not included; they are loaded for the token on screen.
    """
    clip_id = str(clip["clip_id"])
    n_frames = int(clip["n_frames"]) if _is_int(clip.get("n_frames")) else 0
    grouped: dict[tuple[int, int], list[tuple[int, float]]] = {}
    max_window = 0
    for window, tubelet_index, feature_id, activation in timeline:
        max_window = max(max_window, int(window))
        grouped.setdefault((int(window), int(tubelet_index)), []).append((int(feature_id), float(activation)))
    effects: dict[tuple[int, int, int], list[dict]] = {}
    for feature_id, window, tubelet_index, kind, effect in effect_rows:
        max_window = max(max_window, int(window))
        effect_value = _num(effect)
        if effect_value is None or kind is None:
            continue
        effects.setdefault((int(feature_id), int(window), int(tubelet_index)), []).append(
            {"effect_kind": str(kind), "effect": effect_value}
        )
    for bucket in effects.values():
        bucket.sort(key=lambda item: (_KIND_ORDER.get(item["effect_kind"], 2), item["effect_kind"]))
    starts, sample_stride = clip_sampling(clip, grid, max_window, extra)
    hw = _displayed_hw(clip, loaded)
    crop = overlay_rect(hw, extra, adapter)
    frames = []
    for frame in range(n_frames):
        window, tubelet_index = frame_to_token(frame, starts, grid, sample_stride)
        if tubelet_index is None:
            features: list[dict] = []
            shown_t = None
        else:
            features = []
            for feature_id, activation in grouped.get((window, tubelet_index), []):
                meta = meta_by_id.get(feature_id, {})
                features.append(
                    {
                        "id": feature_id,
                        "label": meta.get("label") or "",
                        "badge": meta.get("badge") or "unverified",
                        "status": meta.get("status"),
                        "activation": activation,
                        "effects": effects.get((feature_id, window, tubelet_index), []),
                    }
                )
            features.sort(key=_frame_feature_key)
            shown_t = tubelet_index
        frames.append({"frame": frame, "window": window, "t": shown_t, "features": features})
    if loaded.kind == "video" and loaded.status == "ok":
        media = "video"
    elif loaded.kind == "array" and loaded.status == "ok":
        media = "frames"
    else:
        media = "none"
    return {
        "clip_id": clip_id,
        "fps": float(clip.get("fps") or 0.0),
        "n_frames": n_frames,
        "split": clip.get("split"),
        "pixels": loaded.status,
        "media": media,
        "message": loaded.message,
        "width": loaded.width,
        "height": loaded.height,
        "hw": hw,
        "crop": crop,
        "window_starts": starts,
        "stride": sample_stride,
        "grid": {key: grid[key] for key in ("t", "h", "w", "tubelet", "patch_px") if key in grid},
        "frames": frames,
    }



def _manifest_clip_path(clip: dict, run_dir: Path) -> Path | None:
    raw = clip.get("path") or clip.get("file")
    if not raw or not isinstance(raw, str):
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = run_dir / path
    return path if path.is_file() else None


def _manifest_clips_dir(manifest: dict, run_dir: Path) -> Path | None:
    raw = manifest.get("clips_dir")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = run_dir / path
    return path if path.is_dir() else None


@dataclass
class LoadedClip:
    status: str
    message: str
    kind: str = "none"
    array: np.ndarray | None = None
    path: Path | None = None
    width: int | None = None
    height: int | None = None
    peak: float = 1.0


def _resolve_pixels(manifest: dict, run_dir: Path, clips_dir: Path | None) -> dict[str, LoadedClip]:
    directory = clips_dir if clips_dir is not None else _manifest_clips_dir(manifest, run_dir)
    files = _candidate_files(directory) if directory is not None else []
    by_stem = _index_by_stem(files)
    sha_index: dict[str, Path] | None = None
    resolved: dict[str, LoadedClip] = {}
    for clip in manifest["clips"]:
        if not isinstance(clip, dict) or "clip_id" not in clip:
            continue
        clip_id = str(clip["clip_id"])
        path = _manifest_clip_path(clip, run_dir) if clips_dir is None else None
        expected = clip.get("sha256") if isinstance(clip.get("sha256"), str) else None
        named = _named_clip_file(directory, files, clip)
        stem_path = by_stem.get(clip_id)
        if path is None and _sha_ok(named, expected):
            path = named
        elif path is None and _sha_ok(stem_path, expected):
            path = stem_path
        elif path is None:
            if expected is not None:
                if sha_index is None:
                    sha_index = _index_by_sha(files)
                path = sha_index.get(expected)
            if path is None:
                path = named or stem_path
        resolved[clip_id] = _load_clip_file(path, clip)
    return resolved


def _load_clip_file(path: Path | None, clip: dict) -> LoadedClip:
    if path is None or not path.is_file():
        return LoadedClip("missing", "Clip file not found. Pass --clips, or record a path in the manifest.")
    expected = clip.get("sha256")
    if (
        isinstance(expected, str)
        and path.stat().st_size <= _SHA_EAGER_LIMIT
        and _file_sha256(path) != expected
    ):
        return LoadedClip("sha_mismatch", "Clip file does not match the sha256 recorded in the manifest.", path=path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _load_npy(path, clip)
    if suffix in _NATIVE_VIDEO or suffix in _TRANSCODE_VIDEO:
        return LoadedClip("ok", "", kind="video", path=path)
    return LoadedClip("unreadable", f"Unsupported clip file {path.name}.", path=path)


def _load_npy(path: Path, clip: dict) -> LoadedClip:
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        return LoadedClip("unreadable", f"Clip file could not be read: {exc}", path=path)
    array = np.asarray(array)
    if array.ndim not in (3, 4) or array.shape[0] <= 0:
        return LoadedClip("unreadable", f"Clip array must be T×H×W or T×H×W×C, got {array.shape}.", path=path)
    n_frames = int(clip["n_frames"]) if _is_int(clip.get("n_frames")) else 0
    if n_frames and int(array.shape[0]) != n_frames:
        return LoadedClip(
            "length_mismatch",
            f"Clip file has {array.shape[0]} frames; the manifest says {n_frames}.",
            path=path,
        )
    frames = np.ascontiguousarray(array, dtype=np.float32)
    return LoadedClip(
        "ok",
        "",
        kind="array",
        array=frames,
        path=path,
        width=int(frames.shape[2]),
        height=int(frames.shape[1]),
        peak=_clip_peak(frames),
    )


def _sha_ok(path: Path | None, expected: str | None) -> bool:
    if path is None:
        return False
    if expected is None or path.stat().st_size > _SHA_EAGER_LIMIT:
        return True
    return _file_sha256(path) == expected


def _named_clip_file(directory: Path | None, files: list[Path], clip: dict) -> Path | None:
    """Exact filename from the clip's ``name`` field, before stem or sha256."""
    raw = clip.get("name")
    if not isinstance(raw, str) or not raw.strip():
        return None
    filename = Path(raw).name
    if not filename or filename in (".", ".."):
        return None
    if directory is not None:
        direct = directory / filename
        try:
            direct.relative_to(directory)
        except ValueError:
            return None
        if direct.is_file():
            return direct
    for path in files:
        if path.name == filename:
            return path
    return None


def _candidate_files(directory: Path) -> list[Path]:
    found = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _MEDIA_SUFFIXES:
            found.append(path)
    return found


def _index_by_stem(files: list[Path]) -> dict[str, Path]:
    rank = {suffix: index for index, suffix in enumerate(_MEDIA_SUFFIXES)}
    chosen: dict[str, Path] = {}
    for path in files:
        current = chosen.get(path.stem)
        if current is None or rank[path.suffix.lower()] < rank[current.suffix.lower()]:
            chosen[path.stem] = path
    return chosen


def _index_by_sha(files: list[Path]) -> dict[str, Path]:
    found = {}
    for path in files:
        found[_file_sha256(path)] = path
    return found


def _playable_video(path: Path) -> tuple[Path, str] | None:
    suffix = path.suffix.lower()
    if suffix in _NATIVE_VIDEO:
        return path, _NATIVE_VIDEO[suffix]
    if suffix not in _TRANSCODE_VIDEO:
        return None
    dest = _transcode_cache(path)
    if not dest.is_file() or dest.stat().st_size == 0:
        _transcode_to_mp4(path, dest)
    return dest, "video/mp4"


def _transcode_cache(path: Path) -> Path:
    directory = Path(tempfile.gettempdir()) / "physint-view"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_file_sha256(path)[:16]}.mp4"


def _transcode_to_mp4(src: Path, dest: Path) -> None:
    temporary = dest.with_suffix(".part.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        temporary.unlink(missing_ok=True)
        detail = exc.stderr.decode("utf-8", "replace") if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise RuntimeError(f"ffmpeg could not transcode {src.name}: {detail.strip()}") from exc
    temporary.replace(dest)


def _probe_video(path: Path) -> tuple[int | None, int | None]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None, None
    text = result.stdout.strip()
    if result.returncode != 0 or "x" not in text:
        return None, None
    width, _, height = text.partition("x")
    if width.isdigit() and height.isdigit():
        return int(width), int(height)
    return None, None


def _video_frame_png(path: Path, frame: int, fps: float) -> bytes | None:
    timestamp = max(frame, 0) / fps if fps else 0.0
    # -ss after -i so the thumbnail is that frame, not the previous keyframe.
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ss",
        f"{timestamp:.4f}",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.startswith(b"\x89PNG"):
        return None
    return result.stdout


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip_peak(frames: np.ndarray) -> float:
    if frames.ndim == 3 or frames.shape[-1] in (1, 3, 4):
        return 1.0
    peak = float(np.max(frames)) if frames.size else 1.0
    return peak if peak > 1e-8 else 1.0


def _unit_interval(channel: np.ndarray) -> np.ndarray:
    array = np.asarray(channel, dtype=np.float32)
    peak = float(np.max(array)) if array.size else 0.0
    low = float(np.min(array)) if array.size else 0.0
    if peak <= 1.5 and low >= -0.05:
        return np.clip(array, 0.0, 1.0)
    if peak <= 255.0 and low >= 0.0:
        return np.clip(array / 255.0, 0.0, 1.0)
    span = peak - low if peak > low else 1.0
    return np.clip((array - low) / span, 0.0, 1.0)


def _composite(frame: np.ndarray, peak: float) -> np.ndarray:
    strength = np.max(frame, axis=-1)
    dominant = np.argmax(frame, axis=-1)
    scale = peak if peak > 1e-8 else 1.0
    floor = max(0.05, 0.05 * scale)
    alpha = np.clip((strength - floor) / (scale - floor), 0.0, 1.0)[..., None]
    colors = _PALETTE[dominant % len(_PALETTE)]
    background = np.array([18.0, 17.0, 15.0], dtype=np.float32)
    rgb = background * (1.0 - alpha) + colors * alpha
    return rgb / 255.0


def _upsample(rgb: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return rgb
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def _display_scale(height: int, width: int, target: int = 384) -> int:
    longest = max(height, width, 1)
    if longest >= target:
        return 1
    return max(1, math.ceil(target / longest))


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _require_keys(payload: dict, keys: tuple[str, ...], name: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be an object")
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{name} is missing {', '.join(missing)}")


def _require_columns(table, columns: tuple[str, ...], name: str) -> None:
    present = set(getattr(table, "names", None) or table.column_names)
    missing = [column for column in columns if column not in present]
    if missing:
        raise ValueError(f"{name} is missing columns {', '.join(missing)}")


def _num(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _json_value(value: object):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _num(value)
    return value
