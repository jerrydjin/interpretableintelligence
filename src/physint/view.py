"""Local clip-first viewer over a finished run folder.

Reads ``manifest.json`` and ``features/feature_acts.parquet``.
``features/clip_effects.parquet``, ``features/features.jsonl``, and
``scoreboard.json`` are used when present and skipped when absent. Clip pixels
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
from urllib.parse import unquote, urlparse

import numpy as np
import pyarrow.parquet as pq

from physint.artifacts import read_json, read_jsonl

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
_CLIP_ID = re.compile(r"^/api/clips/(?P<clip>[^/]+)$")
_FRAME_PNG = re.compile(r"^/api/clips/(?P<clip>[^/]+)/frames/(?P<frame>[0-9]+)\.png$")
_MEDIA = re.compile(r"^/api/clips/(?P<clip>[^/]+)/media$")
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


def frame_to_token(frame: int, starts: list[int], grid: dict, sample_stride: int = 1) -> tuple[int, int | None]:
    """Return ``(window, t)``. ``t`` is ``None`` when ``frame`` sits in no window.

    Inside a window, ``t = (frame - window_start) // (tubelet * stride)``.
    """
    if not starts:
        starts = [0]
    window = 0
    for index, start in enumerate(starts):
        if start <= frame:
            window = index
        else:
            break
    offset = frame - starts[window]
    step = tubelet_of(grid) * max(int(sample_stride), 1)
    span = window_span(grid) * max(int(sample_stride), 1)
    if offset < 0 or offset >= span:
        return window, None
    return window, offset // step


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


def badge_for(real: object) -> str:
    """Non-real features are unverified. Only a JSON true counts as real."""
    return "real" if real is True else "unverified"


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
    _png: dict[tuple[str, int], bytes] = field(default_factory=dict)
    _media: dict[str, tuple[Path, str]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def frame_png(self, clip_id: str, frame: int) -> bytes | None:
        clip = self.clips.get(clip_id)
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
        clip = self.clips.get(clip_id)
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
        clip = self.clips[clip_id]
        if loaded is None or loaded.path is None:
            return None
        fps = float(clip.get("fps") or 0.0) or 8.0
        return _video_frame_png(loaded.path, frame, fps)


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
    jsonl = read_jsonl(run_dir / "features" / "features.jsonl") if (run_dir / "features" / "features.jsonl").is_file() else []
    acts = pq.read_table(run_dir / "features" / "feature_acts.parquet")
    _require_columns(
        acts,
        ("feature_id", "clip_id", "window", "t", "h", "w", "value"),
        "feature_acts.parquet",
    )
    effects_path = run_dir / "features" / "clip_effects.parquet"
    if effects_path.is_file():
        effects = pq.read_table(effects_path)
        _require_columns(
            effects,
            ("feature_id", "clip_id", "window", "t", "effect", "effect_kind"),
            "clip_effects.parquet",
        )
        effect_cells = _effect_cells(effects)
    else:
        effect_cells = {}

    grid = _grid_dict(manifest["grid"])
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else None
    act_cells = _act_cells(acts)
    pixels = _resolve_pixels(manifest, run_dir, None if clips_dir is None else Path(clips_dir))
    top_clips = _top_clips(act_cells, manifest, grid, extra)
    features = _feature_records(scoreboard, jsonl, act_cells, top_clips)

    meta_by_id = {feature["id"]: feature for feature in features}
    clip_payloads: dict[str, dict] = {}
    loaded: dict[str, LoadedClip] = {}
    frames: dict[str, np.ndarray] = {}
    peaks: dict[str, float] = {}
    max_window = _max_window(act_cells, effect_cells)
    for clip in manifest["clips"]:
        if not isinstance(clip, dict) or "clip_id" not in clip:
            continue
        clip_id = str(clip["clip_id"])
        file_info = pixels.get(clip_id) or LoadedClip("missing", "Clip file not found. Pass --clips, or record a path in the manifest.")
        n_windows = max_window.get(clip_id, 0)
        payload = _clip_payload(clip, grid, file_info, act_cells, effect_cells, meta_by_id, n_windows, extra)
        clip_payloads[clip_id] = payload
        loaded[clip_id] = file_info
        if file_info.array is not None and file_info.status == "ok":
            frames[clip_id] = file_info.array
            peaks[clip_id] = file_info.peak

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
        "clips": [
            {
                "clip_id": payload["clip_id"],
                "fps": payload["fps"],
                "n_frames": payload["n_frames"],
                "split": payload["split"],
                "pixels": payload["pixels"],
                "media": payload["media"],
            }
            for payload in clip_payloads.values()
        ],
        "features": features,
    }
    return RunModel(run_payload=run_payload, clips=clip_payloads, loaded=loaded, frames=frames, peaks=peaks)


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
                self._route(unquote(urlparse(self.path).path), head=True)
            except BrokenPipeError:
                return
            except Exception as exc:  # noqa: BLE001
                self._send(json.dumps({"error": str(exc)}).encode(), "application/json", status=500, head=True)

        def do_GET(self) -> None:  # noqa: N802
            self.close_connection = True
            path = unquote(urlparse(self.path).path)
            try:
                self._route(path, head=False)
            except BrokenPipeError:
                return
            except Exception as exc:  # noqa: BLE001
                body = json.dumps({"error": str(exc)}).encode()
                self._send(body, "application/json", status=500)

        def _route(self, path: str, head: bool) -> None:
            if path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", head=head)
                return
            if path.startswith("/static/"):
                self._send_static(path[len("/static/") :], head=head)
                return
            if path == "/api/run":
                self._send_json(model.run_payload, head=head)
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


def _feature_records(scoreboard: dict | None, jsonl: list, act_cells: dict, top_clips: dict) -> list[dict]:
    """Every feature seen in the acts, jsonl, or scoreboard.

    A feature with no scoreboard row is unverified. A missing scoreboard makes
    every feature unverified. Extra rows are kept; missing fields stay null.
    """
    scored: dict[int, dict] = {}
    rows = scoreboard.get("features") if isinstance(scoreboard, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and "id" in row and _is_int(row["id"]):
                scored[int(row["id"])] = row
    labelled: dict[int, dict] = {}
    for row in jsonl:
        if isinstance(row, dict) and "id" in row and _is_int(row["id"]):
            labelled[int(row["id"])] = row
    feature_ids = set(scored) | set(labelled) | {key[3] for key in act_cells}
    records = []
    for feature_id in sorted(feature_ids):
        row = scored.get(feature_id)
        meta = labelled.get(feature_id, {})
        jsonl_label = meta.get("label") if isinstance(meta.get("label"), str) else ""
        if row is None:
            record = {
                "id": feature_id,
                "label": jsonl_label,
                "label_model": meta.get("label_model"),
                "badge": "unverified",
                "scored": False,
                "top_clips": top_clips.get(feature_id, []),
            }
            for key in _SCORE_FIELDS:
                record[key] = None
        else:
            label = row.get("label") if isinstance(row.get("label"), str) else jsonl_label
            record = {
                "id": feature_id,
                "label": label or "",
                "label_model": meta.get("label_model"),
                "badge": badge_for(row.get("real")),
                "scored": True,
                "top_clips": top_clips.get(feature_id, []),
            }
            for key in _SCORE_FIELDS:
                record[key] = _json_value(row[key]) if key in row else None
        records.append(record)
    return records


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


def _max_window(act_cells: dict, effect_cells: dict) -> dict[str, int]:
    found: dict[str, int] = {}
    for clip_id, window, _t, _feature_id in act_cells:
        found[clip_id] = max(found.get(clip_id, 0), window)
    for clip_id, window, _t, _feature_id in effect_cells:
        found[clip_id] = max(found.get(clip_id, 0), window)
    return found


def _act_cells(table) -> dict[tuple, dict[tuple[int, int], float]]:
    cells: dict[tuple, dict[tuple[int, int], float]] = {}
    columns = table.select(["feature_id", "clip_id", "window", "t", "h", "w", "value"]).to_pydict()
    for feature_id, clip_id, window, t, h, w, value in zip(
        columns["feature_id"],
        columns["clip_id"],
        columns["window"],
        columns["t"],
        columns["h"],
        columns["w"],
        columns["value"],
        strict=True,
    ):
        key = (str(clip_id), int(window), int(t), int(feature_id))
        site = (int(h), int(w))
        number = float(value)
        bucket = cells.setdefault(key, {})
        bucket[site] = max(bucket.get(site, number), number)
    return cells


def _effect_cells(table) -> dict[tuple, dict[str, list[float]]]:
    grouped: dict[tuple, dict[str, list[float]]] = {}
    columns = table.select(["feature_id", "clip_id", "window", "t", "effect", "effect_kind"]).to_pydict()
    for feature_id, clip_id, window, t, effect, kind in zip(
        columns["feature_id"],
        columns["clip_id"],
        columns["window"],
        columns["t"],
        columns["effect"],
        columns["effect_kind"],
        strict=True,
    ):
        key = (str(clip_id), int(window), int(t), int(feature_id))
        grouped.setdefault(key, {}).setdefault(str(kind), []).append(float(effect))
    return grouped


def _top_clips(cells: dict, manifest: dict, grid: dict, extra: dict | None = None) -> dict[int, list[dict]]:
    clips = {str(clip["clip_id"]): clip for clip in manifest["clips"] if isinstance(clip, dict) and "clip_id" in clip}
    max_window: dict[str, int] = {}
    for clip_id, window, _t, _feature_id in cells:
        max_window[clip_id] = max(max_window.get(clip_id, 0), window)
    peaks: dict[tuple[int, str], tuple] = {}
    for (clip_id, window, t, feature_id), sites in cells.items():
        if not sites:
            continue
        (h, w), value = max(sites.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
        candidate = (value, window, t, h, w)
        key = (feature_id, clip_id)
        current = peaks.get(key)
        if current is None or candidate[0] > current[0] or (candidate[0] == current[0] and candidate[1:] < current[1:]):
            peaks[key] = candidate
    grouped: dict[int, list[dict]] = {}
    for (feature_id, clip_id), (value, window, t, h, w) in peaks.items():
        clip = clips.get(clip_id, {})
        starts, sample_stride = clip_sampling(clip, grid, max(max_window.get(clip_id, 0), window), extra)
        grouped.setdefault(feature_id, []).append(
            {
                "clip_id": clip_id,
                "split": clip.get("split"),
                "window": window,
                "t": t,
                "h": h,
                "w": w,
                "value": value,
                "frame": token_to_frame(window, t, starts, grid, sample_stride),
            }
        )
    for feature_id, rows in grouped.items():
        rows.sort(key=lambda row: (-row["value"], row["clip_id"]))
        del rows[20:]
        grouped[feature_id] = rows
    return grouped


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
    if isinstance(expected, str) and _file_sha256(path) != expected:
        return LoadedClip("sha_mismatch", "Clip file does not match the sha256 recorded in the manifest.", path=path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _load_npy(path, clip)
    if suffix in _NATIVE_VIDEO or suffix in _TRANSCODE_VIDEO:
        width, height = _probe_video(path)
        return LoadedClip("ok", "", kind="video", path=path, width=width, height=height)
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


def _clip_payload(clip, grid, loaded: LoadedClip, act_cells, effect_cells, meta_by_id, n_windows: int, extra: dict | None = None):
    clip_id = str(clip["clip_id"])
    n_frames = int(clip["n_frames"]) if _is_int(clip.get("n_frames")) else 0
    starts, sample_stride = clip_sampling(clip, grid, n_windows, extra)
    frames = []
    for frame in range(n_frames):
        window, tubelet_index = frame_to_token(frame, starts, grid, sample_stride)
        if tubelet_index is None:
            features: list[dict] = []
            shown_t = None
        else:
            features = _features_at(clip_id, window, tubelet_index, grid, act_cells, effect_cells, meta_by_id)
            shown_t = tubelet_index
        frames.append({"frame": frame, "window": window, "t": shown_t, "features": features})
    media = "video" if loaded.kind == "video" and loaded.status == "ok" else "frames" if loaded.kind == "array" and loaded.status == "ok" else "none"
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
        "window_starts": starts,
        "stride": sample_stride,
        "grid": {key: grid[key] for key in ("t", "h", "w", "tubelet", "patch_px") if key in grid},
        "frames": frames,
    }


def _features_at(clip_id, window, t, grid, act_cells, effect_cells, meta_by_id) -> list[dict]:
    height, width = int(grid["h"]), int(grid["w"])
    prefix = (clip_id, window, t)
    feature_ids = {key[3] for key in act_cells if key[:3] == prefix}
    rows = []
    for feature_id in feature_ids:
        sites = act_cells[(clip_id, window, t, feature_id)]
        heatmap = [[0.0 for _ in range(width)] for _ in range(height)]
        activation = 0.0
        for (h, w), value in sites.items():
            activation = max(activation, value)
            if 0 <= h < height and 0 <= w < width:
                heatmap[h][w] = value
        meta = meta_by_id.get(feature_id, {})
        kinds = effect_cells.get((clip_id, window, t, feature_id), {})
        effects = [
            {"effect_kind": kind, "effect": _num(sum(values) / len(values))}
            for kind, values in kinds.items()
            if values
        ]
        effects.sort(key=lambda item: (_KIND_ORDER.get(item["effect_kind"], 2), item["effect_kind"]))
        rows.append(
            {
                "id": feature_id,
                "label": meta.get("label") or "",
                "badge": meta.get("badge") or "unverified",
                "activation": activation,
                "effects": effects,
                "heatmap": heatmap,
            }
        )
    rows.sort(key=lambda row: (-row["activation"], row["id"]))
    return rows


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


def _sha_ok(path: Path | None, expected: str | None) -> bool:
    return path is not None and (expected is None or _file_sha256(path) == expected)


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
    missing = [column for column in columns if column not in table.column_names]
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
