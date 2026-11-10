"""Read-API payload the Rust server matches.

``interpretableintelligence view`` is :mod:`interpretableintelligence.view`. This module is the earlier frame and
feature shape that ``interpretableintelligence.api`` and ``interpretableintelligence-server`` still agree on.


The only run files this module reads are ``manifest.json``,
``features/feature_acts.parquet``, ``features/clip_effects.parquet``,
``features/features.jsonl``, and ``scoreboard.json``. Clip pixels come from a
path recorded on the manifest, or from a directory passed as ``--clips``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import threading
import webbrowser
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pyarrow.parquet as pq

from interpretableintelligence.artifacts import read_json, read_jsonl

STATIC_DIR = Path(__file__).resolve().parent / "static"

_REQUIRED = (
    "manifest.json",
    "features/feature_acts.parquet",
    "features/clip_effects.parquet",
    "features/features.jsonl",
    "scoreboard.json",
)
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
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".json": "application/json",
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


def frame_index(frame: int, grid: dict) -> tuple[int, int]:
    """Map a clip frame to ``(window, t)``.

    One window is ``grid.t`` bins of ``grid.tubelet`` frames. Further windows
    on a longer clip follow back to back. The run folder does not store window
    boundaries, so this is the layout the contract describes.
    """
    tubelet = max(int(grid["tubelet"]), 1)
    bins = max(int(grid["t"]), 1)
    span = bins * tubelet
    window = int(frame) // span
    t = (int(frame) % span) // tubelet
    return window, t


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
    print(f"interpretableintelligence view  {url}", flush=True)
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
    frames: dict[str, np.ndarray] = field(default_factory=dict)
    peaks: dict[str, float] = field(default_factory=dict)
    _png: dict[tuple[str, int], bytes] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def frame_png(self, clip_id: str, frame: int) -> bytes | None:
        clip = self.clips.get(clip_id)
        array = self.frames.get(clip_id)
        if clip is None or array is None or clip["pixels"] != "ok":
            return None
        if frame < 0 or frame >= int(array.shape[0]):
            return None
        key = (clip_id, frame)
        with self._lock:
            cached = self._png.get(key)
            if cached is None:
                image = render_frame(array[frame], self.peaks.get(clip_id, 1.0))
                scale = _display_scale(image.shape[0], image.shape[1])
                cached = encode_png(_upsample(image, scale))
                self._png[key] = cached
            return cached


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
    scoreboard = read_json(run_dir / "scoreboard.json")
    jsonl = read_jsonl(run_dir / "features" / "features.jsonl")
    _require_keys(manifest, ("run_id", "model", "layer", "grid", "clips"), "manifest")
    _require_keys(manifest["grid"], ("t", "h", "w", "tubelet", "patch_px"), "manifest.grid")
    _require_keys(manifest["model"], ("name", "adapter"), "manifest.model")
    _require_keys(scoreboard, ("run", "sae", "features"), "scoreboard")
    _require_keys(scoreboard["run"], ("n_train_clips", "n_heldout_clips"), "scoreboard.run")
    _require_keys(scoreboard["sae"], ("usable", "ev_heldout", "loss_recovered"), "scoreboard.sae")

    acts = pq.read_table(run_dir / "features" / "feature_acts.parquet")
    effects = pq.read_table(run_dir / "features" / "clip_effects.parquet")
    _require_columns(
        acts,
        ("feature_id", "clip_id", "window", "t", "h", "w", "value"),
        "feature_acts.parquet",
    )
    _require_columns(
        effects,
        ("feature_id", "clip_id", "window", "t", "effect", "effect_kind"),
        "clip_effects.parquet",
    )

    grid = {key: int(manifest["grid"][key]) for key in ("t", "h", "w", "tubelet", "patch_px")}
    score_by_id = _score_features(scoreboard["features"])
    label_model = {int(row["id"]): row.get("label_model") for row in jsonl if "id" in row}
    act_cells = _act_cells(acts)
    effect_cells = _effect_cells(effects)
    pixels = _resolve_pixels(manifest, run_dir, None if clips_dir is None else Path(clips_dir))
    top_clips = _top_clips(act_cells, manifest, grid)

    features = []
    for feature_id in sorted(score_by_id):
        row = score_by_id[feature_id]
        record = {
            "id": feature_id,
            "label": row["label"] if isinstance(row["label"], str) else "",
            "label_model": label_model.get(feature_id),
            "badge": badge_for(row["real"]),
            "top_clips": top_clips.get(feature_id, []),
        }
        for key in _SCORE_FIELDS:
            record[key] = _json_value(row[key])
        features.append(record)

    meta_by_id = {feature["id"]: feature for feature in features}
    clip_payloads: dict[str, dict] = {}
    frames: dict[str, np.ndarray] = {}
    peaks: dict[str, float] = {}
    for clip in manifest["clips"]:
        clip_id = str(clip["clip_id"])
        loaded = pixels.get(clip_id)
        payload, array, peak = _clip_payload(clip, grid, loaded, act_cells, effect_cells, meta_by_id)
        clip_payloads[clip_id] = payload
        if array is not None:
            frames[clip_id] = array
            peaks[clip_id] = peak

    model = manifest["model"]
    run_payload = {
        "run_id": manifest["run_id"],
        "model": {
            "adapter": model.get("adapter"),
            "name": model.get("name"),
            "sha": model.get("sha"),
        },
        "layer": int(manifest["layer"]),
        "grid": grid,
        "split": {
            "n_train": int(scoreboard["run"]["n_train_clips"]),
            "n_heldout": int(scoreboard["run"]["n_heldout_clips"]),
        },
        "sae": {
            "usable": scoreboard["sae"]["usable"] is True,
            "ev_heldout": _num(scoreboard["sae"]["ev_heldout"]),
            "loss_recovered": _num(scoreboard["sae"]["loss_recovered"]),
        },
        "clips": [
            {
                "clip_id": payload["clip_id"],
                "fps": payload["fps"],
                "n_frames": payload["n_frames"],
                "split": payload["split"],
                "pixels": payload["pixels"],
            }
            for payload in clip_payloads.values()
        ],
        "features": features,
    }
    return RunModel(run_payload=run_payload, clips=clip_payloads, frames=frames, peaks=peaks)


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

        def do_GET(self) -> None:  # noqa: N802
            self.close_connection = True
            path = unquote(urlparse(self.path).path)
            try:
                self._route(path)
            except BrokenPipeError:
                return
            except Exception as exc:  # noqa: BLE001
                body = json.dumps({"error": str(exc)}).encode()
                self._send(body, "application/json", status=500)

        def _route(self, path: str) -> None:
            if path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html")
                return
            if path.startswith("/static/"):
                self._send_static(path[len("/static/") :])
                return
            if path == "/api/run":
                self._send_json(model.run_payload)
                return
            frame_match = _FRAME_PNG.match(path)
            if frame_match:
                clip_id = frame_match.group("clip")
                frame = int(frame_match.group("frame"))
                png = model.frame_png(clip_id, frame)
                if png is None:
                    self._send_json({"error": "frame not available"}, status=404)
                    return
                self._send(png, "image/png", cache="private, max-age=3600")
                return
            clip_match = _CLIP_ID.match(path)
            if clip_match:
                payload = model.clips.get(clip_match.group("clip"))
                if payload is None:
                    self._send_json({"error": "unknown clip"}, status=404)
                    return
                self._send_json(payload)
                return
            self._send_json({"error": "not found"}, status=404)

        def _send_static(self, relative: str) -> None:
            candidate = _safe_static(relative)
            if candidate is None:
                self._send_json({"error": "not found"}, status=404)
                return
            self._send_file(candidate)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_json({"error": "not found"}, status=404)
                return
            content_type = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
            cache = "public, max-age=86400" if path.suffix.lower() == ".woff2" else "no-cache"
            self._send(path.read_bytes(), content_type, cache=cache)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
            self._send(body, "application/json", status=status)

        def _send(self, body: bytes, content_type: str, status: int = 200, cache: str = "no-cache") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

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


def _score_features(rows: list) -> dict[int, dict]:
    if not isinstance(rows, list):
        raise ValueError("scoreboard.features must be a list")
    scored: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError("scoreboard feature is missing id")
        feature_id = int(row["id"])
        missing = [key for key in ("label", *_SCORE_FIELDS) if key not in row]
        if missing:
            raise ValueError(f"scoreboard feature {feature_id} is missing {', '.join(missing)}")
        scored[feature_id] = row
    return scored


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


def _top_clips(cells: dict, manifest: dict, grid: dict) -> dict[int, list[dict]]:
    splits = {str(clip["clip_id"]): clip.get("split") for clip in manifest["clips"]}
    tubelet = max(int(grid["tubelet"]), 1)
    span = max(int(grid["t"]), 1) * tubelet
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
        grouped.setdefault(feature_id, []).append(
            {
                "clip_id": clip_id,
                "split": splits.get(clip_id),
                "window": window,
                "t": t,
                "h": h,
                "w": w,
                "value": value,
                "frame": window * span + t * tubelet,
            }
        )
    for feature_id, rows in grouped.items():
        rows.sort(key=lambda row: (-row["value"], row["clip_id"]))
        del rows[20:]
        grouped[feature_id] = rows
    return grouped


@dataclass
class _LoadedClip:
    status: str
    message: str
    array: np.ndarray | None = None
    peak: float = 1.0
    path: Path | None = None


def _resolve_pixels(manifest: dict, run_dir: Path, clips_dir: Path | None) -> dict[str, _LoadedClip]:
    from_dir = _index_clip_dir(clips_dir) if clips_dir is not None else {}
    manifest_dir = None if clips_dir is not None else _manifest_clips_dir(manifest, run_dir)
    if manifest_dir is not None and clips_dir is None:
        from_dir = _index_clip_dir(manifest_dir)
    resolved: dict[str, _LoadedClip] = {}
    sha_index: dict[str, Path] | None = None
    for clip in manifest["clips"]:
        clip_id = str(clip["clip_id"])
        path = _manifest_clip_path(clip, run_dir) if clips_dir is None else None
        if path is None:
            path = from_dir.get(clip_id)
        if path is None and clips_dir is not None:
            if sha_index is None:
                sha_index = _index_by_sha(clips_dir)
            expected = clip.get("sha256")
            if isinstance(expected, str):
                path = sha_index.get(expected)
        resolved[clip_id] = _load_clip_file(path, clip)
    return resolved


def _load_clip_file(path: Path | None, clip: dict) -> _LoadedClip:
    if path is None or not path.is_file():
        return _LoadedClip("missing", "Clip file not found. Pass --clips, or record a path in the manifest.")
    expected = clip.get("sha256")
    digest = _file_sha256(path)
    if isinstance(expected, str) and digest != expected:
        return _LoadedClip("sha_mismatch", "Clip file does not match the sha256 recorded in the manifest.")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        return _LoadedClip("unreadable", f"Clip file could not be read: {exc}")
    array = np.asarray(array)
    if array.ndim not in (3, 4) or array.shape[0] <= 0:
        return _LoadedClip("unreadable", f"Clip array must be T×H×W or T×H×W×C, got {array.shape}.")
    n_frames = int(clip.get("n_frames") or 0)
    if n_frames and int(array.shape[0]) != n_frames:
        return _LoadedClip(
            "length_mismatch",
            f"Clip file has {array.shape[0]} frames; the manifest says {n_frames}.",
        )
    frames = np.ascontiguousarray(array, dtype=np.float32)
    return _LoadedClip("ok", "", frames, _clip_peak(frames), path)


def _clip_payload(clip, grid, loaded: _LoadedClip | None, act_cells, effect_cells, meta_by_id):
    loaded = loaded or _LoadedClip("missing", "Clip file not found. Pass --clips, or record a path in the manifest.")
    clip_id = str(clip["clip_id"])
    n_frames = int(clip["n_frames"])
    height = int(loaded.array.shape[1]) if loaded.array is not None else None
    width = int(loaded.array.shape[2]) if loaded.array is not None else None
    frames = []
    for frame in range(n_frames):
        window, t = frame_index(frame, grid)
        features = _features_at(clip_id, window, t, grid, act_cells, effect_cells, meta_by_id)
        frames.append({"frame": frame, "window": window, "t": t, "features": features})
    payload = {
        "clip_id": clip_id,
        "fps": float(clip.get("fps") or 0.0),
        "n_frames": n_frames,
        "split": clip.get("split"),
        "pixels": loaded.status,
        "message": loaded.message,
        "width": width,
        "height": height,
        "grid": grid,
        "frames": frames,
    }
    array = loaded.array if loaded.status == "ok" else None
    return payload, array, loaded.peak


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


def _index_clip_dir(directory: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    index = directory / "clips.json"
    if index.is_file():
        meta = json.loads(index.read_text(encoding="utf-8"))
        for entry in meta.get("clips") or []:
            clip_id = entry.get("clip_id")
            file_name = entry.get("file")
            if clip_id and file_name:
                candidate = directory / str(file_name)
                if candidate.is_file():
                    mapping[str(clip_id)] = candidate
    if mapping:
        return mapping
    for file in sorted(directory.glob("*.npy")):
        mapping[file.stem] = file
    return mapping


def _index_by_sha(directory: Path) -> dict[str, Path]:
    found = {}
    for file in sorted(directory.glob("*.npy")):
        if file.is_file():
            found[_file_sha256(file)] = file
    return found


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
