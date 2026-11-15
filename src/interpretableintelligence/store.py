"""On-disk activation rows. A stage reads or writes one chunk at a time.

The float32 cache under ``acts/stream/`` is what SAE training samples. Contract
shards are emitted from that cache in ``ROWS_PER_SHARD`` slices. Neither path
concatenates the run into one tensor.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from interpretableintelligence.artifacts import ROWS_PER_SHARD

STREAM_CHUNK = 2048
_SPLIT_TRAIN = 0
_SPLIT_HELD = 1


def rss_bytes() -> int:
    """Current resident set, from ``/proc``.

    Platforms without ``/proc`` (macOS) report the process's peak resident set
    from ``getrusage`` instead, which is an upper bound on the stage peak.
    0 only when neither is available.
    """
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return _peak_rss_bytes()
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return _peak_rss_bytes()


def _peak_rss_bytes() -> int:
    try:
        import resource
    except ImportError:  # Windows
        return 0
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024  # bytes on macOS, KiB elsewhere


class RamSampler:
    """Peak resident set while a block of work runs."""

    def __init__(self) -> None:
        self.peak = rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="interpretableintelligence-rss", daemon=True)

    def start(self) -> None:
        self.peak = max(self.peak, rss_bytes())
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak = max(self.peak, rss_bytes())
        return int(self.peak)

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, rss_bytes())


def _close_mmap(array: np.ndarray) -> None:
    mapped = getattr(array, "_mmap", None)
    if mapped is not None:
        mapped.close()


class ActivationWriter:
    """Append windows to a float32 row file, then emit contract shards."""

    def __init__(self, acts_dir: Path, d_model: int) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.acts_dir = Path(acts_dir)
        self.d_model = int(d_model)
        self.stream = self.acts_dir / "stream"
        self.n = 0
        self._handles: dict[str, object] = {}

    def __enter__(self) -> "ActivationWriter":
        self.stream.mkdir(parents=True, exist_ok=True)
        self.acts_dir.mkdir(parents=True, exist_ok=True)
        self._handles = {
            "rows": (self.stream / "rows.f32").open("wb"),
            "clip_idx": (self.stream / "clip_idx.i32").open("wb"),
            "window": (self.stream / "window.i16").open("wb"),
            "t": (self.stream / "t.i16").open("wb"),
            "h": (self.stream / "h.i16").open("wb"),
            "w": (self.stream / "w.i16").open("wb"),
            "split": (self.stream / "split.u8").open("wb"),
        }
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._close_handles()

    def append(self, acts: np.ndarray, clip_index: int, window: int, split_code: int) -> None:
        """Write one ``[t, h, w, d]`` window and drop it. ``split_code`` is 0 or 1."""
        array = np.asarray(acts, dtype=np.float32)
        if array.ndim != 4 or int(array.shape[-1]) != self.d_model:
            raise ValueError(f"expected [t, h, w, {self.d_model}], got {array.shape}")
        t, h, w, _d = (int(dim) for dim in array.shape)
        if min(t, h, w) <= 0:
            raise ValueError(f"empty window {array.shape}")
        if t > 32767 or h > 32767 or w > 32767 or window > 32767 or window < 0:
            raise ValueError("token coordinates do not fit in int16")
        if split_code not in (_SPLIT_TRAIN, _SPLIT_HELD):
            raise ValueError(f"split_code must be 0 or 1, got {split_code}")
        n = t * h * w
        flat = np.ascontiguousarray(array.reshape(n, self.d_model), dtype=np.float32)
        tt, hh, ww = _raster(t, h, w)
        self._handles["rows"].write(flat.tobytes())  # type: ignore[attr-defined]
        self._handles["clip_idx"].write(np.full(n, int(clip_index), dtype=np.int32).tobytes())  # type: ignore[attr-defined]
        self._handles["window"].write(np.full(n, int(window), dtype=np.int16).tobytes())  # type: ignore[attr-defined]
        self._handles["t"].write(tt.tobytes())  # type: ignore[attr-defined]
        self._handles["h"].write(hh.tobytes())  # type: ignore[attr-defined]
        self._handles["w"].write(ww.tobytes())  # type: ignore[attr-defined]
        self._handles["split"].write(np.full(n, int(split_code), dtype=np.uint8).tobytes())  # type: ignore[attr-defined]
        self.n += n

    def append_rows(
        self,
        acts: np.ndarray,
        clip_index: int,
        window: int,
        cells: np.ndarray,
        split_code: int,
    ) -> None:
        """Write rows ``[n, d]`` at explicit grid ``cells`` ``[n, 3]`` (t, h, w).

        For captures that recorded only part of the grid (an imported run lists
        those cells in ``manifest.import.cells``). ``append`` is the full-window form.
        """
        array = np.asarray(acts, dtype=np.float32)
        coords = np.asarray(cells, dtype=np.int64)
        if array.ndim != 2 or int(array.shape[1]) != self.d_model:
            raise ValueError(f"expected [n, {self.d_model}], got {array.shape}")
        n = int(array.shape[0])
        if n <= 0:
            raise ValueError("empty window")
        if coords.shape != (n, 3):
            raise ValueError(f"expected cells [{n}, 3], got {coords.shape}")
        if coords.min() < 0 or coords.max() > 32767 or window > 32767 or window < 0:
            raise ValueError("token coordinates do not fit in int16")
        if split_code not in (_SPLIT_TRAIN, _SPLIT_HELD):
            raise ValueError(f"split_code must be 0 or 1, got {split_code}")
        self._handles["rows"].write(np.ascontiguousarray(array).tobytes())  # type: ignore[attr-defined]
        self._handles["clip_idx"].write(np.full(n, int(clip_index), dtype=np.int32).tobytes())  # type: ignore[attr-defined]
        self._handles["window"].write(np.full(n, int(window), dtype=np.int16).tobytes())  # type: ignore[attr-defined]
        for column, name in enumerate(("t", "h", "w")):
            self._handles[name].write(np.ascontiguousarray(coords[:, column], dtype=np.int16).tobytes())  # type: ignore[attr-defined]
        self._handles["split"].write(np.full(n, int(split_code), dtype=np.uint8).tobytes())  # type: ignore[attr-defined]
        self.n += n

    def finish(self) -> dict:
        """Flush shards and the train/held index. Returns ``{n, d_model, n_train, n_heldout}``."""
        if self.n <= 0:
            raise RuntimeError("no activation rows were captured")
        self._close_handles()
        rows_path = self.stream / "rows.f32"
        expected_bytes = self.n * self.d_model * 4
        if rows_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"row cache is {rows_path.stat().st_size} bytes, expected {expected_bytes}")
        n_train, n_held = _write_split_index(self.stream, self.n)
        _write_shards(self.acts_dir, self.stream, self.n, self.d_model)
        meta = {
            "n": int(self.n),
            "d_model": int(self.d_model),
            "n_train": int(n_train),
            "n_heldout": int(n_held),
        }
        (self.stream / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return meta

    def _close_handles(self) -> None:
        for handle in self._handles.values():
            handle.close()  # type: ignore[attr-defined]
        self._handles = {}


def _raster(t: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = t * h * w
    hw = h * w
    flat = np.arange(n, dtype=np.int32)
    tt = (flat // hw).astype(np.int16)
    rem = flat % hw
    hh = (rem // w).astype(np.int16)
    ww = (rem % w).astype(np.int16)
    return tt, hh, ww


def _write_split_index(stream: Path, n: int) -> tuple[int, int]:
    split = np.memmap(stream / "split.u8", dtype=np.uint8, mode="r", shape=(n,))
    train_path = stream / "train.i64"
    held_path = stream / "held.i64"
    n_train = 0
    n_held = 0
    try:
        with train_path.open("wb") as train_file, held_path.open("wb") as held_file:
            for start in range(0, n, STREAM_CHUNK):
                stop = min(start + STREAM_CHUNK, n)
                part = np.array(split[start:stop], dtype=np.uint8, copy=True)
                train_hits = np.flatnonzero(part == _SPLIT_TRAIN).astype(np.int64) + start
                held_hits = np.flatnonzero(part == _SPLIT_HELD).astype(np.int64) + start
                if train_hits.size:
                    train_file.write(train_hits.tobytes())
                    n_train += int(train_hits.size)
                if held_hits.size:
                    held_file.write(held_hits.tobytes())
                    n_held += int(held_hits.size)
    finally:
        _close_mmap(split)
    if n_train + n_held != n:
        raise RuntimeError(f"split index covers {n_train + n_held} rows, expected {n}")
    return n_train, n_held


def _write_shards(acts_dir: Path, stream: Path, n: int, d_model: int) -> None:
    rows = np.memmap(stream / "rows.f32", dtype=np.float32, mode="r", shape=(n, d_model))
    clip_idx = np.memmap(stream / "clip_idx.i32", dtype=np.int32, mode="r", shape=(n,))
    window = np.memmap(stream / "window.i16", dtype=np.int16, mode="r", shape=(n,))
    t = np.memmap(stream / "t.i16", dtype=np.int16, mode="r", shape=(n,))
    h = np.memmap(stream / "h.i16", dtype=np.int16, mode="r", shape=(n,))
    w = np.memmap(stream / "w.i16", dtype=np.int16, mode="r", shape=(n,))
    try:
        shard = 0
        for start in range(0, n, ROWS_PER_SHARD):
            stop = min(start + ROWS_PER_SHARD, n)
            path = acts_dir / f"shard-{shard:05d}.safetensors"
            save_file(
                {
                    "acts": np.ascontiguousarray(np.array(rows[start:stop], dtype=np.float32, copy=True), dtype=np.float16),
                    "clip_idx": np.ascontiguousarray(clip_idx[start:stop], dtype=np.int32),
                    "window": np.ascontiguousarray(window[start:stop], dtype=np.int16),
                    "t": np.ascontiguousarray(t[start:stop], dtype=np.int16),
                    "h": np.ascontiguousarray(h[start:stop], dtype=np.int16),
                    "w": np.ascontiguousarray(w[start:stop], dtype=np.int16),
                },
                str(path),
            )
            shard += 1
    finally:
        for array in (rows, clip_idx, window, t, h, w):
            _close_mmap(array)


class RowSource:
    """Read-only memmap of a finished activation cache. Slices copy one chunk."""

    def __init__(self, acts_dir: Path) -> None:
        self.acts_dir = Path(acts_dir)
        meta_path = self.acts_dir / "stream" / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"activation cache is missing {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.n = int(meta["n"])
        self.d_model = int(meta["d_model"])
        self.n_train = int(meta["n_train"])
        self.n_heldout = int(meta["n_heldout"])
        self._open: list[np.ndarray] = []
        self.rows: np.ndarray | None = None
        self.clip_idx: np.ndarray | None = None
        self.window: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self.h: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.split: np.ndarray | None = None
        self.train_ids: np.ndarray | None = None
        self.held_ids: np.ndarray | None = None

    def __enter__(self) -> "RowSource":
        stream = self.acts_dir / "stream"
        self.rows = self._map(stream / "rows.f32", np.float32, (self.n, self.d_model))
        self.clip_idx = self._map(stream / "clip_idx.i32", np.int32, (self.n,))
        self.window = self._map(stream / "window.i16", np.int16, (self.n,))
        self.t = self._map(stream / "t.i16", np.int16, (self.n,))
        self.h = self._map(stream / "h.i16", np.int16, (self.n,))
        self.w = self._map(stream / "w.i16", np.int16, (self.n,))
        self.split = self._map(stream / "split.u8", np.uint8, (self.n,))
        self.train_ids = self._map(stream / "train.i64", np.int64, (self.n_train,)) if self.n_train else np.zeros(0, dtype=np.int64)
        self.held_ids = self._map(stream / "held.i64", np.int64, (self.n_heldout,)) if self.n_heldout else np.zeros(0, dtype=np.int64)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for array in self._open:
            _close_mmap(array)
        self._open.clear()

    def chunks(self, size: int):
        step = max(int(size), 1)
        for start in range(0, self.n, step):
            yield start, min(start + step, self.n)

    def gather_train(self, local_index: np.ndarray) -> np.ndarray:
        """Train-local positions -> float32 rows. ``local_index`` is a small batch."""
        if self.train_ids is None or self.rows is None:
            raise RuntimeError("activation cache is not open")
        chosen = np.asarray(local_index, dtype=np.int64).ravel()
        if chosen.size == 0:
            return np.zeros((0, self.d_model), dtype=np.float32)
        if int(chosen.min()) < 0 or int(chosen.max()) >= self.n_train:
            raise IndexError("train row index out of range")
        global_ids = np.asarray(self.train_ids[chosen], dtype=np.int64)
        return np.ascontiguousarray(self.rows[global_ids], dtype=np.float32)

    def _map(self, path: Path, dtype, shape: tuple[int, ...]) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        self._open.append(array)
        return array


def cache_matches_layout(acts_dir: Path, n_clips: int, grid: dict) -> bool:
    """True when the cache and shards are one rectangular token grid, read in chunks."""
    try:
        meta = json.loads((acts_dir / "stream" / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return False
    n = int(meta.get("n", 0))
    d_model = int(meta.get("d_model", 0))
    if n <= 0 or d_model <= 0 or n_clips <= 0:
        return False
    cells = int(grid["t"]) * int(grid["h"]) * int(grid["w"])
    if cells <= 0 or n % (n_clips * cells) != 0:
        return False
    n_windows = n // (n_clips * cells)
    if n_windows <= 0:
        return False
    rows_path = acts_dir / "stream" / "rows.f32"
    if not rows_path.is_file() or rows_path.stat().st_size != n * d_model * 4:
        return False
    if not _shards_match(acts_dir, n, d_model):
        return False
    return _index_is_raster(acts_dir, n, n_clips, n_windows, grid)


def _shards_match(acts_dir: Path, n: int, d_model: int) -> bool:
    paths = sorted(acts_dir.glob("shard-*.safetensors"))
    expected = [f"shard-{index:05d}.safetensors" for index in range(len(paths))]
    if not paths or [path.name for path in paths] != expected:
        return False
    total = 0
    try:
        for path in paths:
            with safe_open(path, framework="np") as handle:
                shape = tuple(handle.get_slice("acts").get_shape())
                if len(shape) != 2 or int(shape[1]) != d_model:
                    return False
                sample = handle.get_slice("acts")[0:1]
                if sample.dtype != np.float16:
                    return False
                for key in ("clip_idx", "window", "t", "h", "w"):
                    column = tuple(handle.get_slice(key).get_shape())
                    if column != (int(shape[0]),):
                        return False
                total += int(shape[0])
    except (OSError, KeyError, ValueError, RuntimeError):
        return False
    return total == n


def _index_is_raster(acts_dir: Path, n: int, n_clips: int, n_windows: int, grid: dict) -> bool:
    t_count = int(grid["t"])
    h_count = int(grid["h"])
    w_count = int(grid["w"])
    cells = t_count * h_count * w_count
    span = n_windows * cells
    stream = acts_dir / "stream"
    try:
        clip_idx = np.memmap(stream / "clip_idx.i32", dtype=np.int32, mode="r", shape=(n,))
        window = np.memmap(stream / "window.i16", dtype=np.int16, mode="r", shape=(n,))
        t = np.memmap(stream / "t.i16", dtype=np.int16, mode="r", shape=(n,))
        h = np.memmap(stream / "h.i16", dtype=np.int16, mode="r", shape=(n,))
        w = np.memmap(stream / "w.i16", dtype=np.int16, mode="r", shape=(n,))
    except (OSError, ValueError):
        return False
    try:
        for start in range(0, n, STREAM_CHUNK):
            stop = min(start + STREAM_CHUNK, n)
            index = np.arange(start, stop, dtype=np.int64)
            exp_clip = index // span
            exp_window = (index % span) // cells
            exp_flat = index % cells
            got_flat = (
                np.array(t[start:stop], dtype=np.int64, copy=True) * h_count
                + np.array(h[start:stop], dtype=np.int64, copy=True)
            ) * w_count + np.array(w[start:stop], dtype=np.int64, copy=True)
            if not np.array_equal(np.array(clip_idx[start:stop], copy=True), exp_clip):
                return False
            if not np.array_equal(np.array(window[start:stop], dtype=np.int64, copy=True), exp_window):
                return False
            if not np.array_equal(got_flat, exp_flat):
                return False
            if int(exp_clip.max()) >= n_clips:
                return False
    finally:
        for array in (clip_idx, window, t, h, w):
            _close_mmap(array)
    return True
