"""Real sample clips for a first run on a real model.

A newcomer who has finished ``ii try`` usually has no folder of video to hand. ``ii get-clips``
downloads one camera of ``lerobot/droid_100`` (MIT licence: 100 real robot episodes in real kitchens
and offices, one 175 MB file) and cuts a short clip from the middle of each episode, where the arm is
working. Each clip keeps its task instruction in ``clips.json``.

With ``with_states`` each clip also gets ``<clip>.states.npy``: the arm's 7-D pose for every frame of
the mp4 (``x y z`` in metres in the robot base frame, extrinsic ``xyz`` Euler angles, gripper 0 open
to 1 closed), read from the dataset's ``observation.state`` at the nearest timestamp. That is the
state V-JEPA 2-AC conditions on (``ii action-effects``).

The dataset stores every episode of a camera back to back in one AV1 file, with each episode's start
and end time in ``meta/episodes``. Clips are re-encoded as H.264 mp4 so any browser plays them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

DATASET = "lerobot/droid_100"
CAMERA = "observation.images.exterior_image_1_left"
LICENSE = "MIT"
DEFAULT_N = 100
DEFAULT_SECONDS = 4.0
INDEX = "clips.json"
STATES_SUFFIX = ".states.npy"


def _episodes() -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset")
    columns = [
        "episode_index",
        "tasks",
        f"videos/{CAMERA}/chunk_index",
        f"videos/{CAMERA}/file_index",
        f"videos/{CAMERA}/from_timestamp",
        f"videos/{CAMERA}/to_timestamp",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    return [
        {
            "episode": int(r["episode_index"]),
            "task": (r["tasks"] or [""])[0],
            "file": (int(r[f"videos/{CAMERA}/chunk_index"]), int(r[f"videos/{CAMERA}/file_index"])),
            "start": float(r[f"videos/{CAMERA}/from_timestamp"]),
            "end": float(r[f"videos/{CAMERA}/to_timestamp"]),
        }
        for r in rows
    ]


def _episode_states() -> dict[int, tuple]:
    """``episode -> (timestamps [N], states [N, 7])`` from the dataset's frame table."""
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, "data/chunk-000/file-000.parquet", repo_type="dataset")
    table = pq.read_table(path, columns=["episode_index", "timestamp", "observation.state"])
    episodes = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
    times = np.asarray(table.column("timestamp").to_numpy(), dtype=np.float64)
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    out = {}
    for ep in np.unique(episodes):
        rows = np.flatnonzero(episodes == ep)
        order = rows[np.argsort(times[rows])]
        out[int(ep)] = (times[order], states[order])
    return out


def states_at(times, states, query):
    """Rows of ``states`` at the timestamps nearest to each of ``query``."""
    import numpy as np

    times = np.asarray(times, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    right = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    left = right - 1
    pick = np.where(np.abs(times[left] - query) <= np.abs(times[right] - query), left, right)
    return np.asarray(states)[pick]


def clips_ready(out_dir: Path, n: int, with_states: bool = False) -> bool:
    """True when ``out_dir`` already holds the ``n`` clips its index lists (and their states)."""
    index = Path(out_dir) / INDEX
    try:
        entries = json.loads(index.read_text()).get("clips", [])
    except (OSError, ValueError, AttributeError):
        return False
    files = [e.get("file", "") for e in entries]
    if with_states:
        files += [e.get("states", "") for e in entries]
    return len(entries) == n and all(f and (Path(out_dir) / f).is_file() for f in files)


def _cut(src, out_path: Path, start: float, end: float, times: list | None = None) -> int:
    """Decode ``[start, end)`` seconds of ``src`` into an H.264 mp4. Returns frames written.

    ``times``, when given, collects each written frame's time in the source file.
    """
    import av

    stream = src.streams.video[0]
    rate = stream.average_rate
    src.seek(int(max(start - 1.0, 0.0) / stream.time_base), stream=stream, backward=True)
    written = 0
    with av.open(str(out_path), "w") as dst:
        video = dst.add_stream("libx264", rate=rate)
        video.width = stream.codec_context.width
        video.height = stream.codec_context.height
        video.pix_fmt = "yuv420p"
        video.time_base = 1 / rate
        video.options = {"crf": "20", "preset": "veryfast", "movflags": "+faststart"}
        for frame in src.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start - 1e-3:
                continue
            if t >= end - 1e-3:
                break
            if times is not None:
                times.append(t)
            image = frame.reformat(format="yuv420p")
            image.pts = written
            image.time_base = 1 / rate
            for packet in video.encode(image):
                dst.mux(packet)
            written += 1
        for packet in video.encode():
            dst.mux(packet)
    return written


def fetch_droid_clips(
    out_dir: Path,
    n: int = DEFAULT_N,
    seconds: float = DEFAULT_SECONDS,
    say: Callable[[str], None] = print,
    with_states: bool = False,
) -> list[dict]:
    """Write ``n`` mp4 clips of ``seconds`` each (one per episode) to ``out_dir``. Idempotent.

    ``with_states`` also writes each clip's per-frame arm pose (module docstring).
    """
    import av
    import numpy as np
    from huggingface_hub import hf_hub_download

    out_dir = Path(out_dir)
    if clips_ready(out_dir, n, with_states=with_states):
        return json.loads((out_dir / INDEX).read_text())["clips"]
    say(f"     Reading the episode list of {DATASET} from huggingface.co. No account is needed: a warning")
    say("     about HF_TOKEN only concerns rate limits and can be ignored.")
    episodes = _episodes()
    if not 1 <= n <= len(episodes):
        raise ValueError(f"{DATASET} has {len(episodes)} episodes; pass --n between 1 and {len(episodes)}")
    # Spread the picks over the whole dataset so a small n still sees many scenes.
    picks = [episodes[round(i * (len(episodes) - 1) / max(n - 1, 1))] for i in range(n)] if n > 1 else episodes[:1]
    out_dir.mkdir(parents=True, exist_ok=True)
    states_by_episode = _episode_states() if with_states else {}

    sources: dict[tuple[int, int], object] = {}
    entries = []
    try:
        for i, ep in enumerate(picks):
            key = ep["file"]
            if key not in sources:
                say(f"     Downloading {DATASET} (one camera, about 175 MB, cached after the first time)")
                path = hf_hub_download(
                    DATASET, f"videos/{CAMERA}/chunk-{key[0]:03d}/file-{key[1]:03d}.mp4", repo_type="dataset"
                )
                sources[key] = av.open(path)
            length = ep["end"] - ep["start"]
            span = min(seconds, length)
            start = ep["start"] + (length - span) / 2
            name = f"droid-{ep['episode']:03d}.mp4"
            times: list[float] = []
            frames = _cut(sources[key], out_dir / name, start, start + span, times)
            entry = {"clip_id": name[:-4], "file": name, "episode": ep["episode"], "task": ep["task"], "frames": frames}
            if with_states:
                if ep["episode"] not in states_by_episode:
                    raise RuntimeError(f"{DATASET} has no state rows for episode {ep['episode']}")
                ep_times, ep_states = states_by_episode[ep["episode"]]
                rel = np.asarray(times) - ep["start"]
                entry["states"] = name[:-4] + STATES_SUFFIX
                np.save(out_dir / entry["states"], states_at(ep_times, ep_states, rel).astype(np.float32))
            entries.append(entry)
            if (i + 1) % 20 == 0 or i + 1 == len(picks):
                say(f"     Cut {i + 1}/{len(picks)} clips")
    finally:
        for src in sources.values():
            src.close()
    index = {"source": DATASET, "camera": CAMERA, "license": LICENSE, "seconds": seconds, "clips": entries}
    if with_states:
        index["states"] = "observation.state per mp4 frame: x y z (m), extrinsic xyz Euler (rad), gripper 0-1"
    (out_dir / INDEX).write_text(json.dumps(index, indent=2) + "\n")
    return entries
