"""Cut short LIBERO-10 clips that every adapter can read, with GR00T sidecars.

Each clip is 16 frames (1.6 s at 10 fps) of the agent-view camera, 256 x 256,
so it is exactly one GR00T observation window (16 executed steps), one V-JEPA 2
window (16 frames), and one DINOv2 window (8 frames at stride 2). All three
models therefore see the same moment, which is what ``ii compare`` needs.

    uv run python scripts/make_libero_clips.py --out clips/libero-moments

Downloads ``--files`` parquet shards of ``lerobot/libero_10_image`` (about
100 MB each), spread across the dataset so every task appears, and takes
``--per-episode`` evenly spaced moments from each episode.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

REPO = "lerobot/libero_10_image"
N_FILES = 147
FRAMES = 16
FPS = 10.0


def _decode(cell) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--files", type=int, default=15, help="Parquet shards to read, spread evenly over the dataset.")
    parser.add_argument("--per-episode", type=int, default=4, help="Moments cut from each episode.")
    args = parser.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    tasks_path = hf_hub_download(REPO, "meta/tasks.parquet", repo_type="dataset")
    tasks = pd.read_parquet(tasks_path)
    task_names = {int(index): str(name) for name, index in zip(tasks.index, tasks["task_index"])}

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    shards = np.unique(np.linspace(0, N_FILES - 1, args.files).round().astype(int))
    for shard in shards:
        path = hf_hub_download(REPO, f"data/chunk-000/file-{shard:03d}.parquet", repo_type="dataset")
        frame = pd.read_parquet(path)
        for episode, rows in frame.groupby("episode_index", sort=True):
            rows = rows.sort_values("frame_index")
            n = len(rows)
            if n < FRAMES * 2:
                continue
            starts = np.linspace(0, n - FRAMES, args.per_episode + 2)[1:-1].round().astype(int)
            for moment, start in enumerate(starts):
                part = rows.iloc[start : start + FRAMES]
                clip_id = f"ep{int(episode):03d}-m{moment}"
                images = np.stack([_decode(cell) for cell in part["observation.images.image"]])
                wrist = np.stack([_decode(cell) for cell in part["observation.images.wrist_image"]])
                np.save(out / f"{clip_id}.npy", images)
                np.save(out / f"{clip_id}.wrist.npy", wrist)
                sidecar = {
                    "task": task_names[int(part["task_index"].iloc[0])],
                    "state": np.stack(part["observation.state"].to_numpy()).astype(float).round(6).tolist(),
                    "wrist": f"{clip_id}.wrist.npy",
                    "source": {"repo": REPO, "episode": int(episode), "start_frame": int(part["frame_index"].iloc[0])},
                }
                (out / f"{clip_id}.groot.json").write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
                entries.append({"clip_id": clip_id, "file": f"{clip_id}.npy", "fps": FPS})
        print(f"shard {shard}: {len(entries)} clips so far", flush=True)
    index = {"fps": FPS, "source": REPO, "n_frames": FRAMES, "clips": entries}
    (out / "clips.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} clips to {out}")


if __name__ == "__main__":
    main()
