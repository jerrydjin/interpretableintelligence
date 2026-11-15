"""Read and write the run-folder contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import load_file, save_file

ROWS_PER_SHARD = 50_000

FEATURE_ACT_COLUMNS = ("feature_id", "clip_id", "window", "t", "h", "w", "value")
CLIP_EFFECT_COLUMNS = ("feature_id", "clip_id", "window", "t", "effect", "effect_kind")
ATTRIBUTION_COLUMNS = ("feature_id", "clip_id", "score")
ROLLOUT_COLUMNS = (
    "episode_id",
    "task_id",
    "seed",
    "condition",
    "feature_ids",
    "layer",
    "kind",
    "value",
    "success",
    "steps",
    "wall_time_s",
    "dictionary",
)
# Written before the random-weights SAE control: no ``dictionary`` column, every edit on the trained SAE.
ROLLOUT_COLUMNS_V1_0 = ROLLOUT_COLUMNS[:-1]


def write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically so a crash cannot leave a half-written contract file."""
    text = json.dumps(payload, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not JSON") from exc
    return rows


def write_activation_shards(directory: Path, acts: np.ndarray, tokens: list) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    n = int(acts.shape[0])
    clip_idx = np.asarray([token.clip_index for token in tokens], dtype=np.int32)
    window = np.asarray([token.window for token in tokens], dtype=np.int16)
    t = np.asarray([token.t for token in tokens], dtype=np.int16)
    h = np.asarray([token.h for token in tokens], dtype=np.int16)
    w = np.asarray([token.w for token in tokens], dtype=np.int16)
    paths = []
    shard = 0
    for start in range(0, n, ROWS_PER_SHARD):
        stop = min(start + ROWS_PER_SHARD, n)
        path = directory / f"shard-{shard:05d}.safetensors"
        save_file(
            {
                "acts": np.ascontiguousarray(acts[start:stop].astype(np.float16)),
                "clip_idx": np.ascontiguousarray(clip_idx[start:stop]),
                "window": np.ascontiguousarray(window[start:stop]),
                "t": np.ascontiguousarray(t[start:stop]),
                "h": np.ascontiguousarray(h[start:stop]),
                "w": np.ascontiguousarray(w[start:stop]),
            },
            str(path),
        )
        paths.append(path)
        shard += 1
    return paths


def load_activation_shards(directory: Path) -> dict[str, np.ndarray]:
    paths = sorted(directory.glob("shard-*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no activation shards in {directory}")
    parts = [load_file(str(path)) for path in paths]
    keys = ("acts", "clip_idx", "window", "t", "h", "w")
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def write_feature_acts(path: Path, rows: dict[str, list]) -> None:
    table = pa.table(
        {
            "feature_id": pa.array(rows["feature_id"], type=pa.int32()),
            "clip_id": pa.array(rows["clip_id"], type=pa.string()),
            "window": pa.array(rows["window"], type=pa.int16()),
            "t": pa.array(rows["t"], type=pa.int16()),
            "h": pa.array(rows["h"], type=pa.int16()),
            "w": pa.array(rows["w"], type=pa.int16()),
            "value": pa.array(rows["value"], type=pa.float32()),
        }
    )
    pq.write_table(table, path)


def write_rollout_episodes(path: Path, rows: list[dict]) -> None:
    """Write one row per episode. Replaces the file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "episode_id": pa.array([row["episode_id"] for row in rows], type=pa.string()),
            "task_id": pa.array([row["task_id"] for row in rows], type=pa.string()),
            "seed": pa.array([row["seed"] for row in rows], type=pa.int64()),
            "condition": pa.array([row["condition"] for row in rows], type=pa.string()),
            "feature_ids": pa.array(
                [list(row["feature_ids"]) for row in rows],
                type=pa.list_(pa.int32()),
            ),
            "layer": pa.array([row["layer"] for row in rows], type=pa.int32()),
            "kind": pa.array([row["kind"] for row in rows], type=pa.string()),
            "value": pa.array([row["value"] for row in rows], type=pa.float64()),
            "success": pa.array([row["success"] for row in rows], type=pa.bool_()),
            "steps": pa.array([row["steps"] for row in rows], type=pa.int32()),
            "wall_time_s": pa.array([row["wall_time_s"] for row in rows], type=pa.float64()),
            "dictionary": pa.array([_row_dictionary(row) for row in rows], type=pa.string()),
        }
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def _row_dictionary(row: dict) -> str | None:
    """Null for baseline, ``trained`` for an edit of the policy's SAE, else ``random:<seed>``."""
    if row["condition"] == "baseline":
        return None
    return row.get("dictionary") or "trained"


def read_rollout_episodes(path: Path) -> list[dict]:
    """Rows as written by :func:`write_rollout_episodes`. A file from before the ``dictionary`` column still reads."""
    table = pq.read_table(path)
    columns = tuple(table.column_names)
    if columns not in (ROLLOUT_COLUMNS, ROLLOUT_COLUMNS_V1_0):
        raise ValueError(f"rollout columns must be {ROLLOUT_COLUMNS}, got {columns}")
    frame = table.to_pydict()
    rows = []
    for index in range(table.num_rows):
        feature_ids = frame["feature_ids"][index] or []
        layer = frame["layer"][index]
        kind = frame["kind"][index]
        value = frame["value"][index]
        rows.append(
            {
                "episode_id": str(frame["episode_id"][index]),
                "task_id": str(frame["task_id"][index]),
                "seed": int(frame["seed"][index]),
                "condition": str(frame["condition"][index]),
                "feature_ids": [int(item) for item in feature_ids],
                "layer": None if layer is None else int(layer),
                "kind": None if kind is None else str(kind),
                "value": None if value is None else float(value),
                "success": bool(frame["success"][index]),
                "steps": int(frame["steps"][index]),
                "wall_time_s": float(frame["wall_time_s"][index]),
            }
        )
        stored = frame["dictionary"][index] if "dictionary" in frame else None
        rows[-1]["dictionary"] = _row_dictionary({**rows[-1], "dictionary": stored})
    return rows


def write_attribution(path: Path, rows: dict[str, list], objective: str = "prediction_loss") -> None:
    """Write attribution scores. ``objective`` is stored in the parquet schema metadata."""
    table = pa.table(
        {
            "feature_id": pa.array(rows["feature_id"], type=pa.int32()),
            "clip_id": pa.array(rows["clip_id"], type=pa.string()),
            "score": pa.array(rows["score"], type=pa.float32()),
        }
    )
    table = table.replace_schema_metadata({b"objective": objective.encode("utf-8")})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_attribution_objective(path: Path) -> str | None:
    """Objective name recorded in the attribution parquet schema metadata."""
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(b"objective")
    if raw is None:
        return None
    return raw.decode("utf-8")


def write_clip_effects(path: Path, rows: dict[str, list]) -> None:
    table = pa.table(
        {
            "feature_id": pa.array(rows["feature_id"], type=pa.int32()),
            "clip_id": pa.array(rows["clip_id"], type=pa.string()),
            "window": pa.array(rows["window"], type=pa.int16()),
            "t": pa.array(rows["t"], type=pa.int16()),
            "effect": pa.array(rows["effect"], type=pa.float32()),
            "effect_kind": pa.array(rows["effect_kind"], type=pa.string()),
        }
    )
    pq.write_table(table, path)
