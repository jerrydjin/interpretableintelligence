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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
