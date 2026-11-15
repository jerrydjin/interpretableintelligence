"""Write an imported capture as a run folder.

Same stages as ``ii run`` where no model is needed:

* activation shards through :class:`~interpretableintelligence.store.ActivationWriter`;
* ``feature_acts.parquet``, top examples, ``ev_heldout`` and ``dead_frac`` from the
  pipeline's own features stage;
* ``attribution.parquet`` and ``candidates.json`` with objective ``activation``
  (sum of |activation|, no backward);
* ``scoreboard.json`` whose gates go through :func:`~interpretableintelligence.stats.feature_is_real`.

What needs the model in the loop (spliced loss, ablations, matched random
features, the random-init control, effect rows) is ``null`` with a reason in
``manifest.import.nulls``. Never ``0.0``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from interpretableintelligence import __version__
from interpretableintelligence.artifacts import write_attribution, write_json, write_jsonl
from interpretableintelligence.attribute import ACTIVATION, candidates_document, summarize_attribution
from interpretableintelligence.labelers import NullLabeller
from interpretableintelligence.sae import SAE, SAEConfig, save_sae
from interpretableintelligence.stats import feature_is_real
from interpretableintelligence.store import ActivationWriter, RowSource
from interpretableintelligence.thresholds import DEFAULT_PREFILTER_K

# Reason keys a validator accepts in ``manifest.import.nulls`` (SCHEMA.md, "Imported runs").
NULL_KEYS = (
    "manifest.split.seed",
    "manifest.split.heldout_frac",
    "scoreboard.run.seed",
    "scoreboard.sae.ev_heldout",
    "scoreboard.sae.loss_recovered",
    "scoreboard.sae.loss_sae",
    "scoreboard.sae.loss_visible_mean",
    "scoreboard.sae.usable",
    "scoreboard.features.label_holds_heldout",
    "scoreboard.features.ablation_effect",
    "scoreboard.features.random_percentile",
    "scoreboard.features.absent_in_random_init",
    "features/clip_effects.parquet",
)

SPLIT_NULL_REASON = (
    "split.json is the source split, copied byte for byte. It was not made by the clip-level split "
    "algorithm, so it has no ii seed or heldout_frac; its own rule is in the file."
)
FEATURE_NULL_REASONS = {
    "scoreboard.features.label_holds_heldout": (
        "no labeller ran: an import has no clip pixels or planted cues to label against, so the held-out label check did not run"
    ),
    "scoreboard.features.ablation_effect": (
        "needs the model's prediction with the feature zeroed; an import has no model in the loop"
    ),
    "scoreboard.features.random_percentile": (
        "needs the ablation effect and matched random-feature ablations; an import has no model in the loop"
    ),
    "scoreboard.features.absent_in_random_init": (
        "needs a capture and SAE from the randomly initialised model; the source has none"
    ),
}


@dataclass
class ImportedClip:
    clip_id: str
    source_id: int | str  # the id the source split.json uses for this clip
    sha256: str  # sha256 of the source file this clip's rows came from
    fps: float
    n_frames: int
    split: str  # train | heldout, from the source split
    n_windows: int
    labels: dict = field(default_factory=dict)  # per-clip outcome fields kept from the source


@dataclass
class ImportedCapture:
    format: str
    model: dict  # adapter, name, sha, random_init
    layer: int
    d_model: int
    grid: dict  # t, h, w, tubelet, patch_px
    cells: dict  # {"t": [...], "h": [...], "w": [...]}: the captured grid cells (their product)
    split_bytes: bytes
    clips: list[ImportedClip]
    # windows(clip_index) yields (window, rows [n, d_model], cells [n, 3]) in window order,
    # cells in row-major (t, h, w) order.
    windows: Callable[[int], Iterator[tuple[int, np.ndarray, np.ndarray]]]
    sae: SAE
    sae_config: SAEConfig
    nulls: dict[str, str]  # reasons for metrics the source cannot support
    source: dict
    notes: dict = field(default_factory=dict)  # format-specific provenance


def captured_cells(cells: dict) -> list[tuple[int, int, int]]:
    return [(int(t), int(h), int(w)) for t in cells["t"] for h in cells["h"] for w in cells["w"]]


def write_run(capture: ImportedCapture, out: Path, *, log: Callable[[str], None] = print) -> dict:
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"refusing to write into non-empty {out}")
    _check_capture(capture)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        return _write(capture, out, log, started)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def _write(capture: ImportedCapture, out: Path, log: Callable[[str], None], started: float) -> dict:
    from interpretableintelligence.pipeline import _features  # noqa: PLC0415 - the ii run features stage

    (out / "split.json").write_bytes(capture.split_bytes)
    split_sha = hashlib.sha256(capture.split_bytes).hexdigest()

    expected_cells = np.asarray(captured_cells(capture.cells), dtype=np.int64)
    n_rows = 0
    with ActivationWriter(out / "acts", capture.d_model) as writer:
        for clip_index, clip in enumerate(capture.clips):
            split_code = 1 if clip.split == "heldout" else 0
            seen = 0
            for window, rows, cells in capture.windows(clip_index):
                if window != seen:
                    raise ValueError(f"{clip.clip_id}: windows must run 0, 1, ...; got {window} after {seen - 1}")
                if not np.array_equal(np.asarray(cells, dtype=np.int64), expected_cells):
                    raise ValueError(f"{clip.clip_id} window {window} does not cover the captured cells in order")
                writer.append_rows(rows, clip_index, window, cells, split_code)
                seen += 1
                n_rows += int(rows.shape[0])
            if seen != clip.n_windows:
                raise ValueError(f"{clip.clip_id}: expected {clip.n_windows} windows, read {seen}")
        writer.finish()
    log(f"acts: {n_rows} rows from {len(capture.clips)} clips")

    save_sae(capture.sae, capture.sae_config, out / "sae")
    device = torch.device("cpu")
    meta = [{"clip_id": clip.clip_id} for clip in capture.clips]
    sparse_rows, _tokens = _features(out, meta, capture.sae_config, device)
    stats = json.loads((out / "features" / "stats.json").read_text(encoding="utf-8"))
    log(f"features: {sparse_rows} nonzero activations  ev_heldout={stats['ev_heldout']:.4f}  dead_frac={stats['dead_frac']:.4f}")

    attribution_started = time.perf_counter()
    by_clip = _activation_magnitude(out, capture, device)
    rows, mean = summarize_attribution(by_clip)
    write_attribution(out / "features" / "attribution.parquet", rows, objective=ACTIVATION)
    n_features = int(capture.sae_config.n_features)
    candidates = candidates_document(
        mean,
        min(DEFAULT_PREFILTER_K, n_features),
        objective=ACTIVATION,
        seconds={"attribution": round(time.perf_counter() - attribution_started, 3)},
    )
    write_json(out / "features" / "candidates.json", candidates)

    nulls = dict(capture.nulls)
    _write_empty_effects(out / "features" / "clip_effects.parquet")
    if "features/clip_effects.parquet" not in nulls:
        raise ValueError("an imported run without effect rows needs a reason for features/clip_effects.parquet")

    manifest_split = {"seed": None, "heldout_frac": None, "sha": split_sha}
    for key in ("manifest.split.seed", "manifest.split.heldout_frac", "scoreboard.run.seed"):
        nulls.setdefault(key, SPLIT_NULL_REASON)

    sae_block = {
        "ev_heldout": float(stats["ev_heldout"]),
        "loss_recovered": None,
        "loss_sae": None,
        "loss_visible_mean": None,
        "dead_frac": float(stats["dead_frac"]),
        "usable": None,
    }
    for key in ("loss_recovered", "loss_sae", "loss_visible_mean", "usable"):
        if f"scoreboard.sae.{key}" not in nulls:
            raise ValueError(f"the {capture.format} reader gave no reason for scoreboard.sae.{key}")

    features = []
    jsonl = []
    for item in stats["features"]:
        record = {
            "id": int(item["id"]),
            "label": "",
            "distinct_clips_top20": int(item["distinct_clips_top20"]),
            "label_holds_heldout": None,
            "ablation_effect": None,
            "random_percentile": None,
            "n_random": 0,
            "absent_in_random_init": None,
        }
        record["real"] = feature_is_real(
            usable=sae_block["usable"] is True,
            distinct_clips_top20=record["distinct_clips_top20"],
            label_holds_heldout=record["label_holds_heldout"],
            random_percentile=record["random_percentile"],
            absent_in_random_init=record["absent_in_random_init"],
        )
        features.append(record)
        jsonl.append({**record, "label_model": NullLabeller.name, "top": item["top"]})
    for key, reason in FEATURE_NULL_REASONS.items():
        nulls.setdefault(key, reason)

    n_train = sum(1 for clip in capture.clips if clip.split == "train")
    n_held = len(capture.clips) - n_train
    scoreboard = {
        "run": {
            "model_sha": capture.model["sha"],
            "layer": int(capture.layer),
            "split_sha": split_sha,
            "seed": None,
            "n_train_clips": n_train,
            "n_heldout_clips": n_held,
        },
        "sae": sae_block,
        "features": features,
    }
    manifest = {
        "run_id": out.name,
        "tool_version": __version__,
        "model": dict(capture.model),
        "layer": int(capture.layer),
        "d_model": int(capture.d_model),
        "grid": dict(capture.grid),
        "split": manifest_split,
        "clips": [
            {
                "clip_id": clip.clip_id,
                "sha256": clip.sha256,
                "fps": float(clip.fps),
                "n_frames": int(clip.n_frames),
                "split": clip.split,
            }
            for clip in capture.clips
        ],
        "import": {
            "format": capture.format,
            "cells": {axis: [int(value) for value in capture.cells[axis]] for axis in ("t", "h", "w")},
            "clips": [
                {"clip_id": clip.clip_id, "source_id": clip.source_id, "n_windows": int(clip.n_windows), **clip.labels}
                for clip in capture.clips
            ],
            "nulls": {key: nulls[key] for key in NULL_KEYS if key in nulls},
            "source": capture.source,
            "notes": capture.notes,
            "wall_s": round(time.perf_counter() - started, 3),
        },
    }
    unknown = sorted(set(nulls) - set(NULL_KEYS))
    if unknown:
        raise ValueError(f"unknown null reason keys {unknown}")
    write_jsonl(out / "features" / "features.jsonl", jsonl)
    write_json(out / "scoreboard.json", scoreboard)
    write_json(out / "manifest.json", manifest)
    # The float32 row cache is a resume artifact (SCHEMA.md); an import never resumes.
    shutil.rmtree(out / "acts" / "stream", ignore_errors=True)
    n_real = sum(1 for record in features if record["real"])
    log(f"wrote {out}  ({len(features)} features, {n_real} real; usable is null until the SAE is spliced into the model)")
    return manifest


def _activation_magnitude(out: Path, capture: ImportedCapture, device: torch.device) -> dict[str, np.ndarray]:
    """Per clip, the sum over tokens and windows of |feature activation| (objective ``activation``)."""
    n_features = int(capture.sae_config.n_features)
    totals = np.zeros((len(capture.clips), n_features), dtype=np.float64)
    with RowSource(out / "acts") as source:
        for start, stop in source.chunks(4096):
            rows = torch.from_numpy(np.array(source.rows[start:stop], dtype=np.float32, copy=True)).to(device)
            clip_idx = np.array(source.clip_idx[start:stop], dtype=np.int64, copy=True)
            with torch.no_grad():
                feats = capture.sae.encode(rows).abs().double().cpu().numpy()
            for clip_index in np.unique(clip_idx):
                totals[int(clip_index)] += feats[clip_idx == clip_index].sum(axis=0)
    return {clip.clip_id: totals[index] for index, clip in enumerate(capture.clips)}


def _write_empty_effects(path: Path) -> None:
    schema = pa.schema(
        [
            ("feature_id", pa.int32()),
            ("clip_id", pa.string()),
            ("window", pa.int16()),
            ("t", pa.int16()),
            ("effect", pa.float32()),
            ("effect_kind", pa.string()),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(schema.empty_table(), path)


def _check_capture(capture: ImportedCapture) -> None:
    if len(capture.clips) < 20:
        raise ValueError(f"the run contract needs at least 20 clips, the source has {len(capture.clips)}")
    ids = [clip.clip_id for clip in capture.clips]
    if len(set(ids)) != len(ids):
        raise ValueError("clip ids repeat")
    if {clip.split for clip in capture.clips} - {"train", "heldout"}:
        raise ValueError("clip splits must be train or heldout")
    if not any(clip.split == "heldout" for clip in capture.clips) or not any(clip.split == "train" for clip in capture.clips):
        raise ValueError("the source split needs at least one train and one held-out clip")
    if int(capture.sae_config.d_model) != int(capture.d_model):
        raise ValueError("SAE d_model does not match the capture")
    for axis in ("t", "h", "w"):
        values = [int(value) for value in capture.cells[axis]]
        if not values or values != sorted(set(values)) or values[0] < 0 or values[-1] >= int(capture.grid[axis]):
            raise ValueError(f"cells.{axis} must be sorted, unique, and inside the grid")
