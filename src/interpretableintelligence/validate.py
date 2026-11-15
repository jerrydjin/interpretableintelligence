"""Validate a run folder against the interpretableintelligence contract."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from interpretableintelligence.artifacts import (
    ATTRIBUTION_COLUMNS,
    CLIP_EFFECT_COLUMNS,
    FEATURE_ACT_COLUMNS,
    ROLLOUT_COLUMNS,
    load_activation_shards,
    read_attribution_objective,
    read_json,
    read_jsonl,
    read_rollout_episodes,
)
from interpretableintelligence.rollout import (
    CONDITION_KEYS,
    FEATURE_KEYS,
    SUMMARY_KEYS,
    config_from_summary,
    make_episode_id,
    summarize,
)
from interpretableintelligence.attribute import METHOD, RANK_OBJECTIVES, SCORED_OBJECTIVES, rank_feature_ids
from interpretableintelligence.policies.base import dictionary_seed
from interpretableintelligence.sae import load_sae
from interpretableintelligence.split import assign_split, split_sha
from interpretableintelligence.stats import feature_is_real, is_usable
from interpretableintelligence.thresholds import (
    EFFECT_KIND_ABLATION,
    EFFECT_KIND_ESTIMATE,
    KNOWN_EFFECT_KINDS,
    N_RANDOM_FEATURES,
    TOP_K_ABLATION_PER_CLIP,
    TOP_K_EXAMPLES,
)

_SHARD_NAME = re.compile(r"shard-\d{5}\.safetensors$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_MANIFEST_KEYS = {"run_id", "tool_version", "model", "layer", "d_model", "grid", "split", "clips"}
# Resume metadata. Readers that only need the contract can ignore both.
# ``import`` marks a run written by ``ii import`` (SCHEMA.md, "Imported runs").
_MANIFEST_OPTIONAL = {"stages", "clips_dir", "import"}
_IMPORT_KEYS = {"format", "cells", "clips", "nulls", "source", "notes"}
_IMPORT_OPTIONAL = {"wall_s"}
_IMPORT_CLIP_KEYS = {"clip_id", "source_id", "n_windows"}
# Fields an imported run may leave null, each with a reason in manifest.import.nulls.
_IMPORT_NULL_KEYS = {
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
}
_STAGE_FIELDS = {"wall_s", "peak_ram_bytes", "device", "rows", "tokens"}
_MODEL_KEYS = {"adapter", "name", "sha", "random_init"}
_GRID_KEYS = {"t", "h", "w", "tubelet", "patch_px"}
_SPLIT_REF_KEYS = {"seed", "heldout_frac", "sha"}
_CLIP_KEYS = {"clip_id", "sha256", "fps", "n_frames", "split"}
_SPLIT_FILE_KEYS = {"seed", "heldout_frac", "train", "heldout", "sha"}
_SCOREBOARD_KEYS = {"run", "sae", "features"}
_RUN_KEYS = {"model_sha", "layer", "split_sha", "seed", "n_train_clips", "n_heldout_clips"}
_SAE_KEYS = {"ev_heldout", "loss_recovered", "loss_sae", "loss_visible_mean", "dead_frac", "usable"}
_FEATURE_KEYS = {
    "id",
    "label",
    "distinct_clips_top20",
    "label_holds_heldout",
    "ablation_effect",
    "random_percentile",
    "n_random",
    "absent_in_random_init",
    "real",
}
_JSONL_KEYS = _FEATURE_KEYS | {"label_model", "top"}
_TOP_KEYS = {"clip_id", "window", "t", "value"}
_CANDIDATE_KEYS = {
    "method",
    "objective",
    "shortlist_only",
    "prefilter_k",
    "rank_by",
    "ranked",
    "candidates",
    "seconds",
}
_RANKED_KEYS = {"rank", "feature_id", "label", "score", "abs_score"}
_SECOND_KEYS = {"attribution", "ablation", "random_controls"}
_SAE_CONFIG_KEYS = {
    "d_model",
    "n_features",
    "l1_coeff",
    "ortho_coeff",
    "lr",
    "steps",
    "batch_size",
    "seed",
    "normalize_decoder",
    "activation",
}


def validate_run(run_dir: Path) -> list[str]:
    """Return a list of contract errors. Empty means the folder is valid."""
    errors: list[str] = []
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return [f"{run_dir} is not a directory"]

    manifest = _read_object(run_dir / "manifest.json", errors)
    split = _read_object(run_dir / "split.json", errors)
    scoreboard = _read_object(run_dir / "scoreboard.json", errors)
    if manifest is None or split is None or scoreboard is None:
        return errors

    schema_version = _schema_version(manifest, errors)
    expected_manifest = set(_MANIFEST_KEYS)
    if "schema_version" in manifest:
        expected_manifest.add("schema_version")
    _allow_keys(manifest, expected_manifest, _MANIFEST_OPTIONAL, "manifest", errors)
    imported = "import" in manifest
    if not imported:
        _exact_keys(split, _SPLIT_FILE_KEYS, "split.json", errors)
    _exact_keys(scoreboard, _SCOREBOARD_KEYS, "scoreboard", errors)
    if errors:
        return errors
    _check_manifest_extras(manifest, errors)
    if imported:
        _check_import_block(manifest, errors)
        if errors:
            return errors

    model = manifest.get("model")
    grid = manifest.get("grid")
    split_ref = manifest.get("split")
    clips = manifest.get("clips")
    if not isinstance(model, dict) or not isinstance(grid, dict) or not isinstance(split_ref, dict):
        errors.append("manifest model, grid, and split must be objects")
        return errors
    if not isinstance(clips, list) or len(clips) < 20:
        errors.append("manifest.clips must list at least 20 clips")
        return errors
    _exact_keys(model, _MODEL_KEYS, "manifest.model", errors)
    _exact_keys(grid, _GRID_KEYS, "manifest.grid", errors)
    _exact_keys(split_ref, _SPLIT_REF_KEYS, "manifest.split", errors)

    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        errors.append("manifest.run_id must be a non-empty string")
    if manifest["run_id"] != run_dir.name:
        errors.append(f"manifest.run_id {manifest.get('run_id')!r} != directory name {run_dir.name!r}")
    if not isinstance(manifest.get("tool_version"), str) or not manifest["tool_version"]:
        errors.append("manifest.tool_version must be a non-empty string")
    if not _is_int(manifest.get("layer")) or manifest["layer"] < 0:
        errors.append("manifest.layer must be a non-negative integer")
    if not _is_int(manifest.get("d_model")) or manifest["d_model"] <= 0:
        errors.append("manifest.d_model must be a positive integer")

    for key in _GRID_KEYS:
        if not _is_int(grid.get(key)) or grid[key] <= 0:
            errors.append(f"manifest.grid.{key} must be a positive integer")
    if not isinstance(model.get("adapter"), str) or not model["adapter"]:
        errors.append("manifest.model.adapter must be a non-empty string")
    if not isinstance(model.get("name"), str) or not model["name"]:
        errors.append("manifest.model.name must be a non-empty string")
    if not isinstance(model.get("random_init"), bool):
        errors.append("manifest.model.random_init must be a boolean")
    _hex_field(model.get("sha"), "manifest.model.sha", errors)

    if not isinstance(split.get("train"), list) or not isinstance(split.get("heldout"), list):
        errors.append("split.json train and heldout must be lists")
        return errors
    if imported:
        _check_imported_split(run_dir, manifest, split, split_ref, clips, errors)
    else:
        if not _is_int(split.get("seed")):
            errors.append("split.json seed must be an integer")
        if not _is_number(split.get("heldout_frac")) or not 0.0 < float(split["heldout_frac"]) < 1.0:
            errors.append("split.json heldout_frac must be in (0, 1)")
        if errors:
            return errors
        _check_split_consistency(manifest, split, split_ref, clips, errors)
    _check_scoreboard_shape(scoreboard, manifest, errors)
    if errors:
        return errors
    if imported:
        _check_null_reasons(run_dir, manifest, scoreboard, errors)

    _check_sae(run_dir, manifest, scoreboard, errors)
    if imported:
        _check_imported_shards(run_dir, manifest, errors)
    else:
        _check_shards(run_dir, manifest, errors)
    _check_feature_files(run_dir, manifest, scoreboard, errors)
    _check_rollouts(run_dir, schema_version, errors)
    from interpretableintelligence.steer_scorecard import check_steering  # noqa: PLC0415

    errors.extend(check_steering(run_dir, manifest, scoreboard))
    from interpretableintelligence.probe import check_probe  # noqa: PLC0415

    errors.extend(check_probe(run_dir, manifest))
    return errors


def _check_split_consistency(manifest: dict, split: dict, split_ref: dict, clips: list, errors: list[str]) -> None:
    clip_ids = []
    seen = set()
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            errors.append(f"manifest.clips[{index}] must be an object")
            continue
        _exact_keys(clip, _CLIP_KEYS, f"manifest.clips[{index}]", errors)
        clip_id = clip.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            errors.append(f"manifest.clips[{index}].clip_id must be a non-empty string")
            continue
        if clip_id in seen:
            errors.append(f"duplicate clip id {clip_id}")
        seen.add(clip_id)
        clip_ids.append(clip_id)
        _hex_field(clip.get("sha256"), f"manifest.clips[{index}].sha256", errors)
        if not _is_number(clip.get("fps")) or float(clip["fps"]) <= 0:
            errors.append(f"{clip_id} fps must be positive")
        if not _is_int(clip.get("n_frames")) or clip["n_frames"] <= 0:
            errors.append(f"{clip_id} n_frames must be a positive integer")
        if clip.get("split") not in {"train", "heldout"}:
            errors.append(f"{clip_id} split must be train or heldout")

    train = list(split["train"])
    heldout = list(split["heldout"])
    if len(train) + len(heldout) != len(set(train) | set(heldout)):
        errors.append("split.json lists overlap or repeat a clip")
    if set(train) | set(heldout) != set(clip_ids):
        errors.append("split.json clip ids do not match manifest.clips")
    if any(not isinstance(item, str) for item in train + heldout):
        errors.append("split.json clip ids must be strings")
        return
    expected_sha = split_sha(int(split["seed"]), float(split["heldout_frac"]), train, heldout)
    if split.get("sha") != expected_sha:
        errors.append("split.json sha does not match the canonical partition")
    if split_ref.get("sha") != split.get("sha"):
        errors.append("manifest.split.sha does not match split.json")
    if split_ref.get("seed") != split.get("seed") or split_ref.get("heldout_frac") != split.get("heldout_frac"):
        errors.append("manifest.split seed/heldout_frac do not match split.json")
    try:
        derived = assign_split(clip_ids, int(split["seed"]), float(split["heldout_frac"]))
    except (TypeError, ValueError) as exc:
        errors.append(f"split could not be rederived: {exc}")
        return
    if derived.train != train or derived.heldout != heldout:
        errors.append("split.json is not the partition produced by the clip-level split algorithm")
    membership = {clip_id: "train" for clip_id in train}
    membership.update({clip_id: "heldout" for clip_id in heldout})
    for clip in clips:
        if isinstance(clip, dict) and clip.get("split") != membership.get(clip.get("clip_id")):
            errors.append(f"{clip.get('clip_id')} split field does not match split.json")


def _check_import_block(manifest: dict, errors: list[str]) -> None:
    """``manifest.import`` from ``ii import``: format, captured cells, per-clip windows, null reasons."""
    block = manifest["import"]
    if not isinstance(block, dict):
        errors.append("manifest.import must be an object")
        return
    _allow_keys(block, _IMPORT_KEYS, _IMPORT_OPTIONAL, "manifest.import", errors)
    if not isinstance(block.get("format"), str) or not block["format"]:
        errors.append("manifest.import.format must be a non-empty string")
    for key in ("source", "notes"):
        if not isinstance(block.get(key), dict):
            errors.append(f"manifest.import.{key} must be an object")
    cells = block.get("cells")
    grid = manifest.get("grid")
    if not isinstance(cells, dict) or set(cells) != {"t", "h", "w"} or not isinstance(grid, dict):
        errors.append("manifest.import.cells must be an object with t, h and w lists")
    else:
        for axis in ("t", "h", "w"):
            values = cells[axis]
            ok = isinstance(values, list) and values and all(_is_int(value) for value in values)
            if not ok or values != sorted(set(values)) or values[0] < 0 or not _is_int(grid.get(axis)) or values[-1] >= grid[axis]:
                errors.append(f"manifest.import.cells.{axis} must be sorted unique integers inside the grid")
    nulls = block.get("nulls")
    if not isinstance(nulls, dict):
        errors.append("manifest.import.nulls must be an object")
    else:
        for key, reason in nulls.items():
            if key not in _IMPORT_NULL_KEYS:
                errors.append(f"manifest.import.nulls has unknown key {key!r}")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"manifest.import.nulls[{key!r}] must be a non-empty reason")
    rows = block.get("clips")
    clips = manifest.get("clips")
    if not isinstance(rows, list) or not isinstance(clips, list) or len(rows) != len(clips):
        errors.append("manifest.import.clips must list one entry per manifest clip")
        return
    for index, (row, clip) in enumerate(zip(rows, clips, strict=True)):
        where = f"manifest.import.clips[{index}]"
        if not isinstance(row, dict) or not isinstance(clip, dict):
            errors.append(f"{where} must be an object")
            continue
        missing = sorted(_IMPORT_CLIP_KEYS - set(row))
        if missing:
            errors.append(f"{where} is missing {missing}")
            continue
        if row["clip_id"] != clip.get("clip_id"):
            errors.append(f"{where}.clip_id does not match manifest.clips[{index}]")
        if not (_is_int(row["source_id"]) or (isinstance(row["source_id"], str) and row["source_id"])):
            errors.append(f"{where}.source_id must be an integer or a non-empty string")
        if not _is_int(row["n_windows"]) or row["n_windows"] < 1:
            errors.append(f"{where}.n_windows must be a positive integer")


def _check_imported_split(
    run_dir: Path, manifest: dict, split: dict, split_ref: dict, clips: list, errors: list[str]
) -> None:
    """The source split.json, copied byte for byte: its lists name ``manifest.import.clips[].source_id``."""
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            errors.append(f"manifest.clips[{index}] must be an object")
            continue
        _exact_keys(clip, _CLIP_KEYS, f"manifest.clips[{index}]", errors)
        clip_id = clip.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            errors.append(f"manifest.clips[{index}].clip_id must be a non-empty string")
            continue
        _hex_field(clip.get("sha256"), f"manifest.clips[{index}].sha256", errors)
        if not _is_number(clip.get("fps")) or float(clip["fps"]) <= 0:
            errors.append(f"{clip_id} fps must be positive")
        if not _is_int(clip.get("n_frames")) or clip["n_frames"] <= 0:
            errors.append(f"{clip_id} n_frames must be a positive integer")
        if clip.get("split") not in {"train", "heldout"}:
            errors.append(f"{clip_id} split must be train or heldout")
    ids = [clip.get("clip_id") for clip in clips if isinstance(clip, dict)]
    if len(ids) != len(set(ids)):
        errors.append("manifest clip ids repeat")
    digest = hashlib.sha256((run_dir / "split.json").read_bytes()).hexdigest()
    if split_ref.get("sha") != digest:
        errors.append("manifest.split.sha must be the sha256 of the split.json bytes on an imported run")
    for key in ("seed", "heldout_frac"):
        value = split_ref.get(key)
        if value is not None and not _is_number(value):
            errors.append(f"manifest.split.{key} must be a number or null")
    train = split["train"]
    heldout = split["heldout"]
    hashable = all(isinstance(item, (int, str)) and not isinstance(item, bool) for item in train + heldout)
    if not hashable:
        errors.append("split.json lists must hold source ids (integers or strings)")
        return
    if len(train) + len(heldout) != len(set(train) | set(heldout)):
        errors.append("split.json lists overlap or repeat a clip")
    source_ids = [row["source_id"] for row in manifest["import"]["clips"]]
    if len(source_ids) != len(set(source_ids)):
        errors.append("manifest.import.clips source ids repeat")
    if set(train) | set(heldout) != set(source_ids):
        errors.append("split.json ids do not match manifest.import.clips source_id")
        return
    membership = {item: "train" for item in train}
    membership.update({item: "heldout" for item in heldout})
    for row, clip in zip(manifest["import"]["clips"], clips, strict=True):
        if isinstance(clip, dict) and clip.get("split") != membership.get(row["source_id"]):
            errors.append(f"{clip.get('clip_id')} split field does not match split.json")


def _check_null_reasons(run_dir: Path, manifest: dict, scoreboard: dict, errors: list[str]) -> None:
    """Every null an imported run writes has a reason, and every reason names a field that is null."""
    null_now: set[str] = set()
    for key in ("seed", "heldout_frac"):
        if manifest["split"].get(key) is None:
            null_now.add(f"manifest.split.{key}")
    if scoreboard["run"].get("seed") is None:
        null_now.add("scoreboard.run.seed")
    for key in ("ev_heldout", "loss_recovered", "loss_sae", "loss_visible_mean", "usable"):
        if scoreboard["sae"].get(key) is None:
            null_now.add(f"scoreboard.sae.{key}")
    for feature in scoreboard["features"]:
        if not isinstance(feature, dict):
            continue
        for key in ("label_holds_heldout", "ablation_effect", "random_percentile", "absent_in_random_init"):
            if key in feature and feature[key] is None:
                null_now.add(f"scoreboard.features.{key}")
    effects = run_dir / "features" / "clip_effects.parquet"
    if effects.is_file():
        try:
            if pq.read_metadata(effects).num_rows == 0:
                null_now.add("features/clip_effects.parquet")
        except Exception:  # noqa: BLE001 - reported by _check_effects
            pass
    reasons = manifest["import"]["nulls"]
    for key in sorted(null_now - set(reasons)):
        errors.append(f"{key} is null on an imported run but manifest.import.nulls gives no reason")
    for key in sorted(set(reasons) - null_now):
        errors.append(f"manifest.import.nulls gives a reason for {key}, which is not null")


def _check_imported_shards(run_dir: Path, manifest: dict, errors: list[str]) -> None:
    """Rows cover each clip's own windows 0..n_windows-1 and exactly the captured cells, once each."""
    directory = run_dir / "acts"
    paths = sorted(directory.glob("shard-*.safetensors")) if directory.is_dir() else []
    if not paths:
        errors.append("acts/ has no shards")
        return
    expected_names = [f"shard-{index:05d}.safetensors" for index in range(len(paths))]
    if [path.name for path in paths] != expected_names:
        errors.append("activation shards must be contiguous shard-00000.safetensors, shard-00001.safetensors, ...")
    try:
        data = load_activation_shards(directory)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"activation shards could not be loaded: {exc}")
        return
    acts = data["acts"]
    if acts.dtype != np.float16 or acts.ndim != 2 or acts.shape[1] != manifest["d_model"]:
        errors.append(f"acts must be float16 [N, d_model={manifest['d_model']}]")
        return
    n = acts.shape[0]
    for key, dtype in (("clip_idx", np.int32), ("window", np.int16), ("t", np.int16), ("h", np.int16), ("w", np.int16)):
        if data[key].dtype != dtype or data[key].shape != (n,):
            errors.append(f"{key} must be {dtype.__name__} [N]")
            return
    if not np.isfinite(acts.astype(np.float32)).all():
        errors.append("activation shard has non-finite values")
    cells = manifest["import"]["cells"]
    width = len(cells["t"]) * len(cells["h"]) * len(cells["w"])
    clip_rows = manifest["import"]["clips"]
    expected_rows = width * sum(int(row["n_windows"]) for row in clip_rows)
    if n != expected_rows:
        errors.append(f"activation rows must cover each clip's windows and the captured cells once (got {n}, expected {expected_rows})")
        return
    cell_t = np.repeat(np.asarray(cells["t"]), len(cells["h"]) * len(cells["w"]))
    cell_h = np.tile(np.repeat(np.asarray(cells["h"]), len(cells["w"])), len(cells["t"]))
    cell_w = np.tile(np.asarray(cells["w"]), len(cells["t"]) * len(cells["h"]))
    want_clip = np.concatenate(
        [np.full(width * int(row["n_windows"]), index, dtype=np.int64) for index, row in enumerate(clip_rows)]
    )
    want_window = np.concatenate([np.repeat(np.arange(int(row["n_windows"])), width) for row in clip_rows])
    reps = expected_rows // width
    order_ok = (
        np.array_equal(data["clip_idx"].astype(np.int64), want_clip)
        and np.array_equal(data["window"].astype(np.int64), want_window)
        and np.array_equal(data["t"].astype(np.int64), np.tile(cell_t, reps))
        and np.array_equal(data["h"].astype(np.int64), np.tile(cell_h, reps))
        and np.array_equal(data["w"].astype(np.int64), np.tile(cell_w, reps))
    )
    if not order_ok:
        errors.append(
            "activation rows must be in manifest clip order, then window, then row-major captured cell, "
            "covering windows 0..n_windows-1 of each clip once"
        )


def _check_scoreboard_shape(scoreboard: dict, manifest: dict, errors: list[str]) -> None:
    run = scoreboard.get("run")
    sae = scoreboard.get("sae")
    features = scoreboard.get("features")
    if not isinstance(run, dict) or not isinstance(sae, dict) or not isinstance(features, list):
        errors.append("scoreboard run, sae, and features have the wrong types")
        return
    _exact_keys(run, _RUN_KEYS, "scoreboard.run", errors)
    _exact_keys(sae, _SAE_KEYS, "scoreboard.sae", errors)
    if run.get("model_sha") != manifest["model"].get("sha"):
        errors.append("scoreboard.run.model_sha does not match manifest.model.sha")
    if run.get("layer") != manifest.get("layer"):
        errors.append("scoreboard.run.layer does not match manifest.layer")
    if run.get("split_sha") != manifest["split"].get("sha"):
        errors.append("scoreboard.run.split_sha does not match manifest.split.sha")
    if run.get("seed") != manifest["split"].get("seed"):
        errors.append("scoreboard.run.seed does not match the split seed")
    n_train = sum(1 for clip in manifest["clips"] if clip.get("split") == "train")
    n_held = sum(1 for clip in manifest["clips"] if clip.get("split") == "heldout")
    if run.get("n_train_clips") != n_train or run.get("n_heldout_clips") != n_held:
        errors.append("scoreboard clip counts do not match manifest")
    imported = "import" in manifest
    if imported and sae.get("ev_heldout") is None:
        pass  # null with a reason (checked in _check_null_reasons)
    elif not _is_number(sae.get("ev_heldout")) or float(sae["ev_heldout"]) > 1.01:
        errors.append("scoreboard.sae.ev_heldout must be a finite number <= 1")
    loss_keys = ("loss_recovered", "loss_sae", "loss_visible_mean")
    if imported and any(sae.get(key) is None for key in loss_keys):
        if sae.get("usable") is not None:
            errors.append("scoreboard.sae.usable must be null when loss_recovered, loss_sae or loss_visible_mean is null")
        for key in loss_keys:
            if sae.get(key) is not None and not _is_number(sae.get(key)):
                errors.append(f"scoreboard.sae.{key} must be finite or null")
    elif not _is_number(sae.get("loss_recovered")):
        errors.append("scoreboard.sae.loss_recovered must be finite")
    elif not _is_number(sae.get("loss_sae")) or not _is_number(sae.get("loss_visible_mean")):
        errors.append("scoreboard.sae.loss_sae and loss_visible_mean must be finite")
    elif not isinstance(sae.get("usable"), bool) or sae["usable"] != is_usable(
        float(sae["loss_recovered"]),
        float(sae["loss_sae"]),
        float(sae["loss_visible_mean"]),
    ):
        errors.append(
            "scoreboard.sae.usable must be true only when loss_recovered >= 0.70 "
            "and loss_sae < loss_visible_mean"
        )
    if not _is_number(sae.get("dead_frac")) or not 0.0 <= float(sae["dead_frac"]) <= 1.0:
        errors.append("scoreboard.sae.dead_frac must be in [0, 1]")
    if not features:
        errors.append("scoreboard.features must be non-empty")
        return
    usable = bool(sae.get("usable")) if isinstance(sae.get("usable"), bool) else False
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            errors.append(f"scoreboard.features[{index}] must be an object")
            continue
        _exact_keys(feature, _FEATURE_KEYS, f"scoreboard.features[{index}]", errors)
        if feature.get("id") != index:
            errors.append(f"scoreboard feature ids must be 0..n-1 in order, index {index} has {feature.get('id')}")
        _check_feature_fields(feature, usable, f"scoreboard.features[{index}]", errors)


def _check_feature_fields(feature: dict, usable: bool, where: str, errors: list[str]) -> None:
    if not isinstance(feature.get("label"), str):
        errors.append(f"{where}.label must be a string")
    if not _is_int(feature.get("distinct_clips_top20")) or feature["distinct_clips_top20"] < 0:
        errors.append(f"{where}.distinct_clips_top20 must be a non-negative integer")
    holds = feature.get("label_holds_heldout")
    if holds is not None and not isinstance(holds, bool):
        errors.append(f"{where}.label_holds_heldout must be true, false, or null")
    effect = feature.get("ablation_effect")
    pct = feature.get("random_percentile")
    if (effect is None) != (pct is None):
        errors.append(f"{where} ablation_effect and random_percentile must both be numbers or both null")
    elif effect is None:
        if feature.get("n_random") != 0:
            errors.append(f"{where}.n_random must be 0 when the feature was not ablated")
    else:
        if not _is_number(effect) or float(effect) < -1e-5:
            errors.append(f"{where}.ablation_effect must be a non-negative number")
        if not _is_number(pct) or not 0.0 <= float(pct) <= 100.0:
            errors.append(f"{where}.random_percentile must be in [0, 100]")
        if not _is_int(feature.get("n_random")) or feature["n_random"] < N_RANDOM_FEATURES:
            errors.append(f"{where}.n_random must be an integer >= {N_RANDOM_FEATURES}")
    absent = feature.get("absent_in_random_init")
    if absent is not None and not isinstance(absent, bool):
        errors.append(f"{where}.absent_in_random_init must be true, false, or null")
    if not isinstance(feature.get("real"), bool):
        errors.append(f"{where}.real must be a boolean")
        return
    if not _is_int(feature.get("distinct_clips_top20")):
        return
    if effect is None and pct is None:
        percentile_arg = None
    elif _is_number(pct):
        percentile_arg = float(pct)
    else:
        return
    expected = feature_is_real(
        usable=usable,
        distinct_clips_top20=int(feature["distinct_clips_top20"]),
        label_holds_heldout=holds if isinstance(holds, bool) or holds is None else None,
        random_percentile=percentile_arg,
        absent_in_random_init=absent if isinstance(absent, bool) or absent is None else None,
    )
    if feature["real"] != expected:
        errors.append(f"{where}.real does not follow the four gates (null never passes)")


def _check_sae(run_dir: Path, manifest: dict, scoreboard: dict, errors: list[str]) -> None:
    config_path = run_dir / "sae" / "config.json"
    weights_path = run_dir / "sae" / "weights.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        errors.append("sae/config.json and sae/weights.safetensors are required")
        return
    try:
        config = read_json(config_path)
    except (OSError, ValueError) as exc:
        errors.append(f"sae/config.json: {exc}")
        return
    topk = config.get("activation") == "topk"
    jumprelu = config.get("activation") == "jumprelu"
    extra = {"k"} if topk else ({"threshold"} if jumprelu else set())
    _exact_keys(config, _SAE_CONFIG_KEYS | extra, "sae/config.json", errors)
    if config.get("d_model") != manifest.get("d_model"):
        errors.append("sae config d_model does not match manifest")
    if config.get("activation") not in ("relu", "topk", "jumprelu"):
        errors.append("sae config activation must be relu, topk or jumprelu")
    if topk and (not _is_int(config.get("k")) or config["k"] <= 0):
        errors.append("topk sae config k must be a positive integer")
    if jumprelu and (not _is_number(config.get("threshold")) or float(config["threshold"]) < 0):
        errors.append("jumprelu sae config threshold must be a finite number >= 0")
    n_features = config.get("n_features")
    if not _is_int(n_features) or n_features <= 0:
        errors.append("sae config n_features must be a positive integer")
        return
    if len(scoreboard["features"]) != n_features:
        errors.append("scoreboard.features length does not match sae n_features")
    try:
        model, _loaded = load_sae(run_dir / "sae", device=_cpu())
    except Exception as exc:  # noqa: BLE001 - surface any unreadable weight file
        errors.append(f"sae weights could not be loaded: {exc}")
        return
    if tuple(model.W_enc.shape) != (n_features, manifest["d_model"]):
        errors.append(f"W_enc shape {tuple(model.W_enc.shape)} != {(n_features, manifest['d_model'])}")
    if tuple(model.W_dec.shape) != (manifest["d_model"], n_features):
        errors.append(f"W_dec shape {tuple(model.W_dec.shape)} != {(manifest['d_model'], n_features)}")
    if tuple(model.b_enc.shape) != (n_features,) or tuple(model.b_dec.shape) != (manifest["d_model"],):
        errors.append("SAE bias shapes do not match d_model and n_features")
    for name, tensor in model.state_dict().items():
        if not torch_finite(tensor):
            errors.append(f"SAE tensor {name} has non-finite values")


def _check_shards(run_dir: Path, manifest: dict, errors: list[str]) -> None:
    directory = run_dir / "acts"
    if not directory.is_dir():
        errors.append("acts/ directory is required")
        return
    paths = sorted(directory.glob("shard-*.safetensors"))
    if not paths:
        errors.append("acts/ has no shards")
        return
    expected_names = [f"shard-{index:05d}.safetensors" for index in range(len(paths))]
    actual = [path.name for path in paths]
    if actual != expected_names or any(_SHARD_NAME.match(name) is None for name in actual):
        errors.append("activation shards must be contiguous shard-00000.safetensors, shard-00001.safetensors, ...")
    try:
        data = load_activation_shards(directory)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"activation shards could not be loaded: {exc}")
        return
    acts = data["acts"]
    shape_ok = acts.dtype == np.float16 and acts.ndim == 2 and acts.shape[1] == manifest["d_model"]
    if not shape_ok:
        errors.append(f"acts must be float16 [N, d_model={manifest['d_model']}]")
        return
    n = acts.shape[0]
    for key, dtype in (("clip_idx", np.int32), ("window", np.int16), ("t", np.int16), ("h", np.int16), ("w", np.int16)):
        column = data[key]
        if column.dtype != dtype or column.shape != (n,):
            errors.append(f"{key} must be {dtype.__name__} [N]")
            shape_ok = False
    if not shape_ok:
        return
    if not np.isfinite(acts.astype(np.float32)).all():
        errors.append("activation shard has non-finite values")
    grid = manifest["grid"]
    clip_idx = data["clip_idx"].astype(np.int64)
    window = data["window"].astype(np.int64)
    t = data["t"].astype(np.int64)
    h = data["h"].astype(np.int64)
    w = data["w"].astype(np.int64)
    if clip_idx.size and (clip_idx.min() < 0 or clip_idx.max() >= len(manifest["clips"])):
        errors.append("clip_idx is outside manifest.clips")
        return
    windows = sorted(set(window.tolist()))
    if windows != list(range(len(windows))):
        errors.append("window indices must be contiguous from 0")
        return
    present = set(zip(clip_idx.tolist(), window.tolist(), t.tolist(), h.tolist(), w.tolist(), strict=True))
    expected = {
        (ci, wi, ti, hi, wi_)
        for ci in range(len(manifest["clips"]))
        for wi in windows
        for ti in range(int(grid["t"]))
        for hi in range(int(grid["h"]))
        for wi_ in range(int(grid["w"]))
    }
    if present != expected:
        errors.append(
            f"activation rows must cover every clip, window, and grid cell once "
            f"(got {len(present)}, expected {len(expected)})"
        )


def _check_feature_files(run_dir: Path, manifest: dict, scoreboard: dict, errors: list[str]) -> None:
    jsonl_path = run_dir / "features" / "features.jsonl"
    acts_path = run_dir / "features" / "feature_acts.parquet"
    effects_path = run_dir / "features" / "clip_effects.parquet"
    for path in (jsonl_path, acts_path, effects_path):
        if not path.is_file():
            errors.append(f"missing {path.relative_to(run_dir)}")
    if not jsonl_path.is_file():
        return
    try:
        rows = read_jsonl(jsonl_path)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return
    features = scoreboard["features"]
    if len(rows) != len(features):
        errors.append("features.jsonl line count does not match scoreboard.features")
    clip_ids = {clip["clip_id"] for clip in manifest["clips"] if isinstance(clip, dict)}
    grid = manifest["grid"]
    for index, (row, feature) in enumerate(zip(rows, features, strict=False)):
        if not isinstance(row, dict):
            errors.append(f"features.jsonl line {index + 1} must be an object")
            continue
        _exact_keys(row, _JSONL_KEYS, f"features.jsonl[{index}]", errors)
        for key in _FEATURE_KEYS:
            if row.get(key) != feature.get(key):
                errors.append(f"features.jsonl[{index}].{key} does not match the scoreboard")
        if not isinstance(row.get("label_model"), str):
            errors.append(f"features.jsonl[{index}].label_model must be a string")
        top = row.get("top")
        if not isinstance(top, list) or len(top) > TOP_K_EXAMPLES:
            errors.append(f"features.jsonl[{index}].top must be a list of at most {TOP_K_EXAMPLES}")
            continue
        seen_clips = set()
        for item in top:
            if not isinstance(item, dict) or set(item) != _TOP_KEYS:
                errors.append(f"features.jsonl[{index}].top entries must have clip_id, window, t, value")
                break
            if item["clip_id"] not in clip_ids:
                errors.append(f"features.jsonl[{index}] top clip_id is unknown")
            if not _is_int(item["window"]) or not _is_int(item["t"]) or not 0 <= item["t"] < grid["t"]:
                errors.append(f"features.jsonl[{index}] top window/t is out of range")
            if not _is_number(item["value"]):
                errors.append(f"features.jsonl[{index}] top value must be numeric")
            seen_clips.add(item["clip_id"])
        else:
            if _is_int(row.get("distinct_clips_top20")) and row["distinct_clips_top20"] != len(seen_clips):
                errors.append(f"features.jsonl[{index}] distinct_clips_top20 does not match top")

    if acts_path.is_file():
        _check_feature_acts(acts_path, manifest, len(features), errors)
    measured = [
        int(feature["id"])
        for feature in features
        if isinstance(feature, dict) and feature.get("ablation_effect") is not None
    ]
    if effects_path.is_file():
        _check_effects(effects_path, manifest, len(features), measured, errors)
    _check_attribution(run_dir, manifest, len(features), errors)


def _check_feature_acts(path: Path, manifest: dict, n_features: int, errors: list[str]) -> None:
    try:
        table = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"feature_acts.parquet: {exc}")
        return
    if tuple(table.column_names) != FEATURE_ACT_COLUMNS:
        errors.append(f"feature_acts columns must be {FEATURE_ACT_COLUMNS}")
        return
    frame = table.to_pydict()
    clip_ids = {clip["clip_id"] for clip in manifest["clips"]}
    grid = manifest["grid"]
    for fid, clip_id, window, t, h, w, value in zip(
        frame["feature_id"],
        frame["clip_id"],
        frame["window"],
        frame["t"],
        frame["h"],
        frame["w"],
        frame["value"],
        strict=True,
    ):
        if not _is_int(fid) or not 0 <= int(fid) < n_features:
            errors.append("feature_acts feature_id out of range")
            return
        if clip_id not in clip_ids:
            errors.append("feature_acts clip_id is not in the manifest")
            return
        if not _is_int(window) or int(window) < 0:
            errors.append("feature_acts window must be a non-negative integer")
            return
        if not (_is_int(t) and _is_int(h) and _is_int(w)):
            errors.append("feature_acts coordinates must be integers")
            return
        if not (0 <= int(t) < grid["t"] and 0 <= int(h) < grid["h"] and 0 <= int(w) < grid["w"]):
            errors.append("feature_acts coordinate outside the grid")
            return
        if not _is_number(value) or float(value) <= 0.0:
            errors.append("feature_acts values must be strictly positive (sparse nonzeros only)")
            return


def _check_effects(path: Path, manifest: dict, n_features: int, measured: list[int], errors: list[str]) -> None:
    try:
        table = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"clip_effects.parquet: {exc}")
        return
    if tuple(table.column_names) != CLIP_EFFECT_COLUMNS:
        errors.append(f"clip_effects columns must be {CLIP_EFFECT_COLUMNS}")
        return
    if table.num_rows == 0 and "import" in manifest:
        # An imported run with no model in the loop; the reason is checked in _check_null_reasons.
        return
    frame = table.to_pydict()
    grid = manifest["grid"]
    try:
        shards = load_activation_shards(path.parent.parent / "acts")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"clip_effects could not be aligned to activation shards: {exc}")
        return
    windows = sorted(set(int(item) for item in shards["window"].tolist()))
    clip_ids = [clip["clip_id"] for clip in manifest["clips"]]
    estimate_keys = []
    ablation_keys = []
    extra_keys: dict[str, list] = {kind: [] for kind in KNOWN_EFFECT_KINDS if kind not in {EFFECT_KIND_ESTIMATE, EFFECT_KIND_ABLATION}}
    known_kinds = set(KNOWN_EFFECT_KINDS)
    for fid, clip_id, window, t, effect, kind in zip(
        frame["feature_id"],
        frame["clip_id"],
        frame["window"],
        frame["t"],
        frame["effect"],
        frame["effect_kind"],
        strict=True,
    ):
        if kind not in known_kinds:
            errors.append(f"effect_kind must be one of {KNOWN_EFFECT_KINDS}, got {kind!r}")
            return
        if not _is_int(fid) or not 0 <= int(fid) < n_features:
            errors.append("clip_effects feature_id out of range")
            return
        if clip_id not in clip_ids:
            errors.append("clip_effects clip_id is not in the manifest")
            return
        if int(window) not in windows or not 0 <= int(t) < grid["t"]:
            errors.append("clip_effects window/t is outside the captured grid")
            return
        if not _is_number(effect):
            errors.append("clip_effects effect must be finite")
            return
        key = (int(fid), clip_id, int(window), int(t))
        if kind == EFFECT_KIND_ESTIMATE:
            estimate_keys.append(key)
        elif kind == EFFECT_KIND_ABLATION:
            ablation_keys.append(key)
        else:
            extra_keys[kind].append(key)
    expected_estimates = {
        (fid, clip_id, window, t)
        for fid in range(n_features)
        for clip_id in clip_ids
        for window in windows
        for t in range(int(grid["t"]))
    }
    if set(estimate_keys) != expected_estimates or len(estimate_keys) != len(expected_estimates):
        errors.append("estimate rows must cover every feature, clip, window, and frame exactly once")
    if len(ablation_keys) != len(set(ablation_keys)):
        errors.append("ablation rows must be unique per feature, clip, window, and frame")
    from collections import defaultdict

    per_clip: dict[str, set[int]] = defaultdict(set)
    per_clip_rows: dict[str, list[tuple]] = defaultdict(list)
    for fid, clip_id, window, t in ablation_keys:
        per_clip[clip_id].add(fid)
        per_clip_rows[clip_id].append((fid, window, t))
    if len(measured) == n_features:
        expected_k = min(TOP_K_ABLATION_PER_CLIP, n_features)
        for clip_id in clip_ids:
            got = per_clip.get(clip_id, set())
            if len(got) != expected_k:
                errors.append(f"{clip_id} must ablate exactly {expected_k} features, got {len(got)}")
                continue
            expected_rows = {(fid, window, t) for fid in got for window in windows for t in range(int(grid["t"]))}
            if set(per_clip_rows[clip_id]) != expected_rows:
                errors.append(f"{clip_id} ablation rows must cover every frame of the ablated features")
    else:
        expected_ids = set(measured)
        for clip_id in clip_ids:
            got = per_clip.get(clip_id, set())
            if got != expected_ids:
                errors.append(
                    f"{clip_id} must ablate exactly the prefiltered features {sorted(expected_ids)}, got {sorted(got)}"
                )
                continue
            expected_rows = {
                (fid, window, t) for fid in expected_ids for window in windows for t in range(int(grid["t"]))
            }
            if set(per_clip_rows[clip_id]) != expected_rows:
                errors.append(f"{clip_id} ablation rows must cover every frame of the prefiltered features")
    if not estimate_keys or not ablation_keys:
        errors.append("clip_effects must contain both estimate and ablation rows")
    for kind, keys in extra_keys.items():
        if len(keys) != len(set(keys)):
            errors.append(f"{kind} rows must be unique per feature, clip, window, and frame")


def _schema_version(manifest: dict, errors: list[str]) -> int:
    if "schema_version" not in manifest:
        return 0
    value = manifest["schema_version"]
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        errors.append("manifest.schema_version must be 0 or 1")
        return -1
    return int(value)


def _check_rollouts(run_dir: Path, schema_version: int, errors: list[str]) -> None:
    rollout_dir = run_dir / "rollouts"
    has_rollouts = rollout_dir.exists()
    if schema_version == 1 and not has_rollouts:
        errors.append("schema_version 1 requires rollouts/")
        return
    if has_rollouts and schema_version != 1:
        errors.append("rollouts/ requires manifest.schema_version 1")
        return
    if schema_version != 1:
        return
    if not rollout_dir.is_dir():
        errors.append("rollouts/ must be a directory")
        return
    episodes_path = rollout_dir / "episodes.parquet"
    summary_path = rollout_dir / "summary.json"
    if not episodes_path.is_file() or not summary_path.is_file():
        errors.append("rollouts/episodes.parquet and rollouts/summary.json are required")
        return
    try:
        rows = read_rollout_episodes(episodes_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rollouts/episodes.parquet: {exc}")
        return
    try:
        summary = read_json(summary_path)
    except (OSError, ValueError) as exc:
        errors.append(f"rollouts/summary.json: {exc}")
        return
    if not isinstance(summary, dict):
        errors.append("rollouts/summary.json must be a JSON object")
        return
    _exact_keys(summary, SUMMARY_KEYS, "rollouts/summary.json", errors)
    if errors and any(error.startswith("rollouts/summary.json keys") for error in errors):
        return
    if summary.get("schema_version") != 1:
        errors.append("rollouts/summary.json schema_version must be 1")
    _check_rollout_rows(rows, summary, errors)
    if any(error.startswith("rollouts/episodes") for error in errors):
        return
    try:
        config = config_from_summary(summary)
        from interpretableintelligence.rollout_groot import summarize_rollout_rows

        rebuilt = summarize_rollout_rows(rows, config)
    except (TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"rollouts summary does not match the episode schedule: {exc}")
        return
    _same_json(summary, rebuilt, "rollouts/summary.json", errors)


def _check_rollout_rows(rows: list[dict], summary: dict, errors: list[str]) -> None:
    if not rows:
        errors.append("rollouts/episodes must contain at least one episode")
        return
    seen: set[str] = set()
    policy = summary.get("policy")
    for index, row in enumerate(rows):
        where = f"rollouts/episodes[{index}]"
        if tuple(row) != ROLLOUT_COLUMNS and set(row) != set(ROLLOUT_COLUMNS):
            errors.append(f"{where} is missing rollout columns")
            return
        episode_id = row["episode_id"]
        if not isinstance(episode_id, str) or not episode_id:
            errors.append(f"{where}.episode_id must be a non-empty string")
            return
        if episode_id in seen:
            errors.append(f"duplicate episode_id {episode_id}")
            return
        seen.add(episode_id)
        if not isinstance(row["task_id"], str) or not row["task_id"]:
            errors.append(f"{where}.task_id must be a non-empty string")
            return
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
            errors.append(f"{where}.seed must be an integer")
            return
        condition = row["condition"]
        feature_ids = row["feature_ids"]
        if not isinstance(feature_ids, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in feature_ids):
            errors.append(f"{where}.feature_ids must be a list of integers")
            return
        if condition == "baseline":
            if feature_ids or row["kind"] is not None or row["value"] is not None or row["layer"] is not None or row["dictionary"] is not None:
                errors.append(f"{where} baseline rows have an empty intervention spec")
                return
        elif condition == "intervention":
            if not feature_ids:
                errors.append(f"{where} intervention rows need at least one feature id")
                return
            if row["kind"] not in {"ablate", "scale", "clamp"}:
                errors.append(f"{where}.kind must be ablate, scale, or clamp")
                return
            if isinstance(row["layer"], bool) or not isinstance(row["layer"], int) or row["layer"] < 0:
                errors.append(f"{where}.layer must be a non-negative integer")
                return
            if not _is_number(row["value"]):
                errors.append(f"{where}.value must be finite")
                return
            try:
                dictionary_seed(row["dictionary"])
            except ValueError:
                errors.append(f"{where}.dictionary must be trained or random:<seed>")
                return
        else:
            errors.append(f"{where}.condition must be baseline or intervention")
            return
        if not isinstance(row["success"], bool):
            errors.append(f"{where}.success must be a boolean")
            return
        if isinstance(row["steps"], bool) or not isinstance(row["steps"], int) or row["steps"] < 1:
            errors.append(f"{where}.steps must be a positive integer")
            return
        if not _is_number(row["wall_time_s"]) or float(row["wall_time_s"]) < 0.0:
            errors.append(f"{where}.wall_time_s must be >= 0")
            return
        if isinstance(policy, str):
            expected_id = make_episode_id(
                policy=policy,
                condition=condition,
                task_id=row["task_id"],
                seed=row["seed"],
                feature_ids=feature_ids,
                layer=row["layer"],
                kind=row["kind"],
                value=row["value"],
                dictionary=row["dictionary"],
            )
            if episode_id != expected_id:
                errors.append(f"{where}.episode_id does not match the intervention spec")
                return
    for block_name, keys in (("conditions", CONDITION_KEYS), ("features", FEATURE_KEYS)):
        block = summary.get(block_name)
        if not isinstance(block, list) or not block:
            errors.append(f"rollouts/summary.json {block_name} must be a non-empty list")
            continue
        for index, item in enumerate(block):
            if not isinstance(item, dict):
                errors.append(f"rollouts/summary.json {block_name}[{index}] must be an object")
                continue
            _exact_keys(item, keys, f"rollouts/summary.json {block_name}[{index}]", errors)


def _same_json(left: object, right: object, where: str, errors: list[str]) -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            errors.append(f"{where} expected {right!r} but stored {left!r}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) or isinstance(right, float):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-8):
                errors.append(f"{where} expected {right!r} but stored {left!r}")
        elif left != right:
            errors.append(f"{where} expected {right!r} but stored {left!r}")
        return
    if left is None or right is None:
        if left is not right:
            errors.append(f"{where} expected {right!r} but stored {left!r}")
        return
    if isinstance(left, str) and isinstance(right, str):
        if left != right:
            errors.append(f"{where} expected {right!r} but stored {left!r}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{where} length {len(left)} != {len(right)}")
            return
        for index, (item_left, item_right) in enumerate(zip(left, right, strict=True)):
            _same_json(item_left, item_right, f"{where}[{index}]", errors)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            errors.append(f"{where} keys mismatch missing={sorted(set(right) - set(left))} extra={sorted(set(left) - set(right))}")
            return
        for key in left:
            _same_json(left[key], right[key], f"{where}.{key}", errors)
        return
    if left != right:
        errors.append(f"{where} expected {right!r} but stored {left!r}")



def _check_attribution(run_dir: Path, manifest: dict, n_features: int, errors: list[str]) -> None:
    path = run_dir / "features" / "attribution.parquet"
    candidates_path = run_dir / "features" / "candidates.json"
    if not path.is_file():
        errors.append("missing features/attribution.parquet")
    if not candidates_path.is_file():
        errors.append("missing features/candidates.json")
        return
    if not path.is_file():
        return
    try:
        table = pq.read_table(path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"attribution.parquet: {exc}")
        return
    if tuple(table.column_names) != ATTRIBUTION_COLUMNS:
        errors.append(f"attribution columns must be {ATTRIBUTION_COLUMNS}")
        return
    recorded = read_attribution_objective(path)
    imported = "import" in manifest
    # A scored run ranks by the adapter's gradient objective. An import has no model to backprop
    # through, so it may rank by any recorded objective (in practice ``activation``).
    allowed = RANK_OBJECTIVES if imported else SCORED_OBJECTIVES
    if recorded not in allowed:
        errors.append(
            f"attribution.parquet objective metadata must be one of {allowed}, got {recorded!r}"
        )
    frame = table.to_pydict()
    clip_ids = [clip["clip_id"] for clip in manifest["clips"] if isinstance(clip, dict)]
    seen: set[tuple[int, str]] = set()
    sums = {feature_id: 0.0 for feature_id in range(n_features)}
    counts = {feature_id: 0 for feature_id in range(n_features)}
    for feature_id, clip_id, score in zip(frame["feature_id"], frame["clip_id"], frame["score"], strict=True):
        if not _is_int(feature_id) or not 0 <= int(feature_id) < n_features:
            errors.append("attribution feature_id out of range")
            return
        if clip_id not in clip_ids:
            errors.append("attribution clip_id is not in the manifest")
            return
        if not _is_number(score):
            errors.append("attribution score must be finite")
            return
        key = (int(feature_id), str(clip_id))
        if key in seen:
            errors.append(f"duplicate attribution row for feature {feature_id} clip {clip_id}")
            return
        seen.add(key)
        sums[int(feature_id)] += float(score)
        counts[int(feature_id)] += 1
    expected = {(feature_id, clip_id) for feature_id in range(n_features) for clip_id in clip_ids}
    if seen != expected:
        errors.append("attribution rows must cover every feature and clip exactly once")
        return
    mean = np.array([sums[feature_id] / counts[feature_id] for feature_id in range(n_features)], dtype=np.float64)
    try:
        document = read_json(candidates_path)
    except (OSError, ValueError) as exc:
        errors.append(f"candidates.json: {exc}")
        return
    if not isinstance(document, dict):
        errors.append("candidates.json must be a JSON object")
        return
    _exact_keys(document, _CANDIDATE_KEYS, "candidates.json", errors)
    if document.get("method") != METHOD:
        errors.append(f"candidates.json method must be {METHOD}")
    if document.get("objective") != recorded:
        errors.append(f"candidates.json objective must match attribution.parquet ({recorded!r})")
    if document.get("shortlist_only") is not True:
        errors.append("candidates are a shortlist and must be confirmed by a real ablation")
    if not isinstance(document.get("rank_by"), str) or not document["rank_by"]:
        errors.append("candidates.json rank_by must be a non-empty string")
    prefilter_k = document.get("prefilter_k")
    ranked = document.get("ranked")
    shortlist = document.get("candidates")
    if not _is_int(prefilter_k) or not 1 <= int(prefilter_k) <= n_features:
        errors.append("candidates.json prefilter_k must be an integer in 1..n_features")
        return
    if not isinstance(ranked, list) or len(ranked) != n_features:
        errors.append("candidates.json ranked must list every feature once")
        return
    if not isinstance(shortlist, list) or shortlist != [row.get("feature_id") for row in ranked[: int(prefilter_k)]]:
        errors.append("candidates.json candidates must be the first prefilter_k ranked feature ids")
    expected_order = rank_feature_ids(mean)
    got_ids = []
    for index, row in enumerate(ranked):
        if not isinstance(row, dict):
            errors.append(f"candidates.json ranked[{index}] must be an object")
            return
        _exact_keys(row, _RANKED_KEYS, f"candidates.json ranked[{index}]", errors)
        if row.get("rank") != index + 1:
            errors.append(f"candidates.json ranks must be 1..n in order")
        if not isinstance(row.get("label"), str):
            errors.append(f"candidates.json ranked[{index}].label must be a string")
        feature_id = row.get("feature_id")
        if not _is_int(feature_id):
            errors.append(f"candidates.json ranked[{index}].feature_id must be an integer")
            return
        got_ids.append(int(feature_id))
        if not _is_number(row.get("score")) or not _is_number(row.get("abs_score")):
            errors.append(f"candidates.json ranked[{index}] scores must be finite")
            continue
        if abs(float(row["abs_score"]) - abs(float(row["score"]))) > 1e-8:
            errors.append(f"candidates.json ranked[{index}].abs_score must be |score|")
        if abs(float(row["score"]) - float(mean[int(feature_id)])) > 1e-5:
            errors.append(
                f"candidates.json ranked[{index}] score does not match the mean attribution parquet row"
            )
    if got_ids != expected_order:
        errors.append("candidates.json is not sorted by descending |mean attribution| (ties: smaller feature id)")
    seconds = document.get("seconds")
    if not isinstance(seconds, dict):
        errors.append("candidates.json seconds must be an object")
        return
    _exact_keys(seconds, _SECOND_KEYS, "candidates.json seconds", errors)
    for key in _SECOND_KEYS:
        value = seconds.get(key)
        if value is not None and (not _is_number(value) or float(value) < 0):
            errors.append(f"candidates.json seconds.{key} must be a non-negative number or null")




def _read_object(path: Path, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"missing {path.name}")
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path.name} must be a JSON object")
        return None
    return payload


def _exact_keys(payload: dict, expected: set[str], where: str, errors: list[str]) -> None:
    _allow_keys(payload, expected, set(), where, errors)


def _allow_keys(payload: dict, required: set[str], optional: set[str], where: str, errors: list[str]) -> None:
    got = set(payload)
    missing = sorted(required - got)
    extra = sorted(got - required - optional)
    if missing or extra:
        errors.append(f"{where} keys mismatch missing={missing} extra={extra}")


def _check_manifest_extras(manifest: dict, errors: list[str]) -> None:
    if "clips_dir" in manifest and (not isinstance(manifest["clips_dir"], str) or not manifest["clips_dir"]):
        errors.append("manifest.clips_dir must be a non-empty string")
    if "stages" not in manifest:
        return
    stages = manifest["stages"]
    if not isinstance(stages, dict):
        errors.append("manifest.stages must be an object")
        return
    for name, stage in stages.items():
        if not isinstance(name, str) or not name:
            errors.append("manifest.stages keys must be non-empty strings")
            continue
        if not isinstance(stage, dict):
            errors.append(f"manifest.stages.{name} must be an object")
            continue
        _exact_keys(stage, _STAGE_FIELDS, f"manifest.stages.{name}", errors)
        if not _is_number(stage.get("wall_s")) or float(stage.get("wall_s", -1)) < 0:
            errors.append(f"manifest.stages.{name}.wall_s must be a non-negative number")
        if not _is_int(stage.get("peak_ram_bytes")) or stage.get("peak_ram_bytes", -1) < 0:
            errors.append(f"manifest.stages.{name}.peak_ram_bytes must be a non-negative integer")
        if not isinstance(stage.get("device"), str) or not stage["device"]:
            errors.append(f"manifest.stages.{name}.device must be a non-empty string")
        if not _is_int(stage.get("rows")) or stage.get("rows", -1) < 0:
            errors.append(f"manifest.stages.{name}.rows must be a non-negative integer")
        if not _is_int(stage.get("tokens")) or stage.get("tokens", -1) < 0:
            errors.append(f"manifest.stages.{name}.tokens must be a non-negative integer")


def _hex_field(value: object, where: str, errors: list[str]) -> None:
    if not isinstance(value, str) or _HEX64.match(value) is None:
        errors.append(f"{where} must be a 64-char lowercase hex sha256")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _cpu():
    import torch

    return torch.device("cpu")


def torch_finite(tensor) -> bool:
    return bool(tensor.detach().isfinite().all())
