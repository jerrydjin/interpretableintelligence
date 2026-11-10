"""Resumable capture → SAE → features → score → viewer.

Activation rows stay in an on-disk cache. Each stage reads or writes a chunk,
records wall time, peak RSS, device, and row/token counts on the manifest,
and skips itself when those outputs are still valid for the same clips.
"""

from __future__ import annotations

import heapq
import json
import math
import shutil
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from interpretableintelligence import __version__
from interpretableintelligence.adapters import get_adapter
from interpretableintelligence.adapters.fake import CUE_NAMES, cue_volume
from interpretableintelligence.artifacts import (
    ATTRIBUTION_COLUMNS,
    CLIP_EFFECT_COLUMNS,
    FEATURE_ACT_COLUMNS,
    read_json,
    write_attribution,
    write_json,
)
from interpretableintelligence.attribute import (
    PREDICTION_LOSS,
    attribution_by_clip,
    candidates_document,
    rank_feature_ids,
    summarize_attribution,
)
from interpretableintelligence.clips import file_sha256, list_clip_meta, load_clip
from interpretableintelligence.device import pick_device
from interpretableintelligence.labelers import FakeCueLabeller, NullLabeller
from interpretableintelligence.sae import SAEConfig, load_sae, save_sae, train_sae_fetched
from interpretableintelligence.score import (
    ablate_edit,
    check_extra_effects,
    cosine_distance,
    estimate_effects,
    extra_effect_rows,
    predictor_sse,
    splice_edit,
)
from interpretableintelligence.split import assign_split
from interpretableintelligence.stats import feature_is_real, is_usable, loss_recovered, percentile_le
from interpretableintelligence.store import (
    STREAM_CHUNK,
    ActivationWriter,
    RamSampler,
    RowSource,
    cache_matches_layout,
)
from interpretableintelligence.thresholds import (
    DEAD_ACT_EPS,
    DEFAULT_HELDOUT_FRAC,
    DEFAULT_LAYER,
    DEFAULT_PREFILTER_K,
    DEFAULT_SEED,
    LABEL_ASSIGN_MIN_CORR,
    LABEL_HOLD_MIN_CORR,
    MIN_CLIPS,
    N_RANDOM_FEATURES,
    PATTERN_ABSENT_MAX_CORR,
    TOP_K_ABLATION_PER_CLIP,
    TOP_K_EXAMPLES,
)
from interpretableintelligence.view import bind_server, load_run

STAGES = ("capture", "sae", "features", "score", "view")
REPORT_BEGIN = "--- interpretableintelligence cost report ---"
REPORT_END = "--- end cost report ---"

_ACT_SCHEMA = pa.schema(
    [
        ("feature_id", pa.int32()),
        ("clip_id", pa.string()),
        ("window", pa.int16()),
        ("t", pa.int16()),
        ("h", pa.int16()),
        ("w", pa.int16()),
        ("value", pa.float32()),
    ]
)
_EFFECT_SCHEMA = pa.schema(
    [
        ("feature_id", pa.int32()),
        ("clip_id", pa.string()),
        ("window", pa.int16()),
        ("t", pa.int16()),
        ("effect", pa.float32()),
        ("effect_kind", pa.string()),
    ]
)


def footage_cost(stage_seconds: float, video_seconds: float, gpu_cost_per_hour: float) -> float:
    """Price of one hour of footage from measured stage time.

    ``(stage_seconds / 3600) * gpu_cost_per_hour / (video_seconds / 3600)``
    simplifies to ``stage_seconds * gpu_cost_per_hour / video_seconds``.
    """
    if not math.isfinite(stage_seconds) or stage_seconds < 0:
        raise ValueError("stage_seconds must be a non-negative finite number")
    if not math.isfinite(video_seconds) or video_seconds <= 0:
        raise ValueError("video_seconds must be positive to price an hour of footage")
    if not math.isfinite(gpu_cost_per_hour) or gpu_cost_per_hour < 0:
        raise ValueError("gpu_cost_per_hour must be a non-negative finite number")
    return float(stage_seconds) * float(gpu_cost_per_hour) / float(video_seconds)


def format_report(report: dict) -> str:
    lines = [
        REPORT_BEGIN,
        f"clips: {int(report['clips'])}",
        f"video_s: {float(report['video_s']):.6f}",
        f"tokens: {int(report['tokens'])}",
    ]
    stages = report["stages"]
    for name in STAGES:
        metrics = stages.get(name)
        if metrics is None:
            lines.append(f"stage {name}: not run")
        else:
            lines.append(f"stage {name}: {float(metrics['wall_s']):.6f}s")
    if report.get("gpu_cost_per_hour") is not None:
        lines.append(f"gpu_cost_per_hour: {format(float(report['gpu_cost_per_hour']), 'g')}")
        lines.append(f"cost_per_hour_footage: {float(report['cost_per_hour_footage']):.6f}")
    lines.append(REPORT_END)
    return "\n".join(lines)


def execute_pipeline(
    adapter_name: str,
    clips_dir: Path,
    out_dir: Path,
    layer: int = DEFAULT_LAYER,
    seed: int = DEFAULT_SEED,
    heldout_frac: float = DEFAULT_HELDOUT_FRAC,
    tiny: bool = False,
    prefilter: int | None = None,
    extra_effects: tuple[str, ...] = (),
    gpu_cost_per_hour: float | None = None,
    score: bool = True,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = False,
    log=print,
) -> dict:
    """Run every missing stage and, unless ``serve`` is false, block on the viewer."""
    if gpu_cost_per_hour is not None and (not math.isfinite(gpu_cost_per_hour) or gpu_cost_per_hour < 0):
        raise ValueError("gpu_cost_per_hour must be a non-negative finite number")
    if not 0.0 < float(heldout_frac) < 1.0:
        raise ValueError(f"heldout_frac must be in (0, 1), got {heldout_frac}")
    if tiny and adapter_name != "vjepa2":
        raise ValueError("--tiny is only supported for the vjepa2 adapter")
    if prefilter is not None and int(prefilter) < 1:
        raise ValueError(f"prefilter K must be >= 1, got {prefilter}")
    prefilter = None if prefilter is None else int(prefilter)
    extra_effects = check_extra_effects(extra_effects)

    torch.set_num_threads(1)
    device = pick_device()
    clips_dir = Path(clips_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = list_clip_meta(clips_dir)
    if len(meta) < MIN_CLIPS:
        raise ValueError(f"need at least {MIN_CLIPS} clips, found {len(meta)} in {clips_dir}")

    adapter_cls = get_adapter(adapter_name)
    adapter = adapter_cls(layer=layer, device=device, tiny=tiny)
    adapter.load(random_init=False)
    grid = adapter.grid().as_dict()
    d_model = int(adapter.d_model)
    model_sha = adapter.model_sha()
    capture_inputs = _capture_inputs(
        adapter_name=adapter.name,
        layer=layer,
        seed=seed,
        heldout_frac=heldout_frac,
        model_sha=model_sha,
        d_model=d_model,
        grid=grid,
        meta=meta,
        tiny=tiny,
    )
    config = SAEConfig(d_model=d_model, seed=int(seed))
    sae_inputs = _with_kind(capture_inputs, "sae", sae=config.as_dict())

    if _capture_ready(out, capture_inputs, grid, len(meta)):
        log("skip capture")
    else:
        log("run capture")
        _invalidate(out, "capture")
        with _measure(device) as measured:
            rows = _capture(adapter, meta, out, layer, seed, heldout_frac, model_sha, capture_inputs, clips_dir)
            measured["rows"] = rows
            measured["tokens"] = rows
        _record_stage(out, "capture", measured["metrics"])
        _write_complete(out / "acts" / "complete.json", capture_inputs, measured["metrics"])

    if _sae_ready(out, sae_inputs, config):
        log("skip sae")
    else:
        log("run sae")
        _invalidate(out, "sae")
        with _measure(device) as measured:
            trained = _train(out / "acts", config, device, log)
            measured["rows"] = trained
            measured["tokens"] = trained
        _record_stage(out, "sae", measured["metrics"])
        _write_complete(out / "sae" / "complete.json", sae_inputs, measured["metrics"])

    weights_sha = file_sha256(out / "sae" / "weights.safetensors")
    feature_inputs = _with_kind(sae_inputs, "features", weights_sha=weights_sha)
    if _features_ready(out, feature_inputs):
        log("skip features")
    else:
        log("run features")
        _invalidate(out, "features")
        with _measure(device) as measured:
            sparse_rows, tokens = _features(out, meta, config, device)
            measured["rows"] = sparse_rows
            measured["tokens"] = tokens
        _record_stage(out, "features", measured["metrics"])
        _write_complete(out / "features" / "complete.json", feature_inputs, measured["metrics"])

    stats_path = out / "features" / "stats.json"
    score_inputs = _with_kind(
        feature_inputs,
        "score",
        stats_sha=file_sha256(stats_path),
        prefilter=prefilter,
        extra_effects=list(extra_effects),
    )
    scoreboard = _read_scoreboard(out)
    if not score:
        log("skip score")
    elif _score_ready(out, score_inputs, config.n_features, len(meta), grid):
        log("skip score")
        scoreboard = _read_scoreboard(out)
    else:
        log("run score")
        _invalidate(out, "score")
        with _measure(device) as measured:
            scoreboard, effect_rows = _score(
                adapter_cls,
                adapter,
                meta,
                out,
                config,
                device,
                layer,
                seed,
                model_sha,
                capture_inputs,
                log,
                tiny=tiny,
                prefilter=prefilter,
                extra_effects=extra_effects,
            )
            measured["rows"] = effect_rows
            measured["tokens"] = int(read_json(out / "acts" / "stream" / "meta.json")["n"])
        _record_stage(out, "score", measured["metrics"])
        _write_complete(out / "score.complete.json", score_inputs, measured["metrics"])

    view_inputs = _with_kind(feature_inputs, "view")
    if (out / "scoreboard.json").is_file():
        view_inputs = _with_kind(view_inputs, "view", scoreboard_sha=file_sha256(out / "scoreboard.json"))
    httpd = None
    thread = None
    if _view_ready(out, view_inputs):
        log("skip view")
        if serve:
            _model, httpd, thread, _url = _open_viewer(out, clips_dir, host, port, open_browser, log)
    else:
        log("run view")
        with _measure(device) as measured:
            model, httpd, thread, _url = _open_viewer(out, clips_dir, host, port, open_browser, log, serve=serve)
            measured["rows"] = len(model.run_payload["clips"])
            measured["tokens"] = 0
        _record_stage(out, "view", measured["metrics"])
        _write_complete(out / "view.complete.json", view_inputs, measured["metrics"])

    manifest = read_json(out / "manifest.json")
    report = _report(manifest, gpu_cost_per_hour)
    for line in format_report(report).splitlines():
        log(line)
    if not serve:
        log("viewer not started")
    try:
        if thread is not None:
            thread.join()
    except KeyboardInterrupt:
        raise
    finally:
        if httpd is not None:
            httpd.shutdown()
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)
            httpd.server_close()
    return {"report": report, "scoreboard": scoreboard, "manifest": manifest}


def _report(manifest: dict, gpu_cost_per_hour: float | None) -> dict:
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), dict) else {}
    capture = stages.get("capture")
    if not isinstance(capture, dict) or "tokens" not in capture:
        raise RuntimeError("manifest is missing capture token counts")
    video_s = 0.0
    for clip in manifest["clips"]:
        video_s += float(clip["n_frames"]) / float(clip["fps"])
    report = {
        "clips": len(manifest["clips"]),
        "video_s": video_s,
        "tokens": int(capture["tokens"]),
        "stages": stages,
        "gpu_cost_per_hour": None if gpu_cost_per_hour is None else float(gpu_cost_per_hour),
        "cost_per_hour_footage": None,
    }
    if gpu_cost_per_hour is not None:
        elapsed = sum(float(stage["wall_s"]) for stage in stages.values())
        report["cost_per_hour_footage"] = footage_cost(elapsed, video_s, float(gpu_cost_per_hour))
    return report


def _capture(adapter, meta, out, layer, seed, heldout_frac, model_sha, inputs, clips_dir: Path) -> int:
    split = assign_split([item["clip_id"] for item in meta], int(seed), float(heldout_frac))
    membership = split.membership()
    grid = adapter.grid()
    expected_windows: list[int] | None = None
    with ActivationWriter(out / "acts", int(adapter.d_model)) as writer:
        for clip_index, item in enumerate(meta):
            clip = load_clip(item)
            try:
                windows = adapter.windows(clip)
                indexes = [int(window.index) for window in windows]
                if indexes != list(range(len(indexes))):
                    raise RuntimeError(f"{clip.clip_id} windows must be 0..n-1 in order, got {indexes}")
                if expected_windows is None:
                    expected_windows = indexes
                elif indexes != expected_windows:
                    raise RuntimeError("every clip must yield the same windows")
                code = 0 if membership[clip.clip_id] == "train" else 1
                for window in windows:
                    with adapter.hook(adapter.layer, mode="capture") as handle:
                        adapter.encode(clip, window)
                    if len(handle.captured) != 1:
                        raise RuntimeError(f"expected one captured tensor for {clip.clip_id}")
                    acts = handle.captured[0].detach().to(dtype=torch.float32, device="cpu").numpy()
                    if tuple(int(dim) for dim in acts.shape[:3]) != (grid.t, grid.h, grid.w):
                        raise RuntimeError(f"{clip.clip_id} activation grid {acts.shape} != {grid.as_dict()}")
                    if int(acts.shape[-1]) != int(adapter.d_model):
                        raise RuntimeError(f"captured d_model {acts.shape[-1]} != {adapter.d_model}")
                    writer.append(acts, clip_index, int(window.index), code)
                    del acts
            finally:
                _clear_cache(adapter)
                del clip
        store_meta = writer.finish()
    if not cache_matches_layout(out / "acts", len(meta), grid.as_dict()):
        raise RuntimeError("captured rows are not a rectangular token grid")
    manifest = {
        "run_id": out.name,
        "tool_version": __version__,
        "model": {
            "adapter": adapter.name,
            "name": getattr(adapter, "model_name", adapter.name),
            "sha": model_sha,
            "random_init": False,
        },
        "layer": int(layer),
        "d_model": int(adapter.d_model),
        "grid": grid.as_dict(),
        "split": {"seed": split.seed, "heldout_frac": split.heldout_frac, "sha": split.sha},
        "clips": [
            {
                "clip_id": item["clip_id"],
                "sha256": item["sha256"],
                "fps": float(item["fps"]),
                "n_frames": int(item["n_frames"]),
                "split": membership[item["clip_id"]],
            }
            for item in meta
        ],
        "clips_dir": str(Path(clips_dir).resolve()),
        "stages": {},
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "split.json", split.as_dict())
    del inputs
    return int(store_meta["n"])


def _train(acts_dir: Path, config: SAEConfig, device: torch.device, log) -> int:
    with RowSource(acts_dir) as source:
        if source.n_train < 2:
            raise RuntimeError("need at least 2 train activation rows")
        mean = _mean_rows(source, source.train_ids, source.n_train)
        model = train_sae_fetched(
            source.n_train,
            torch.from_numpy(mean),
            source.gather_train,
            config,
            device,
            log=log,
        )
        trained = source.n_train
    save_sae(model, config, acts_dir.parent / "sae")
    return int(trained)


def _features(out: Path, meta: list[dict], config: SAEConfig, device: torch.device) -> tuple[int, int]:
    features_dir = out / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    model, _loaded = load_sae(out / "sae", device)
    n_features = int(model.n_features)
    with RowSource(out / "acts") as source:
        if source.n_heldout < 1:
            raise RuntimeError("capture has no held-out rows")
        mean = _mean_rows(source, source.held_ids, source.n_heldout)
        mu = torch.from_numpy(mean).view(1, -1)
        chunk = _chunk_rows(n_features, source.d_model)
        tops = [_TopK(TOP_K_EXAMPLES) for _ in range(n_features)]
        freq = np.zeros(n_features, dtype=np.int64)
        max_act = np.zeros(n_features, dtype=np.float64)
        sst = 0.0
        sse = 0.0
        writer = _ColumnBatch(features_dir / "feature_acts.parquet", _ACT_SCHEMA)
        try:
            for start, stop in source.chunks(chunk):
                acts = np.array(source.rows[start:stop], dtype=np.float32, copy=True)
                split = np.array(source.split[start:stop], dtype=np.uint8, copy=True)
                with torch.no_grad():
                    encoded = model.encode(torch.from_numpy(acts).to(device)).detach().cpu()
                    held = split == 1
                    if bool(held.any()):
                        x = torch.from_numpy(np.ascontiguousarray(acts[held]))
                        held_features = encoded[torch.from_numpy(np.ascontiguousarray(held))]
                        xhat = model.decode(held_features.to(device)).detach().cpu()
                        sst += float(((x - mu) ** 2).sum())
                        sse += float(((x - xhat) ** 2).sum())
                feats = encoded.numpy()
                token_index, feature_index = np.nonzero(feats > DEAD_ACT_EPS)
                if token_index.size == 0:
                    continue
                global_rows = token_index + start
                clip_index = np.array(source.clip_idx[global_rows], dtype=np.int64, copy=True)
                window = np.array(source.window[global_rows], dtype=np.int16, copy=True)
                t = np.array(source.t[global_rows], dtype=np.int16, copy=True)
                h = np.array(source.h[global_rows], dtype=np.int16, copy=True)
                w = np.array(source.w[global_rows], dtype=np.int16, copy=True)
                values = feats[token_index, feature_index]
                writer.add(
                    feature_id=feature_index.astype(np.int32, copy=False),
                    clip_id=[meta[int(index)]["clip_id"] for index in clip_index],
                    window=window,
                    t=t,
                    h=h,
                    w=w,
                    value=np.ascontiguousarray(values, dtype=np.float32),
                )
                for feature_id in np.unique(feature_index):
                    selected = feature_index == feature_id
                    chosen = values[selected]
                    freq[int(feature_id)] += int(chosen.size)
                    max_act[int(feature_id)] = max(float(max_act[int(feature_id)]), float(chosen.max()))
                    rows = global_rows[selected]
                    clips = clip_index[selected]
                    wins = window[selected]
                    times = t[selected]
                    for offset in range(int(chosen.size)):
                        tops[int(feature_id)].add(
                            float(chosen[offset]),
                            int(rows[offset]),
                            meta[int(clips[offset])]["clip_id"],
                            int(wins[offset]),
                            int(times[offset]),
                        )
            sparse_rows = writer.close()
        except Exception:
            writer.abort()
            raise
        tokens = int(source.n)
        n_held = int(source.n_heldout)
    ev = 0.0 if sst <= 1e-12 else float(1.0 - (sse / sst))
    dead = float(np.mean(freq == 0)) if n_features else 0.0
    feature_stats = []
    candidates = []
    for feature_id in range(n_features):
        top, distinct = tops[feature_id].export()
        row = {
            "id": feature_id,
            "freq": int(freq[feature_id]),
            "max_act": float(max_act[feature_id]),
            "distinct_clips_top20": int(distinct),
            "top": top,
        }
        feature_stats.append(row)
        if row["freq"] > 0:
            candidates.append(
                {
                    "feature_id": feature_id,
                    "distinct_clips_top20": row["distinct_clips_top20"],
                    "freq": row["freq"],
                    "max_act": row["max_act"],
                }
            )
    candidates.sort(key=lambda item: (-float(item["max_act"]), int(item["feature_id"])))
    _write_any(
        features_dir / "stats.json",
        {
            "n_tokens": tokens,
            "n_heldout": n_held,
            "n_features": n_features,
            "sparse_rows": int(sparse_rows),
            "dead_frac": dead,
            "ev_heldout": ev,
            "features": feature_stats,
        },
    )
    _write_any(features_dir / "candidates.json", candidates)
    del model
    return int(sparse_rows), tokens


def _score(
    adapter_cls,
    adapter,
    meta,
    out,
    config,
    device,
    layer,
    seed,
    model_sha,
    capture_inputs,
    log,
    tiny: bool = False,
    prefilter: int | None = None,
    extra_effects: tuple[str, ...] = (),
) -> tuple[dict, int]:
    stats = json.loads((out / "features" / "stats.json").read_text(encoding="utf-8"))
    manifest = read_json(out / "manifest.json")
    split = read_json(out / "split.json")
    model, _loaded = load_sae(out / "sae", device)
    n_features = int(model.n_features)
    if len(stats["features"]) != n_features:
        raise RuntimeError("feature stats do not match the SAE")
    control_inputs = {"kind": "control", "capture": capture_inputs, "sae": config.as_dict()}
    control_dir = out / "control"
    if _control_acts_ready(control_dir / "acts", control_inputs, manifest["grid"], len(meta)):
        log("score: reuse random-init activations")
    else:
        _rm(control_dir)
        control = adapter_cls(layer=layer, device=device, tiny=tiny)
        control.load(random_init=True)
        _capture_control(control, meta, control_dir / "acts", manifest)
        _write_complete(control_dir / "acts" / "complete.json", control_inputs, {})
        _clear_cache(control)
        del control
    if _control_sae_ready(control_dir / "sae", control_inputs, config):
        log("score: reuse random-init SAE")
    else:
        _rm(control_dir / "sae")
        with RowSource(control_dir / "acts") as source:
            mean = _mean_rows(source, source.train_ids, source.n_train)
            control_model = train_sae_fetched(
                source.n_train,
                torch.from_numpy(mean),
                source.gather_train,
                config,
                device,
                log=log,
            )
        save_sae(control_model, config, control_dir / "sae")
        del control_model
        _write_complete(control_dir / "sae" / "complete.json", control_inputs, {})
    control_model, _control_config = load_sae(control_dir / "sae", device)
    labels, absent = _correlate(
        out / "acts",
        control_dir / "acts",
        model,
        control_model,
        meta,
        device,
        label_cues=adapter.name == "fake",
    )
    del control_model

    membership = {item["clip_id"]: item for item in manifest["clips"]}
    heldout_meta = [item for item in meta if membership[item["clip_id"]]["split"] == "heldout"]
    loss_clean = _mean_loss(adapter, heldout_meta)
    loss_zero = _mean_loss(adapter, heldout_meta, zero=True)
    loss_sae = _mean_loss(adapter, heldout_meta, edit=splice_edit(model))
    loss_visible_mean = _mean_visible_loss(adapter, heldout_meta)
    recovered = loss_recovered(loss_zero, loss_sae, loss_clean)
    usable = is_usable(recovered, loss_sae, loss_visible_mean)
    ev = float(stats["ev_heldout"])
    dead = float(stats["dead_frac"])
    log(
        f"ev_heldout={ev:.4f}  loss_clean={loss_clean:.6f}  loss_zero={loss_zero:.6f}  "
        f"loss_sae={loss_sae:.6f}  loss_visible_mean={loss_visible_mean:.6f}  "
        f"loss_recovered={recovered:.4f}  dead_frac={dead:.3f}  usable={usable}"
    )

    _clear_cache(adapter)
    attribution_started = time.perf_counter()
    by_clip = attribution_by_clip(adapter, model, [load_clip(item) for item in meta], objective=PREDICTION_LOSS)
    attribution_seconds = time.perf_counter() - attribution_started
    _clear_cache(adapter)
    attr_rows, mean_scores = summarize_attribution(by_clip)
    ranked_ids = rank_feature_ids(mean_scores)
    if prefilter is None:
        selected = None
        reported_k = min(DEFAULT_PREFILTER_K, n_features)
    else:
        reported_k = min(int(prefilter), n_features)
        selected = set(ranked_ids[:reported_k])
        log(f"prefilter K={reported_k}  candidates={ranked_ids[:reported_k]}")

    top_k = min(TOP_K_ABLATION_PER_CLIP, n_features)
    effects_path = out / "features" / "clip_effects.parquet"
    effect_writer = _ColumnBatch(effects_path, _EFFECT_SCHEMA)
    try:
        rank = _estimate_rows(adapter, model, meta, effect_writer, n_features)
        chosen = {
            clip_id: [int(index) for index in np.argsort(-np.asarray(scores), kind="stable")[:top_k]]
            for clip_id, scores in rank.items()
        }
        ablation_started = time.perf_counter()
        held_total, held_count = _ablation_rows(
            adapter,
            model,
            meta,
            membership,
            chosen,
            n_features,
            effect_writer,
            only=selected,
        )
        ablation_seconds = time.perf_counter() - ablation_started
        if extra_effects:
            clips = [load_clip(item) for item in meta]
            features = _feature_matrix(out / "acts", model, device)
            ablated: set[tuple[str, int]] = set()
            for item in meta:
                clip_id = item["clip_id"]
                ids = selected if selected is not None else chosen[clip_id]
                for feature_id in ids:
                    ablated.add((clip_id, int(feature_id)))
            extra = extra_effect_rows(adapter, model, clips, features, ablated, extra_effects, log)
            effect_writer.add(**extra)
        effects = effect_writer.close()
    except Exception:
        effect_writer.abort()
        raise
    fire_source = out / "features" / "feature_acts.parquet"
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("CREATE TABLE held_clips (clip_id VARCHAR)")
        connection.executemany("INSERT INTO held_clips VALUES (?)", [(item["clip_id"],) for item in heldout_meta])
        with RowSource(out / "acts") as source:
            pool = _heldout_pool(source, len(meta))
        records = []
        jsonl = []
        real_labels = []
        control_seconds = 0.0
        for feature_id in range(n_features):
            info = stats["features"][feature_id]
            labelled = labels[feature_id]
            measured = selected is None or feature_id in selected
            if not measured:
                effect = None
                percentile = None
                n_random = 0
            else:
                if int(held_count[feature_id]) == 0:
                    effect = 0.0
                else:
                    effect = float(held_total[feature_id] / held_count[feature_id])
                started_controls = time.perf_counter()
                fires = _fire_values(connection, fire_source, feature_id)
                percentile = _random_percentile(
                    adapter=adapter,
                    sae=model,
                    feature_id=feature_id,
                    real_effect=effect,
                    fires=fires,
                    pool=pool,
                    meta=meta,
                    heldout_meta=heldout_meta,
                    seed=int(seed),
                )
                control_seconds += time.perf_counter() - started_controls
                n_random = N_RANDOM_FEATURES
            absent_flag = bool(absent[feature_id])
            real = feature_is_real(
                usable=usable,
                distinct_clips_top20=int(info["distinct_clips_top20"]),
                label_holds_heldout=labelled["label_holds_heldout"],
                random_percentile=percentile,
                absent_in_random_init=absent_flag,
            )
            record = {
                "id": feature_id,
                "label": labelled["label"],
                "distinct_clips_top20": int(info["distinct_clips_top20"]),
                "label_holds_heldout": labelled["label_holds_heldout"],
                "ablation_effect": effect,
                "random_percentile": None if percentile is None else float(percentile),
                "n_random": n_random,
                "absent_in_random_init": absent_flag,
                "real": bool(real),
            }
            records.append(record)
            jsonl.append(
                {
                    "id": feature_id,
                    "label": labelled["label"],
                    "label_model": labelled["label_model"],
                    "top": info["top"],
                    "distinct_clips_top20": int(info["distinct_clips_top20"]),
                    "label_holds_heldout": labelled["label_holds_heldout"],
                    "ablation_effect": effect,
                    "random_percentile": None if percentile is None else float(percentile),
                    "n_random": n_random,
                    "absent_in_random_init": absent_flag,
                    "real": bool(real),
                }
            )
            if real:
                real_labels.append(labelled["label"])
            pct_text = f"{float(percentile):6.1f}" if percentile is not None else "     -"
            abl_text = f"{effect:.5f}" if effect is not None else "skipped"
            log(
                f"feature {feature_id:02d}  label={labelled['label'] or '-':<8}  real={str(bool(real)).lower():<5}  "
                f"distinct={int(info['distinct_clips_top20']):<3}  holds={labelled['label_holds_heldout']}  "
                f"pct={pct_text}  absent={absent_flag}  abl={abl_text}"
            )
    finally:
        connection.close()
        _clear_cache(adapter)
        del model

    scoreboard = {
        "run": {
            "model_sha": model_sha,
            "layer": int(layer),
            "split_sha": split["sha"],
            "seed": int(seed),
            "n_train_clips": len(split["train"]),
            "n_heldout_clips": len(split["heldout"]),
        },
        "sae": {
            "ev_heldout": ev,
            "loss_recovered": float(recovered),
            "loss_sae": float(loss_sae),
            "loss_visible_mean": float(loss_visible_mean),
            "dead_frac": dead,
            "usable": bool(usable),
        },
        "features": records,
    }
    write_json(out / "scoreboard.json", scoreboard)
    _write_jsonl(out / "features" / "features.jsonl", jsonl)
    write_attribution(out / "features" / "attribution.parquet", attr_rows, objective=PREDICTION_LOSS)
    write_json(
        out / "features" / "candidates.json",
        candidates_document(
            mean_scores,
            reported_k,
            objective=PREDICTION_LOSS,
            labels=[row["label"] for row in labels],
            seconds={
                "attribution": attribution_seconds,
                "ablation": ablation_seconds,
                "random_controls": control_seconds,
            },
        ),
    )
    log(
        f"attribution {attribution_seconds:.3f}s  ablation {ablation_seconds:.3f}s  "
        f"random_controls {control_seconds:.3f}s"
    )
    log(f"wrote {out}  real={len(real_labels)} {real_labels}")
    return scoreboard, int(effects)


def _capture_control(adapter, meta, acts_dir: Path, manifest: dict) -> None:
    grid = adapter.grid()
    manifest_grid = manifest["grid"]
    with ActivationWriter(acts_dir, int(adapter.d_model)) as writer:
        for clip_index, item in enumerate(meta):
            clip = load_clip(item)
            try:
                recorded = manifest["clips"][clip_index]
                if recorded["clip_id"] != item["clip_id"]:
                    raise RuntimeError("control capture clip order does not match the manifest")
                code = 0 if recorded["split"] == "train" else 1
                for window in adapter.windows(clip):
                    with adapter.hook(adapter.layer, mode="capture") as handle:
                        adapter.encode(clip, window)
                    acts = handle.captured[0].detach().to(dtype=torch.float32, device="cpu").numpy()
                    if tuple(int(dim) for dim in acts.shape[:3]) != (
                        int(manifest_grid["t"]),
                        int(manifest_grid["h"]),
                        int(manifest_grid["w"]),
                    ):
                        raise RuntimeError("random-init activations are not aligned with the run")
                    writer.append(acts, clip_index, int(window.index), code)
                    del acts
            finally:
                _clear_cache(adapter)
                del clip
        writer.finish()
    if not cache_matches_layout(acts_dir, len(meta), grid.as_dict()):
        raise RuntimeError("random-init activations are not aligned with the run")
    main_acts = acts_dir.parent.parent / "acts"
    if not _same_coordinates(main_acts, acts_dir):
        raise RuntimeError("random-init activations are not aligned with the run")


def _correlate(
    main_acts: Path,
    control_acts: Path,
    model,
    control_model,
    meta,
    device,
    label_cues: bool = True,
) -> tuple[list[dict], list[bool]]:
    n_features = int(model.n_features)
    n_cues = len(CUE_NAMES)
    label_n = np.zeros(2, dtype=np.int64)
    sum_a = np.zeros((2, n_features), dtype=np.float64)
    sum_a2 = np.zeros((2, n_features), dtype=np.float64)
    sum_c = np.zeros((2, n_cues), dtype=np.float64)
    sum_c2 = np.zeros((2, n_cues), dtype=np.float64)
    sum_ac = np.zeros((2, n_features, n_cues), dtype=np.float64)
    sum_main = np.zeros(n_features, dtype=np.float64)
    sum_main2 = np.zeros(n_features, dtype=np.float64)
    sum_ctrl = np.zeros(n_features, dtype=np.float64)
    sum_ctrl2 = np.zeros(n_features, dtype=np.float64)
    cross = np.zeros((n_features, n_features), dtype=np.float64)
    seen = 0
    cues = _CueCache()
    chunk = _chunk_rows(n_features, model.d_model)
    with RowSource(main_acts) as main, RowSource(control_acts) as control:
        if main.n != control.n or main.d_model != control.d_model:
            raise RuntimeError("random-init activations are not aligned with the run")
        for start, stop in main.chunks(chunk):
            main_rows = np.array(main.rows[start:stop], dtype=np.float32, copy=True)
            control_rows = np.array(control.rows[start:stop], dtype=np.float32, copy=True)
            split = np.array(main.split[start:stop], dtype=np.uint8, copy=True)
            clip_index = np.array(main.clip_idx[start:stop], dtype=np.int64, copy=True)
            t = np.array(main.t[start:stop], dtype=np.int64, copy=True)
            h = np.array(main.h[start:stop], dtype=np.int64, copy=True)
            w = np.array(main.w[start:stop], dtype=np.int64, copy=True)
            with torch.no_grad():
                main_f = model.encode(torch.from_numpy(main_rows).to(device)).detach().cpu().numpy()
                ctrl_f = control_model.encode(torch.from_numpy(control_rows).to(device)).detach().cpu().numpy()
            if label_cues:
                cue_rows = cues.rows(meta, clip_index, t, h, w)
                _accumulate_labels(label_n, sum_a, sum_a2, sum_c, sum_c2, sum_ac, main_f, cue_rows, split)
            main64 = main_f.astype(np.float64, copy=False)
            ctrl64 = ctrl_f.astype(np.float64, copy=False)
            sum_main += main64.sum(axis=0)
            sum_main2 += np.square(main64).sum(axis=0)
            sum_ctrl += ctrl64.sum(axis=0)
            sum_ctrl2 += np.square(ctrl64).sum(axis=0)
            cross += main64.T @ ctrl64
            seen += int(main_f.shape[0])
        if seen != main.n:
            raise RuntimeError("correlation pass did not see every token")
        n_tokens = main.n
    if not label_cues:
        labels = [
            {"label": "", "label_model": NullLabeller.name, "label_holds_heldout": False}
            for _feature_id in range(n_features)
        ]
    else:
        labels = []
    for feature_id in range(n_features) if label_cues else []:
        best_name = ""
        best_r = LABEL_ASSIGN_MIN_CORR
        for cue_index, name in enumerate(CUE_NAMES):
            corr = _corr(
                sum_a[0, feature_id],
                sum_c[0, cue_index],
                sum_a2[0, feature_id],
                sum_c2[0, cue_index],
                sum_ac[0, feature_id, cue_index],
                int(label_n[0]),
            )
            if corr > best_r:
                best_r = corr
                best_name = name
        if not best_name:
            labels.append({"label": "", "label_model": FakeCueLabeller.name, "label_holds_heldout": False})
            continue
        cue_index = CUE_NAMES.index(best_name)
        held = _corr(
            sum_a[1, feature_id],
            sum_c[1, cue_index],
            sum_a2[1, feature_id],
            sum_c2[1, cue_index],
            sum_ac[1, feature_id, cue_index],
            int(label_n[1]),
        )
        labels.append(
            {
                "label": best_name,
                "label_model": FakeCueLabeller.name,
                "label_holds_heldout": bool(held >= LABEL_HOLD_MIN_CORR),
            }
        )
    absent = []
    for feature_id in range(n_features):
        std = _std(sum_main[feature_id], sum_main2[feature_id], n_tokens)
        if std <= DEAD_ACT_EPS:
            absent.append(False)
            continue
        best = 0.0
        for other in range(n_features):
            other_std = _std(sum_ctrl[other], sum_ctrl2[other], n_tokens)
            if other_std <= DEAD_ACT_EPS:
                continue
            corr = _corr(
                sum_main[feature_id],
                sum_ctrl[other],
                sum_main2[feature_id],
                sum_ctrl2[other],
                cross[feature_id, other],
                n_tokens,
            )
            best = max(best, corr)
        absent.append(bool(best < PATTERN_ABSENT_MAX_CORR))
    return labels, absent


def _estimate_rows(adapter, sae, meta, writer: "_ColumnBatch", n_features: int) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for item in meta:

        def visit(clip, item=item):
            acc = np.zeros(n_features, dtype=np.float64)
            for window in adapter.windows(clip):
                effect = estimate_effects(adapter, sae, clip, window).detach().cpu()
                acc += effect.abs().mean(dim=(0, 1, 2)).numpy()
                frame = effect.mean(dim=(1, 2)).numpy()
                for t_index in range(int(frame.shape[0])):
                    width = n_features
                    writer.add(
                        feature_id=np.arange(width, dtype=np.int32),
                        clip_id=[clip.clip_id] * width,
                        window=np.full(width, int(window.index), dtype=np.int16),
                        t=np.full(width, t_index, dtype=np.int16),
                        effect=np.ascontiguousarray(frame[t_index], dtype=np.float32),
                        effect_kind=["estimate"] * width,
                    )
            scores[clip.clip_id] = acc

        _with_clip(adapter, item, visit)
    return scores


def _ablation_rows(
    adapter,
    sae,
    meta,
    membership,
    chosen,
    n_features: int,
    writer: "_ColumnBatch",
    only: set[int] | None = None,
):
    held_total = np.zeros(n_features, dtype=np.float64)
    held_count = np.zeros(n_features, dtype=np.int64)
    for item in meta:
        top = set(chosen[item["clip_id"]])
        heldout = membership[item["clip_id"]]["split"] == "heldout"
        if only is None:
            feature_ids = list(range(n_features)) if heldout else sorted(top)
            record_ids = top
        else:
            feature_ids = sorted(only)
            record_ids = set(only)

        def visit(clip, record_ids=record_ids, heldout=heldout, feature_ids=feature_ids):
            windows = adapter.windows(clip)
            baselines = []
            masks = []
            for window in windows:
                baseline, _target = adapter.predict_masked(clip, window)
                baselines.append(baseline.detach())
                masks.append(adapter.mask(window).detach())
            for feature_id in feature_ids:
                edit = ablate_edit(sae, feature_id)
                for window, baseline, mask in zip(windows, baselines, masks, strict=True):
                    edited, _target = adapter.predict_masked(clip, window, edit=edit)
                    dist = cosine_distance(baseline, edited).detach().cpu()
                    if heldout:
                        held_total[feature_id] += float(dist.sum())
                        held_count[feature_id] += int(dist.numel())
                    if feature_id not in record_ids:
                        continue
                    coords = mask.detach().cpu().nonzero(as_tuple=False)
                    for t_index in range(int(adapter.grid().t)):
                        selected = coords[:, 0] == t_index
                        value = float(dist[selected].mean()) if int(selected.sum()) else 0.0
                        writer.add(
                            feature_id=np.asarray([feature_id], dtype=np.int32),
                            clip_id=[clip.clip_id],
                            window=np.asarray([int(window.index)], dtype=np.int16),
                            t=np.asarray([t_index], dtype=np.int16),
                            effect=np.asarray([value], dtype=np.float32),
                            effect_kind=["ablation"],
                        )

        _with_clip(adapter, item, visit)
    return held_total, held_count


def _random_percentile(*, adapter, sae, feature_id, real_effect, fires, pool, meta, heldout_meta, seed: int) -> float:
    if fires.size == 0:
        return percentile_le(real_effect, np.zeros(N_RANDOM_FEATURES, dtype=np.float64))
    decoder = sae.W_dec[:, feature_id].detach()
    dec_norm = float(decoder.norm().clamp(min=1e-8))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(feature_id), 17]))
    d_model = int(decoder.shape[0])
    samples = np.empty(N_RANDOM_FEATURES, dtype=np.float64)
    clip_index, t, h, w = pool
    for trial in range(N_RANDOM_FEATURES):
        choice = rng.choice(int(clip_index.shape[0]), size=int(fires.size), replace=False)
        shuffled = rng.permutation(fires)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(rng.integers(0, 2**31 - 1)))
        direction = torch.randn(d_model, generator=generator, dtype=torch.float32)
        direction = direction / direction.norm().clamp(min=1e-8) * dec_norm
        direction = direction.to(adapter.device)
        by_clip: dict[int, list[tuple[int, int, int, float]]] = defaultdict(list)
        for slot, value in zip(choice, shuffled, strict=True):
            by_clip[int(clip_index[int(slot)])].append(
                (int(t[int(slot)]), int(h[int(slot)]), int(w[int(slot)]), float(value))
            )
        index_of = {item["clip_id"]: index for index, item in enumerate(meta)}
        total = 0.0
        count = 0
        for item in heldout_meta:
            positions = by_clip.get(index_of[item["clip_id"]], [])

            def edit(acts: torch.Tensor, positions=positions, direc=direction) -> torch.Tensor:
                out = acts.clone()
                for ti, hi, wi, value in positions:
                    out[ti, hi, wi] = out[ti, hi, wi] - value * direc
                return out

            def visit(clip, edit=edit):
                nonlocal total, count
                with torch.no_grad():
                    for window in adapter.windows(clip):
                        edited, _target = adapter.predict_masked(clip, window, edit=edit)
                        baseline, _target = adapter.predict_masked(clip, window)
                        dist = cosine_distance(baseline, edited)
                        total += float(dist.sum())
                        count += int(dist.numel())

            _with_clip(adapter, item, visit)
        samples[trial] = 0.0 if count == 0 else total / count
    return percentile_le(real_effect, samples)


def _heldout_pool(source: RowSource, n_clips: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del n_clips
    clips: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    h_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    step = STREAM_CHUNK
    for start in range(0, source.n_heldout, step):
        stop = min(start + step, source.n_heldout)
        ids = np.asarray(source.held_ids[start:stop], dtype=np.int64)
        clips.append(np.array(source.clip_idx[ids], dtype=np.int32, copy=True))
        t_parts.append(np.array(source.t[ids], dtype=np.int16, copy=True))
        h_parts.append(np.array(source.h[ids], dtype=np.int16, copy=True))
        w_parts.append(np.array(source.w[ids], dtype=np.int16, copy=True))
    if not clips:
        empty_i = np.zeros(0, dtype=np.int32)
        empty_s = np.zeros(0, dtype=np.int16)
        return empty_i, empty_s, empty_s, empty_s
    return (
        np.concatenate(clips),
        np.concatenate(t_parts),
        np.concatenate(h_parts),
        np.concatenate(w_parts),
    )


def _fire_values(connection, path: Path, feature_id: int) -> np.ndarray:
    rows = connection.execute(
        """
        SELECT value
        FROM read_parquet(?)
        WHERE feature_id = ?
          AND clip_id IN (SELECT clip_id FROM held_clips)
        """,
        [str(path), int(feature_id)],
    ).fetchall()
    if not rows:
        return np.zeros(0, dtype=np.float64)
    return np.asarray([row[0] for row in rows], dtype=np.float64)


def _mean_visible_loss(adapter, metas: list[dict]) -> float:
    """Streamed visible-mean baseline. Same reduction as ``_mean_loss``."""
    sse = 0.0
    count = 0

    def visit(clip):
        nonlocal sse, count
        with torch.no_grad():
            for window in adapter.windows(clip):
                baseline = adapter.visible_mean_baseline(clip, window)
                _pred, target = adapter.predict_masked(clip, window)
                if tuple(baseline.shape) != tuple(target.shape):
                    raise RuntimeError(
                        f"visible-mean baseline shape {tuple(baseline.shape)} != target {tuple(target.shape)}"
                    )
                sse += float(((baseline - target) ** 2).sum())
                count += int(target.shape[0])

    for item in metas:
        _with_clip(adapter, item, visit)
    if count == 0:
        raise RuntimeError("no masked tokens to score")
    return sse / count


def _mean_loss(adapter, metas: list[dict], edit=None, zero: bool = False) -> float:
    sse = 0.0
    count = 0

    def visit(clip, edit=edit, zero=zero):
        nonlocal sse, count
        part_sse, part_n = predictor_sse(adapter, [clip], edit=edit, zero=zero)
        sse += part_sse
        count += part_n

    for item in metas:
        _with_clip(adapter, item, visit)
    if count == 0:
        raise RuntimeError("no masked tokens to score")
    return sse / count


def _mean_rows(source: RowSource, ids: np.ndarray, n_rows: int) -> np.ndarray:
    if n_rows <= 0:
        raise RuntimeError("cannot average an empty activation split")
    acc = np.zeros(source.d_model, dtype=np.float64)
    seen = 0
    for start in range(0, n_rows, STREAM_CHUNK):
        stop = min(start + STREAM_CHUNK, n_rows)
        chosen = np.asarray(ids[start:stop], dtype=np.int64)
        rows = np.array(source.rows[chosen], dtype=np.float64, copy=True)
        acc += rows.sum(axis=0)
        seen += int(rows.shape[0])
    if seen != n_rows:
        raise RuntimeError(f"averaged {seen} rows, expected {n_rows}")
    return (acc / seen).astype(np.float32)


def _open_viewer(out, clips_dir, host, port, open_browser, log, serve: bool = True):
    model = load_run(Path(out), Path(clips_dir))
    if not serve:
        return model, None, None, None
    httpd = bind_server(model, host, int(port))
    thread = threading.Thread(target=httpd.serve_forever, name="interpretableintelligence-view", daemon=False)
    thread.start()
    bound_host, bound_port = httpd.server_address[:2]
    url = f"http://{bound_host}:{bound_port}/"
    try:
        payload = _wait_health(url)
    except Exception:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        raise
    if payload.get("run_id") != model.run_payload["run_id"]:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        raise RuntimeError("viewer /api/run returned a different run")
    log(f"viewer {url}")
    log("health /api/run ok")
    if open_browser:
        try:
            webbrowser.open(url)
        except OSError:
            pass
    return model, httpd, thread, url


def _wait_health(url: str) -> dict:
    target = url + "api/run"
    last = "viewer did not answer"
    for _ in range(100):
        try:
            with urllib.request.urlopen(target, timeout=2) as response:
                if response.status != 200:
                    last = f"status {response.status}"
                else:
                    payload = json.loads(response.read().decode("utf-8"))
                    if isinstance(payload, dict) and payload.get("run_id"):
                        return payload
                    last = "missing run_id"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(0.05)
    raise RuntimeError(f"/api/run health check failed: {last}")


class _CueCache:
    def __init__(self) -> None:
        self.index: int | None = None
        self.volume: np.ndarray | None = None

    def rows(self, meta, clip_index, t, h, w) -> np.ndarray:
        out = np.zeros((int(clip_index.shape[0]), len(CUE_NAMES)), dtype=np.float64)
        for chosen in np.unique(clip_index):
            mask = clip_index == chosen
            volume = self._volume(int(chosen), meta[int(chosen)])
            out[mask] = np.transpose(volume[:, t[mask], h[mask], w[mask]])
        return out

    def _volume(self, clip_index: int, item: dict) -> np.ndarray:
        if self.index != clip_index or self.volume is None:
            clip = load_clip(item)
            try:
                self.volume = cue_volume(clip.frames)
            finally:
                del clip
            self.index = clip_index
        return self.volume


class _TopK:
    def __init__(self, k: int) -> None:
        self.k = int(k)
        self.heap: list[tuple[float, int, str, int, int]] = []

    def add(self, value: float, row: int, clip_id: str, window: int, t: int) -> None:
        item = (float(value), int(row), clip_id, int(window), int(t))
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif item[0] > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def export(self) -> tuple[list[dict], int]:
        ordered = sorted(self.heap, key=lambda item: (-item[0], item[1]))
        top = [
            {"clip_id": item[2], "window": item[3], "t": item[4], "value": float(item[0])}
            for item in ordered
        ]
        return top, len({item[2] for item in ordered})


class _ColumnBatch:
    """Write one parquet column batch at a time, then publish the file."""

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(f".{self.path.name}.partial")
        self.schema = schema
        self.columns = {field.name: [] for field in schema}
        self.pending = 0
        self.total = 0
        self._writer = None

    def add(self, **columns) -> None:
        width = None
        materialized = {}
        for name, value in columns.items():
            if isinstance(value, np.ndarray):
                data = value.tolist()
            else:
                data = list(value)
            if width is None:
                width = len(data)
            elif len(data) != width:
                raise ValueError(f"column {name} has {len(data)} rows, expected {width}")
            materialized[name] = data
        if not width:
            return
        for name, data in materialized.items():
            self.columns[name].extend(data)
        self.pending += width
        if self.pending >= 20_000:
            self.flush()

    def flush(self) -> None:
        if self.pending == 0:
            return
        arrays = {
            field.name: pa.array(self.columns[field.name], type=field.type)
            for field in self.schema
        }
        for name in self.columns:
            self.columns[name] = []
        table = pa.table(arrays, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.tmp, self.schema)
        self._writer.write_table(table)
        self.total += self.pending
        self.pending = 0

    def close(self) -> int:
        self.flush()
        if self._writer is None:
            empty = {field.name: pa.array([], type=field.type) for field in self.schema}
            pq.write_table(pa.table(empty, schema=self.schema), self.tmp)
        else:
            self._writer.close()
            self._writer = None
        self.tmp.replace(self.path)
        return int(self.total)

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.tmp.exists():
            self.tmp.unlink()


def _accumulate_labels(label_n, sum_a, sum_a2, sum_c, sum_c2, sum_ac, feats, cues, split) -> None:
    for split_id in (0, 1):
        mask = split == split_id
        if not np.any(mask):
            continue
        acts = feats[mask].astype(np.float64, copy=False)
        cue_rows = cues[mask].astype(np.float64, copy=False)
        label_n[split_id] += int(acts.shape[0])
        sum_a[split_id] += acts.sum(axis=0)
        sum_a2[split_id] += np.square(acts).sum(axis=0)
        sum_c[split_id] += cue_rows.sum(axis=0)
        sum_c2[split_id] += np.square(cue_rows).sum(axis=0)
        sum_ac[split_id] += acts.T @ cue_rows


def _corr(sum_a, sum_b, sum_a2, sum_b2, sum_ab, n: int) -> float:
    if n <= 0:
        return 0.0
    mean_a = float(sum_a) / n
    mean_b = float(sum_b) / n
    da = math.sqrt(max(float(sum_a2) - n * mean_a * mean_a, 0.0))
    db = math.sqrt(max(float(sum_b2) - n * mean_b * mean_b, 0.0))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float((float(sum_ab) - n * mean_a * mean_b) / (da * db))


def _std(sum_, sum2, n: int) -> float:
    if n <= 0:
        return 0.0
    mean = float(sum_) / n
    var = float(sum2) / n - mean * mean
    return math.sqrt(max(var, 0.0))


def _chunk_rows(n_features: int, d_model: int) -> int:
    by_feature = max(1, 250_000 // max(int(n_features), 1))
    by_width = max(1, 1_000_000 // max(int(d_model), 1))
    return max(1, min(STREAM_CHUNK, by_feature, by_width))


def _feature_matrix(acts_dir: Path, model, device: torch.device) -> np.ndarray:
    """Dense SAE codes in capture order. Group loss only needs co-occurrence."""
    chunks: list[np.ndarray] = []
    with RowSource(acts_dir) as source:
        width = _chunk_rows(model.n_features, source.d_model)
        for start, stop in source.chunks(width):
            acts = np.array(source.rows[start:stop], dtype=np.float32, copy=True)
            with torch.no_grad():
                encoded = model.encode(torch.from_numpy(acts).to(device)).detach().cpu().numpy()
            chunks.append(np.asarray(encoded, dtype=np.float32))
    if not chunks:
        raise RuntimeError("no activations to score extra effects")
    return np.concatenate(chunks, axis=0)


class _measure:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.box: dict = {}
        self.sampler = RamSampler()
        self.started = 0.0

    def __enter__(self) -> dict:
        self.sampler.start()
        self.started = time.perf_counter()
        return self.box

    def __exit__(self, exc_type, exc, tb) -> None:
        peak = self.sampler.stop()
        self.box["metrics"] = {
            "wall_s": float(time.perf_counter() - self.started),
            "peak_ram_bytes": int(peak),
            "device": str(self.device),
            "rows": int(self.box.get("rows", 0)),
            "tokens": int(self.box.get("tokens", 0)),
        }


def _with_clip(adapter, item: dict, visit) -> None:
    clip = load_clip(item)
    try:
        visit(clip)
    finally:
        _clear_cache(adapter)
        del clip


def _clear_cache(adapter) -> None:
    cache = getattr(adapter, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _capture_inputs(*, adapter_name, layer, seed, heldout_frac, model_sha, d_model, grid, meta, tiny: bool = False) -> dict:
    return {
        "kind": "capture",
        "adapter": adapter_name,
        "tiny": bool(tiny),
        "layer": int(layer),
        "seed": int(seed),
        "heldout_frac": float(heldout_frac),
        "model_sha": model_sha,
        "d_model": int(d_model),
        "grid": {key: int(value) for key, value in grid.items()},
        "tool_version": __version__,
        "clips": [
            {
                "clip_id": item["clip_id"],
                "sha256": item["sha256"],
                "fps": float(item["fps"]),
                "n_frames": int(item["n_frames"]),
            }
            for item in meta
        ],
    }


def _with_kind(inputs: dict, kind: str, **extra) -> dict:
    payload = dict(inputs)
    payload["kind"] = kind
    payload.update(extra)
    return payload


def _capture_ready(out: Path, inputs: dict, grid: dict, n_clips: int) -> bool:
    if not _same_inputs(out / "acts" / "complete.json", inputs):
        return False
    if not (out / "split.json").is_file() or not (out / "manifest.json").is_file():
        return False
    if not cache_matches_layout(out / "acts", n_clips, grid):
        return False
    return _stage_recorded(out, "capture")


def _sae_ready(out: Path, inputs: dict, config: SAEConfig) -> bool:
    if not _same_inputs(out / "sae" / "complete.json", inputs):
        return False
    config_path = out / "sae" / "config.json"
    weights = out / "sae" / "weights.safetensors"
    if not config_path.is_file() or not weights.is_file():
        return False
    try:
        saved = read_json(config_path)
    except (OSError, ValueError):
        return False
    if saved != config.as_dict():
        return False
    try:
        model, _loaded = load_sae(out / "sae", torch.device("cpu"))
    except (OSError, ValueError, RuntimeError, KeyError):
        return False
    ok = (
        tuple(model.W_enc.shape) == (config.n_features, config.d_model)
        and tuple(model.W_dec.shape) == (config.d_model, config.n_features)
    )
    del model
    return ok and _stage_recorded(out, "sae")


def _features_ready(out: Path, inputs: dict) -> bool:
    if not _same_inputs(out / "features" / "complete.json", inputs):
        return False
    stats_path = out / "features" / "stats.json"
    acts_path = out / "features" / "feature_acts.parquet"
    if not stats_path.is_file() or not acts_path.is_file():
        return False
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        schema = pq.read_schema(acts_path)
        rows = pq.ParquetFile(acts_path).metadata.num_rows
        store = json.loads((out / "acts" / "stream" / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, AttributeError):
        return False
    if tuple(schema.names) != FEATURE_ACT_COLUMNS:
        return False
    if not isinstance(stats.get("features"), list):
        return False
    if int(stats.get("n_tokens", -1)) != int(store["n"]):
        return False
    if int(stats.get("sparse_rows", -1)) != int(rows):
        return False
    return _stage_recorded(out, "features")


def _score_ready(out: Path, inputs: dict, n_features: int, n_clips: int, grid: dict) -> bool:
    if not _same_inputs(out / "score.complete.json", inputs):
        return False
    effects = out / "features" / "clip_effects.parquet"
    jsonl = out / "features" / "features.jsonl"
    attribution = out / "features" / "attribution.parquet"
    candidates = out / "features" / "candidates.json"
    if (
        not (out / "scoreboard.json").is_file()
        or not effects.is_file()
        or not jsonl.is_file()
        or not attribution.is_file()
        or not candidates.is_file()
    ):
        return False
    try:
        scoreboard = read_json(out / "scoreboard.json")
        schema = pq.read_schema(effects)
        effect_rows = pq.ParquetFile(effects).metadata.num_rows
        lines = _count_lines(jsonl)
        store = json.loads((out / "acts" / "stream" / "meta.json").read_text(encoding="utf-8"))
        attr_schema = pq.read_schema(attribution)
        document = json.loads(candidates.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, AttributeError):
        return False
    if tuple(schema.names) != CLIP_EFFECT_COLUMNS:
        return False
    if tuple(attr_schema.names) != ATTRIBUTION_COLUMNS:
        return False
    ranked = document.get("ranked") if isinstance(document, dict) else None
    if not isinstance(ranked, list) or len(ranked) != n_features:
        return False
    features = scoreboard.get("features")
    if not isinstance(features, list) or len(features) != n_features or lines != n_features:
        return False
    cells = int(grid["t"]) * int(grid["h"]) * int(grid["w"])
    if cells <= 0 or int(store["n"]) % (n_clips * cells) != 0:
        return False
    n_windows = int(store["n"]) // (n_clips * cells)
    minimum = n_features * n_clips * n_windows * int(grid["t"])
    return int(effect_rows) >= minimum and _stage_recorded(out, "score")


def _view_ready(out: Path, inputs: dict) -> bool:
    if not _same_inputs(out / "view.complete.json", inputs):
        return False
    if not (out / "manifest.json").is_file() or not (out / "features" / "feature_acts.parquet").is_file():
        return False
    return _stage_recorded(out, "view")


def _control_acts_ready(acts_dir: Path, inputs: dict, grid: dict, n_clips: int) -> bool:
    if not _same_inputs(acts_dir / "complete.json", inputs):
        return False
    return cache_matches_layout(acts_dir, n_clips, grid)


def _control_sae_ready(directory: Path, inputs: dict, config: SAEConfig) -> bool:
    if not _same_inputs(directory / "complete.json", inputs):
        return False
    config_path = directory / "config.json"
    if not config_path.is_file() or not (directory / "weights.safetensors").is_file():
        return False
    try:
        return read_json(config_path) == config.as_dict()
    except (OSError, ValueError):
        return False


def _stage_recorded(out: Path, name: str) -> bool:
    path = out / "manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = read_json(path)
    except (OSError, ValueError):
        return False
    stage = manifest.get("stages", {}).get(name) if isinstance(manifest.get("stages"), dict) else None
    if not isinstance(stage, dict):
        return False
    return all(key in stage for key in ("wall_s", "peak_ram_bytes", "device", "rows", "tokens"))


def _same_inputs(path: Path, inputs: dict) -> bool:
    if not path.is_file():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return _canon(saved.get("inputs")) == _canon(inputs)


def _canon(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_complete(path: Path, inputs: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"inputs": inputs, "metrics": metrics})


def _record_stage(out: Path, name: str, metrics: dict) -> None:
    path = out / "manifest.json"
    manifest = read_json(path)
    stages = manifest.setdefault("stages", {})
    stages[name] = metrics
    write_json(path, manifest)


def _read_scoreboard(out: Path) -> dict | None:
    path = out / "scoreboard.json"
    if not path.is_file():
        return None
    return read_json(path)


def _invalidate(out: Path, stage: str) -> None:
    if stage == "capture":
        targets = [
            out / "acts",
            out / "sae",
            out / "features",
            out / "control",
            out / "split.json",
            out / "manifest.json",
            out / "scoreboard.json",
            out / "score.complete.json",
            out / "view.complete.json",
        ]
    elif stage == "sae":
        targets = [
            out / "sae",
            out / "features",
            out / "scoreboard.json",
            out / "score.complete.json",
            out / "view.complete.json",
        ]
    elif stage == "features":
        targets = [
            out / "features",
            out / "scoreboard.json",
            out / "score.complete.json",
            out / "view.complete.json",
        ]
    elif stage == "score":
        targets = [
            out / "scoreboard.json",
            out / "score.complete.json",
            out / "view.complete.json",
            out / "features" / "features.jsonl",
            out / "features" / "clip_effects.parquet",
            out / "features" / "attribution.parquet",
            out / "features" / "candidates.json",
        ]
    else:
        targets = []
    for target in targets:
        _rm(target)


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file() or path.is_symlink():
        path.unlink()


def _same_coordinates(left: Path, right: Path) -> bool:
    left_meta = json.loads((left / "stream" / "meta.json").read_text(encoding="utf-8"))
    right_meta = json.loads((right / "stream" / "meta.json").read_text(encoding="utf-8"))
    if int(left_meta["n"]) != int(right_meta["n"]):
        return False
    n = int(left_meta["n"])
    for name, dtype in (
        ("clip_idx.i32", np.int32),
        ("window.i16", np.int16),
        ("t.i16", np.int16),
        ("h.i16", np.int16),
        ("w.i16", np.int16),
        ("split.u8", np.uint8),
    ):
        a = np.memmap(left / "stream" / name, dtype=dtype, mode="r", shape=(n,))
        b = np.memmap(right / "stream" / name, dtype=dtype, mode="r", shape=(n,))
        try:
            for start in range(0, n, STREAM_CHUNK):
                stop = min(start + STREAM_CHUNK, n)
                if not np.array_equal(np.array(a[start:stop], copy=True), np.array(b[start:stop], copy=True)):
                    return False
        finally:
            for array in (a, b):
                mapped = getattr(array, "_mmap", None)
                if mapped is not None:
                    mapped.close()
    return True


def _count_lines(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_any(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
