"""Resumable capture → SAE → features → score → viewer.

Activation rows stay in an on-disk cache. Each stage reads or writes a chunk,
records wall time, peak RSS, device, and row/token counts on the manifest,
and skips itself when those outputs are still valid for the same clips.
"""

from __future__ import annotations

import heapq
import json
import math
import os
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
from interpretableintelligence.act_cache import ActivationCache, captured_layer
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
    default_objective,
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
    LABEL_ASSIGN_MIN_AUROC,
    LABEL_HOLD_MIN_AUROC,
    HF_SAE_BATCH,
    HF_SAE_EXPANSION,
    HF_SAE_LR,
    HF_SAE_STEPS,
    HF_SAE_TOPK,
    SCORE_ALL_MAX_FEATURES,
    DEFAULT_SEED,
    LABEL_ASSIGN_MIN_CORR,
    LABEL_HOLD_MIN_CORR,
    MIN_CLIPS,
    N_RANDOM_FEATURES,
    PATTERN_ABSENT_MAX_CORR,
    TOP_K_ABLATION_PER_CLIP,
    TOP_K_EXAMPLES,
)
from interpretableintelligence.view import bind_server, enable_steering, load_run

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


def sae_dictionary_size(adapter_name: str, d_model: int, requested: int | None = None) -> int:
    """SAE width. ``fake`` and ``vjepa2`` keep the 12-feature acceptance dictionary;
    a real model behind ``hf`` gets ``HF_SAE_EXPANSION`` features per activation dimension."""
    if requested is not None:
        if int(requested) < 1:
            raise ValueError(f"--sae-features must be >= 1, got {requested}")
        return int(requested)
    if adapter_name.startswith("hf:"):
        return HF_SAE_EXPANSION * int(d_model)
    return SAEConfig(d_model=int(d_model)).n_features


def sae_config_for(adapter_name: str, d_model: int, seed: int, requested: int | None = None) -> SAEConfig:
    """``fake`` and ``vjepa2``: the L1 ReLU acceptance SAE. ``hf``, or any wider
    dictionary asked for with ``--sae-features``: a TopK SAE.

    The L1 and decoder-orthogonality terms are tuned for the 12-feature
    acceptance dictionary; the orthogonality term sums over every pair of
    features, so at thousands of features it swamps the reconstruction loss.
    TopK fixes sparsity directly and needs neither term.
    """
    n_features = sae_dictionary_size(adapter_name, d_model, requested)
    if not adapter_name.startswith("hf:") and n_features <= SCORE_ALL_MAX_FEATURES:
        return SAEConfig(d_model=int(d_model), n_features=n_features, seed=int(seed))
    return SAEConfig(
        d_model=int(d_model),
        n_features=n_features,
        seed=int(seed),
        activation="topk",
        k=min(HF_SAE_TOPK, n_features),
        l1_coeff=0.0,
        ortho_coeff=0.0,
        lr=HF_SAE_LR,
        steps=HF_SAE_STEPS,
        batch_size=HF_SAE_BATCH,
    )


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
    sae_features: int | None = None,
    concepts_file: Path | None = None,
    score: bool = True,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = False,
    steer: bool = False,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    log=print,
    on_stage=None,
    manifest_clips_dir: str | None = None,
) -> dict:
    """Run every missing stage and, unless ``serve`` is false, block on the viewer.

    ``steer`` turns on the viewer's playground (it loads the model and SAE once more).

    Captured layers go through the shared activation cache (``act_cache``) unless
    ``use_cache`` is false, so a rerun into a new folder, a new seed or split, or a
    new SAE size reads them from disk instead of running the model again.

    ``on_stage(name)`` is called after each stage has run or been skipped, once its outputs
    are on disk (``ii run --remote`` uses it to ship finished stages back). ``manifest_clips_dir``
    is the ``clips_dir`` recorded on the manifest instead of ``clips_dir`` itself, for a run whose
    clips live at a different path on the machine that will view it.
    """
    stage_done = on_stage if on_stage is not None else (lambda _name: None)
    if gpu_cost_per_hour is not None and (not math.isfinite(gpu_cost_per_hour) or gpu_cost_per_hour < 0):
        raise ValueError("gpu_cost_per_hour must be a non-negative finite number")
    if not 0.0 < float(heldout_frac) < 1.0:
        raise ValueError(f"heldout_frac must be in (0, 1), got {heldout_frac}")
    if tiny and adapter_name != "vjepa2":
        raise ValueError("--tiny is only supported for the vjepa2 adapter")
    if concepts_file is not None and concept_labelling(adapter_name, tiny) is None:
        raise ValueError("--concepts applies to real models (hf, vjepa2, groot); fake labels from its planted cues")
    if prefilter is not None and int(prefilter) < 1:
        raise ValueError(f"prefilter K must be >= 1, got {prefilter}")
    prefilter = None if prefilter is None else int(prefilter)
    extra_effects = check_extra_effects(extra_effects)

    # One thread keeps runs bit-reproducible; II_THREADS trades that for speed on a CPU-only box.
    torch.set_num_threads(max(1, int(os.environ.get("II_THREADS", "1"))))
    device = pick_device()
    clips_dir = Path(clips_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = list_clip_meta(clips_dir)
    if len(meta) < MIN_CLIPS:
        raise ValueError(f"need at least {MIN_CLIPS} clips, found {len(meta)} in {clips_dir}")

    cache = ActivationCache.open(cache_dir, enabled=use_cache)
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
    config = sae_config_for(adapter.name, d_model, int(seed), sae_features)
    n_features = config.n_features
    if prefilter is None and n_features > SCORE_ALL_MAX_FEATURES:
        prefilter = DEFAULT_PREFILTER_K
        log(
            f"{n_features} SAE features: scoring the {prefilter} with the largest attribution "
            "(pass --prefilter K to change). The rest stay unverified."
        )
    sae_inputs = _with_kind(capture_inputs, "sae", sae=config.as_dict())

    if _capture_ready(out, capture_inputs, grid, len(meta)):
        log("skip capture")
    else:
        log("run capture")
        _invalidate(out, "capture")
        with _measure(device) as measured:
            rows = _capture(
                adapter,
                meta,
                out,
                layer,
                seed,
                heldout_frac,
                model_sha,
                capture_inputs,
                clips_dir,
                cache,
                manifest_clips_dir,
                log=log,
            )
            measured["rows"] = rows
            measured["tokens"] = rows
        _record_stage(out, "capture", measured["metrics"])
        _write_complete(out / "acts" / "complete.json", capture_inputs, measured["metrics"])
    stage_done("capture")

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
    stage_done("sae")

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
    stage_done("features")

    stats_path = out / "features" / "stats.json"
    score_inputs = _with_kind(
        feature_inputs,
        "score",
        stats_sha=file_sha256(stats_path),
        prefilter=prefilter,
        extra_effects=list(extra_effects),
        **_concept_inputs(adapter.name, concepts_file, tiny),
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
                concepts_file=concepts_file,
                cache=cache,
            )
            measured["rows"] = effect_rows
            measured["tokens"] = int(read_json(out / "acts" / "stream" / "meta.json")["n"])
        _record_stage(out, "score", measured["metrics"])
        _write_complete(out / "score.complete.json", score_inputs, measured["metrics"])
    if score:
        stage_done("score")

    view_inputs = _with_kind(feature_inputs, "view")
    if (out / "scoreboard.json").is_file():
        view_inputs = _with_kind(view_inputs, "view", scoreboard_sha=file_sha256(out / "scoreboard.json"))
    httpd = None
    thread = None
    if _view_ready(out, view_inputs):
        log("skip view")
        if serve:
            _model, httpd, thread, _url = _open_viewer(out, clips_dir, host, port, open_browser, log, steer=steer)
    else:
        log("run view")
        with _measure(device) as measured:
            model, httpd, thread, _url = _open_viewer(
                out, clips_dir, host, port, open_browser, log, serve=serve, steer=steer
            )
            measured["rows"] = len(model.run_payload["clips"])
            measured["tokens"] = 0
        _record_stage(out, "view", measured["metrics"])
        _write_complete(out / "view.complete.json", view_inputs, measured["metrics"])

    if cache.hits or cache.stored:
        log(cache.summary())
    stage_done("view")

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


def _progress(log, stage: str, done: int, total: int, started: float) -> None:
    """Log ``done/total`` about every tenth of the way (and at the end) with a time estimate."""
    if log is None or total < 10:
        return
    step = max(total // 10, 1)
    if done % step and done != total:
        return
    elapsed = time.perf_counter() - started
    left = elapsed / done * (total - done)
    tail = f", about {_duration(left)} left" if done < total else f" in {_duration(elapsed)}"
    log(f"{stage}: {done}/{total} clips{tail}")


def _duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f} min"


def _capture(
    adapter,
    meta,
    out,
    layer,
    seed,
    heldout_frac,
    model_sha,
    inputs,
    clips_dir: Path,
    cache: ActivationCache | None = None,
    recorded_clips_dir: str | None = None,
    log=None,
) -> int:
    cache = cache if cache is not None else ActivationCache(None)
    split = assign_split([item["clip_id"] for item in meta], int(seed), float(heldout_frac))
    membership = split.membership()
    grid = adapter.grid()
    expected_windows: list[int] | None = None
    started = time.perf_counter()
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
                    acts = captured_layer(adapter, clip, window, cache=cache, model_sha=model_sha).numpy()
                    if tuple(int(dim) for dim in acts.shape[:3]) != (grid.t, grid.h, grid.w):
                        raise RuntimeError(f"{clip.clip_id} activation grid {acts.shape} != {grid.as_dict()}")
                    if int(acts.shape[-1]) != int(adapter.d_model):
                        raise RuntimeError(f"captured d_model {acts.shape[-1]} != {adapter.d_model}")
                    writer.append(acts, clip_index, int(window.index), code)
                    del acts
            finally:
                _clear_cache(adapter)
                del clip
            _progress(log, "capture", clip_index + 1, len(meta), started)
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
        "clips_dir": recorded_clips_dir or str(Path(clips_dir).resolve()),
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
    concepts_file: Path | None = None,
    cache: ActivationCache | None = None,
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
        _capture_control(control, meta, control_dir / "acts", manifest, cache)
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
        concepts=_concept_source(adapter, concepts_file, device, log, tiny),
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
    top_k = min(TOP_K_ABLATION_PER_CLIP, n_features)
    effects_path = out / "features" / "clip_effects.parquet"
    effect_writer = _ColumnBatch(effects_path, _EFFECT_SCHEMA)
    try:
        # One backward pass per window gives both the per-cell effect estimates and the
        # prediction-loss attribution that picks which features to ablate.
        attribution_started = time.perf_counter()
        objective = default_objective(adapter)
        rank, by_clip = _estimate_rows(adapter, model, meta, effect_writer, n_features, objective=objective, log=log)
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
        _autointerp(adapter, model, meta, out, labels, selected, ranked_ids, device, seed, log)
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
            log=log,
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
        controls_done = 0
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
                controls_done += 1
                if selected is not None:
                    log(
                        f"score: {N_RANDOM_FEATURES} random controls for feature {feature_id} "
                        f"({controls_done}/{len(selected)})"
                    )
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
            if not measured and not real:
                continue
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
    write_attribution(out / "features" / "attribution.parquet", attr_rows, objective=objective)
    write_json(
        out / "features" / "candidates.json",
        candidates_document(
            mean_scores,
            reported_k,
            objective=objective,
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


def _capture_control(adapter, meta, acts_dir: Path, manifest: dict, cache: ActivationCache | None = None) -> None:
    cache = cache if cache is not None else ActivationCache(None)
    control_sha = adapter.model_sha() if cache.enabled else ""
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
                    acts = captured_layer(adapter, clip, window, cache=cache, model_sha=control_sha).numpy()
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
    concepts=None,
) -> tuple[list[dict], list[bool]]:
    """Labels and the random-init absence check, in one pass over the activations.

    ``concepts`` supplies per-token concept strengths (``names``, ``name`` for
    ``label_model``, ``rows(meta, clip_index, t, h, w)``). With ``label_cues``
    and no ``concepts``, the fake adapter's planted cue channels are the concepts.
    Without either, every label is empty.
    """
    n_features = int(model.n_features)
    if concepts is None and label_cues:
        concepts = _CueCache()
    label_cues = concepts is not None
    concept_names = tuple(concepts.names) if concepts is not None else CUE_NAMES
    label_model = concepts.name if concepts is not None else NullLabeller.name
    n_cues = len(concept_names)
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
    cues = concepts
    auroc = _AurocLabels(n_features, n_cues) if getattr(concepts, "statistic", "pearson") == "auroc" else None
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
                window = np.array(main.window[start:stop], dtype=np.int64, copy=True)
                cue_rows = cues.rows(meta, clip_index, t, h, w, window=window)
                if auroc is not None:
                    auroc.add_concepts(cue_rows, split)
                else:
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
    if auroc is not None:
        labels = auroc.labels(main_acts, model, device, chunk, concept_names, label_model)
    for feature_id in range(n_features) if label_cues and auroc is None else []:
        best_name = ""
        best_r = LABEL_ASSIGN_MIN_CORR
        for cue_index, name in enumerate(concept_names):
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
            labels.append({"label": "", "label_model": label_model, "label_holds_heldout": False})
            continue
        cue_index = concept_names.index(best_name)
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
                "label_model": label_model,
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


def _estimate_rows(
    adapter, sae, meta, writer: "_ColumnBatch", n_features: int, objective: str | None = None, log=None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Write per-cell effect estimates. Returns per-clip ranking scores (mean |effect|) and the
    per-clip attribution (summed effect, as ``attribute.attribution_by_clip``), from the same
    backward passes. ``objective=None`` takes the adapter's default (``score.default_objective``)."""
    scores: dict[str, np.ndarray] = {}
    attribution: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for done, item in enumerate(meta, start=1):

        def visit(clip, item=item):
            acc = np.zeros(n_features, dtype=np.float64)
            total = np.zeros(n_features, dtype=np.float64)
            windows = adapter.windows(clip)
            if not windows:
                raise RuntimeError(f"{clip.clip_id} has no windows")
            for window in windows:
                effect = estimate_effects(adapter, sae, clip, window, objective=objective).detach().cpu()
                acc += effect.abs().mean(dim=(0, 1, 2)).numpy()
                total += np.asarray(effect.numpy(), dtype=np.float64).reshape(-1, n_features).sum(axis=0)
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
            attribution[clip.clip_id] = total

        _with_clip(adapter, item, visit)
        _progress(log, "score: attribution", done, len(meta), started)
    return scores, attribution


def _ablation_rows(
    adapter,
    sae,
    meta,
    membership,
    chosen,
    n_features: int,
    writer: "_ColumnBatch",
    only: set[int] | None = None,
    log=None,
):
    held_total = np.zeros(n_features, dtype=np.float64)
    held_count = np.zeros(n_features, dtype=np.int64)
    started = time.perf_counter()
    for done, item in enumerate(meta, start=1):
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
            # Every feature's edit on a window runs as one batch when the adapter supports it.
            edits = [ablate_edit(sae, feature_id) for feature_id in feature_ids]
            dists_by_window = []
            for window, baseline in zip(windows, baselines, strict=True):
                with torch.no_grad():
                    edited_all = _predict_many(adapter, clip, window, edits)
                dists_by_window.append(
                    [cosine_distance(baseline, edited).detach().cpu() for edited in edited_all]
                )
            for slot, feature_id in enumerate(feature_ids):
                for window, mask, dists in zip(windows, masks, dists_by_window, strict=True):
                    dist = dists[slot]
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
        _progress(log, "score: ablations", done, len(meta), started)
    return held_total, held_count


def _random_percentile(*, adapter, sae, feature_id, real_effect, fires, pool, meta, heldout_meta, seed: int) -> float:
    """Percentile of ``real_effect`` among ``N_RANDOM_FEATURES`` matched random directions.

    Every trial draws its directions and positions up front, in the same RNG
    order as one trial at a time. Each held-out clip is then loaded once, its
    unedited prediction is computed once per window, and every trial's edit is
    measured against that. Per-trial sums still add clips and windows in the
    same order, so the samples are unchanged.
    """
    if fires.size == 0:
        return percentile_le(real_effect, np.zeros(N_RANDOM_FEATURES, dtype=np.float64))
    decoder = sae.W_dec[:, feature_id].detach()
    dec_norm = float(decoder.norm().clamp(min=1e-8))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(feature_id), 17]))
    d_model = int(decoder.shape[0])
    clip_index, t, h, w = pool
    index_of = {item["clip_id"]: index for index, item in enumerate(meta)}
    trials: list[tuple[torch.Tensor, dict[int, list[tuple[int, int, int, float]]]]] = []
    for _trial in range(N_RANDOM_FEATURES):
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
        trials.append((direction, by_clip))

    totals = [0.0] * N_RANDOM_FEATURES
    counts = [0] * N_RANDOM_FEATURES
    for item in heldout_meta:
        clip_key = index_of[item["clip_id"]]

        def visit(clip):
            with torch.no_grad():
                for window in adapter.windows(clip):
                    baseline, _target = adapter.predict_masked(clip, window)
                    edits = []
                    for direction, by_clip in trials:
                        positions = by_clip.get(clip_key, [])

                        def edit(acts: torch.Tensor, positions=positions, direc=direction) -> torch.Tensor:
                            out = acts.clone()
                            for ti, hi, wi, value in positions:
                                out[ti, hi, wi] = out[ti, hi, wi] - value * direc
                            return out

                        edits.append(edit)
                    for trial, edited in enumerate(_predict_many(adapter, clip, window, edits)):
                        dist = cosine_distance(baseline, edited)
                        totals[trial] += float(dist.sum())
                        counts[trial] += int(dist.numel())

        _with_clip(adapter, item, visit)
    samples = np.array(
        [0.0 if count == 0 else total / count for total, count in zip(totals, counts, strict=True)],
        dtype=np.float64,
    )
    return percentile_le(real_effect, samples)


def _predict_many(adapter, clip, window, edits) -> list[torch.Tensor]:
    """Edited predictions for one window, batched when the adapter supports it."""
    many = getattr(adapter, "predict_masked_many", None)
    if many is not None:
        return many(clip, window, edits)
    return [adapter.predict_masked(clip, window, edit=edit)[0] for edit in edits]


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


def _open_viewer(out, clips_dir, host, port, open_browser, log, serve: bool = True, steer: bool = False):
    model = load_run(Path(out), Path(clips_dir))
    if not serve:
        return model, None, None, None
    if steer:
        enable_steering(model, Path(clips_dir))
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


def concept_labelling(adapter_name: str, tiny: bool = False) -> str | None:
    """How a real model's features get CLIP concept labels.

    ``"dense"``: per-patch concept maps (``hf`` and ``vjepa2``, whose tokens
    are image patches). ``"window"``: one whole-frame score per window
    (``groot``, whose tokens are state and action steps). ``None``: the fake
    adapter's planted cues, or no labels for a random stand-in.
    """
    if tiny:
        return None
    if adapter_name.startswith("hf:") or adapter_name == "vjepa2":
        return "dense"
    if adapter_name.startswith("groot:") and adapter_name != "groot:tiny":
        return "window"
    return None


def _concept_inputs(adapter_name: str, concepts_file, tiny: bool = False) -> dict:
    """Score-stage cache key for the labeller, so a new concept list relabels."""
    kind = concept_labelling(adapter_name, tiny)
    if kind is None:
        return {}
    from interpretableintelligence.autointerp import AUTOINTERP_VERSION, provider_from_env
    from interpretableintelligence.labelers import CLIP_LABEL_MODEL, load_concepts

    inputs = {"labeller": CLIP_LABEL_MODEL, "concepts": load_concepts(concepts_file), "concept_level": kind}
    picked = provider_from_env()
    if picked is not None and adapter_name.startswith("hf:"):
        inputs["autointerp"] = [AUTOINTERP_VERSION, *picked]
    return inputs


def _autointerp(adapter, sae, meta, out: Path, labels: list[dict], selected, ranked_ids, device, seed, log) -> None:
    """Replace the CLIP labels of the features being scored with explain-then-simulate labels.

    Only for hf and only when a vision-model key is set (``autointerp.client_from_env``).
    The label still has to hold on held-out clips at the same AUROC bar. A provider
    error keeps the CLIP labels and says so; it never fails the run.
    """
    if not adapter.name.startswith("hf:"):
        return
    from interpretableintelligence.autointerp import MAX_FEATURES, client_from_env, label_features

    client = client_from_env()
    if client is None:
        return
    targets = sorted(selected) if selected is not None else sorted(ranked_ids[: min(MAX_FEATURES, len(ranked_ids))])
    log(f"score: auto-interp labels from {client.name} for {len(targets)} features")
    try:
        new, evidence = label_features(
            client=client, adapter=adapter, meta=meta, acts_dir=out / "acts", sae=sae,
            feature_ids=targets, device=device, seed=int(seed), log=log,
        )
    except RuntimeError as exc:
        log(f"score: auto-interp failed ({exc}); keeping CLIP labels")
        return
    for feature_id, row in new.items():
        labels[feature_id] = row
    write_json(out / "features" / "autointerp.json", evidence)


def _concept_source(adapter, concepts_file, device, log, tiny: bool = False):
    """CLIP concepts for a real model; ``None`` keeps the default labelling."""
    kind = concept_labelling(adapter.name, tiny)
    if kind is None:
        return None
    from interpretableintelligence.labelers import ClipConceptSource, WindowConceptSource, load_concepts

    names = load_concepts(concepts_file)
    log(f"score: labelling with {len(names)} CLIP concepts ({kind})")
    source = ClipConceptSource if kind == "dense" else WindowConceptSource
    return source(adapter, names, device=device)


class _AurocLabels:
    """Concept labels for sparse features: how well a concept separates tokens where
    the feature fires from tokens where it does not (ROC AUC, Mann-Whitney).

    Pearson against a dense concept map is capped far below 1 when a feature fires
    on about 1% of tokens, however cleanly the concept predicts it. Pass one keeps
    the concept rows; ``labels`` ranks them per split and re-encodes the
    activations, so the firing x rank sums are one matrix product per chunk.
    """

    def __init__(self, n_features: int, n_concepts: int) -> None:
        self.n_features = n_features
        self.n_concepts = n_concepts
        self.concepts: list[np.ndarray] = []
        self.splits: list[np.ndarray] = []

    def add_concepts(self, rows: np.ndarray, split: np.ndarray) -> None:
        self.concepts.append(np.asarray(rows, dtype=np.float32))
        self.splits.append(np.asarray(split, dtype=np.uint8))

    def labels(self, main_acts, model, device, chunk, names, label_model) -> list[dict]:
        concepts = np.concatenate(self.concepts)
        split = np.concatenate(self.splits)
        ranks = np.zeros_like(concepts, dtype=np.float64)
        n_split = np.zeros(2, dtype=np.int64)
        for split_id in (0, 1):
            mask = split == split_id
            n_split[split_id] = int(mask.sum())
            if n_split[split_id]:
                ranks[mask] = np.argsort(np.argsort(concepts[mask], axis=0, kind="stable"), axis=0) + 1.0
        rank_sum = np.zeros((2, self.n_features, self.n_concepts), dtype=np.float64)
        n_fire = np.zeros((2, self.n_features), dtype=np.float64)
        with RowSource(main_acts) as main:
            for start, stop in main.chunks(chunk):
                rows = np.array(main.rows[start:stop], dtype=np.float32, copy=True)
                with torch.no_grad():
                    fire = (model.encode(torch.from_numpy(rows).to(device)) > DEAD_ACT_EPS).cpu().numpy()
                part = split[start:stop]
                for split_id in (0, 1):
                    mask = part == split_id
                    if np.any(mask):
                        chosen = fire[mask].astype(np.float64)
                        rank_sum[split_id] += chosen.T @ ranks[start:stop][mask]
                        n_fire[split_id] += chosen.sum(axis=0)
        auc = np.full((2, self.n_features, self.n_concepts), 0.5)
        for split_id in (0, 1):
            pos = n_fire[split_id][:, None]
            neg = n_split[split_id] - pos
            ok = (pos > 0) & (neg > 0)
            value = (rank_sum[split_id] - pos * (pos + 1) / 2) / np.where(ok, pos * neg, 1.0)
            auc[split_id] = np.where(ok, value, 0.5)
        out = []
        for feature_id in range(self.n_features):
            best = int(np.argmax(auc[0, feature_id]))
            if auc[0, feature_id, best] < LABEL_ASSIGN_MIN_AUROC:
                out.append({"label": "", "label_model": label_model, "label_holds_heldout": False})
                continue
            out.append(
                {
                    "label": names[best],
                    "label_model": label_model,
                    "label_holds_heldout": bool(auc[1, feature_id, best] >= LABEL_HOLD_MIN_AUROC),
                }
            )
        return out


class _CueCache:
    names = CUE_NAMES
    name = FakeCueLabeller.name

    def __init__(self) -> None:
        self.index: int | None = None
        self.volume: np.ndarray | None = None

    def rows(self, meta, clip_index, t, h, w, window=None) -> np.ndarray:
        del window
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
            out / "features" / "autointerp.json",
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
