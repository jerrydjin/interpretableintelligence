"""Attribution-patching prefilter.

One forward and one backward per window scores every SAE feature. The score
on a clip is the sum over that clip's windows and tokens of

    activation * d(mean masked predictor MSE) / d(activation)

with the same straight-through decoder path as ``estimate_effects``. Features
are ranked by the absolute value of the mean score across clips, so a feature
that fires hard but does not move the predictor stays near the bottom.
Expensive ablations then run only on the top-K.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from physint.artifacts import ATTRIBUTION_COLUMNS
from physint.score import estimate_effects
from physint.stats import spearman
from physint.thresholds import DEFAULT_PREFILTER_K

METHOD = "attribution_patching"
RANK_BY = (
    "abs(mean over clips of the per-clip score); "
    "the per-clip score sums activation * d(loss)/d(activation) over tokens and windows; "
    "ties break toward the smaller feature id"
)


def attribution_by_clip(adapter, sae, clips) -> dict[str, np.ndarray]:
    """Per-clip attribution, shape ``[n_features]``, summed over windows and tokens."""
    n_features = int(sae.n_features)
    if n_features < 1:
        raise RuntimeError("SAE has no features to attribute")
    scores: dict[str, np.ndarray] = {}
    for clip in clips:
        windows = adapter.windows(clip)
        if not windows:
            raise RuntimeError(f"{clip.clip_id} has no windows")
        total = np.zeros(n_features, dtype=np.float64)
        for window in windows:
            effect = estimate_effects(adapter, sae, clip, window).detach().cpu().numpy()
            flat = np.asarray(effect, dtype=np.float64).reshape(-1, effect.shape[-1])
            if flat.shape[1] != n_features:
                raise RuntimeError(
                    f"attribution width {flat.shape[1]} != SAE n_features {n_features} on {clip.clip_id}"
                )
            total += flat.sum(axis=0)
        scores[clip.clip_id] = total
    if not scores:
        raise RuntimeError("no clips to attribute")
    return scores


def summarize_attribution(by_clip: dict[str, np.ndarray]) -> tuple[dict[str, list], np.ndarray]:
    """Parquet rows and the mean score used for ranking.

    Each per-clip vector is stored as float32. The ranking mean is the mean of
    those stored values, so a reader of the parquet recomputes the same order.
    """
    if not by_clip:
        raise ValueError("no attribution scores")
    clip_ids = sorted(by_clip)
    stored = [np.asarray(by_clip[clip_id], dtype=np.float64).astype(np.float32) for clip_id in clip_ids]
    width = stored[0].shape[0]
    if any(vec.shape != (width,) for vec in stored):
        raise ValueError("attribution vectors differ in width")
    rows: dict[str, list] = {key: [] for key in ATTRIBUTION_COLUMNS}
    mean = np.zeros(width, dtype=np.float64)
    n_clips = len(stored)
    for feature_id in range(width):
        total = 0.0
        for clip_id, vec in zip(clip_ids, stored, strict=True):
            value = float(vec[feature_id])
            rows["feature_id"].append(feature_id)
            rows["clip_id"].append(clip_id)
            rows["score"].append(value)
            total += value
        mean[feature_id] = total / n_clips
    return rows, mean


def rank_feature_ids(mean_scores: np.ndarray) -> list[int]:
    """Feature ids sorted by descending ``|score|``, then ascending id."""
    scores = np.asarray(mean_scores, dtype=np.float64).ravel()
    if scores.ndim != 1:
        raise ValueError(f"expected one score per feature, got {scores.shape}")
    feature_ids = np.arange(scores.shape[0])
    order = np.lexsort((feature_ids, -np.abs(scores)))
    return [int(feature_id) for feature_id in order]


def candidates_document(
    mean_scores: np.ndarray,
    prefilter_k: int,
    *,
    labels: list[str] | None = None,
    seconds: dict | None = None,
) -> dict:
    """Full ranking plus the top-K shortlist expensive ablations should run on."""
    scores = np.asarray(mean_scores, dtype=np.float64).ravel()
    n_features = int(scores.shape[0])
    if prefilter_k < 1:
        raise ValueError(f"prefilter K must be >= 1, got {prefilter_k}")
    if labels is not None and len(labels) != n_features:
        raise ValueError(f"expected {n_features} labels, got {len(labels)}")
    k = min(int(prefilter_k), n_features)
    ranked_ids = rank_feature_ids(scores)
    ranked = []
    for rank, feature_id in enumerate(ranked_ids, start=1):
        score = float(scores[feature_id])
        ranked.append(
            {
                "rank": rank,
                "feature_id": feature_id,
                "label": "" if labels is None else labels[feature_id],
                "score": score,
                "abs_score": abs(score),
            }
        )
    document = {
        "method": METHOD,
        "prefilter_k": k,
        "rank_by": RANK_BY,
        "ranked": ranked,
        "candidates": [feature_id for feature_id in ranked_ids[:k]],
        "seconds": {
            "attribution": None if seconds is None else seconds.get("attribution"),
            "ablation": None if seconds is None else seconds.get("ablation"),
            "random_controls": None if seconds is None else seconds.get("random_controls"),
        },
    }
    return document


def compare_run(run_dir: Path, k: int | None = None) -> dict:
    """Spearman and top-K overlap between attribution scores and ablation effects.

    Feature-level ablation effects come from ``scoreboard.json`` when that
    file has numeric ``ablation_effect`` values, and otherwise from the mean
    ``effect_kind="ablation"`` row in ``features/clip_effects.parquet``.
    Attribution is ``|mean score over clips|`` (held-out clips when
    ``split.json`` lists them). Clip-level Spearman pairs the per-clip
    attribution score with the mean ablation effect on that clip.
    """
    run_dir = Path(run_dir)
    by_clip = _read_attribution(run_dir / "features" / "attribution.parquet")
    feature_ids = sorted({feature_id for scores in by_clip.values() for feature_id in scores})
    if len(feature_ids) < 2:
        raise ValueError("need at least two features to compare rankings")

    heldout = _read_heldout(run_dir / "split.json")
    if heldout is not None:
        missing = [clip_id for clip_id in heldout if clip_id not in by_clip]
        if missing:
            raise ValueError(f"attribution.parquet is missing held-out clips: {missing[:3]}")
    all_mean = _mean_scores(by_clip, list(by_clip))
    held_mean = _mean_scores(by_clip, heldout) if heldout else None

    scoreboard_effects = _read_scoreboard_effects(run_dir / "scoreboard.json")
    clip_effects = _read_clip_ablation(run_dir / "features" / "clip_effects.parquet")
    if scoreboard_effects is not None:
        feature_effects = {fid: value for fid, value in scoreboard_effects.items() if value is not None}
        feature_source = "scoreboard.json ablation_effect"
    elif clip_effects is not None:
        feature_effects = _mean_effect_by_feature(clip_effects)
        feature_source = "clip_effects.parquet ablation rows, mean over clips"
    else:
        raise FileNotFoundError(
            f"{run_dir} has features/attribution.parquet but no scoreboard.json ablation_effect "
            "and no features/clip_effects.parquet ablation rows"
        )
    if len(feature_effects) < 2:
        raise ValueError("need at least two measured ablation effects to compare rankings")

    labels = _read_labels(run_dir)
    requested_k = DEFAULT_PREFILTER_K if k is None else int(k)
    if requested_k < 1:
        raise ValueError(f"K must be >= 1, got {requested_k}")
    width = max(feature_ids) + 1
    mean_vector = np.zeros(width, dtype=np.float64)
    for feature_id, score in all_mean.items():
        mean_vector[feature_id] = score
    attribution_order = [feature_id for feature_id in rank_feature_ids(mean_vector) if feature_id in all_mean]
    effective_k = min(requested_k, len(attribution_order), len(feature_effects))
    attribution_topk = attribution_order[:effective_k]
    ablation_order = _rank_effects(feature_effects)
    ablation_topk = ablation_order[:effective_k]
    overlap = len(set(attribution_topk) & set(ablation_topk))

    held_spearman, held_n = _feature_spearman(held_mean, feature_effects)
    all_spearman, all_n = _feature_spearman(all_mean, feature_effects)
    clip_spearman, clip_n = _clip_spearman(by_clip, clip_effects)

    rows = []
    rank_of = {feature_id: rank for rank, feature_id in enumerate(attribution_order, start=1)}
    for feature_id in attribution_order:
        rows.append(
            {
                "feature_id": feature_id,
                "label": labels.get(feature_id, ""),
                "rank": rank_of[feature_id],
                "score": all_mean[feature_id],
                "abs_score": abs(all_mean[feature_id]),
                "ablation_effect": feature_effects.get(feature_id),
            }
        )
    return {
        "feature_source": feature_source,
        "spearman_feature_heldout": held_spearman,
        "n_feature_heldout": held_n,
        "spearman_feature_all": all_spearman,
        "n_feature_all": all_n,
        "spearman_clip": clip_spearman,
        "n_clip": clip_n,
        "k": effective_k,
        "overlap": overlap,
        "attribution_topk": attribution_topk,
        "ablation_topk": ablation_topk,
        "features": rows,
    }


def format_report(result: dict) -> str:
    """Text report printed by ``scripts/compare_attribution_vs_ablation.py``."""
    primary_name, primary, primary_n = _primary_spearman(result)
    lines = [
        (
            f"Spearman |attribution| vs ablation ({primary_name}, n={primary_n}): "
            f"{_fmt_corr(primary)}"
        ),
        f"ablation source: {result['feature_source']}",
        (
            f"Spearman |mean attribution over held-out clips| vs ablation_effect "
            f"(n={result['n_feature_heldout']}): {_fmt_corr(result['spearman_feature_heldout'])}"
        ),
        (
            f"Spearman |mean attribution over all clips| vs ablation_effect "
            f"(n={result['n_feature_all']}): {_fmt_corr(result['spearman_feature_all'])}"
        ),
        (
            f"Spearman |per-clip attribution| vs per-clip ablation "
            f"(n={result['n_clip']}): {_fmt_corr(result['spearman_clip'])}"
        ),
        (
            f"top-{result['k']} overlap: {result['overlap']}/{result['k']}"
        ),
        f"attribution top-{result['k']}: {result['attribution_topk']}",
        f"ablation top-{result['k']}: {result['ablation_topk']}",
        "rank feature_id label score ablation_effect",
    ]
    for row in result["features"]:
        effect = row["ablation_effect"]
        effect_text = "null" if effect is None else f"{effect:.6e}"
        label = row["label"] or "-"
        lines.append(
            f"{row['rank']:4d} {row['feature_id']:4d} {label:<8} {row['score']:+.6e} {effect_text}"
        )
    return "\n".join(lines)


def _primary_spearman(result: dict) -> tuple[str, float | None, int]:
    if result["spearman_feature_heldout"] is not None:
        return "held-out feature mean", result["spearman_feature_heldout"], result["n_feature_heldout"]
    if result["spearman_feature_all"] is not None:
        return "all-clip feature mean", result["spearman_feature_all"], result["n_feature_all"]
    return "per-clip pairs", result["spearman_clip"], result["n_clip"]


def _fmt_corr(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _feature_spearman(
    mean_scores: dict[int, float] | None,
    effects: dict[int, float],
) -> tuple[float | None, int]:
    if mean_scores is None:
        return None, 0
    shared = sorted(set(mean_scores) & set(effects))
    if len(shared) < 2:
        return None, len(shared)
    correlation = spearman(
        [abs(mean_scores[feature_id]) for feature_id in shared],
        [effects[feature_id] for feature_id in shared],
    )
    return correlation, len(shared)


def _clip_spearman(
    by_clip: dict[str, dict[int, float]],
    clip_effects: dict[tuple[int, str], float] | None,
) -> tuple[float | None, int]:
    if not clip_effects:
        return None, 0
    left: list[float] = []
    right: list[float] = []
    for clip_id, scores in by_clip.items():
        for feature_id, score in scores.items():
            effect = clip_effects.get((feature_id, clip_id))
            if effect is None:
                continue
            left.append(abs(score))
            right.append(effect)
    if len(left) < 2:
        return None, len(left)
    return spearman(left, right), len(left)


def _mean_effect_by_feature(clip_effects: dict[tuple[int, str], float]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for (feature_id, _clip_id), effect in clip_effects.items():
        grouped[feature_id].append(effect)
    return {feature_id: float(np.mean(values)) for feature_id, values in grouped.items()}


def _rank_effects(effects: dict[int, float]) -> list[int]:
    feature_ids = np.array(sorted(effects), dtype=np.int64)
    values = np.array([effects[int(feature_id)] for feature_id in feature_ids], dtype=np.float64)
    order = np.lexsort((feature_ids, -values))
    return [int(feature_ids[index]) for index in order]


def _mean_scores(by_clip: dict[str, dict[int, float]], clip_ids: list[str]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for clip_id in clip_ids:
        for feature_id, score in by_clip[clip_id].items():
            grouped[feature_id].append(score)
    return {feature_id: float(np.mean(values)) for feature_id, values in grouped.items()}


def _read_attribution(path: Path) -> dict[str, dict[int, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    table = pq.read_table(path)
    if tuple(table.column_names) != ATTRIBUTION_COLUMNS:
        raise ValueError(f"{path} columns must be {ATTRIBUTION_COLUMNS}")
    frame = table.to_pydict()
    by_clip: dict[str, dict[int, float]] = {}
    for feature_id, clip_id, score in zip(
        frame["feature_id"], frame["clip_id"], frame["score"], strict=True
    ):
        clip_scores = by_clip.setdefault(str(clip_id), {})
        feature_id = int(feature_id)
        if feature_id in clip_scores:
            raise ValueError(f"duplicate attribution row for feature {feature_id} clip {clip_id}")
        value = float(score)
        if not np.isfinite(value):
            raise ValueError(f"non-finite attribution for feature {feature_id} clip {clip_id}")
        clip_scores[feature_id] = value
    if not by_clip:
        raise ValueError(f"{path} has no rows")
    widths = {len(scores) for scores in by_clip.values()}
    if len(widths) != 1:
        raise ValueError("attribution.parquet must list the same features for every clip")
    return by_clip


def _read_heldout(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    heldout = payload.get("heldout") if isinstance(payload, dict) else None
    if not isinstance(heldout, list) or not heldout:
        return None
    return [str(clip_id) for clip_id in heldout]


def _read_scoreboard_effects(path: Path) -> dict[int, float | None] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise ValueError(f"{path} features must be a list")
    effects: dict[int, float | None] = {}
    for row in features:
        if not isinstance(row, dict) or "id" not in row or "ablation_effect" not in row:
            raise ValueError(f"{path} feature rows need id and ablation_effect")
        feature_id = int(row["id"])
        effect = row["ablation_effect"]
        effects[feature_id] = None if effect is None else float(effect)
    return effects


def _read_clip_ablation(path: Path) -> dict[tuple[int, str], float] | None:
    if not path.is_file():
        return None
    table = pq.read_table(path, columns=["feature_id", "clip_id", "effect", "effect_kind"])
    frame = table.to_pydict()
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for feature_id, clip_id, effect, kind in zip(
        frame["feature_id"], frame["clip_id"], frame["effect"], frame["effect_kind"], strict=True
    ):
        if kind != "ablation":
            continue
        grouped[(int(feature_id), str(clip_id))].append(float(effect))
    if not grouped:
        return None
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _read_labels(run_dir: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    candidates_path = run_dir / "features" / "candidates.json"
    if candidates_path.is_file():
        payload = json.loads(candidates_path.read_text(encoding="utf-8"))
        for row in payload.get("ranked", []):
            if isinstance(row, dict) and "feature_id" in row:
                labels[int(row["feature_id"])] = str(row.get("label") or "")
    scoreboard_path = run_dir / "scoreboard.json"
    if scoreboard_path.is_file():
        payload = json.loads(scoreboard_path.read_text(encoding="utf-8"))
        for row in payload.get("features", []):
            if isinstance(row, dict) and "id" in row and row.get("label"):
                labels[int(row["id"])] = str(row["label"])
    return labels
