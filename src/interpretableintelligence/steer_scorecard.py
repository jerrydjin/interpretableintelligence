"""Steering scorecard: how reliably, and how much, editing a feature moves the model's output.

``ii steer-scorecard RUN`` takes the run's top verified features (scoreboard
``real: true``, strongest ablation effect first), or the ids passed with
``--features``, and on every held-out window where a feature fires it

* **removes** the feature (scale 0, the scorer's single-feature ablation), and
* **boosts** it (scale ``STEER_BOOST_FACTOR``),

then measures how far the prediction moved against the clean prediction. For
GR00T the prediction is the planned action chunk, so this is how much the
planned actions change. Every edit is paired with ``STEER_CONTROLS`` matched
random-feature controls on the same window: a random direction at the
feature's decoder norm, carrying the same per-token edit sizes as the feature
edit, placed on a random permutation of that window's tokens. Each control
arm keeps one direction across every window, so it behaves like one random
feature.

The edits are the playground's (``steer.steer_edit``): only the chosen
feature's contribution changes, and the SAE's reconstruction error stays in.

Output, under the run folder::

    steering/effects.parquet   one row per (feature, kind, arm, clip, window)
    steering/scorecard.json    derived from the parquet; ``validate`` rebuilds it

This never reads or writes ``scoreboard.json``'s gates and never changes a
badge. It is an offline measurement on held-out clips, not a closed-loop
rollout, so for a robot policy it is not a finding under the rollout rules
(SPRT rise, McNemar, controls, random-weights SAE).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from interpretableintelligence.artifacts import read_json, write_json
from interpretableintelligence.stats import monte_carlo_pvalue, percentile_le
from interpretableintelligence.thresholds import (
    DEAD_ACT_EPS,
    STEER_ALPHA,
    STEER_BOOST_FACTOR,
    STEER_BOOTSTRAP,
    STEER_CONFIDENCE,
    STEER_CONTROLS,
    STEER_TIE_EPS,
    STEER_TOP_FEATURES,
)

SCHEMA_VERSION = 1
STEERING_DIR = "steering"
SCORECARD_FILE = "scorecard.json"
EFFECTS_FILE = "effects.parquet"
EFFECT_COLUMNS = ("feature_id", "kind", "arm", "control", "clip_id", "window", "rel_change", "abs_change")
KINDS = ("remove", "boost")
ARMS = ("feature", "control")
METRIC = "relative_l2_change"
CONTROL_DESCRIPTION = (
    "random direction at the feature's decoder norm, the same per-token edit sizes, "
    "on a random permutation of the same window's tokens; one direction per control arm"
)
NOTE = (
    "Offline steering on held-out clips. It does not change scoreboard.json or any badge, "
    "and for a robot policy it is not a rollout finding."
)
SCORECARD_KEYS = {
    "schema_version",
    "run_id",
    "model_sha",
    "layer",
    "split_sha",
    "adapter",
    "target",
    "metric",
    "selection",
    "feature_ids",
    "boost_factor",
    "n_controls",
    "control",
    "seed",
    "bootstrap",
    "confidence",
    "alpha",
    "sae_usable",
    "n_episodes",
    "n_windows",
    "rows",
    "rollout_finding",
    "note",
}
ROW_KEYS = {
    "feature_id",
    "label",
    "real",
    "kind",
    "factor",
    "n_windows",
    "n_episodes",
    "coverage",
    "action_change",
    "control_change",
    "excess",
    "ratio",
    "abs_change",
    "reliability",
    "control_percentile",
    "p_value",
    "beats_controls",
}
INTERVAL_KEYS = {"mean", "ci_low", "ci_high"}
RELIABILITY_KEYS = {"rate", "ci_low", "ci_high", "chance"}


# ---------------------------------------------------------------- selection


def select_features(scoreboard: dict, requested: list[int] | None, top: int, n_features: int) -> tuple[list[int], str]:
    """``(feature_ids, selection)``. Without ``requested``, the top verified features only."""
    if requested:
        bad = [fid for fid in requested if fid < 0 or fid >= n_features]
        if bad:
            raise ValueError(f"unknown feature ids {bad}; this SAE has features 0..{n_features - 1}")
        return list(dict.fromkeys(int(fid) for fid in requested)), "requested"
    real = [
        feature
        for feature in scoreboard.get("features") or []
        if isinstance(feature, dict) and feature.get("real") is True and _is_int(feature.get("id"))
    ]
    if not real:
        raise ValueError(
            "this run has no verified (real) features to score. Pass --features to measure unverified ones; "
            "the scorecard records that they are unverified and the badge does not change"
        )

    def strength(feature: dict) -> tuple[float, int]:
        effect = feature.get("ablation_effect")
        value = float(effect) if isinstance(effect, (int, float)) and not isinstance(effect, bool) and math.isfinite(effect) else -math.inf
        return (-value, int(feature["id"]))

    ordered = sorted(real, key=strength)
    return [int(feature["id"]) for feature in ordered[: max(1, int(top))]], "verified"


def parse_feature_ids(text: str) -> list[int]:
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as exc:
            raise ValueError(f"--features must be comma-separated integers, got {part!r}") from exc
    return out


# ---------------------------------------------------------------- measuring


def execute_steer_scorecard(
    run_dir: Path,
    *,
    clips_dir: Path | None = None,
    features: list[int] | None = None,
    top: int = STEER_TOP_FEATURES,
    boost: float = STEER_BOOST_FACTOR,
    controls: int = STEER_CONTROLS,
    seed: int = 0,
    bootstrap: int = STEER_BOOTSTRAP,
    max_episodes: int | None = None,
    log=print,
) -> dict:
    """Measure, write ``steering/``, and return the scorecard."""
    import torch  # noqa: PLC0415 (validate and view read steering/ without torch)

    from interpretableintelligence.steer import Steerer  # noqa: PLC0415 (loads torch models)

    run_dir = Path(run_dir)
    least = min_controls(STEER_ALPHA)
    if controls < least:
        raise ValueError(
            f"--controls must be at least {least}: with {controls} controls the smallest possible p-value is "
            f"1/{controls + 1}, so no edit could beat the controls at alpha {STEER_ALPHA}"
        )
    if bootstrap < 1:
        raise ValueError("--bootstrap must be at least 1")
    if not math.isfinite(boost) or boost == 1.0:
        raise ValueError("--boost must be a finite factor other than 1 (1 leaves the feature unchanged)")
    split = read_json(run_dir / "split.json")
    heldout = sorted(str(clip_id) for clip_id in split.get("heldout") or [])
    if not heldout:
        raise ValueError(f"{run_dir}/split.json has no held-out clips")
    if max_episodes is not None:
        heldout = heldout[: max(1, int(max_episodes))]

    steerer = Steerer(run_dir, clips_dir)
    ctx, adapter, sae = steerer.ctx, steerer.adapter, steerer.sae
    feature_ids, selection = select_features(ctx.scoreboard, features, top, int(sae.n_features))
    clip_order = {str(clip["clip_id"]): index for index, clip in enumerate(ctx.manifest.get("clips") or [])}
    directions = {fid: _control_directions(sae, fid, controls, seed) for fid in feature_ids}
    log(f"steering {len(feature_ids)} feature(s) {feature_ids} on {len(heldout)} held-out clip(s), {controls} controls each")

    columns: dict[str, list] = {name: [] for name in EFFECT_COLUMNS}
    n_windows = 0
    for clip_number, clip_id in enumerate(heldout, start=1):
        clip = steerer._clip(clip_id)
        windows = adapter.windows(clip)
        for window in windows:
            n_windows += 1
            clean = steerer._clean(clip, window)
            acts_shape = _acts_shape(clean)
            feats = clean.features.reshape(-1, clean.features.shape[-1])
            baseline = clean.baseline.detach().to(torch.float32)
            edits, keys = [], []
            for fid in feature_ids:
                column = feats[:, fid].to(torch.float32)
                if not bool((column > DEAD_ACT_EPS).any()):
                    continue
                for kind_index, kind in enumerate(KINDS):
                    factor = 0.0 if kind == "remove" else float(boost)
                    edits.append(_feature_edit(sae, fid, factor))
                    keys.append((fid, kind, "feature", -1))
                    coef = (factor - 1.0) * column  # what the feature edit adds along its decoder column, per token
                    rng = np.random.default_rng(
                        np.random.SeedSequence([int(seed), int(fid), kind_index, clip_order.get(clip_id, 0), _stable(clip_id), int(window.index)])
                    )
                    for control in range(controls):
                        perm = torch.as_tensor(rng.permutation(coef.shape[0]), dtype=torch.long, device=coef.device)
                        delta = torch.zeros(coef.shape[0], directions[fid].shape[1], dtype=torch.float32, device=coef.device)
                        delta[perm] = coef[:, None] * directions[fid][control][None, :].to(coef.device)
                        edits.append(_delta_edit(delta.reshape(*acts_shape, -1)))
                        keys.append((fid, kind, "control", control))
            if not edits:
                continue
            with torch.no_grad():
                preds = _predict_many(adapter, clip, window, edits)
            base_norm = float(baseline.norm().clamp(min=1e-12))
            for (fid, kind, arm, control), pred in zip(keys, preds, strict=True):
                diff = pred.detach().to(torch.float32).reshape(baseline.shape) - baseline
                columns["feature_id"].append(fid)
                columns["kind"].append(kind)
                columns["arm"].append(arm)
                columns["control"].append(control)
                columns["clip_id"].append(clip_id)
                columns["window"].append(int(window.index))
                columns["rel_change"].append(_finite(float(diff.norm()) / base_norm))
                columns["abs_change"].append(_finite(float(diff.reshape(diff.shape[0], -1).norm(dim=-1).mean())) if diff.numel() else 0.0)
        log(f"  [{clip_number}/{len(heldout)}] {clip_id}: {len(windows)} window(s)")

    table = effects_table(columns)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": ctx.run_id,
        "model_sha": ctx.model_sha,
        "layer": ctx.layer,
        "split_sha": str((ctx.manifest.get("split") or {}).get("sha", "")),
        "adapter": ctx.adapter_name,
        "target": "action_chunk" if ctx.adapter_name.startswith("groot:") else "masked_prediction",
        "metric": METRIC,
        "selection": selection,
        "feature_ids": feature_ids,
        "boost_factor": float(boost),
        "n_controls": int(controls),
        "control": CONTROL_DESCRIPTION,
        "seed": int(seed),
        "bootstrap": int(bootstrap),
        "confidence": STEER_CONFIDENCE,
        "alpha": STEER_ALPHA,
        "sae_usable": bool(steerer.sae_usable),
        "n_episodes": len(heldout),
        "n_windows": n_windows,
        "rollout_finding": False,
        "note": NOTE,
    }
    scorecard = summarize(table, meta, ctx.features_by_id)
    out = run_dir / STEERING_DIR
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{EFFECTS_FILE}.tmp"
    pq.write_table(table, tmp)
    tmp.replace(out / EFFECTS_FILE)
    write_json(out / SCORECARD_FILE, scorecard)
    return scorecard


def min_controls(alpha: float) -> int:
    """Fewest controls for which ``1 / (controls + 1) <= alpha``, the best p-value a feature can reach."""
    return max(1, math.ceil(1.0 / float(alpha) - 1.0 - 1e-9))


def effects_table(columns: dict[str, list]) -> pa.Table:
    return pa.table(
        {
            "feature_id": pa.array(columns["feature_id"], type=pa.int32()),
            "kind": pa.array(columns["kind"], type=pa.string()),
            "arm": pa.array(columns["arm"], type=pa.string()),
            "control": pa.array(columns["control"], type=pa.int16()),
            "clip_id": pa.array(columns["clip_id"], type=pa.string()),
            "window": pa.array(columns["window"], type=pa.int16()),
            "rel_change": pa.array(columns["rel_change"], type=pa.float32()),
            "abs_change": pa.array(columns["abs_change"], type=pa.float32()),
        }
    )


def _feature_edit(sae, fid: int, factor: float):
    from interpretableintelligence.steer import steer_edit  # noqa: PLC0415

    return steer_edit(sae, (int(fid),), "ablate" if factor == 0.0 else "scale", float(factor))


def _delta_edit(delta):
    import torch  # noqa: PLC0415

    def edit(acts: torch.Tensor, delta: torch.Tensor = delta) -> torch.Tensor:
        return acts + delta.to(device=acts.device, dtype=acts.dtype).reshape(acts.shape)

    return edit


def _control_directions(sae, fid: int, controls: int, seed: int) -> torch.Tensor:
    """``[controls, d_model]`` random directions, each at feature ``fid``'s decoder norm."""
    import torch  # noqa: PLC0415

    column = sae.W_dec[:, int(fid)].detach().to(torch.float32)
    norm = float(column.norm().clamp(min=1e-8))
    generator = torch.Generator(device="cpu").manual_seed(int(np.random.SeedSequence([int(seed), int(fid), 23]).generate_state(1)[0]))
    raw = torch.randn(int(controls), int(column.shape[0]), generator=generator, dtype=torch.float32)
    return (raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8) * norm).to(column.device)


def _acts_shape(clean) -> tuple[int, ...]:
    return tuple(int(size) for size in clean.features.shape[:-1])


def _predict_many(adapter, clip, window, edits) -> list:
    many = getattr(adapter, "predict_masked_many", None)
    if many is not None:
        return list(many(clip, window, edits))
    return [adapter.predict_masked(clip, window, edit=edit)[0] for edit in edits]


def _stable(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


# ---------------------------------------------------------------- summary


def summarize(table: pa.Table, meta: dict, features_by_id: dict[int, dict]) -> dict:
    """The scorecard, derived only from ``effects.parquet`` rows and the run settings in ``meta``.

    Per (feature, kind), over held-out windows where the feature fires:

    * ``action_change``: mean relative L2 change of the prediction (the planned chunk for GR00T).
    * ``control_change``: the same for the matched random controls, averaged over control arms.
    * ``excess``: ``action_change - control_change``. ``ratio``: ``action_change / control_change``.
    * ``reliability``: share of windows where the feature edit moved the output more than every
      control did on that window. Chance is ``1 / (n_controls + 1)``.
    * ``p_value``: Monte Carlo p-value of the feature's mean against the control arms' means.

    Intervals are percentile bootstraps that resample held-out episodes (clips), since windows of
    one episode are not independent. They are null with fewer than two episodes.
    """
    columns = {name: table.column(name).to_pylist() for name in ("feature_id", "kind", "arm", "control", "clip_id", "window")}
    rel = table.column("rel_change").to_numpy(zero_copy_only=False).astype(np.float64)
    absolute = table.column("abs_change").to_numpy(zero_copy_only=False).astype(np.float64)
    n_controls = int(meta["n_controls"])
    rows = []
    for fid in meta["feature_ids"]:
        scored = features_by_id.get(int(fid)) or {}
        for kind_index, kind in enumerate(KINDS):
            factor = 0.0 if kind == "remove" else float(meta["boost_factor"])
            cand: dict[tuple[str, int], float] = {}
            cand_abs: dict[tuple[str, int], float] = {}
            ctrl: dict[tuple[str, int], np.ndarray] = {}
            for index in range(len(rel)):
                if columns["feature_id"][index] != fid or columns["kind"][index] != kind:
                    continue
                key = (columns["clip_id"][index], int(columns["window"][index]))
                if columns["arm"][index] == "feature":
                    cand[key] = rel[index]
                    cand_abs[key] = absolute[index]
                else:
                    ctrl.setdefault(key, np.full(n_controls, np.nan))[int(columns["control"][index])] = rel[index]
            keys = sorted(cand)
            missing = [key for key in keys if key not in ctrl or np.isnan(ctrl[key]).any()]
            if missing or set(ctrl) - set(cand):
                raise ValueError(f"feature {fid} {kind}: every window needs one feature row and {n_controls} control rows")
            rows.append(
                _row(
                    fid=int(fid),
                    scored=scored,
                    kind=kind,
                    factor=factor,
                    keys=keys,
                    cand=np.array([cand[key] for key in keys], dtype=np.float64),
                    cand_abs=np.array([cand_abs[key] for key in keys], dtype=np.float64),
                    ctrl=np.stack([ctrl[key] for key in keys]) if keys else np.zeros((0, n_controls)),
                    meta=meta,
                    rng=np.random.default_rng(np.random.SeedSequence([int(meta["seed"]), int(fid), kind_index, 99])),
                )
            )
    return {**{key: meta[key] for key in SCORECARD_KEYS - {"rows"}}, "rows": rows}


def _row(*, fid, scored, kind, factor, keys, cand, cand_abs, ctrl, meta, rng) -> dict:
    n_controls = int(meta["n_controls"])
    n = len(keys)
    episodes = sorted({clip_id for clip_id, _window in keys})
    total = int(meta["n_windows"])
    base = {
        "feature_id": fid,
        "label": str(scored.get("label") or ""),
        "real": scored.get("real") is True,
        "kind": kind,
        "factor": factor,
        "n_windows": n,
        "n_episodes": len(episodes),
        "coverage": (n / total) if total else 0.0,
    }
    if n == 0:
        empty = {"mean": None, "ci_low": None, "ci_high": None}
        return {
            **base,
            "action_change": dict(empty),
            "control_change": dict(empty),
            "excess": dict(empty),
            "ratio": dict(empty),
            "abs_change": None,
            "reliability": {"rate": None, "ci_low": None, "ci_high": None, "chance": 1.0 / (n_controls + 1)},
            "control_percentile": None,
            "p_value": None,
            "beats_controls": False,
        }
    ctrl_mean = ctrl.mean(axis=1)
    beats = (cand > ctrl.max(axis=1) + STEER_TIE_EPS).astype(np.float64)
    arm_means = ctrl.mean(axis=0)
    cand_mean = float(cand.mean())
    ctrl_overall = float(ctrl_mean.mean())

    # Cluster bootstrap over episodes: per-episode sums, resampled with replacement.
    episode_index = {clip_id: index for index, clip_id in enumerate(episodes)}
    owner = np.array([episode_index[clip_id] for clip_id, _window in keys], dtype=np.int64)
    n_ep = len(episodes)
    sums = np.zeros((n_ep, 4), dtype=np.float64)  # cand, ctrl, beats, count
    np.add.at(sums, owner, np.stack([cand, ctrl_mean, beats, np.ones(n)], axis=1))
    intervals: dict[str, tuple[float | None, float | None]] = {}
    if n_ep >= 2:
        draws = rng.integers(0, n_ep, size=(int(meta["bootstrap"]), n_ep))
        weights = np.zeros((draws.shape[0], n_ep), dtype=np.float64)
        np.add.at(weights, (np.repeat(np.arange(draws.shape[0]), n_ep), draws.ravel()), 1.0)
        boot = weights @ sums
        count = boot[:, 3]
        stats = {
            "action_change": boot[:, 0] / count,
            "control_change": boot[:, 1] / count,
            "excess": (boot[:, 0] - boot[:, 1]) / count,
            "reliability": boot[:, 2] / count,
        }
        ctrl_boot = boot[:, 1]
        stats["ratio"] = np.where(ctrl_boot > 1e-12, boot[:, 0] / np.maximum(ctrl_boot, 1e-12), np.nan)
        lo_q, hi_q = (1.0 - float(meta["confidence"])) / 2.0, 1.0 - (1.0 - float(meta["confidence"])) / 2.0
        for name, values in stats.items():
            finite = values[np.isfinite(values)]
            if finite.size == 0 or (name == "ratio" and finite.size < values.size):
                intervals[name] = (None, None)
            else:
                intervals[name] = (float(np.quantile(finite, lo_q)), float(np.quantile(finite, hi_q)))

    def interval(name: str, mean: float | None) -> dict:
        low, high = intervals.get(name, (None, None))
        return {"mean": mean, "ci_low": low, "ci_high": high}

    ratio = cand_mean / ctrl_overall if ctrl_overall > 1e-12 else None
    excess = interval("excess", cand_mean - ctrl_overall)
    p_value = monte_carlo_pvalue(cand_mean, arm_means)
    reliability = interval("reliability", float(beats.mean()))
    return {
        **base,
        "action_change": interval("action_change", cand_mean),
        "control_change": interval("control_change", ctrl_overall),
        "excess": excess,
        "ratio": interval("ratio", ratio) if ratio is not None else {"mean": None, "ci_low": None, "ci_high": None},
        "abs_change": float(cand_abs.mean()),
        "reliability": {"rate": reliability["mean"], "ci_low": reliability["ci_low"], "ci_high": reliability["ci_high"], "chance": 1.0 / (n_controls + 1)},
        "control_percentile": percentile_le(cand_mean, arm_means),
        "p_value": p_value,
        "beats_controls": bool(p_value <= float(meta["alpha"]) and excess["ci_low"] is not None and excess["ci_low"] > 0.0),
    }


# ---------------------------------------------------------------- reading and checking


def read_scorecard(run_dir: Path) -> dict | None:
    """The stored scorecard, or None when the run has no ``steering/``."""
    path = Path(run_dir) / STEERING_DIR / SCORECARD_FILE
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def check_steering(run_dir: Path, manifest: dict, scoreboard: dict) -> list[str]:
    """Contract errors for ``steering/``. Empty when it is absent or consistent."""
    folder = Path(run_dir) / STEERING_DIR
    if not folder.exists():
        return []
    errors: list[str] = []
    if not folder.is_dir():
        return ["steering/ must be a directory"]
    card_path, effects_path = folder / SCORECARD_FILE, folder / EFFECTS_FILE
    if not card_path.is_file() or not effects_path.is_file():
        return [f"steering/{SCORECARD_FILE} and steering/{EFFECTS_FILE} are required"]
    try:
        card = read_json(card_path)
    except (OSError, ValueError) as exc:
        return [f"steering/{SCORECARD_FILE}: {exc}"]
    if not isinstance(card, dict):
        return [f"steering/{SCORECARD_FILE} must be a JSON object"]
    if set(card) != SCORECARD_KEYS:
        return [f"steering/{SCORECARD_FILE} keys mismatch missing={sorted(SCORECARD_KEYS - set(card))} extra={sorted(set(card) - SCORECARD_KEYS)}"]
    if card["schema_version"] != SCHEMA_VERSION:
        errors.append(f"steering/{SCORECARD_FILE} schema_version must be {SCHEMA_VERSION}")
    if card["rollout_finding"] is not False:
        errors.append(f"steering/{SCORECARD_FILE} rollout_finding must be false")
    model = manifest.get("model") or {}
    expected = {
        "run_id": manifest.get("run_id"),
        "model_sha": model.get("sha"),
        "layer": manifest.get("layer"),
        "split_sha": (manifest.get("split") or {}).get("sha"),
        "adapter": model.get("adapter"),
    }
    for key, value in expected.items():
        if card[key] != value:
            errors.append(f"steering/{SCORECARD_FILE} {key} {card[key]!r} does not match the manifest ({value!r})")
    if errors:
        return errors
    try:
        table = pq.read_table(effects_path)
    except (OSError, pa.ArrowInvalid) as exc:
        return [f"steering/{EFFECTS_FILE}: {exc}"]
    if tuple(table.column_names) != EFFECT_COLUMNS:
        return [f"steering/{EFFECTS_FILE} columns must be {list(EFFECT_COLUMNS)}, got {table.column_names}"]
    kinds = set(table.column("kind").to_pylist())
    arms = set(table.column("arm").to_pylist())
    if not kinds <= set(KINDS) or not arms <= set(ARMS):
        return [f"steering/{EFFECTS_FILE} kind must be in {list(KINDS)} and arm in {list(ARMS)}"]
    features_by_id = {int(f["id"]): f for f in scoreboard.get("features") or [] if isinstance(f, dict) and _is_int(f.get("id"))}
    try:
        rebuilt = summarize(table, {key: card[key] for key in SCORECARD_KEYS - {"rows"}}, features_by_id)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return [f"steering/{EFFECTS_FILE} does not rebuild the scorecard: {exc}"]
    from interpretableintelligence.validate import _same_json  # noqa: PLC0415 (validate imports this module lazily)

    _same_json(card, rebuilt, f"steering/{SCORECARD_FILE}", errors)
    return errors


def format_report(card: dict) -> str:
    target = "planned actions" if card.get("target") == "action_chunk" else "prediction"
    lines = [
        f"steering scorecard: {card['run_id']} ({card['selection']} features, {card['n_controls']} matched random controls, "
        f"{card['n_episodes']} held-out episodes, {card['n_windows']} windows)",
        f"metric: relative L2 change of the {target}; {int(round(100 * card['confidence']))}% intervals resample episodes",
    ]
    for row in card["rows"]:
        name = f"feature {row['feature_id']}" + (f" {row['label']}" if row["label"] else "") + (" [real]" if row["real"] else " [unverified]")
        verb = "remove" if row["kind"] == "remove" else f"boost x{row['factor']:g}"
        if not row["n_windows"]:
            lines.append(f"  {name:<32} {verb:<10} never fires on held-out windows")
            continue
        ratio = row["ratio"]
        rel = row["reliability"]
        lines.append(
            f"  {name:<32} {verb:<10} change {_fmt(row['action_change']['mean'])} vs controls {_fmt(row['control_change']['mean'])}"
            f"  ratio {_fmt(ratio['mean'], 2)} [{_fmt(ratio['ci_low'], 2)}, {_fmt(ratio['ci_high'], 2)}]"
            f"  beats every control on {_pct(rel['rate'])} [{_pct(rel['ci_low'])}, {_pct(rel['ci_high'])}] (chance {_pct(rel['chance'])})"
            f"  p={row['p_value']:.3f}  {'beats controls' if row['beats_controls'] else 'does not beat controls'}"
        )
    lines.append(NOTE)
    return "\n".join(lines)


def _fmt(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def _finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
