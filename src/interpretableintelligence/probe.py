"""Held-out failure probe on SAE latents, against a random-weights SAE and raw activations.

``ii probe RUN --labels FILE`` asks whether the trained SAE's latents predict
episode failure ahead of time better than two baselines built from the same
activations:

* ``trained_sae``: the run's SAE (``sae/``).
* ``random_sae``: :func:`interpretableintelligence.sae.random_weights_sae`, the
  same architecture, width and ``k`` with untrained, seeded weights.
* ``raw``: the layer activations themselves (``acts/``).

Each episode is a clip. For a failed episode the probe may only look at tokens
whose time bin ends at least ``lead_s`` seconds before the failure onset; for a
successful one, ``lead_s`` before the end of the clip. Each input is max-pooled
over those tokens, standardised on the train split, and fed to the same L2
logistic regression, fit on ``split.json``'s train clips only. Held-out clips are
scored once.

The gate (``sae_gate``) is ``pass`` only when, at a lead time of at least
``PROBE_MIN_LEAD_S``, the trained SAE's held-out AUROC beats **both** baselines
with a one-sided paired bootstrap p-value below ``PROBE_ALPHA`` (episodes
resampled with replacement, within class). It is ``not_run`` when it cannot be
evaluated (too few episodes of a class, or a lead below the floor), and ``fail``
otherwise. ``positive`` is ``sae_gate == "pass"``.

Output, under the run folder::

    probe/scores.parquet   one row per (input, episode): probe logit and lead time
    probe/probe.json       derived from the parquet; ``validate`` rebuilds it

This never reads or writes ``scoreboard.json``'s gates, never sets ``real``, and
is not a rollout finding. Time convention: windows tile a clip back to back, so
bin ``t`` of window ``k`` ends at ``(k * grid.t + t + 1) * grid.tubelet / fps``
seconds. That holds for the ``fake`` adapter; other adapters need their own
mapping before their probe results mean anything in seconds.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from interpretableintelligence.artifacts import read_json, write_json
from interpretableintelligence.thresholds import (
    PROBE_ALPHA,
    PROBE_BOOTSTRAP,
    PROBE_CONFIDENCE,
    PROBE_L2,
    PROBE_MIN_CLASS,
    PROBE_MIN_LEAD_S,
)

SCHEMA_VERSION = 1
PROBE_DIR = "probe"
RESULT_FILE = "probe.json"
SCORES_FILE = "scores.parquet"
INPUTS = ("trained_sae", "random_sae", "raw")
BASELINES = ("random_sae", "raw")
SPLITS = ("train", "heldout")
GATE_VALUES = ("pass", "fail", "not_run")
SCORE_COLUMNS = ("input", "clip_id", "split", "failure", "ref_s", "cutoff_s", "lead_s", "score")
POOL = "max"
PROBE_KIND = "logistic_l2"
RANDOM_INIT = "train_sae initializer, zero optimiser steps"
TIME_EPS = 1e-9
ROWS_PER_CHUNK = 8192
NOTE = (
    "Offline failure probe on held-out clips. It does not change scoreboard.json or any badge, "
    "never sets real, and for a robot policy it is not a rollout finding."
)
RESULT_KEYS = {
    "schema_version",
    "run_id",
    "model_sha",
    "layer",
    "split_sha",
    "adapter",
    "target",
    "labels",
    "pool",
    "probe",
    "lead_s",
    "min_lead_s",
    "bootstrap",
    "confidence",
    "alpha",
    "seed",
    "n_episodes",
    "inputs",
    "comparisons",
    "sae_gate",
    "positive",
    "reason",
    "real",
    "rollout_finding",
    "note",
}
INPUT_KEYS = {"n_dims", "auroc", "ci_low", "ci_high", "lead_time_s"}
TRAINED_EXTRA = {"sae_usable", "loss_recovered"}
RANDOM_EXTRA = {"seed", "init", "weights_sha256", "sae_usable", "loss_recovered", "clip_effects", "real"}
COMPARISON_KEYS = {"delta", "ci_low", "ci_high", "p_value", "beats"}


# ---------------------------------------------------------------- labels


def read_labels(path: Path, clip_ids: list[str]) -> dict:
    """``{"source", "sha256", "episodes": {clip_id: (failure, onset_s | None)}}``. Every clip needs a label."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise ValueError(f"{path}: labels must be an object with an 'episodes' list")
    known = set(clip_ids)
    episodes: dict[str, tuple[bool, float | None]] = {}
    for index, row in enumerate(payload["episodes"]):
        where = f"{path}: episodes[{index}]"
        if not isinstance(row, dict) or not isinstance(row.get("clip_id"), str):
            raise ValueError(f"{where} needs a string clip_id")
        clip_id = row["clip_id"]
        if clip_id not in known:
            raise ValueError(f"{where}: clip {clip_id!r} is not in this run")
        if clip_id in episodes:
            raise ValueError(f"{where}: duplicate clip {clip_id!r}")
        failure = row.get("failure")
        if not isinstance(failure, bool):
            raise ValueError(f"{where}: failure must be true or false")
        onset = row.get("onset_s")
        if failure:
            if isinstance(onset, bool) or not isinstance(onset, (int, float)) or not math.isfinite(onset) or onset < 0:
                raise ValueError(f"{where}: a failed episode needs a non-negative onset_s")
            onset = float(onset)
        elif onset is not None:
            raise ValueError(f"{where}: a successful episode has onset_s null")
        episodes[clip_id] = (failure, onset)
    missing = sorted(known - set(episodes))
    if missing:
        raise ValueError(f"{path}: no label for {len(missing)} clip(s), e.g. {missing[:3]}")
    source = payload.get("source")
    return {
        "source": source if isinstance(source, str) else None,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "episodes": episodes,
    }


def default_labels_path(manifest: dict) -> Path | None:
    """``<clips_dir>/failures.json`` when the manifest records a clip directory that has one."""
    clips_dir = manifest.get("clips_dir")
    if not isinstance(clips_dir, str):
        return None
    from interpretableintelligence.adapters.fake import FAILURE_LABELS_FILE  # noqa: PLC0415

    candidate = Path(clips_dir) / FAILURE_LABELS_FILE
    return candidate if candidate.is_file() else None


# ---------------------------------------------------------------- pooling


def bin_end_s(window: np.ndarray, t: np.ndarray, grid: dict, fps: np.ndarray) -> np.ndarray:
    """End time in seconds of each ``(window, t)`` bin (windows tile the clip back to back)."""
    frames = (np.asarray(window, dtype=np.float64) * int(grid["t"]) + np.asarray(t, dtype=np.float64) + 1.0) * int(grid["tubelet"])
    return frames / np.asarray(fps, dtype=np.float64)


def _bin_keys(clip_idx: np.ndarray, window: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (clip_idx.astype(np.int64) << 32) | (window.astype(np.int64) << 16) | t.astype(np.int64)


def _shard_paths(acts_dir: Path) -> list[Path]:
    paths = sorted(acts_dir.glob("shard-*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no activation shards in {acts_dir}")
    return paths


def train_mean(acts_dir: Path, train_clip_idx: np.ndarray, d_model: int) -> np.ndarray:
    """float32 mean of the train rows (the random-weights SAE's ``b_dec``)."""
    from safetensors.numpy import load_file  # noqa: PLC0415

    total = np.zeros(d_model, dtype=np.float64)
    count = 0
    for path in _shard_paths(acts_dir):
        shard = load_file(str(path))
        keep = np.isin(shard["clip_idx"], train_clip_idx)
        if keep.any():
            total += shard["acts"][keep].astype(np.float64).sum(axis=0)
            count += int(keep.sum())
    if count < 2:
        raise RuntimeError("need at least 2 train activation rows")
    return (total / count).astype(np.float32)


def pooled_bins(acts_dir: Path, encoders: dict, device) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Max over ``(h, w)`` of each input, per ``(clip, window, t)`` bin.

    ``encoders`` maps an input name to an SAE (or ``None`` for raw activations).
    Returns sorted bin keys and ``{input: float32 [n_bins, dims]}``.
    """
    import torch  # noqa: PLC0415
    from safetensors.numpy import load_file  # noqa: PLC0415

    key_parts: list[np.ndarray] = []
    value_parts: dict[str, list[np.ndarray]] = {name: [] for name in encoders}
    for path in _shard_paths(acts_dir):
        shard = load_file(str(path))
        keys = _bin_keys(shard["clip_idx"], shard["window"], shard["t"])
        if keys.size == 0:
            continue
        # Rows are raster ordered, so each bin is one contiguous run inside a shard.
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        key_parts.append(keys[starts])
        # Encode in groups of whole bins of about ROWS_PER_CHUNK rows, so a wide SAE never
        # materialises a shard's full latent matrix.
        firsts = np.unique(np.searchsorted(starts, np.arange(0, keys.size, ROWS_PER_CHUNK), side="right") - 1)
        bounds = np.r_[starts[firsts], keys.size]
        for group, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            stop = firsts[group + 1] if group + 1 < firsts.size else starts.size
            local = starts[firsts[group] : stop] - lo
            x = torch.from_numpy(shard["acts"][lo:hi].astype(np.float32)).to(device)
            for name, sae in encoders.items():
                if sae is None:
                    values = x
                else:
                    with torch.no_grad():
                        values = sae.encode(x)
                dense = values.detach().cpu().numpy().astype(np.float32, copy=False)
                value_parts[name].append(np.maximum.reduceat(dense, local, axis=0))
    if not key_parts:
        raise RuntimeError(f"activation shards in {acts_dir} hold no rows")
    keys = np.concatenate(key_parts)
    unique, inverse = np.unique(keys, return_inverse=True)
    pooled: dict[str, np.ndarray] = {}
    for name, parts in value_parts.items():
        stacked = np.concatenate(parts, axis=0)
        out = np.full((unique.shape[0], stacked.shape[1]), -np.inf, dtype=np.float32)
        np.maximum.at(out, inverse, stacked)  # a bin split across two shards
        pooled[name] = out
    return unique, pooled


# ---------------------------------------------------------------- probe


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, iters: int = 100) -> tuple[np.ndarray, float]:
    """L2 logistic regression by Newton's method; the bias is not penalised. Deterministic."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    design = np.hstack([x, np.ones((n, 1))])
    penalty = np.full(d + 1, float(l2))
    penalty[-1] = 0.0
    beta = np.zeros(d + 1)
    for _ in range(iters):
        z = np.clip(design @ beta, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = design.T @ (p - y) + penalty * beta
        weights = p * (1.0 - p)
        hessian = (design * weights[:, None]).T @ design + np.diag(penalty) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hessian, grad)
        beta = beta - step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta[:-1], float(beta[-1])


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a failure > score of a success), ties counted half."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.sort(np.asarray(neg, dtype=np.float64))
    if pos.size == 0 or neg.size == 0:
        raise ValueError("auroc needs at least one episode of each class")
    below = np.searchsorted(neg, pos, side="left")
    at_or_below = np.searchsorted(neg, pos, side="right")
    return float((below.sum() + 0.5 * (at_or_below - below).sum()) / (pos.size * neg.size))


def _stable(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little")


# ---------------------------------------------------------------- run


def execute_probe(
    run_dir: Path,
    *,
    labels_path: Path | None = None,
    lead_s: float = PROBE_MIN_LEAD_S,
    seed: int = 0,
    bootstrap: int = PROBE_BOOTSTRAP,
    l2: float = PROBE_L2,
    random_seed: int | None = None,
    log=print,
) -> dict:
    """Fit the three probes, write ``probe/``, and return ``probe.json``."""
    import torch  # noqa: PLC0415

    from interpretableintelligence.device import pick_device  # noqa: PLC0415
    from interpretableintelligence.sae import load_sae, random_weights_sae  # noqa: PLC0415

    run_dir = Path(run_dir)
    if not math.isfinite(lead_s) or lead_s < 0:
        raise ValueError(f"--lead-s must be a non-negative number of seconds, got {lead_s}")
    if bootstrap < 1:
        raise ValueError("--bootstrap must be at least 1")
    if not math.isfinite(l2) or l2 <= 0:
        raise ValueError("--l2 must be positive")
    manifest = read_json(run_dir / "manifest.json")
    split = read_json(run_dir / "split.json")
    if not (run_dir / "sae" / "weights.safetensors").is_file():
        raise FileNotFoundError(f"{run_dir} has no trained SAE (sae/weights.safetensors); run `ii run` first")
    scoreboard_path = run_dir / "scoreboard.json"
    scoreboard = read_json(scoreboard_path) if scoreboard_path.is_file() else None

    clips = manifest["clips"]
    clip_ids = [str(clip["clip_id"]) for clip in clips]
    membership = {clip_id: "train" for clip_id in split["train"]}
    membership.update({clip_id: "heldout" for clip_id in split["heldout"]})
    if set(membership) != set(clip_ids):
        raise ValueError("split.json does not cover exactly the manifest's clips")

    if labels_path is None:
        labels_path = default_labels_path(manifest)
        if labels_path is None:
            raise FileNotFoundError("pass --labels FILE (no failures.json next to the run's clips)")
    labels = read_labels(labels_path, clip_ids)

    device = pick_device()
    trained, config = load_sae(run_dir / "sae", device)
    control_seed = int(config.seed if random_seed is None else random_seed)
    train_idx = np.asarray([i for i, cid in enumerate(clip_ids) if membership[cid] == "train"], dtype=np.int32)
    mean = train_mean(run_dir / "acts", train_idx, int(config.d_model))
    control = random_weights_sae(config, torch.from_numpy(mean), control_seed, device)
    log(
        f"probe: trained SAE {config.n_features} features ({config.activation}"
        f"{'' if config.k is None else f', k={config.k}'}), random-weights SAE seed {control_seed}, "
        f"raw d_model {config.d_model}"
    )
    keys, pooled = pooled_bins(run_dir / "acts", {"trained_sae": trained, "random_sae": control, "raw": None}, device)

    grid = manifest["grid"]
    bin_clip = (keys >> 32).astype(np.int64)
    bin_window = ((keys >> 16) & 0xFFFF).astype(np.int64)
    bin_t = (keys & 0xFFFF).astype(np.int64)
    fps = np.asarray([float(clips[i]["fps"]) for i in bin_clip])
    ends = bin_end_s(bin_window, bin_t, grid, fps)

    columns: dict[str, list] = {name: [] for name in SCORE_COLUMNS}
    episodes = []
    for index, clip_id in enumerate(clip_ids):
        failure, onset = labels["episodes"][clip_id]
        duration = float(clips[index]["n_frames"]) / float(clips[index]["fps"])
        ref = float(onset) if failure else duration
        cutoff = ref - float(lead_s)
        in_clip = np.flatnonzero(bin_clip == index)
        seen = in_clip[ends[in_clip] <= cutoff + TIME_EPS]
        if seen.size == 0:
            continue  # nothing observed that far ahead; the episode cannot be scored
        episodes.append(
            {
                "clip_id": clip_id,
                "split": membership[clip_id],
                "failure": failure,
                "ref_s": ref,
                "cutoff_s": cutoff,
                "lead_s": ref - float(ends[seen].max()),
                "rows": seen,
            }
        )
    y_all = np.asarray([ep["failure"] for ep in episodes], dtype=bool)
    is_train = np.asarray([ep["split"] == "train" for ep in episodes], dtype=bool)
    fit_ok = bool(is_train.any()) and len(set(y_all[is_train].tolist())) == 2
    dims = {name: int(pooled[name].shape[1]) for name in INPUTS}
    if fit_ok:
        for name in INPUTS:
            features = np.stack([pooled[name][ep["rows"]].max(axis=0) for ep in episodes]).astype(np.float64)
            mu = features[is_train].mean(axis=0)
            sd = features[is_train].std(axis=0)
            sd[sd < 1e-8] = 1.0
            z = (features - mu) / sd
            weights, bias = fit_logistic(z[is_train], y_all[is_train].astype(np.float64), l2)
            logits = z @ weights + bias
            for ep, logit in zip(episodes, logits, strict=True):
                columns["input"].append(name)
                columns["clip_id"].append(ep["clip_id"])
                columns["split"].append(ep["split"])
                columns["failure"].append(bool(ep["failure"]))
                columns["ref_s"].append(float(ep["ref_s"]))
                columns["cutoff_s"].append(float(ep["cutoff_s"]))
                columns["lead_s"].append(float(ep["lead_s"]))
                columns["score"].append(float(logit))
    else:
        log("probe: the train split needs both failed and successful episodes to fit; the gate is not run")

    table = scores_table(columns)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "model_sha": manifest["model"]["sha"],
        "layer": manifest["layer"],
        "split_sha": manifest["split"]["sha"],
        "adapter": manifest["model"]["adapter"],
        "target": "failure",
        "labels": {
            "source": labels["source"],
            "sha256": labels["sha256"],
            "n_failures": int(sum(1 for f, _ in labels["episodes"].values() if f)),
            "n_episodes": len(labels["episodes"]),
            "n_unscored": len(clip_ids) - len(episodes),
        },
        "pool": POOL,
        "probe": {"kind": PROBE_KIND, "l2": float(l2), "standardize": "train mean and std"},
        "lead_s": float(lead_s),
        "min_lead_s": PROBE_MIN_LEAD_S,
        "bootstrap": int(bootstrap),
        "confidence": PROBE_CONFIDENCE,
        "alpha": PROBE_ALPHA,
        "seed": int(seed),
        "dims": dims,
        "trained": _trained_sae_fields(scoreboard),
        "random": {
            "seed": control_seed,
            "init": RANDOM_INIT,
            "weights_sha256": sae_weights_sha(control),
        },
    }
    result = summarize(table, meta)
    out = run_dir / PROBE_DIR
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / SCORES_FILE)
    write_json(out / RESULT_FILE, result)
    return result


def scores_table(columns: dict[str, list]) -> pa.Table:
    return pa.table(
        {
            "input": pa.array(columns["input"], type=pa.string()),
            "clip_id": pa.array(columns["clip_id"], type=pa.string()),
            "split": pa.array(columns["split"], type=pa.string()),
            "failure": pa.array(columns["failure"], type=pa.bool_()),
            "ref_s": pa.array(columns["ref_s"], type=pa.float64()),
            "cutoff_s": pa.array(columns["cutoff_s"], type=pa.float64()),
            "lead_s": pa.array(columns["lead_s"], type=pa.float64()),
            "score": pa.array(columns["score"], type=pa.float64()),
        }
    )


def sae_weights_sha(sae) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(sae.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32)).tobytes())
    return digest.hexdigest()


def _trained_sae_fields(scoreboard: dict | None) -> dict:
    """Copied from the scoreboard when it has them; null otherwise (never a 0.0 stand-in)."""
    sae = scoreboard.get("sae") if isinstance(scoreboard, dict) else None
    if not isinstance(sae, dict):
        return {"sae_usable": None, "loss_recovered": None}
    usable = sae.get("usable")
    recovered = sae.get("loss_recovered")
    return {
        "sae_usable": usable if isinstance(usable, bool) else None,
        "loss_recovered": float(recovered) if isinstance(recovered, (int, float)) and not isinstance(recovered, bool) else None,
    }


# ---------------------------------------------------------------- summary


def summarize(table: pa.Table, meta: dict) -> dict:
    """``probe.json`` from ``scores.parquet`` and the settings. ``validate`` calls this too."""
    data = table.to_pydict()
    held: dict[str, dict] = {}
    for name in INPUTS:
        rows = [i for i, (inp, spl) in enumerate(zip(data["input"], data["split"], strict=True)) if inp == name and spl == "heldout"]
        rows.sort(key=lambda i: data["clip_id"][i])
        held[name] = {
            "clip_id": [data["clip_id"][i] for i in rows],
            "failure": np.asarray([data["failure"][i] for i in rows], dtype=bool),
            "score": np.asarray([data["score"][i] for i in rows], dtype=np.float64),
            "lead": np.asarray([data["lead_s"][i] for i in rows], dtype=np.float64),
        }
    reference = held["trained_sae"]
    aligned = all(held[name]["clip_id"] == reference["clip_id"] for name in INPUTS)
    y = reference["failure"]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    reasons = []
    if not data["input"]:
        reasons.append("the train split lacks failed or successful episodes, so no probe was fit")
    elif not aligned:
        reasons.append("held-out episodes differ between inputs")
    if n_pos < PROBE_MIN_CLASS or n_neg < PROBE_MIN_CLASS:
        reasons.append(f"held-out split has {n_pos} failed and {n_neg} successful scored episodes; the gate needs {PROBE_MIN_CLASS} of each")
    lead_time = float(reference["lead"][y].min()) if n_pos and aligned else None
    if float(meta["lead_s"]) + TIME_EPS < PROBE_MIN_LEAD_S:
        reasons.append(f"lead {meta['lead_s']:g} s is below the {PROBE_MIN_LEAD_S:g} s the gate requires")
    elif lead_time is not None and lead_time + TIME_EPS < PROBE_MIN_LEAD_S:
        reasons.append(f"held-out lead time {lead_time:g} s is below {PROBE_MIN_LEAD_S:g} s")

    evaluable = aligned and n_pos >= 1 and n_neg >= 1
    point: dict[str, float | None] = {name: None for name in INPUTS}
    boot: dict[str, np.ndarray] = {}
    if evaluable:
        for name in INPUTS:
            point[name] = auroc(held[name]["score"][y], held[name]["score"][~y])
        rng = np.random.default_rng(np.random.SeedSequence([int(meta["seed"]), _stable("probe-bootstrap") % (2**32)]))
        pos_idx = np.flatnonzero(y)
        neg_idx = np.flatnonzero(~y)
        b = int(meta["bootstrap"])
        boot = {name: np.empty(b, dtype=np.float64) for name in INPUTS}
        for i in range(b):
            pos = pos_idx[rng.integers(0, pos_idx.size, pos_idx.size)]
            neg = neg_idx[rng.integers(0, neg_idx.size, neg_idx.size)]
            for name in INPUTS:
                scores = held[name]["score"]
                boot[name][i] = auroc(scores[pos], scores[neg])

    tail = (1.0 - float(meta["confidence"])) / 2.0

    def interval(samples: np.ndarray | None) -> tuple[float | None, float | None]:
        if samples is None:
            return None, None
        return float(np.quantile(samples, tail)), float(np.quantile(samples, 1.0 - tail))

    inputs: dict[str, dict] = {}
    for name in INPUTS:
        low, high = interval(boot.get(name))
        block = {
            "n_dims": int(meta["dims"][name]),
            "auroc": point[name],
            "ci_low": low,
            "ci_high": high,
            "lead_time_s": lead_time,
        }
        if name == "trained_sae":
            block.update(meta["trained"])
        elif name == "random_sae":
            block.update(meta["random"])
            # The control is never spliced, scored or ablated: no usable flag, no
            # loss_recovered, no clip effects, and it can never be real.
            block.update({"sae_usable": None, "loss_recovered": None, "clip_effects": None, "real": False})
        inputs[name] = block

    comparisons: dict[str, dict] = {}
    for name in BASELINES:
        if evaluable:
            deltas = boot["trained_sae"] - boot[name]
            delta = float(point["trained_sae"] - point[name])
            p_value = float((1 + np.count_nonzero(deltas <= 0.0)) / (deltas.size + 1))
            low, high = interval(deltas)
            beats = bool(delta > 0.0 and p_value < float(meta["alpha"]))
        else:
            delta = p_value = low = high = None
            beats = False
        comparisons[name] = {"delta": delta, "ci_low": low, "ci_high": high, "p_value": p_value, "beats": beats}

    if reasons:
        gate = "not_run"
        reason = "; ".join(reasons)
    elif all(comparisons[name]["beats"] for name in BASELINES):
        gate = "pass"
        reason = (
            f"trained SAE beats the random-weights SAE and raw activations on held-out AUROC "
            f"(p < {meta['alpha']:g}) at a {lead_time:g} s lead"
        )
    else:
        gate = "fail"
        lost = [name for name in BASELINES if not comparisons[name]["beats"]]
        reason = "trained SAE does not beat " + " or ".join(lost) + f" on held-out AUROC at p < {meta['alpha']:g}"

    return {
        "schema_version": meta["schema_version"],
        "run_id": meta["run_id"],
        "model_sha": meta["model_sha"],
        "layer": meta["layer"],
        "split_sha": meta["split_sha"],
        "adapter": meta["adapter"],
        "target": meta["target"],
        "labels": meta["labels"],
        "pool": meta["pool"],
        "probe": meta["probe"],
        "lead_s": meta["lead_s"],
        "min_lead_s": meta["min_lead_s"],
        "bootstrap": meta["bootstrap"],
        "confidence": meta["confidence"],
        "alpha": meta["alpha"],
        "seed": meta["seed"],
        "n_episodes": {"heldout": len(reference["clip_id"]), "heldout_failures": n_pos, "heldout_successes": n_neg},
        "inputs": inputs,
        "comparisons": comparisons,
        "sae_gate": gate,
        "positive": gate == "pass",
        "reason": reason,
        "real": False,
        "rollout_finding": False,
        "note": NOTE,
    }


def meta_from_result(result: dict) -> dict:
    """The settings ``summarize`` needs, read back from a stored ``probe.json``."""
    inputs = result["inputs"]
    return {
        **{key: result[key] for key in (
            "schema_version", "run_id", "model_sha", "layer", "split_sha", "adapter", "target", "labels",
            "pool", "probe", "lead_s", "min_lead_s", "bootstrap", "confidence", "alpha", "seed",
        )},
        "dims": {name: inputs[name]["n_dims"] for name in INPUTS},
        "trained": {key: inputs["trained_sae"][key] for key in TRAINED_EXTRA},
        "random": {key: inputs["random_sae"][key] for key in ("seed", "init", "weights_sha256")},
    }


# ---------------------------------------------------------------- validate


def check_probe(run_dir: Path, manifest: dict) -> list[str]:
    """Contract errors for ``probe/``. Empty when it is absent or valid."""
    out = Path(run_dir) / PROBE_DIR
    if not out.exists():
        return []
    errors: list[str] = []
    result_path = out / RESULT_FILE
    scores_path = out / SCORES_FILE
    if not result_path.is_file() or not scores_path.is_file():
        return [f"{PROBE_DIR}/ needs both {RESULT_FILE} and {SCORES_FILE}"]
    try:
        result = read_json(result_path)
    except ValueError as exc:
        return [f"{PROBE_DIR}/{RESULT_FILE} is not JSON: {exc}"]
    if not isinstance(result, dict) or set(result) != RESULT_KEYS:
        return [f"{PROBE_DIR}/{RESULT_FILE} keys must be exactly {sorted(RESULT_KEYS)}"]
    inputs = result.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(INPUTS):
        return [f"{PROBE_DIR}/{RESULT_FILE} inputs must be exactly {list(INPUTS)}"]
    expected_keys = {"trained_sae": INPUT_KEYS | TRAINED_EXTRA, "random_sae": INPUT_KEYS | RANDOM_EXTRA, "raw": INPUT_KEYS}
    for name, keys in expected_keys.items():
        if not isinstance(inputs[name], dict) or set(inputs[name]) != keys:
            errors.append(f"{PROBE_DIR}/{RESULT_FILE} inputs.{name} keys must be exactly {sorted(keys)}")
    if errors:
        return errors
    for key, expected in (
        ("run_id", manifest.get("run_id")),
        ("model_sha", (manifest.get("model") or {}).get("sha")),
        ("layer", manifest.get("layer")),
        ("split_sha", (manifest.get("split") or {}).get("sha")),
        ("adapter", (manifest.get("model") or {}).get("adapter")),
    ):
        if result.get(key) != expected:
            errors.append(f"{PROBE_DIR}/{RESULT_FILE} {key} {result.get(key)!r} != manifest {expected!r}")
    if result.get("real") is not False or inputs["random_sae"].get("real") is not False:
        errors.append(f"{PROBE_DIR}/{RESULT_FILE} must never set real")
    for key in ("sae_usable", "loss_recovered", "clip_effects"):
        if inputs["random_sae"].get(key) is not None:
            errors.append(f"{PROBE_DIR}/{RESULT_FILE} inputs.random_sae.{key} must be null")
    if result.get("sae_gate") not in GATE_VALUES:
        errors.append(f"{PROBE_DIR}/{RESULT_FILE} sae_gate must be one of {list(GATE_VALUES)}")
    if errors:
        return errors
    table = pq.read_table(scores_path)
    if tuple(table.column_names) != SCORE_COLUMNS:
        return [f"{PROBE_DIR}/{SCORES_FILE} columns must be {list(SCORE_COLUMNS)}, got {table.column_names}"]
    split_of = {str(clip["clip_id"]): clip.get("split") for clip in manifest.get("clips") or []}
    for clip_id, split in zip(table.column("clip_id").to_pylist(), table.column("split").to_pylist(), strict=True):
        if split_of.get(clip_id) != split:
            errors.append(f"{PROBE_DIR}/{SCORES_FILE} clip {clip_id!r} split {split!r} != manifest {split_of.get(clip_id)!r}")
            break
    if not set(table.column("input").to_pylist()) <= set(INPUTS):
        errors.append(f"{PROBE_DIR}/{SCORES_FILE} input must be one of {list(INPUTS)}")
    if errors:
        return errors
    try:
        rebuilt = summarize(table, meta_from_result(result))
    except (KeyError, TypeError, ValueError) as exc:
        return [f"{PROBE_DIR}/{RESULT_FILE} cannot be rebuilt from {SCORES_FILE}: {exc}"]
    from interpretableintelligence.validate import _same_json  # noqa: PLC0415 (validate imports this module lazily)

    _same_json(result, rebuilt, f"{PROBE_DIR}/{RESULT_FILE}", errors)
    return errors


# ---------------------------------------------------------------- report


def format_report(result: dict) -> str:
    lines = [
        f"probe: {result['run_id']}  labels {result['labels']['source'] or 'unnamed'}  "
        f"lead {result['lead_s']:g} s  held-out {result['n_episodes']['heldout_failures']} failed / "
        f"{result['n_episodes']['heldout_successes']} ok",
        f"  {'input':<12} {'dims':>5} {'AUROC':>7} {'95% CI':>17} {'lead s':>7}",
    ]
    for name in INPUTS:
        block = result["inputs"][name]
        ci = "-" if block["ci_low"] is None else f"[{block['ci_low']:.3f}, {block['ci_high']:.3f}]"
        lines.append(
            f"  {name:<12} {block['n_dims']:>5} {_fmt(block['auroc']):>7} {ci:>17} {_fmt(block['lead_time_s'], 2):>7}"
        )
    for name in BASELINES:
        comp = result["comparisons"][name]
        lines.append(
            f"  trained vs {name:<10} delta {_fmt(comp['delta'])}  p {_fmt(comp['p_value'])}  "
            f"{'beats' if comp['beats'] else 'does not beat'}"
        )
    lines.append(f"  sae_gate {result['sae_gate']}  positive {str(result['positive']).lower()}: {result['reason']}")
    lines.append(f"  {result['note']}")
    return "\n".join(lines)


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"
