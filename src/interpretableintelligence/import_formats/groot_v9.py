"""Import GR00T rollcap-v0 (v9) capture folders into the ``ii`` run contract."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.numpy import load_file, save_file

from interpretableintelligence import __version__
from interpretableintelligence.artifacts import (
    write_activation_shards,
    write_attribution,
    write_clip_effects,
    write_feature_acts,
    write_json,
    write_jsonl,
)
from interpretableintelligence.attribute import ACTIVATION, METHOD, rank_feature_ids
from interpretableintelligence.sae import SAEConfig, save_sae
from interpretableintelligence.thresholds import (
    DEAD_ACT_EPS,
    EFFECT_KIND_ABLATION,
    EFFECT_KIND_ESTIMATE,
)

DEFAULT_ABLATION_SUMMARY = (
    Path(__file__).resolve().parents[3] / "demo-data" / "groot" / "groot-1028-ablate-summary.json"
)


@dataclass(frozen=True)
class _Token:
    clip_index: int
    clip_id: str
    window: int
    t: int
    h: int
    w: int


def import_groot_v9(
    *,
    src: Path,
    out: Path,
    ablation_summary: Path | None = None,
) -> None:
    src = src.resolve()
    out = out.resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output {out}")
    out.mkdir(parents=True)

    rollcap_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if rollcap_manifest.get("format") != "rollcap-v0":
        raise ValueError(f"expected rollcap-v0 manifest in {src}, got format={rollcap_manifest.get('format')!r}")
    episodes = _read_episodes(src)
    if len(episodes) < 20:
        raise ValueError(f"need at least 20 episodes for the ii contract, got {len(episodes)}")

    split_src = src / "split.json"
    if not split_src.is_file():
        raise FileNotFoundError(split_src)
    split_bytes = split_src.read_bytes()
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    (out / "split.json").write_bytes(split_bytes)

    d_model = int(rollcap_manifest.get("d_model") or rollcap_manifest.get("hook", {}).get("d_model") or episodes[0].get("d_model", 0))
    if d_model <= 0:
        raise ValueError("could not determine d_model from rollcap manifest")
    layer = int(rollcap_manifest.get("layer", 0))
    tokens_per_step = int(rollcap_manifest.get("tokens_per_step") or episodes[0].get("tokens_per_step", 32))
    run_id = out.name
    heldout_eps = set(int(x) for x in json.loads(split_bytes)["heldout"])

    clip_rows, max_window = _build_clip_rows(episodes, heldout_eps, src)
    acts_path = out / ".import_acts.memmap"
    acts, tokens = _convert_activations_streaming(
        src, episodes, clip_rows, d_model, tokens_per_step, max_window, acts_path
    )
    model_sha = _model_sha(rollcap_manifest)
    manifest = {
        "run_id": run_id,
        "tool_version": __version__,
        "model": {
            "adapter": "groot",
            "name": str(rollcap_manifest.get("model", {}).get("name", "groot")),
            "sha": model_sha,
            "random_init": False,
        },
        "layer": layer,
        "d_model": d_model,
        "grid": {"t": tokens_per_step, "h": 1, "w": 1, "tubelet": 1, "patch_px": 1},
        "split": {"kind": "episode_index", "sha": split_sha},
        "clips": clip_rows,
        "import": {
            "format": "groot-v9",
            "src_run_id": str(rollcap_manifest.get("run_id", src.name)),
            "src_path": str(src),
            "max_window": max_window,
            "failure_fields": "episodes.jsonl fail_kind/fail_t (pot0-v1); not fail_kind_v9_active",
        },
        "rollcap": {"format": "rollcap-v0", "preserved_files": ["split.json"]},
    }
    write_json(out / "manifest.json", manifest)

    write_activation_shards(out / "acts", acts, tokens)
    (out / "features").mkdir(parents=True, exist_ok=True)
    _copy_sae(src, out, d_model)
    sae_model, n_features = _load_import_sae(out)

    means, per_clip = _encode_feature_acts_streaming(
        out / "features" / "feature_acts.parquet", acts, tokens, sae_model, n_features, clip_rows
    )
    attr_rows = _attribution_from_clip_sums(per_clip, clip_rows, n_features)
    write_attribution(out / "features" / "attribution.parquet", attr_rows, objective=ACTIVATION)

    ablation_path = ablation_summary or DEFAULT_ABLATION_SUMMARY
    prefilter_k = min(3, n_features)
    prefilter_ids = [int(x) for x in rank_feature_ids(means)[:prefilter_k]]

    ranked = _build_ranked(means, n_features)
    write_json(
        out / "features" / "candidates.json",
        {
            "method": METHOD,
            "objective": ACTIVATION,
            "shortlist_only": True,
            "prefilter_k": len(prefilter_ids),
            "rank_by": (
                "mean over clips of sum |activation| over tokens and windows; "
                "ties break toward the smaller feature id"
            ),
            "ranked": ranked,
            "candidates": prefilter_ids,
            "seconds": {"attribution": None, "ablation": None, "random_controls": None},
        },
    )

    effects = _clip_effects_prefilter(prefilter_ids, clip_rows, max_window, tokens_per_step)
    write_clip_effects(out / "features" / "clip_effects.parquet", effects)

    scoreboard = _scoreboard(
        manifest,
        split_sha,
        n_features,
        src,
        ablation_path,
    )
    jsonl_rows = _features_jsonl_from_scoreboard(scoreboard["features"])
    write_json(out / "scoreboard.json", scoreboard)
    write_jsonl(out / "features" / "features.jsonl", jsonl_rows)

    ep_copy = src / "episodes.jsonl"
    if ep_copy.is_file():
        shutil.copy2(ep_copy, out / "episodes.jsonl")
    if acts_path.is_file():
        acts_path.unlink()


def _read_episodes(src: Path) -> list[dict]:
    rows = []
    for line in (src / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["episode_idx"]))
    return rows


def _model_sha(rollcap_manifest: dict) -> str:
    name = json.dumps(rollcap_manifest.get("model", {}), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _clip_id(episode_idx: int) -> str:
    return f"ep-{episode_idx:05d}"


def _max_policy_window(src: Path, episodes: list[dict]) -> int:
    max_window = 0
    for ep in episodes:
        idx = int(ep["episode_idx"])
        data = load_file(str(src / "acts" / f"shard-{idx:05d}.safetensors"))
        steps = data["step"][data["tok"] >= 0]
        max_window = max(max_window, len(set(int(s) for s in steps.tolist())))
    return max_window


def _build_clip_rows(
    episodes: list[dict],
    heldout_eps: set[int],
    src: Path,
) -> tuple[list[dict], int]:
    clip_rows: list[dict] = []
    for ep in episodes:
        idx = int(ep["episode_idx"])
        split = "heldout" if idx in heldout_eps else "train"
        clip_sha = hashlib.sha256(json.dumps(ep, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        clip_rows.append(
            {
                "clip_id": _clip_id(idx),
                "sha256": clip_sha,
                "fps": 1.0,
                "n_frames": int(ep.get("n_steps", 1)),
                "split": split,
                "episode_idx": idx,
                "success": bool(ep.get("success")),
                "pot0_fail_kind": ep.get("fail_kind"),
                "pot0_fail_t": ep.get("fail_t"),
                "pot0_fail_subkind": ep.get("fail_subkind"),
                "pot0_fail_rule": ep.get("fail_rule"),
            }
        )
    max_window = _max_policy_window(src, episodes)
    return clip_rows, max_window


def _load_episode_shard(src: Path, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_file(str(src / "acts" / f"shard-{idx:05d}.safetensors"))
    steps = data["step"].astype(np.int64)
    toks = data["tok"].astype(np.int64)
    acts = data["acts"]
    order = np.lexsort((toks, steps))
    mask = toks[order] >= 0
    return steps[order][mask], toks[order][mask], acts[order][mask]


def _convert_activations_streaming(
    src: Path,
    episodes: list[dict],
    clip_rows: list[dict],
    d_model: int,
    tokens_per_step: int,
    max_window: int,
    acts_path: Path,
) -> tuple[np.ndarray, list[_Token]]:
    token_records: list[_Token] = []
    n_rows = len(episodes) * max_window * tokens_per_step
    acts = np.memmap(acts_path, dtype=np.float16, mode="w+", shape=(n_rows, d_model))
    offset = 0
    for ep_index, ep in enumerate(episodes):
        idx = int(ep["episode_idx"])
        steps, toks, ep_acts = _load_episode_shard(src, idx)
        window_of_step = {int(s): i for i, s in enumerate(sorted(set(steps.tolist())))}
        clip_id = clip_rows[ep_index]["clip_id"]
        inv_steps = sorted(window_of_step, key=lambda s: window_of_step[s])
        block = max_window * tokens_per_step
        for window in range(max_window):
            for tok in range(tokens_per_step):
                local = window * tokens_per_step + tok
                if window < len(inv_steps):
                    step = inv_steps[window]
                    pick = (steps == step) & (toks == tok)
                    if pick.any():
                        acts[offset + local] = ep_acts[pick][0].astype(np.float16)
                token_records.append(_Token(ep_index, clip_id, window, tok, 0, 0))
        offset += block
        del steps, toks, ep_acts
        gc.collect()
    acts.flush()
    return acts, token_records


def _copy_sae(src: Path, out: Path, d_model: int) -> None:
    sae_dir = out / "sae"
    sae_dir.mkdir(parents=True, exist_ok=True)
    raw_config = json.loads((src / "sae" / "config.json").read_text(encoding="utf-8"))
    n_features = int(raw_config.get("n_latents") or raw_config.get("d_sae") or raw_config.get("n_features"))
    config = SAEConfig(
        d_model=d_model,
        n_features=n_features,
        l1_coeff=float(raw_config.get("l1_coeff", 1e-3)),
        ortho_coeff=float(raw_config.get("ortho_coeff", 1e-3)),
        lr=float(raw_config.get("train", {}).get("lr", 4e-4) if isinstance(raw_config.get("train"), dict) else 4e-4),
        steps=int(raw_config.get("train", {}).get("steps", 884) if isinstance(raw_config.get("train"), dict) else 884),
        batch_size=int(raw_config.get("train", {}).get("batch", 4096) if isinstance(raw_config.get("train"), dict) else 4096),
        seed=int(raw_config.get("train", {}).get("seed", 0) if isinstance(raw_config.get("train"), dict) else 0),
        normalize_decoder=True,
        activation="relu",
    )
    weights = load_file(str(src / "sae" / "weights.safetensors"))
    w_enc = np.asarray(weights["W_enc"], dtype=np.float32)
    w_dec = np.asarray(weights["W_dec"], dtype=np.float32)
    if w_enc.shape == (d_model, n_features):
        w_enc = w_enc.T
    if w_dec.shape == (n_features, d_model):
        w_dec = w_dec.T
    model = torch.nn.Module()
    model.register_buffer("W_enc", torch.from_numpy(w_enc.copy()))
    model.register_buffer("b_enc", torch.from_numpy(np.asarray(weights["b_enc"], dtype=np.float32)))
    model.register_buffer("W_dec", torch.from_numpy(w_dec.copy()))
    model.register_buffer("b_dec", torch.from_numpy(np.asarray(weights["b_dec"], dtype=np.float32)))
    from interpretableintelligence.sae import SAE

    sae = SAE(d_model, n_features)
    sae.W_enc.data.copy_(model.W_enc)
    sae.b_enc.data.copy_(model.b_enc)
    sae.W_dec.data.copy_(model.W_dec)
    sae.b_dec.data.copy_(model.b_dec)
    save_sae(sae, config, sae_dir)


def _load_import_sae(out: Path) -> tuple[torch.nn.Module, int]:
    from interpretableintelligence.sae import load_sae

    model, config = load_sae(out / "sae", device=torch.device("cpu"))
    return model, int(config.n_features)


def _encode_feature_acts_streaming(
    path: Path,
    acts: np.ndarray,
    tokens: list[_Token],
    sae: torch.nn.Module,
    n_features: int,
    clip_rows: list[dict],
) -> np.ndarray:
    clip_ids = [row["clip_id"] for row in clip_rows]
    clip_index = {clip_id: index for index, clip_id in enumerate(clip_ids)}
    per_clip = np.zeros((n_features, len(clip_ids)), dtype=np.float64)
    schema = pa.schema(
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
    batch = 128
    with pq.ParquetWriter(path, schema) as writer:
        with torch.no_grad():
            for start in range(0, acts.shape[0], batch):
                stop = min(start + batch, acts.shape[0])
                x = torch.from_numpy(acts[start:stop].astype(np.float32))
                feats = sae.encode(x).cpu().numpy()
                block = {key: [] for key in ("feature_id", "clip_id", "window", "t", "h", "w", "value")}
                for local, feat_row in enumerate(feats):
                    token = tokens[start + local]
                    ci = clip_index[token.clip_id]
                    active = np.flatnonzero(feat_row > DEAD_ACT_EPS)
                    for fid in active:
                        value = float(feat_row[int(fid)])
                        per_clip[int(fid), ci] += value
                        block["feature_id"].append(int(fid))
                        block["clip_id"].append(token.clip_id)
                        block["window"].append(int(token.window))
                        block["t"].append(int(token.t))
                        block["h"].append(int(token.h))
                        block["w"].append(int(token.w))
                        block["value"].append(value)
                if block["feature_id"]:
                    writer.write_table(pa.table(block, schema=schema))
    means = per_clip.mean(axis=1)
    return means, per_clip


def _attribution_from_clip_sums(
    per_clip: np.ndarray,
    clip_rows: list[dict],
    n_features: int,
) -> dict[str, list]:
    clip_ids = [row["clip_id"] for row in clip_rows]
    attr = {key: [] for key in ("feature_id", "clip_id", "score")}
    for fid in range(n_features):
        for ci, clip_id in enumerate(clip_ids):
            attr["feature_id"].append(fid)
            attr["clip_id"].append(clip_id)
            attr["score"].append(float(per_clip[fid, ci]))
    return attr


def _build_ranked(means: np.ndarray, n_features: int) -> list[dict]:
    order = rank_feature_ids(means)
    ranked = []
    for rank, fid in enumerate(order, start=1):
        score = float(np.float32(means[fid]))
        ranked.append(
            {
                "rank": rank,
                "feature_id": int(fid),
                "label": "",
                "score": score,
                "abs_score": abs(score),
            }
        )
    if len(ranked) != n_features:
        raise RuntimeError("ranked length mismatch")
    return ranked


def _clip_effects_prefilter(
    feature_ids: list[int],
    clip_rows: list[dict],
    max_window: int,
    tokens_per_step: int,
) -> dict[str, list]:
    rows = {key: [] for key in ("feature_id", "clip_id", "window", "t", "effect", "effect_kind")}
    for fid in feature_ids:
        for clip in clip_rows:
            for window in range(max_window):
                for t in range(tokens_per_step):
                    for kind in (EFFECT_KIND_ESTIMATE, EFFECT_KIND_ABLATION):
                        rows["feature_id"].append(int(fid))
                        rows["clip_id"].append(clip["clip_id"])
                        rows["window"].append(int(window))
                        rows["t"].append(int(t))
                        rows["effect"].append(0.0 if kind == EFFECT_KIND_ESTIMATE else 0.0)
                        rows["effect_kind"].append(kind)
    return rows


def _scoreboard(
    manifest: dict,
    split_sha: str,
    n_features: int,
    src: Path,
    ablation_summary: Path,
) -> dict:
    heldout_eval = {}
    path = src / "sae" / "heldout_eval.json"
    if path.is_file():
        heldout_eval = json.loads(path.read_text(encoding="utf-8"))
    ev = float(heldout_eval.get("heldout", {}).get("ev_normalised", 0.0))
    dead_latents = float(heldout_eval.get("heldout", {}).get("dead_latents", 0))
    n_latents = float(heldout_eval.get("rollout_io_load", {}).get("sae_M", n_features))
    dead_frac = dead_latents / n_latents if n_latents else 1.0
    n_train = sum(1 for clip in manifest["clips"] if clip["split"] == "train")
    n_held = sum(1 for clip in manifest["clips"] if clip["split"] == "heldout")
    features = []
    for fid in range(n_features):
        features.append(
            {
                "id": fid,
                "label": "",
                "distinct_clips_top20": 0,
                "label_holds_heldout": None,
                "ablation_effect": None,
                "random_percentile": None,
                "n_random": 0,
                "absent_in_random_init": None,
                "real": False,
            }
        )
    groot_ablation = None
    if ablation_summary.is_file():
        groot_ablation = json.loads(ablation_summary.read_text(encoding="utf-8"))
    scoreboard = {
        "run": {
            "model_sha": manifest["model"]["sha"],
            "layer": int(manifest["layer"]),
            "split_sha": split_sha,
            "seed": 0,
            "n_train_clips": n_train,
            "n_heldout_clips": n_held,
        },
        "sae": {
            "ev_heldout": ev,
            "loss_recovered": 0.0,
            "loss_sae": 1.0,
            "loss_visible_mean": 1.0,
            "dead_frac": float(dead_frac),
            "usable": False,
        },
        "features": features,
    }
    if groot_ablation is not None:
        scoreboard["groot_ablation"] = groot_ablation
    return scoreboard


def _features_jsonl_from_scoreboard(scoreboard_features: list[dict]) -> list[dict]:
    rows = []
    for feature in scoreboard_features:
        rows.append(
            {
                **feature,
                "label_model": "groot-import-v1",
                "top": [],
                "distinct_clips_top20": int(feature.get("distinct_clips_top20", 0)),
            }
        )
    return rows
