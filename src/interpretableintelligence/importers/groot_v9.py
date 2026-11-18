"""``ii import --format groot-v9``: a GR00T v9 rollout capture (rollcap-v0 layout).

The layout is the one ``adapters/groot/v9`` writes and the ``data-groot-1028-v1``
release ships: ``manifest.json`` (``format: rollcap-v0``), ``episodes.jsonl``,
``split.json`` (episode indices), ``acts/shard-NNNNN.safetensors`` (one per
episode: ``acts`` f16 ``[N, d]``, ``episode_idx``, ``step``, ``tok``), and a
BatchTopK SAE in ``sae/``.

Mapping onto ``ii run groot`` (``adapters/groot/adapter.py``):

* adapter ``groot:<checkpoint>``, the same DiT block (``layer``) on the final denoise step;
* one clip per episode, one window per policy call (``window = step // action_queue_steps``);
* grid ``t=1, h=1, w=1+action_horizon``, ``tubelet = action_queue_steps``. ``w`` is the DiT
  token index (0 = state token, k = action step k-1). The capture kept a fixed subset of
  tokens (``token_indices``); those are ``manifest.import.cells.w``.

The SAE's input normalisation ``(x - norm_mu) / norm_scale`` is folded into the ii
weights, so ``f = jumprelu(W_enc (x - b_dec) + b_enc)`` with the source's learned
inference threshold reproduces the source's active set exactly, and every value is
``norm_scale`` times the source's (raw activation units). Feature ids are unchanged.

Failure labels are the pot-0 detector's (``fail_rule`` ``pot0-v1``, ``tools/derive_pot0_fail.py``).
The v9 active-object labels (``*_v9_active``) are never read.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from interpretableintelligence.importers.common import ImportedCapture, ImportedClip
from interpretableintelligence.sae import SAE, SAEConfig

FORMAT = "groot-v9"
LAYOUT = "rollcap-v0"
POT0_RULE = "pot0-v1"
# LIBERO steps the simulator at its control frequency, 20 Hz; the capture does not record it.
LIBERO_CONTROL_HZ = 20.0
_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Outcome fields kept per clip. All come from the pot-0 relabel or the episode record.
LABEL_FIELDS = ("seed", "init_state_id", "success", "fail_kind", "fail_t", "fail_subkind", "fail_rule")

LOSS_REASON = (
    "needs the GR00T action head (DiT blocks after the hooked one and the action decoder, conditioned on the "
    "VLM backbone features and robot state of each policy call) run with the SAE reconstruction spliced in; "
    "the capture stores the hooked block's output only, not those inputs"
)
EFFECTS_REASON = (
    "estimate and ablation rows need the GR00T action head in the loop (a backward pass and an edited forward "
    "pass per policy call); the capture stores the hooked block's output only, so no effect row is written"
)
USABLE_REASON = "usable is decided by loss_recovered and loss_sae, which are null here (see their reasons)"


def read_groot_v9(
    src: Path,
    *,
    model_sha: str | None = None,
    fps: float = LIBERO_CONTROL_HZ,
    log: Callable[[str], None] = print,
) -> ImportedCapture:
    src = Path(src)
    manifest = _read_json(src / "manifest.json")
    if manifest.get("format") != LAYOUT:
        raise ValueError(f"{src}: manifest format is {manifest.get('format')!r}, expected {LAYOUT!r}")
    episodes = _read_episodes(src / "episodes.jsonl")
    _check_pot0(manifest, episodes)

    split_bytes = (src / "split.json").read_bytes()
    split = json.loads(split_bytes)
    membership = _membership(split, [int(ep["episode_idx"]) for ep in episodes])

    layer = int(manifest["layer"])
    d_model = int(manifest["d_model"])
    chunk = int(manifest["action_queue_steps"])
    horizon = int(manifest["action_horizon"])
    n_tok = int(manifest.get("n_tok", 1 + horizon))
    if n_tok != 1 + horizon:
        raise ValueError(f"n_tok {n_tok} != 1 + action_horizon {horizon}")
    token_indices = [int(tok) for tok in manifest["token_indices"]]
    if token_indices != sorted(set(token_indices)) or token_indices[0] < 0 or token_indices[-1] >= n_tok:
        raise ValueError("token_indices must be sorted, unique, and inside the DiT sequence")
    hook = manifest.get("hook") or {}
    if hook and int(hook.get("block_index", layer)) != layer:
        raise ValueError("manifest layer and hook.block_index disagree")

    checkpoint = str((manifest.get("model") or {}).get("name") or "")
    if not checkpoint:
        raise ValueError("manifest.model.name (the GR00T checkpoint) is missing")
    from interpretableintelligence.adapters.groot.adapter import make_spec  # noqa: PLC0415

    adapter = make_spec(checkpoint)
    sha, sha_source = _model_sha(adapter, checkpoint, manifest, model_sha, log)

    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("fps must be positive")
    clips: list[ImportedClip] = []
    shard_paths: list[Path] = []
    for ep in episodes:
        idx = int(ep["episode_idx"])
        path = src / "acts" / f"shard-{idx:05d}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(path)
        clip_id = str(ep.get("episode_id") or f"ep-{idx:05d}")
        if _EPISODE_ID.match(clip_id) is None:
            raise ValueError(f"episode_id {clip_id!r} is not a usable clip id")
        n_windows = int(ep["n_chunks"])
        n_steps = int(ep["n_steps"])
        if n_windows < 1 or n_steps < 1:
            raise ValueError(f"{clip_id} has no policy calls")
        clips.append(
            ImportedClip(
                clip_id=clip_id,
                source_id=idx,
                sha256=_file_sha256(path),
                fps=float(fps),
                n_frames=n_steps,
                split=membership[idx],
                n_windows=n_windows,
                labels={key: ep.get(key) for key in LABEL_FIELDS},
            )
        )
        shard_paths.append(path)

    cells = {"t": [0], "h": [0], "w": token_indices}
    cell_array = np.asarray([(0, 0, tok) for tok in token_indices], dtype=np.int64)

    def windows(clip_index: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        clip = clips[clip_index]
        yield from _episode_windows(shard_paths[clip_index], int(clip.source_id), clip.n_windows, chunk, token_indices, d_model, cell_array)

    sae, sae_config, sae_notes = _fold_sae(src / "sae", d_model)
    heldout_eval = _read_json(src / "sae" / "heldout_eval.json") if (src / "sae" / "heldout_eval.json").is_file() else {}
    loss_reason = str(heldout_eval.get("loss_recovered_note") or LOSS_REASON)
    if heldout_eval.get("loss_recovered") is not None:
        log("note: the source records a loss_recovered, but ii cannot recompute it without the model; writing null")
    nulls = {
        "scoreboard.sae.loss_recovered": f"source sae/heldout_eval.json: {loss_reason}" if heldout_eval else LOSS_REASON,
        "scoreboard.sae.loss_sae": LOSS_REASON,
        "scoreboard.sae.loss_visible_mean": LOSS_REASON,
        "scoreboard.sae.usable": USABLE_REASON,
        "features/clip_effects.parquet": EFFECTS_REASON,
    }

    relabel = manifest.get("fail_relabel") or {}
    files = ["manifest.json", "episodes.jsonl", "split.json", "sae/config.json", "sae/weights.safetensors"]
    source = {
        "layout": LAYOUT,
        "run_id": manifest.get("run_id"),
        "source_run_id": manifest.get("source_run_id"),
        "task_id": manifest.get("task_id"),
        "task_name": manifest.get("task_name"),
        "provenance": manifest.get("provenance"),
        "files": {name: _file_sha256(src / name) for name in files if (src / name).is_file()},
    }
    notes = {
        "hook": hook.get("module") or manifest.get("hook_target"),
        "denoise_step_index": hook.get("denoise_step_index"),
        "num_inference_timesteps": hook.get("num_inference_timesteps"),
        "token_layout": manifest.get("token_layout"),
        "token_subsample": manifest.get("token_subsample"),
        "window": f"one policy call; window k is the action chunk predicted at env step k * {chunk}",
        "clip_sha256": "sha256 of the episode's source acts/shard-NNNNN.safetensors (the capture has no clip pixels)",
        "fps": f"{float(fps):g} env steps per second (LIBERO control frequency; not recorded in the capture)",
        "n_frames": "env steps executed in the episode",
        "mean_pool_rows": "acts_meanpool/ (one mean over all DiT tokens per policy call) is not imported",
        "split_rule": split.get("rule") if isinstance(split, dict) else None,
        "failure_labels": {
            "detector": POT0_RULE,
            "script": relabel.get("script"),
            "fields": list(LABEL_FIELDS),
            "units": relabel.get("units") or "env steps",
            "not_used": ["fail_kind_v9_active", "fail_t_v9_active"],
        },
        "model_sha_source": sha_source,
        "sae": sae_notes,
    }
    model = {"adapter": adapter, "name": adapter, "sha": sha, "random_init": False}
    log(
        f"{FORMAT}: {len(clips)} episodes ({sum(c.split == 'heldout' for c in clips)} held out), "
        f"{sum(c.n_windows for c in clips)} policy calls, {len(token_indices)} of {n_tok} tokens, "
        f"SAE {sae_config.n_features} features"
    )
    return ImportedCapture(
        format=FORMAT,
        model=model,
        layer=layer,
        d_model=d_model,
        grid={"t": 1, "h": 1, "w": n_tok, "tubelet": chunk, "patch_px": 1},
        cells=cells,
        split_bytes=split_bytes,
        clips=clips,
        windows=windows,
        sae=sae,
        sae_config=sae_config,
        nulls=nulls,
        source=source,
        notes=notes,
    )


def _episode_windows(
    path: Path,
    episode_idx: int,
    n_windows: int,
    chunk: int,
    token_indices: list[int],
    d_model: int,
    cells: np.ndarray,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    data = load_file(str(path))
    acts = data["acts"]
    if acts.ndim != 2 or acts.shape[1] != d_model:
        raise ValueError(f"{path.name}: acts must be [N, {d_model}], got {acts.shape}")
    episode = data["episode_idx"].astype(np.int64)
    if episode.size and (episode != episode_idx).any():
        raise ValueError(f"{path.name} holds rows of another episode")
    step = data["step"].astype(np.int64)
    tok = data["tok"].astype(np.int64)
    if (tok < 0).any():
        raise ValueError(f"{path.name} has mean-pool rows (tok < 0); acts/ must hold token rows only")
    if (step % chunk != 0).any():
        raise ValueError(f"{path.name}: every step must be a chunk start (multiple of {chunk})")
    if not np.isfinite(acts.astype(np.float32)).all():
        raise ValueError(f"{path.name} has non-finite activations")
    order = np.lexsort((tok, step))
    step, tok, acts = step[order], tok[order], acts[order]
    expected_tok = np.asarray(token_indices, dtype=np.int64)
    width = expected_tok.size
    if step.size != n_windows * width:
        raise ValueError(f"{path.name}: {step.size} rows, expected {n_windows} policy calls x {width} tokens")
    for window in range(n_windows):
        part = slice(window * width, (window + 1) * width)
        if (step[part] != window * chunk).any() or not np.array_equal(tok[part], expected_tok):
            raise ValueError(f"{path.name}: policy call {window} (step {window * chunk}) does not hold each captured token once")
        yield window, acts[part].astype(np.float32), cells


def _fold_sae(directory: Path, d_model: int) -> tuple[SAE, SAEConfig, dict]:
    raw = _read_json(directory / "config.json")
    arch = str(raw.get("arch", ""))
    if arch != "batchtopk":
        raise ValueError(f"sae/config.json arch {arch!r}: the groot-v9 importer reads BatchTopK SAEs")
    norm = raw.get("norm") or {}
    if norm.get("pos_mean") is not None:
        raise ValueError("this SAE subtracts a per-position mean, which cannot be folded into one encoder")
    weights = load_file(str(directory / "weights.safetensors"))
    n_features = int(raw.get("d_sae") or raw["n_latents"])
    if int(raw.get("d_in", d_model)) != d_model:
        raise ValueError("SAE d_in does not match the capture's d_model")
    w_enc = weights["W_enc"].astype(np.float32)  # [d, M]
    w_dec = weights["W_dec"].astype(np.float32)  # [M, d]
    b_enc = weights["b_enc"].astype(np.float32)
    b_dec = weights["b_dec"].astype(np.float32)
    if w_enc.shape != (d_model, n_features) or w_dec.shape != (n_features, d_model):
        raise ValueError(f"SAE weight shapes {w_enc.shape}, {w_dec.shape} do not match d={d_model}, M={n_features}")
    mu = weights["norm_mu"].astype(np.float32).reshape(-1) if "norm_mu" in weights else np.zeros(d_model, np.float32)
    scales = np.asarray(weights["norm_scale"], dtype=np.float32).reshape(-1) if "norm_scale" in weights else np.ones(1, np.float32)
    if scales.size not in (1, d_model):
        raise ValueError(f"norm_scale has {scales.size} values; expected 1 or d_model={d_model}")
    per_channel = scales.size > 1  # ii retrain-sae --norm channel
    scale = float(scales[0]) if not per_channel else float(np.median(scales))
    if "threshold" in weights:
        threshold = float(np.asarray(weights["threshold"]).reshape(-1)[0])
    else:
        threshold = float(raw["inference_threshold"])
    stated = raw.get("inference_threshold")
    if stated is not None and not math.isclose(float(stated), threshold, rel_tol=1e-5, abs_tol=1e-9):
        raise ValueError(f"inference_threshold {stated} in config.json != threshold {threshold} in the weights")
    if not (np.isfinite(scales).all() and (scales > 0).all() and math.isfinite(threshold) and threshold >= 0):
        raise ValueError("norm_scale must be positive and the threshold finite and >= 0")
    norms = np.linalg.norm(w_dec, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("SAE decoder rows are not unit-norm")

    # Source: pre = relu(((x - mu)/s - b_dec) @ W_enc + b_enc); f = pre * (pre > thr);
    #         x_hat = s * (f @ W_dec + b_dec) + mu.
    # ii with f' = s * f: pre' = relu(W_enc' (x - b_dec') + b_enc'), W_enc' = W_enc^T,
    #         b_enc' = s * b_enc, b_dec' = mu + s * b_dec, threshold' = s * thr, W_dec' = W_dec^T.
    # Per-channel s [d]: one scale cannot move into the threshold, so the values stay the source's own:
    #         W_enc' = (W_enc / s[:, None])^T, b_enc' = b_enc, threshold' = thr, b_dec' = mu + s * b_dec,
    #         W_dec' = (W_dec * s)^T (rows in raw units, so not unit-norm).
    fold_thr = threshold if per_channel else scale * threshold
    sae = SAE(d_model, n_features, activation="jumprelu", threshold=fold_thr)
    with torch.no_grad():
        if per_channel:
            sae.W_enc.copy_(torch.from_numpy(np.ascontiguousarray((w_enc / scales[:, None]).T)))
            sae.b_enc.copy_(torch.from_numpy(b_enc))
            sae.W_dec.copy_(torch.from_numpy(np.ascontiguousarray((w_dec * scales[None, :]).T)))
            sae.b_dec.copy_(torch.from_numpy(mu + scales * b_dec))
        else:
            sae.W_enc.copy_(torch.from_numpy(np.ascontiguousarray(w_enc.T)))
            sae.b_enc.copy_(torch.from_numpy(scale * b_enc))
            sae.W_dec.copy_(torch.from_numpy(np.ascontiguousarray(w_dec.T)))
            sae.b_dec.copy_(torch.from_numpy(mu + scale * b_dec))
    sae.eval()
    for param in sae.parameters():
        param.requires_grad_(False)
    train = raw.get("train") or {}
    config = SAEConfig(
        d_model=d_model,
        n_features=n_features,
        l1_coeff=0.0,
        ortho_coeff=0.0,
        lr=float(train.get("lr", 0.0)),
        steps=int(train.get("steps", 0)),
        batch_size=int(train.get("batch", 0)),
        seed=int(train.get("seed", 0)),
        normalize_decoder=not per_channel,
        activation="jumprelu",
        threshold=fold_thr,
    )
    notes = {
        "source_arch": arch,
        "source_k": raw.get("k"),
        "source_threshold": threshold,
        "norm_scale": scale,
        "fold": (
            "x_n = (x - norm_mu) / norm_scale is folded into W_enc, b_enc and b_dec; the threshold is scaled by "
            "norm_scale. The active set equals the source's, and each value is norm_scale times the source's."
        ),
    }
    if per_channel:
        notes["norm_scale"] = {"per_channel": True, "median": scale, "min": float(scales.min()), "max": float(scales.max())}
        notes["fold"] = (
            "Per-channel norm_scale s: x_n = (x - norm_mu) / s is folded into W_enc (rows divided by s) and b_dec "
            "(norm_mu + s * b_dec); W_dec rows are multiplied by s, so they are in raw units and not unit-norm. "
            "The active set and every value equal the source's."
        )
    return sae, config, notes


def _model_sha(adapter: str, checkpoint: str, manifest: dict, given: str | None, log) -> tuple[str, str]:
    if given is not None:
        if re.fullmatch(r"[0-9a-f]{64}", given) is None:
            raise ValueError("--model-sha must be a 64-char lowercase hex sha256")
        return given, "argument"
    from interpretableintelligence.adapters.groot.adapter import _checkpoint_fingerprint, model_sha_for  # noqa: PLC0415

    try:
        fingerprint = _checkpoint_fingerprint(checkpoint)
    except Exception as exc:  # noqa: BLE001 - offline or the Hub refused; fall back below
        log(f"note: could not read the {checkpoint} weights fingerprint ({type(exc).__name__}); model.sha is a digest of the declared model")
    else:
        return model_sha_for(adapter, fingerprint), "hub (same as ii run groot)"
    declared = json.dumps(
        {"adapter": adapter, "model": manifest.get("model"), "hook": manifest.get("hook")},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(b"ii-import-declared-model-v1\0" + declared.encode("utf-8")).hexdigest()
    return digest, "declared (sha256 of the source's model and hook record; the weights were not hashed at capture)"


def _check_pot0(manifest: dict, episodes: list[dict]) -> None:
    relabel = manifest.get("fail_relabel")
    if not isinstance(relabel, dict) or relabel.get("rule") != POT0_RULE:
        raise ValueError(
            f"manifest.fail_relabel.rule is not {POT0_RULE!r}: run tools/derive_pot0_fail.py on the capture first. "
            "The v9 active-object failure labels are not used."
        )
    for ep in episodes:
        rule = ep.get("fail_rule")
        if not isinstance(rule, str) or not rule.startswith(POT0_RULE):
            raise ValueError(f"episode {ep.get('episode_idx')} has no pot-0 failure label (fail_rule {rule!r})")
        if bool(ep.get("success")) and ep.get("fail_kind") is not None:
            raise ValueError(f"episode {ep.get('episode_idx')} succeeded but has fail_kind {ep.get('fail_kind')!r}")
        if not bool(ep.get("success")) and ep.get("fail_kind") is None:
            raise ValueError(f"episode {ep.get('episode_idx')} failed but has no pot-0 fail_kind")
    rows = manifest.get("episodes")
    if isinstance(rows, list):
        by_idx = {int(row["episode_idx"]): row for row in rows}
        for ep in episodes:
            row = by_idx.get(int(ep["episode_idx"]))
            if row is None or any(row.get(key) != ep.get(key) for key in LABEL_FIELDS):
                raise ValueError(f"episode {ep['episode_idx']}: manifest.json and episodes.jsonl disagree on its labels")


def _membership(split: object, episode_ids: list[int]) -> dict[int, str]:
    if not isinstance(split, dict) or not isinstance(split.get("train"), list) or not isinstance(split.get("heldout"), list):
        raise ValueError("split.json must have train and heldout lists")
    train, heldout = split["train"], split["heldout"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in train + heldout):
        raise ValueError("split.json lists must hold episode indices (integers)")
    if len(set(train) | set(heldout)) != len(train) + len(heldout):
        raise ValueError("split.json lists overlap or repeat an episode")
    if set(train) | set(heldout) != set(episode_ids):
        raise ValueError("split.json episodes do not match episodes.jsonl")
    membership = {int(item): "train" for item in train}
    membership.update({int(item): "heldout" for item in heldout})
    return membership


def _read_episodes(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["episode_idx"]))
    ids = [int(row["episode_idx"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("episodes.jsonl repeats an episode_idx")
    bad = [row["episode_idx"] for row in rows if row.get("status", "done") != "done"]
    if bad:
        raise ValueError(f"episodes not finished: {bad[:5]}")
    return rows


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
