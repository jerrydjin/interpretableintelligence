"""Retrain the GR00T capture SAE with dead-latent revival, and evaluate it against the one it replaces.

Why: the ``groot-1028-v9-n100`` BatchTopK SAE (8192 latents, ``k`` = 32) has 7,736 latents that never fire on its
train split and 7,730 on held-out (``demo-data/groot/sae-heldout-eval.json``). Only about 460 latents carry the
code, so most of the dictionary the picker and the random controls drew from is empty. This module trains a
replacement on the same capture rows (train split only) and writes it in the same ``sae/`` layout, so
``ablate_v9.load_sae``, the picker's ``rollout_io.PolicySAE``, ``ii import --format groot-v9`` and
``ii rollouts --pick`` (with its random-weights SAE control) read it unchanged.

Training (all in the SAE's normalised space ``x_n = (x - norm_mu) / norm_scale``):

- Init is the trainer's own init that the random-weights SAE control redraws (``random_sae.untrained_weights``):
  ``randn(D, M)`` with unit-norm columns as ``W_enc``, its transpose as ``W_dec``, zero biases. So
  ``random:<seed>`` with the training seed is exactly this SAE before its first step, as research rule 4 assumes.
- BatchTopK (Bussmann et al. 2024): over a batch of ``B`` tokens keep the ``k * B`` largest ReLU pre-activations.
- Loss: normalised MSE. A latent is dead once it has not fired for ``dead_tokens`` training tokens. Revival
  (``revive``) is one of:

  - ``auxk`` (Gao et al. 2024): add ``aux_coeff`` times the AuxK loss, in which the ``aux_k`` largest ReLU
    pre-activations of dead latents per token reconstruct the main reconstruction's residual (normalised by the
    residual's variance; ``b_dec`` is held fixed in that term since every latent shares it).
  - ``resample`` (Bricken et al. 2023): every ``resample_every`` steps in the first 60% of training, point each dead
    latent at the residual of a high-error train token and reset its Adam state.
  - ``none``: plain BatchTopK, the old SAE's recipe.
- Decoder rows are renormalised to unit length after every step (the importer requires unit-norm rows).
- Inference threshold: the ``(k * n)``-th largest pre-activation over the ``n`` train tokens
  (``random_sae.batchtopk_threshold``, the same rule the untrained controls use), so mean L0 on train is ``k``.

Evaluation (:func:`evaluate`) runs the JumpReLU encoder at the threshold, the way the hook and the picker do, on the
train and held-out episodes of ``split.json``, and reports the keys of ``sae-heldout-eval.json`` plus dense-latent
counts, the mean per-channel EV, and how much of the input variance sits in its largest channels (on synthetic data,
a few outlier channels reproduce the old SAE's dead-latent pattern, and revival does not fix it). It does not compute ``loss_recovered``: that needs the action head with the reconstruction spliced in.

Nothing here touches ``split.json``, a scoreboard, or a rollout result. A retrained SAE is a new dictionary: its
feature ids do not match the old one's (720, 7081, 5852, ...), and nothing found on the old SAE carries over.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

# The trainer and evaluation live in core (``interpretableintelligence.sae_batchtopk``); re-exported for old callers.
from interpretableintelligence.sae_batchtopk import (  # noqa: F401
    CHANNEL_SCALE_FLOOR,
    CHUNK_ROWS,
    DENSE_FRAC,
    INIT_RULE,
    NORMS,
    RESAMPLE_ENC_SCALE,
    RESAMPLE_TOKENS,
    RESAMPLE_UNTIL,
    REVIVE,
    TOP_DIMS,
    RetrainConfig,
    Split,
    _batchtopk,
    _quantiles,
    _resample,
    evaluate,
    load_tensors,
    norm_stats,
    save_sae_dir,
    train,
)

# Entries of a capture folder that do not depend on the SAE; the output folder links to these.
# features/, candidates*, picker* and ablation outputs were computed with the old dictionary and are left out.
CAPTURE_LINKS = ("manifest.json", "split.json", "episodes.jsonl", "episodes.json", "acts", "acts_meanpool", "actions.safetensors", "actions", "sim")
SAE_DIR = "sae"
REPORT = "retrain.json"
EVAL_FILE = "heldout_eval.json"


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_capture(capture: Path, device: torch.device, log: Callable[[str], None] = print) -> tuple[Split, Split]:
    """Token rows (``tok >= 0``) of the capture's train and held-out episodes, as float16 on ``device``."""
    from safetensors.torch import load_file

    capture = Path(capture)
    split = _read_json(capture / "split.json")
    train_ids, held_ids = set(int(i) for i in split["train"]), set(int(i) for i in split["heldout"])
    if train_ids & held_ids:
        raise ValueError("split.json: an episode is in both train and heldout")
    shards = sorted((capture / "acts").glob("shard-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{capture}: no acts/shard-*.safetensors")
    parts: dict[str, list[list[torch.Tensor]]] = {"train": [[], [], []], "heldout": [[], [], []]}
    for shard in shards:
        tensors = load_file(str(shard))
        episode = tensors["episode_idx"].long()
        tok = tensors["tok"].long() if "tok" in tensors else torch.zeros_like(episode)
        keep_tok = tok >= 0
        for name, ids in (("train", train_ids), ("heldout", held_ids)):
            keep = keep_tok & torch.tensor([int(e) in ids for e in episode.tolist()], dtype=torch.bool)
            if keep.any():
                parts[name][0].append(tensors["acts"][keep].to(torch.float16))
                parts[name][1].append(tok[keep])
                parts[name][2].append(episode[keep])
    out = []
    for name in ("train", "heldout"):
        acts, toks, eps = parts[name]
        if not acts:
            raise ValueError(f"{capture}: no {name} rows")
        result = Split(torch.cat(acts).to(device), torch.cat(toks).to(device), torch.cat(eps).to(device))
        log(f"{name}: {result.n} tokens from {len(torch.unique(result.episode))} episodes")
        out.append(result)
    return out[0], out[1]



def _link_capture(capture: Path, out: Path) -> dict:
    linked, skipped = [], []
    for entry in sorted(Path(capture).iterdir()):
        if entry.name in CAPTURE_LINKS:
            target = out / entry.name
            if not target.exists() and not target.is_symlink():
                os.symlink(entry.resolve(), target)
            linked.append(entry.name)
        else:
            skipped.append(entry.name)
    return {"linked": linked, "not_linked": skipped}


def sae_config(config: RetrainConfig, d: int, threshold: float, scale: float, capture: Path, n_train: int) -> dict:
    return {
        "arch": "batchtopk",
        "d_in": d,
        "d_sae": config.n_latents,
        "n_latents": config.n_latents,
        "k": config.k,
        "inference_threshold": threshold,
        "norm": {"pos_mean": None, "norm_scale": scale, "mode": config.norm, "norm_scale_shape": "per_channel" if config.norm == "channel" else "scalar"},
        "train": {
            "trainer": "ii retrain-sae (adapters/groot/retrain_sae.py)",
            "steps": config.steps,
            "batch": config.batch_tokens,
            "lr": config.lr,
            "seed": config.seed,
            "aux_coeff": config.aux_coeff,
            "aux_k": config.aux_k,
            "dead_tokens": config.dead_tokens or n_train,
            "init": INIT_RULE.split("; threshold")[0],
            "threshold_rule": "(k * n)-th largest pre-activation over the n train-split tokens",
            "source_capture": str(capture),
        },
    }


def retrain(
    capture: Path,
    out: Path,
    config: RetrainConfig,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> dict:
    """Train, evaluate old and new on train and held-out, and write ``out`` as a capture folder with the new ``sae/``."""
    config.check()
    capture, out = Path(capture), Path(out)
    if (out / SAE_DIR).exists():
        raise FileExistsError(f"{out / SAE_DIR} already exists; pick a new --out")
    source_path = capture / SAE_DIR / "weights.safetensors"
    source = load_tensors(source_path) if source_path.is_file() else None
    train_split, held_split = load_capture(capture, device, log)
    mu, scale = norm_stats(train_split, source, config.norm)
    log(f"norm: {config.norm} (norm_scale {float(scale.min()):.4g}..{float(scale.max()):.4g} over {scale.numel()} value(s))")
    started = time.perf_counter()
    weights, curve = train(train_split, mu, scale, config, log)
    seconds = time.perf_counter() - started
    new = {"train": evaluate(weights, train_split), "heldout": evaluate(weights, held_split)}
    old = None
    if source is not None:
        old = {"train": evaluate(source, train_split), "heldout": evaluate(source, held_split)}
    out.mkdir(parents=True, exist_ok=True)
    links = _link_capture(capture, out)
    d = int(train_split.acts.shape[1])
    threshold = float(weights["threshold"])
    scale_note = float(weights["norm_scale"].reshape(-1)[0]) if config.norm != "channel" else float(weights["norm_scale"].median())
    save_sae_dir(weights, out / SAE_DIR, sae_config(config, d, threshold, scale_note, capture, train_split.n))
    eval_doc = dict(new, loss_recovered=None, loss_recovered_note="not computed: needs the action head with the reconstruction spliced in")
    (out / SAE_DIR / EVAL_FILE).write_text(json.dumps(eval_doc, indent=1) + "\n", encoding="utf-8")
    report = {
        "capture": str(capture),
        "config": asdict(config),
        "device": str(device),
        "train_seconds": round(seconds, 1),
        "threshold": threshold,
        "new": new,
        "old": old,
        "curve": curve,
        "capture_links": links,
        "note": (
            "A retrained SAE is a new dictionary: feature ids do not match the old SAE's, and no finding carries over. "
            "Offline only: nothing here is a rollout result or changes a scoreboard."
        ),
    }
    (out / REPORT).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def format_report(report: dict) -> str:
    rows = [("old", report.get("old")), ("new", report["new"])]
    lines = [f"{'SAE':<4} {'split':<8} {'latents':>7} {'alive':>6} {'dead':>6} {'dense':>6} {'L0':>6} {'EV':>7}"]
    for name, result in rows:
        if result is None:
            continue
        for split in ("train", "heldout"):
            r = result[split]
            total = r["alive_latents"] + r["dead_latents"]
            lines.append(
                f"{name:<4} {split:<8} {total:>7} {r['alive_latents']:>6} {r['dead_latents']:>6} "
                f"{r['dense_latents']:>6} {r['l0_mean']:>6.2f} {r['ev_raw']:>7.4f}"
            )
    return "\n".join(lines)
