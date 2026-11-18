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
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from interpretableintelligence.adapters.groot import random_sae

DENSE_FRAC = 0.1
# Entries of a capture folder that do not depend on the SAE; the output folder links to these.
# features/, candidates*, picker* and ablation outputs were computed with the old dictionary and are left out.
CAPTURE_LINKS = ("manifest.json", "split.json", "episodes.jsonl", "episodes.json", "acts", "acts_meanpool", "actions.safetensors", "actions", "sim")
SAE_DIR = "sae"
REPORT = "retrain.json"
EVAL_FILE = "heldout_eval.json"
CHUNK_ROWS = 8192
NORMS = ("keep", "fit", "channel")
CHANNEL_SCALE_FLOOR = 1e-3  # a channel's scale is at least this times the median channel std
REVIVE = ("auxk", "resample", "none")
TOP_DIMS = 8
RESAMPLE_UNTIL = 0.6  # no resampling in the last 40% of steps, so revived latents settle before the threshold is set
RESAMPLE_TOKENS = 65536
RESAMPLE_ENC_SCALE = 0.2


@dataclass(frozen=True)
class RetrainConfig:
    n_latents: int = 2048
    k: int = 32
    steps: int = 6000
    batch_tokens: int = 4096
    lr: float = 4e-4
    aux_coeff: float = 1 / 32
    aux_k: int = 512
    # A latent is dead after this many training tokens without firing. 0 means one pass over the train split.
    dead_tokens: int = 0
    seed: int = 0
    # How dead latents are revived: "auxk" (AuxK loss), "resample" (reinitialise them on high-error tokens),
    # or "none" (plain BatchTopK, the old SAE's recipe).
    revive: str = "auxk"
    # Resample every this many steps during the first RESAMPLE_UNTIL of training. 0: steps // 10.
    resample_every: int = 0
    # "keep": reuse the capture SAE's norm_mu / norm_scale (data statistics, not weights), so old and new SAEs see
    # the same inputs. "fit": recompute them from the train rows. "channel": per-channel standardisation (train
    # mean and per-channel std): norm_scale is then a [D] vector, which the v9 hook, the picker and the importer read.
    norm: str = "keep"

    def check(self) -> None:
        if self.n_latents < 1 or not 1 <= self.k <= self.n_latents:
            raise ValueError(f"need 1 <= k <= n_latents, got k={self.k}, n_latents={self.n_latents}")
        if self.steps < 1 or self.batch_tokens < 1:
            raise ValueError("steps and batch_tokens must be >= 1")
        if not (self.lr > 0 and math.isfinite(self.lr)):
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.aux_coeff < 0 or self.aux_k < 0 or self.dead_tokens < 0:
            raise ValueError("aux_coeff, aux_k and dead_tokens must be >= 0")
        if self.revive not in REVIVE:
            raise ValueError(f"revive must be one of {REVIVE}, got {self.revive!r}")
        if self.resample_every < 0:
            raise ValueError("resample_every must be >= 0")
        if self.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {self.norm!r}")


@dataclass
class Split:
    acts: torch.Tensor  # [n, D] float16 raw rows
    tok: torch.Tensor  # [n] token position
    episode: torch.Tensor  # [n] episode index

    @property
    def n(self) -> int:
        return int(self.acts.shape[0])

    def chunks(self, size: int = CHUNK_ROWS):
        for start in range(0, self.n, size):
            yield self.acts[start : start + size].float()


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


def norm_stats(train: Split, source: dict[str, torch.Tensor] | None, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``norm_mu`` [D] and ``norm_scale``. ``fit``: train mean, and the scalar scale (shape [1]) that makes a
    centred row's mean squared norm ``D``. ``channel``: train mean and each channel's own std (shape [D], floored at
    ``CHANNEL_SCALE_FLOOR`` times the median std), so every channel has unit variance."""
    device = train.acts.device
    if mode == "keep":
        if source is None:
            raise ValueError("norm='keep' needs the capture's sae/weights.safetensors; use norm='fit'")
        return source["norm_mu"].float().reshape(-1).to(device), source["norm_scale"].float().reshape(-1)[:1].to(device)
    d = int(train.acts.shape[1])
    total = torch.zeros(d, dtype=torch.float64, device=device)
    for x in train.chunks():
        total += x.double().sum(0)
    mean = total / train.n
    sq = torch.zeros(d, dtype=torch.float64, device=device)
    for x in train.chunks():
        sq += ((x.double() - mean) ** 2).sum(0)
    var = sq / train.n
    if mode == "channel":
        std = torch.sqrt(var)
        floor = CHANNEL_SCALE_FLOOR * float(std.median())
        if not floor > 0:
            raise ValueError("train rows have zero variance in most channels")
        return mean.float(), std.clamp_min(floor).float()
    scale = torch.sqrt(var.mean()).reshape(1)
    if not (bool((scale > 0).all()) and bool(torch.isfinite(scale).all())):
        raise ValueError("train rows have zero variance")
    return mean.float(), scale.float()


def _batchtopk(pre: torch.Tensor, k: int) -> torch.Tensor:
    flat = pre.reshape(-1)
    top = flat.topk(min(k * pre.shape[0], flat.numel()), sorted=False)
    return torch.zeros_like(flat).scatter(0, top.indices, top.values).reshape(pre.shape)


def train(
    train_split: Split,
    mu: torch.Tensor,
    scale: torch.Tensor,
    config: RetrainConfig,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    """Train a BatchTopK SAE with AuxK on the train rows. Returns weights in the capture ``sae/`` layout and a curve."""
    config.check()
    device = train_split.acts.device
    d = int(train_split.acts.shape[1])
    template = {"W_enc": torch.empty(d, config.n_latents), "norm_mu": mu.cpu(), "norm_scale": torch.ones(1)}
    init = random_sae.untrained_weights(template, int(config.seed))
    params = {key: torch.nn.Parameter(init[key].to(device)) for key in ("W_enc", "b_enc", "W_dec", "b_dec")}
    opt = torch.optim.Adam(params.values(), lr=config.lr)
    dead_tokens = config.dead_tokens or train_split.n
    since_fired = torch.zeros(config.n_latents, dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(int(config.seed))
    batch = min(config.batch_tokens, train_split.n)
    mu = mu.to(device)
    scale = scale.to(device=device, dtype=torch.float32).reshape(-1)
    curve: list[dict] = []
    resample_every = config.resample_every or max(config.steps // 10, 1)
    resampled = 0
    every = max(config.steps // 20, 1)
    started = time.perf_counter()
    for step in range(config.steps):
        idx = torch.randint(0, train_split.n, (batch,), device=device, generator=gen)
        x = (train_split.acts[idx].float() - mu) / scale
        raw = (x - params["b_dec"]) @ params["W_enc"] + params["b_enc"]
        f = _batchtopk(torch.relu(raw), config.k)
        recon = f @ params["W_dec"] + params["b_dec"]
        err = x - recon
        var = ((x - x.mean(0)) ** 2).sum(-1).mean().clamp_min(1e-12)
        mse = (err**2).sum(-1).mean() / var
        loss = mse
        fired = (f > 0).any(0)
        since_fired += batch
        since_fired[fired] = 0
        dead = since_fired >= dead_tokens
        n_dead = int(dead.sum())
        aux = torch.zeros((), device=device)
        if config.revive == "auxk" and config.aux_coeff and config.aux_k and n_dead:
            k_aux = min(config.aux_k, n_dead)
            # b_dec is shared by every latent, so the aux loss does not move it.
            aux_pre = torch.relu((x - params["b_dec"].detach()) @ params["W_enc"] + params["b_enc"])
            masked = aux_pre.masked_fill(~dead, -1.0)
            top = masked.topk(k_aux, dim=-1)
            e_hat = torch.zeros_like(aux_pre).scatter(-1, top.indices, top.values) @ params["W_dec"]
            target = err.detach()
            aux = ((e_hat - target) ** 2).sum(-1).mean() / ((target - target.mean(0)) ** 2).sum(-1).mean().clamp_min(1e-12)
            loss = loss + config.aux_coeff * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            params["W_dec"].div_(params["W_dec"].norm(dim=1, keepdim=True).clamp_min(1e-8))
        if (
            config.revive == "resample"
            and resample_every
            and (step + 1) % resample_every == 0
            and step + 1 <= RESAMPLE_UNTIL * config.steps
        ):
            dead_now = since_fired >= dead_tokens
            if dead_now.any():
                n_resampled = _resample(params, opt, dead_now, train_split, mu, scale, config.k, gen)
                since_fired[dead_now] = 0
                resampled += n_resampled
        if step % every == 0 or step + 1 == config.steps:
            point = {"step": step + 1, "fvu": float(mse.detach()), "aux": float(aux.detach()), "dead": n_dead, "resampled": resampled}
            curve.append(point)
            rate = (time.perf_counter() - started) / (step + 1)
            log(
                f"step {step + 1}/{config.steps} fvu={point['fvu']:.5f} aux={point['aux']:.4f} "
                f"dead={n_dead}/{config.n_latents} ({rate * (config.steps - step - 1):.0f}s left)"
            )
    weights = {key: value.detach().float().contiguous() for key, value in params.items()}
    weights["norm_mu"] = mu.float().contiguous()
    weights["norm_scale"] = scale.clone()
    threshold, _rows = random_sae.batchtopk_threshold(weights, train_split.chunks, config.k)
    weights["threshold"] = torch.tensor(threshold, dtype=torch.float32, device=device)
    return {key: value.cpu() for key, value in weights.items()}, curve


@torch.no_grad()
def _resample(params: dict, opt: torch.optim.Adam, dead: torch.Tensor, data: Split, mu: torch.Tensor, scale: torch.Tensor, k: int, gen: torch.Generator) -> int:
    """Point each dead latent at the residual of a badly reconstructed train token (Bricken et al. 2023).

    Tokens are drawn with probability proportional to their squared reconstruction error. The decoder row becomes the
    unit residual, the encoder column the same direction at 0.2x the mean alive encoder norm, the encoder bias 0, and
    the Adam moments of those entries are reset.
    """
    device = data.acts.device
    n = min(data.n, RESAMPLE_TOKENS)
    idx = torch.randint(0, data.n, (n,), device=device, generator=gen)
    x = (data.acts[idx].float() - mu) / scale
    f = _batchtopk(torch.relu((x - params["b_dec"]) @ params["W_enc"] + params["b_enc"]), k)
    err = x - (f @ params["W_dec"] + params["b_dec"])
    weight = (err**2).sum(-1)
    if not float(weight.sum()) > 0:
        return 0
    ids = torch.nonzero(dead).flatten()
    pick = torch.multinomial(weight, ids.numel(), replacement=ids.numel() > n, generator=gen)
    dirs = err[pick]
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    alive = ~dead
    enc_norm = float(params["W_enc"][:, alive].norm(dim=0).mean()) if alive.any() else 1.0
    params["W_dec"][ids] = dirs
    params["W_enc"][:, ids] = dirs.T * (RESAMPLE_ENC_SCALE * enc_norm)
    params["b_enc"][ids] = 0.0
    for name, view in (("W_dec", (ids, slice(None))), ("W_enc", (slice(None), ids)), ("b_enc", (ids,))):
        state = opt.state.get(params[name])
        if state:
            state["exp_avg"][view] = 0.0
            state["exp_avg_sq"][view] = 0.0
    return int(ids.numel())


@torch.no_grad()
def evaluate(weights: dict[str, torch.Tensor], split: Split) -> dict:
    """The ``sae-heldout-eval.json`` metrics for one split, with the JumpReLU-at-threshold encoder."""
    device = split.acts.device
    w = {key: value.float().to(device) for key, value in weights.items()}
    scale = w["norm_scale"].reshape(-1)  # [1] or [D] (norm="channel")
    # Errors are also measured in raw units (scaled back by norm_scale), so SAEs with different normalisations
    # compare on the same footing. For a scalar scale the two spaces give the same EV.
    s2 = (scale.double() ** 2).expand(int(split.acts.shape[1]))
    thr = float(w["threshold"].reshape(-1)[0])
    mu = w["norm_mu"]
    d = int(split.acts.shape[1])
    n_latents = int(w["W_enc"].shape[1])
    # Pass 1: means (overall, per token position) in normalised space.
    sum_all = torch.zeros(d, dtype=torch.float64, device=device)
    positions = torch.unique(split.tok).tolist()
    sum_pos = {p: torch.zeros(d, dtype=torch.float64, device=device) for p in positions}
    count_pos = {p: 0 for p in positions}
    for start in range(0, split.n, CHUNK_ROWS):
        xn = (split.acts[start : start + CHUNK_ROWS].float() - mu) / scale
        tok = split.tok[start : start + CHUNK_ROWS]
        sum_all += xn.double().sum(0)
        for p in positions:
            sel = tok == p
            sum_pos[p] += xn[sel].double().sum(0)
            count_pos[p] += int(sel.sum())
    mean_all = (sum_all / split.n).float()
    mean_pos = {p: (sum_pos[p] / count_pos[p]).float() for p in positions}
    # Pass 2: errors, variances, firing counts.
    sse = sst = 0.0
    sse_raw = sst_raw = 0.0
    sse_dim = torch.zeros(d, dtype=torch.float64, device=device)
    sst_dim = torch.zeros(d, dtype=torch.float64, device=device)
    sse_pos = {p: 0.0 for p in positions}
    sst_pos = {p: 0.0 for p in positions}
    fires = torch.zeros(n_latents, dtype=torch.long, device=device)
    l0 = 0.0
    for start in range(0, split.n, CHUNK_ROWS):
        xn = (split.acts[start : start + CHUNK_ROWS].float() - mu) / scale
        tok = split.tok[start : start + CHUNK_ROWS]
        z = torch.relu((xn - w["b_dec"]) @ w["W_enc"] + w["b_enc"])
        f = z * (z > thr)
        recon = f @ w["W_dec"] + w["b_dec"]
        sq_err = (xn - recon) ** 2
        err = sq_err.sum(-1)
        dev = (xn - mean_all) ** 2
        sse += float(err.sum())
        sst += float(dev.sum())
        sse_raw += float((sq_err.double() @ s2).sum())
        sst_raw += float((dev.double() @ s2).sum())
        sse_dim += sq_err.double().sum(0)
        sst_dim += dev.double().sum(0)
        on = f > 0
        fires += on.sum(0)
        l0 += float(on.sum())
        for p in positions:
            sel = tok == p
            sse_pos[p] += float((sq_err[sel].double() @ s2).sum())
            sst_pos[p] += float((((xn[sel] - mean_pos[p]) ** 2).double() @ s2).sum())
    ev = 1.0 - sse / sst if sst > 0 else float("nan")
    ev_raw = 1.0 - sse_raw / sst_raw if sst_raw > 0 else float("nan")
    raw_dim = sst_dim * s2
    freq = fires.float() / split.n
    alive = int((fires > 0).sum())
    return {
        "n_tokens": split.n,
        "n_episodes": int(torch.unique(split.episode).numel()),
        # EV in the SAE's normalised space, and in raw activation units. Equal for a scalar norm_scale; for
        # norm="channel" ev_normalised weighs every channel equally and ev_raw is the one to compare across SAEs.
        "ev_normalised": round(ev, 4),
        "ev_raw": round(ev_raw, 4),
        "fvu": round(1.0 - ev_raw, 4),
        # Mean over dimensions of each dimension's own EV. A few high-variance dimensions can carry ev_normalised;
        # this weighs every dimension equally, so it drops when the SAE only reconstructs the large ones.
        "ev_per_dim_mean": round(float((1.0 - sse_dim / sst_dim.clamp_min(1e-30)).mean()), 4),
        # Share of the split's total variance in its TOP_DIMS largest-variance dimensions (input statistic, the same for
        # every SAE). Near 1 means a few outlier channels dominate and ev_normalised mostly measures those.
        "variance_share_top_dims": round(float(raw_dim.topk(min(TOP_DIMS, d)).values.sum() / raw_dim.sum().clamp_min(1e-30)), 4),
        "top_dims": [int(i) for i in raw_dim.topk(min(TOP_DIMS, d)).indices.tolist()],
        "l0_mean": round(l0 / split.n, 2),
        "alive_latents": alive,
        "dead_latents": n_latents - alive,
        "alive_frac": round(alive / n_latents, 4),
        "dense_latents": int((freq > DENSE_FRAC).sum()),
        "dense_rule": f"fires on more than {DENSE_FRAC:.0%} of tokens",
        "firing_freq_quantiles_alive": _quantiles(freq[fires > 0]),
        "ev_within_token_position": {
            str(p): round(1.0 - sse_pos[p] / sst_pos[p], 4) if sst_pos[p] > 0 else None for p in positions
        },
    }


def _quantiles(values: torch.Tensor) -> dict:
    if values.numel() == 0:
        return {}
    qs = torch.tensor([0.1, 0.5, 0.9, 0.99], device=values.device)
    return {f"q{round(q * 100)}": float(f"{v:.3g}") for q, v in zip(qs.tolist(), torch.quantile(values.float(), qs).tolist(), strict=True)}


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
            "init": random_sae.INIT_RULE.split("; threshold")[0],
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
    source = random_sae.load_tensors(source_path) if source_path.is_file() else None
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
    random_sae.save_sae_dir(weights, out / SAE_DIR, sae_config(config, d, threshold, scale_note, capture, train_split.n))
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
