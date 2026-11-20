"""BatchTopK SAE trainer with dead-latent revival, its evaluation, and its untrained twin (any model).

The weights use one flat layout (``SAE_KEYS``): ``W_enc`` [D, M], ``W_dec`` [M, D], ``b_enc``, ``b_dec``, and the
input normalisation ``norm_mu`` / ``norm_scale`` (a scalar, or one value per channel), plus the JumpReLU inference
``threshold``. Inputs are normalised as ``x_n = (x - norm_mu) / norm_scale`` before encoding.

- :func:`train`: BatchTopK (Bussmann et al. 2024) on normalised MSE, with AuxK (Gao et al. 2024) or resampling
  (Bricken et al. 2023) to revive dead latents; decoder rows renormalised every step. The inference threshold is the
  ``(k * n)``-th largest pre-activation over the ``n`` train rows (:func:`batchtopk_threshold`), so mean L0 there is ``k``.
- :func:`evaluate`: EV (normalised and raw units), per-dimension EV, alive / dead / dense latents and firing quantiles,
  with the JumpReLU-at-threshold encoder.
- :func:`untrained_weights` / :func:`build_untrained_sae`: the trainer's own init (``INIT_RULE``) with the same
  normalisation and a threshold set for the same ``k``. That is the random-weights SAE control of research rule 4.

Rows come in as a :class:`Split` (``acts`` [n, D], a token position and an episode index per row). Nothing here knows
about a particular model; ``ii retrain-sae`` (``adapters/groot/retrain_sae.py``) reads a GR00T capture into a Split and
writes the result as a capture ``sae/`` folder.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

SAE_KEYS = ("W_enc", "W_dec", "b_enc", "b_dec", "norm_mu", "norm_scale", "threshold")
INIT_RULE = (
    "W = randn(D, M, generator=manual_seed(seed)); columns unit-normalised; W_enc = W, W_dec = W.T; b_enc = b_dec = 0; "
    "norm_mu, norm_scale copied from the trained SAE; threshold = the (k * n)-th largest pre-activation over the n "
    "calibration tokens (BatchTopK over one batch), so mean L0 there is k"
)

DENSE_FRAC = 0.1
CHUNK_ROWS = 8192
NORMS = ("keep", "fit", "channel")
CHANNEL_SCALE_FLOOR = 1e-3  # a channel's scale is at least this times the median channel std
REVIVE = ("auxk", "resample", "none")
TOP_DIMS = 8
RESAMPLE_UNTIL = 0.6  # no resampling in the last 40% of steps, so revived latents settle before the threshold is set
RESAMPLE_TOKENS = 65536
RESAMPLE_ENC_SCALE = 0.2


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    missing = [key for key in SAE_KEYS if key not in tensors]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    return tensors


def untrained_weights(trained: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """The trainer's init drawn with ``seed``, with the trained SAE's normalisation. ``threshold`` is not set yet."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    d_in, n_latents = (int(item) for item in trained["W_enc"].shape)
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(d_in, n_latents, generator=generator, dtype=torch.float32)
    matrix = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return {
        "W_enc": matrix.contiguous(),
        "W_dec": matrix.T.contiguous(),
        "b_enc": torch.zeros(n_latents, dtype=torch.float32),
        "b_dec": torch.zeros(d_in, dtype=torch.float32),
        "norm_mu": trained["norm_mu"].float().clone(),
        "norm_scale": trained["norm_scale"].float().reshape(-1).clone(),
    }


def pre_activations(weights: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """ReLU pre-activations ``relu(((x - norm_mu) / norm_scale - b_dec) @ W_enc + b_enc)`` of raw rows ``x`` [n, D]."""
    scale = weights["norm_scale"].float().reshape(-1).to(x.device)  # [1], or [D] for a channel-normalised SAE
    xn = (x.float() - weights["norm_mu"]) / scale
    return torch.relu((xn - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"])


def batchtopk_threshold(weights: dict[str, torch.Tensor], chunks, k: int) -> tuple[float, int]:
    """The (k * n)-th largest pre-activation over the n rows ``chunks()`` yields, and n.

    ``chunks`` is called twice (count, then select), so it must return a fresh iterable each time. Exact: the global
    top (k * n) values are among each chunk's own top (k * n), so a buffer of that size suffices.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rows = sum(int(x.shape[0]) for x in chunks())
    if rows == 0:
        raise ValueError("no calibration rows")
    target = int(k) * rows
    best = torch.empty(0, dtype=torch.float32)
    seen = 0
    for x in chunks():
        if x.numel() == 0:
            continue
        seen += int(x.shape[0])
        pre = pre_activations(weights, x).reshape(-1)
        merged = torch.cat([best.to(pre.device), pre])
        best = torch.topk(merged, min(target, merged.numel()), sorted=False).values
    if seen != rows or best.numel() < target:
        raise RuntimeError(f"calibration rows changed between passes ({rows} then {seen})")
    return float(best.min()), rows


def build_untrained_sae(trained_path: Path, seed: int, chunks, *, k: int) -> tuple[dict[str, torch.Tensor], dict]:
    """Untrained weights for ``seed`` with a BatchTopK threshold over ``chunks()`` (a callable returning a fresh iterable
    of raw rows); and a provenance record."""
    trained = load_tensors(Path(trained_path))
    weights = untrained_weights(trained, int(seed))
    threshold, rows = batchtopk_threshold(weights, chunks, int(k))
    weights["threshold"] = torch.tensor(threshold, dtype=torch.float32)
    record = {
        "dictionary": f"random:{int(seed)}",
        "seed": int(seed),
        "k": int(k),
        "threshold": threshold,
        "calibration_rows": rows,
        "init": INIT_RULE,
        "trained_threshold": float(trained["threshold"].float().reshape(-1)[0]),
    }
    return weights, record


def save_sae_dir(weights: dict[str, torch.Tensor], directory: Path, config: dict) -> Path:
    """Write ``weights.safetensors`` and ``config.json`` in the capture's ``sae/`` layout. Returns the weights path."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "weights.safetensors"
    save_file({key: weights[key].contiguous() for key in SAE_KEYS}, str(path))
    (directory / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    return path




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
    init = untrained_weights(template, int(config.seed))
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
    threshold, _rows = batchtopk_threshold(weights, train_split.chunks, config.k)
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
