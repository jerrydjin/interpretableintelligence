"""GR00T v9 SAE feature-ablation runner (rise test, task 1028). Labels only, nothing here is a validated cause.

Intervention (LatentAblator, a forward hook on _groot_model.action_head.model.transformer_blocks.{layer}):
    x_n = (x - norm_mu) / norm_scale                       (the SAE's training normalisation)
    f   = JumpReLU_thr((x_n - b_dec) @ W_enc + b_enc)      (SAE inference rule, latents independent)
    out = x - norm_scale * sum_{i in S} f_i * W_dec[i]     (raw-space decoder contribution of the ablated latents)
so everything else, including the SAE's reconstruction error, is kept. S = {} goes through the same code path and
must return x bit-for-bit (recon-only check; asserted at every edited call). Applied at every policy call
(predict_action_chunk), at the denoise steps in plan["hook"]["denoise_steps"] ("final" = step index
num_inference_timesteps-1, the step the SAE was trained on; "all" = every step), on plan["hook"]["tokens"]
("all" = every DiT token, or a list of token positions).

Conditions (plan["conditions"]): baseline (no hook edit, fixed n, no SPRT), recon_only (S = {}, a few eps),
candidate / control (S = latents, SPRT), random_sae (S = latents of an untrained SAE, the random-weights SAE control).
A condition's optional "dictionary" names the SAE it edits: "trained" (default) or "random:<seed>", whose subset file
run_ablation takes in `dictionaries`. The edit rule is the same for every dictionary, so an untrained SAE's large
reconstruction error is kept and only the ablated latents' decoder contributions are removed. Every condition uses the same seeds: episode i -> seed 40000+i,
init_state_id i % 50, v9 per-episode noise generator, so results are paired by episode index.

SPRT (per condition, on the contiguous prefix of finished episodes in index order, so completion order and
resume do not change it): Wald tests H0 p=p0 vs H1 p=p1_rise and H0 p=p0 vs H1 p=p1_fall, both with
alpha/beta; stop at the first n where either test crosses a boundary, or at cap. Episodes that finish after
the stop (in flight at the time) are logged with used=false and excluded from n / rate / deltas.

Logs: <out>/<arm>/<cond>/episodes.jsonl (append-only, header line first, one line per episode, fsync'd;
broken last line ignored on resume), summary.json per condition, <out>/summary.json, schedule.json, plan.json.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ABLATE_VERSION = "ablate-v1"
PLAN_SCHEMA = "groot-ablate-plan-v1"
SAE_KEYS = ("W_enc", "W_dec", "b_enc", "b_dec", "norm_mu", "norm_scale", "threshold")


# --------------------------------------------------------------------------- SPRT / stats (pure)
# The implementation lives in core (interpretableintelligence.rollout_stats), the repo's one stats module.
from interpretableintelligence.rollout_stats import (  # noqa: E402,F401
    _closed_llr,
    _root,
    _sprt_key,
    _stop_arrays,
    _STOP_CACHE,
    _stop_probs,
    _tails,
    _VERDICTS_STOPPED,
    arm_inference,
    control_comparison,
    holm,
    llr_step,
    mcnemar_exact_p,
    min_boundary_gap,
    paired_delta,
    sequential_ci,
    sequential_decision,
    sequential_inference,
    sequential_pvalues,
    sprt_bounds,
    sprt_stop_points,
    wilson,
)


# --------------------------------------------------------------------------- SAE
@dataclass
class SAESubset:
    """Encoder columns / decoder rows of the latents that can be ablated; exact for a JumpReLU-at-threshold encoder
    (each latent's activation depends only on its own encoder column)."""
    latent_ids: list[int]
    W_enc: torch.Tensor   # [D, m]
    b_enc: torch.Tensor   # [m]
    W_dec: torch.Tensor   # [m, D]
    b_dec: torch.Tensor   # [D]
    norm_mu: torch.Tensor  # [D]
    norm_scale: "float | torch.Tensor"  # a float, or a [D] tensor for a per-channel normalised SAE
    threshold: float
    meta: dict

    @property
    def d_in(self) -> int:
        return int(self.W_enc.shape[0])

    def pos(self, latents) -> list[int]:
        ix = {l: j for j, l in enumerate(self.latent_ids)}
        missing = [int(l) for l in latents if int(l) not in ix]
        if missing:
            raise KeyError(f"latents {missing} not in the SAE subset {self.latent_ids}")
        return [ix[int(l)] for l in latents]

    def to(self, device) -> "SAESubset":
        f = lambda t: t.to(device=device, dtype=torch.float32)  # noqa: E731
        return SAESubset(list(self.latent_ids), f(self.W_enc), f(self.b_enc), f(self.W_dec), f(self.b_dec),
                         f(self.norm_mu), f(self.norm_scale) if torch.is_tensor(self.norm_scale) else self.norm_scale,
                         self.threshold, dict(self.meta))

    def encode(self, x32: torch.Tensor) -> torch.Tensor:
        """x32 [..., D] raw block output (fp32) -> f [..., m]."""
        xn = (x32 - self.norm_mu) / self.norm_scale
        z = torch.relu((xn - self.b_dec) @ self.W_enc + self.b_enc)
        return z * (z > self.threshold)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_sae(path, latents=None, norm_stats=None) -> SAESubset:
    """Full sae/weights.safetensors or a subset file (has latent_ids). norm_stats (sae/norm_stats.safetensors), if given,
    must equal the copies in the weights file; those are the values used."""
    from safetensors.torch import load_file  # noqa: PLC0415

    d = load_file(str(path))
    for k in SAE_KEYS:
        if k not in d:
            raise KeyError(f"{path}: missing {k}")
    mu, scale = d["norm_mu"].float(), d["norm_scale"].float().reshape(-1)
    if norm_stats is not None:
        n = load_file(str(norm_stats))
        if not (torch.equal(n["norm_mu"].float(), mu) and torch.equal(n["norm_scale"].float().reshape(-1), scale)):
            raise ValueError(f"{norm_stats} != norm_mu / norm_scale in {path}")
    ids = [int(i) for i in d["latent_ids"].tolist()] if "latent_ids" in d else list(range(int(d["W_enc"].shape[1])))
    want = sorted({int(l) for l in (latents if latents is not None else ids)})
    ix = {l: j for j, l in enumerate(ids)}
    sel = torch.tensor([ix[l] for l in want], dtype=torch.long)
    return SAESubset(want, d["W_enc"].float()[:, sel].contiguous(), d["b_enc"].float()[sel].contiguous(),
                     d["W_dec"].float()[sel].contiguous(), d["b_dec"].float().contiguous(), mu.contiguous(),
                     _scale_value(scale, int(d["W_enc"].shape[0]), path), float(d["threshold"].float().reshape(-1)[0]),
                     {"path": str(path), "subset_file": "latent_ids" in d})


def _scale_value(scale: torch.Tensor, d_in: int, path) -> "float | torch.Tensor":
    """norm_scale as stored: one value (the v9 SAEs) -> float; d_in values (ii retrain-sae --norm channel) -> [D] tensor."""
    if scale.numel() == 1:
        return float(scale[0])
    if scale.numel() != d_in:
        raise ValueError(f"{path}: norm_scale has {scale.numel()} values; expected 1 or d_in={d_in}")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError(f"{path}: norm_scale must be finite and positive")
    return scale.contiguous()


def write_subset(full_path, latents, out_path, norm_stats=None) -> str:
    """Save the subset (same keys + latent_ids) and return its sha256."""
    from safetensors.torch import save_file  # noqa: PLC0415

    s = load_sae(full_path, latents, norm_stats)
    t = {"W_enc": s.W_enc, "W_dec": s.W_dec, "b_enc": s.b_enc, "b_dec": s.b_dec, "norm_mu": s.norm_mu,
         "norm_scale": s.norm_scale.float() if torch.is_tensor(s.norm_scale) else torch.tensor([s.norm_scale], dtype=torch.float32),
         "threshold": torch.tensor(s.threshold, dtype=torch.float32),
         "latent_ids": torch.tensor(s.latent_ids, dtype=torch.int64)}
    # one metadata key: safetensors orders multiple metadata keys nondeterministically (the sha would change per run)
    prov = json.dumps({"source": str(full_path), "source_sha256": sha256_file(full_path),
                       "rule": "rows/cols of the full SAE for latent_ids; JumpReLU latents are independent"}, sort_keys=True)
    save_file(t, str(out_path), metadata={"provenance": prov})
    return sha256_file(out_path)


def blob_from_file(p) -> str:
    return base64.b85encode(zlib.compress(Path(p).read_bytes(), 9)).decode("ascii")


def file_from_blob(blob: str, out_path, sha256: str | None = None) -> str:
    raw = zlib.decompress(base64.b85decode(blob.encode("ascii")))
    Path(out_path).write_bytes(raw)
    got = hashlib.sha256(raw).hexdigest()
    if sha256 and got != sha256:
        raise ValueError(f"SAE subset sha256 {got} != plan {sha256}")
    return got


# --------------------------------------------------------------------------- hook editor
TRAINED = "trained"


def cond_dictionary(cond: dict) -> str:
    """The SAE a condition edits: "trained" unless the plan names an untrained "random:<seed>" dictionary."""
    name = cond.get("dictionary") or TRAINED
    if name != TRAINED and not (name.startswith("random:") and name[len("random:"):].isdigit()):
        raise ValueError(f"condition {cond.get('id')!r}: dictionary must be 'trained' or 'random:<seed>', got {name!r}")
    return name


@dataclass(frozen=True)
class Edit:
    """Per-episode edit: sel = SAE-subset positions to ablate (empty = recon-only identity), track = positions whose
    firing is recorded, dictionary = which SAE (trained or random:<seed>). Short repr because Job.meta is printed in
    TIMING lines."""
    cond: str
    sel: tuple
    track: tuple
    dictionary: str = TRAINED

    def __repr__(self) -> str:
        return f"Edit({self.cond}, n_sel={len(self.sel)})"


class LatentAblator:
    """Forward hook on transformer_blocks.{layer}. Rows whose edit is None are left untouched (baseline).

    `sae` is one SAESubset (every edit is on the trained SAE) or {dictionary name: SAESubset}; each row is edited
    through the SAE its Edit.dictionary names."""

    def __init__(self, sae, *, denoise_steps="final", tokens="all", stat_tokens=None):
        if denoise_steps not in ("final", "all"):
            raise ValueError(f"denoise_steps must be 'final' or 'all', got {denoise_steps!r}")
        self.saes_cpu = dict(sae) if isinstance(sae, dict) else {TRAINED: sae}
        if not self.saes_cpu:
            raise ValueError("LatentAblator needs at least one SAE")
        self.saes: dict = {}
        self.denoise_steps = denoise_steps
        self.tokens = tokens
        self.stat_tokens = None if stat_tokens is None else [int(t) for t in stat_tokens]
        self.n_denoise = None
        self.rows = None
        self.calls = 0
        self.edited_calls = 0
        self.stats = None
        self._h = None
        self.block = None

    # wiring
    def install(self, policy, layer: int):
        import physint_inproc_v8 as v8  # noqa: PLC0415

        _n, block, _info = v8.resolve_block(policy, layer)
        self.install_on(block, int(policy._groot_model.action_head.num_inference_timesteps))
        return self

    def install_on(self, block, n_denoise: int):
        self.remove()
        self.block, self.n_denoise = block, int(n_denoise)
        self._h = block.register_forward_hook(self._hook)
        return self

    def remove(self):
        if self._h is not None:
            self._h.remove()
            self._h = None

    def begin(self, rows: list):
        self.rows = list(rows)
        self.calls = self.edited_calls = 0
        self.stats = [None] * len(rows)

    def disarm(self):
        self.rows = None

    def end(self) -> list:
        if self.rows is None:
            raise RuntimeError("LatentAblator.end() without begin()")
        if any(r is not None for r in self.rows):
            if self.calls != self.n_denoise:
                raise RuntimeError(f"edit hook fired {self.calls}x, expected num_inference_timesteps={self.n_denoise}")
            want = self.n_denoise if self.denoise_steps == "all" else 1
            if self.edited_calls != want:
                raise RuntimeError(f"edited {self.edited_calls} denoise steps, expected {want}")
        out, self.rows = self.stats, None
        return out

    # the edit
    def _tok_mask(self, T: int, device) -> torch.Tensor | None:
        if self.tokens == "all":
            return None
        m = torch.zeros(T, dtype=torch.bool, device=device)
        m[[int(t) for t in self.tokens]] = True
        return m

    @torch.no_grad()
    def edit_tensor(self, t: torch.Tensor) -> tuple[torch.Tensor, list]:
        """t [B, T, D] -> (edited t, per-row stats). Pure function of t and self.rows (used by the hook and tests)."""
        if t.ndim != 3 or t.shape[0] != len(self.rows):
            raise RuntimeError(f"edit hook: output {tuple(t.shape)} vs {len(self.rows)} batch rows")
        B, T, _D = t.shape
        x32 = t.float()
        codes = {}  # dictionary -> f [B, T, m], encoded once per call for the dictionaries this batch uses
        for e in self.rows:
            if e is None or e.dictionary in codes:
                continue
            if e.dictionary not in self.saes_cpu:
                raise RuntimeError(f"edit hook: no SAE loaded for dictionary {e.dictionary!r}")
            sae = self.saes.get(e.dictionary)
            if sae is None or sae.norm_mu.device != t.device:
                sae = self.saes[e.dictionary] = self.saes_cpu[e.dictionary].to(t.device)
            if t.shape[-1] != sae.d_in:
                raise RuntimeError(f"edit hook: d_model {t.shape[-1]} != SAE d_in {sae.d_in}")
            codes[e.dictionary] = sae.encode(x32)
        tm = self._tok_mask(T, t.device)
        st_ix = None if self.stat_tokens is None else torch.tensor([i for i in self.stat_tokens if i < T], device=t.device)
        new_rows, stats = [], []
        for b, e in enumerate(self.rows):
            if e is None:
                new_rows.append(t[b])
                stats.append(None)
                continue
            sae, f = self.saes[e.dictionary], codes[e.dictionary]
            sel = torch.tensor(list(e.sel), dtype=torch.long, device=t.device)
            delta = (f[b][:, sel] @ sae.W_dec[sel]) * sae.norm_scale  # [T, D]; zeros when sel is empty
            if tm is not None:
                delta = delta * tm[:, None]
            nb = (x32[b] - delta).to(t.dtype)
            change = float((nb.float() - x32[b]).abs().max())
            if not len(e.sel) and change != 0.0:
                raise RuntimeError(f"recon-only edit changed the block output (max |diff| {change})")
            tr = torch.tensor(list(e.track), dtype=torch.long, device=t.device)
            ft = f[b][:, tr]  # [T, n_track]
            fs = ft if st_ix is None else ft[st_ix]
            xn = x32[b].norm(dim=-1).clamp_min(1e-12)
            stats.append({"fired_stat": (fs > 0).sum(0).tolist(), "f_sum_stat": fs.sum(0).tolist(), "n_stat": int(fs.shape[0]),
                          "fired_all": (ft > 0).sum(0).tolist(), "n_all": int(T),
                          "rel_delta_sum": float((delta.norm(dim=-1) / xn).sum()), "max_abs_change": change})
            new_rows.append(nb)
        return torch.stack(new_rows, 0), stats

    def _hook(self, _m, _i, out):
        if self.rows is None:
            return None
        self.calls += 1
        if self.denoise_steps == "final" and self.calls != self.n_denoise:
            return None
        if all(r is None for r in self.rows):
            return None
        t = out[0] if isinstance(out, (tuple, list)) else out
        new, st = self.edit_tensor(t)
        self.edited_calls += 1
        for b, s in enumerate(st):
            if s is not None:
                self.stats[b] = s if self.stats[b] is None else self.merge(self.stats[b], s)
        if isinstance(out, tuple):
            return (new, *out[1:])
        if isinstance(out, list):
            return [new, *out[1:]]
        return new

    # per-episode accumulation
    @staticmethod
    def merge(a: dict, s: dict) -> dict:
        return {"fired_stat": [x + y for x, y in zip(a["fired_stat"], s["fired_stat"])],
                "f_sum_stat": [x + y for x, y in zip(a["f_sum_stat"], s["f_sum_stat"])], "n_stat": a["n_stat"] + s["n_stat"],
                "fired_all": [x + y for x, y in zip(a["fired_all"], s["fired_all"])], "n_all": a["n_all"] + s["n_all"],
                "rel_delta_sum": a["rel_delta_sum"] + s["rel_delta_sum"], "max_abs_change": max(a["max_abs_change"], s["max_abs_change"]),
                "n_calls": a.get("n_calls", 1) + s.get("n_calls", 1)}

    def accumulate(self, acc, s):
        if s is None:
            return acc
        s = dict(s, n_calls=1)
        return s if acc is None else self.merge(acc, s)

    @staticmethod
    def summarize(acc) -> dict | None:
        if acc is None:
            return None
        ns, na = max(1, acc["n_stat"]), max(1, acc["n_all"])
        return {"n_calls": acc.get("n_calls", 1), "n_tok_stat": acc["n_stat"], "n_tok_all": acc["n_all"],
                "fired_stat": acc["fired_stat"], "fire_frac_stat": [round(x / ns, 5) for x in acc["fired_stat"]],
                "fire_frac_all": [round(x / na, 5) for x in acc["fired_all"]],
                "mean_f_when_fired_stat": [round(f / c, 5) if c else None for f, c in zip(acc["f_sum_stat"], acc["fired_stat"])],
                "mean_rel_delta": round(acc["rel_delta_sum"] / na, 6), "max_abs_change": acc["max_abs_change"]}


# --------------------------------------------------------------------------- plan
def _arm_dir(arm: str) -> str:
    return {"A": "arm-A", "B": "arm-B", "baseline": "baseline"}[arm]


def default_plan(candidates_path, *, sae_meta: dict, token_indices: list[int], task: dict, picker_commit: str,
                 recon_n: int = 8, cap: int = 75, baseline_n: int = 75) -> dict:
    cf = json.loads(Path(candidates_path).read_text())
    ctrl = {int(k): [int(x) for x in v] for k, v in cf["matched_random_controls"].items()}
    a_ctrl, b1, b2 = ctrl[720], ctrl[7081], ctrl[5852]
    if not (len(a_ctrl) == len(b1) == len(b2) == 10):
        raise ValueError("expected 10 matched controls per candidate")
    conds = [{"id": "baseline", "arm": "baseline", "kind": "baseline", "latents": None, "n_max": baseline_n, "sprt": False,
              "dir": "baseline"},
             {"id": "recon-only", "arm": "baseline", "kind": "recon_only", "latents": [], "n_max": recon_n, "sprt": False,
              "dir": "baseline/recon-only"},
             {"id": "A-cand-720", "arm": "A", "kind": "candidate", "latents": [720], "n_max": cap, "sprt": True, "dir": "arm-A/cand-720"},
             {"id": "B-cand-7081+5852", "arm": "B", "kind": "candidate", "latents": [7081, 5852], "n_max": cap, "sprt": True,
              "dir": "arm-B/cand-7081+5852"}]
    for j in range(10):
        conds.append({"id": f"A-ctrl{j + 1:02d}-{a_ctrl[j]}", "arm": "A", "kind": "control", "latents": [a_ctrl[j]], "n_max": cap,
                      "sprt": True, "dir": f"arm-A/ctrl{j + 1:02d}-{a_ctrl[j]}", "control_for": [720], "control_rank": j + 1})
        conds.append({"id": f"B-ctrl{j + 1:02d}-{b1[j]}+{b2[j]}", "arm": "B", "kind": "control", "latents": [b1[j], b2[j]],
                      "n_max": cap, "sprt": True, "dir": f"arm-B/ctrl{j + 1:02d}-{b1[j]}+{b2[j]}", "control_for": [7081, 5852],
                      "control_rank": j + 1})
    # dispatch order within a round: baseline, recon-only, both candidates, then control 1 of A and B, control 2...
    order = [c["id"] for c in conds]
    return {
        "schema": PLAN_SCHEMA, "version": ABLATE_VERSION, "unvalidated": True, "real": False,
        "run_id": "groot-1028-v9-ablate", "task": task,
        "hook": {"module": "_groot_model.action_head.model.transformer_blocks.5", "layer": 5,
                 "denoise_steps": "final", "tokens": "all", "stat_tokens": token_indices,
                 "rule": "out = x - norm_scale * sum_{i in S} f_i W_dec[i]; f = JumpReLU_thr(((x-norm_mu)/norm_scale - b_dec) W_enc + b_enc)",
                 "denoise_steps_note": "SAE trained on the final denoise step (index 3 of 4) only; edits applied there, at every policy call",
                 "tokens_note": "all 41 DiT tokens (state + 40 action tokens); SAE trained on 32 of 41 (stat_tokens), firing stats are on those"},
        "sae": sae_meta,
        "seeds": {"rule": "seed = 40000 + i; init_state_id = i % n_init_states", "base": 40000,
                  "noise": "v9 per-episode torch.Generator(seed) routed into GR00T sample_noise"},
        "sprt": {"p0": 0.56, "p1_rise": 0.86, "p1_fall": 0.26, "alpha": 0.05, "beta": 0.05, "cap": cap,
                 "stop_rule": "h1_or_both_h0",
                 "stop_rule_note": ("stop when rise or fall accepts H1, or both accept H0 (Sobel-Wald), or at cap. The literal "
                                    "'stop when either test decides' can never declare a rise with these p0/p1/alpha/beta "
                                    "(the fall test always accepts H0 first); see ablate_v9.sequential_decision"),
                 "source": "brief defaults: no p0/p1/alpha/beta given in Andrej's picker_onset_pot0 spec or candidates_final.json",
                 "p0_note": "0.56 = capture success rate 56/100 on seeds 20000-20099; the baseline condition measures it on 40000+i"},
        "labels": "pot_events rule pot0-v1 via physint_inproc_v9.label_episode",
        "conditions": conds,
        "schedule": {"rule": "lockstep: each free env slot gets the next episode index of the active condition with the fewest "
                             "episodes dispatched (ties: `order`); a condition stops dispatching once its SPRT stops or n_max is reached",
                     "order": order},
        "provenance": {"candidates_final": str(candidates_path), "candidates_final_sha256": sha256_file(candidates_path),
                       "picker": "vjepa-interp/pick_groot_pot0.py", "picker_commit": picker_commit,
                       "arm_B_control_pairing": "B control j = (matched_random_controls[7081][j], matched_random_controls[5852][j])"},
    }


def plan_sha(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------- state / logs
class AblationState:
    """Per-condition append-only logs + lockstep scheduler + SPRT. No torch / env needed (unit-testable)."""

    MAX_ERRORS = 3

    def __init__(self, plan: dict, out: Path, *, only: list[str] | None = None, log: Callable[[str], None] = print):
        self.plan, self.out, self.log = plan, Path(out), log
        self.sha = plan_sha(plan)
        self.conds = {c["id"]: c for c in plan["conditions"]}
        self.order = list(plan["schedule"]["order"])
        self.only = only
        self.rows: dict[str, dict[int, dict]] = {c: {} for c in self.conds}
        self.inflight: dict[str, set] = {c: set() for c in self.conds}
        self.errors: dict[tuple, int] = {}
        self.dups = 0
        self.out.mkdir(parents=True, exist_ok=True)
        pp = self.out / "plan.json"
        if pp.is_file():
            old = json.loads(pp.read_text())
            if plan_sha(old) != self.sha:
                raise RuntimeError(f"{pp} is a different plan (sha {plan_sha(old)[:12]} != {self.sha[:12]}); use a new out dir")
        else:
            _atomic_json(pp, plan)
        _atomic_json(self.out / "schedule.json", {"plan_sha256": self.sha, **plan["schedule"],
                                                  "positions": {c: i for i, c in enumerate(self.order)}})
        for cid in self.conds:
            self._open(cid)

    def path(self, cid: str) -> Path:
        return self.out / self.conds[cid]["dir"] / "episodes.jsonl"

    def _open(self, cid: str):
        p = self.path(cid)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_file():
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last write after a disconnect
                if r.get("type") == "header":
                    if r.get("plan_sha256") != self.sha:
                        raise RuntimeError(f"{p}: header plan_sha256 differs from this plan")
                    continue
                if r.get("type") != "episode":
                    continue
                if int(r["ep"]) in self.rows[cid]:
                    self.dups += 1
                    continue
                self.rows[cid][int(r["ep"])] = r
        if not p.is_file() or not p.read_text().strip():
            c = self.conds[cid]
            self._append(p, {"type": "header", "cond": cid, "plan_sha256": self.sha, "version": ABLATE_VERSION,
                             "arm": c["arm"], "kind": c["kind"], "latents": c["latents"], "n_max": c["n_max"], "sprt": c["sprt"],
                             "sprt_params": self.plan["sprt"] if c["sprt"] else None, "seeds": self.plan["seeds"],
                             "schedule_position": self.order.index(cid), "schedule_order": self.order,
                             "schedule_rule": self.plan["schedule"]["rule"], "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")})

    @staticmethod
    def _append(p: Path, row: dict):
        with open(p, "ab") as f:
            f.seek(0, os.SEEK_END)
            if f.tell():  # never glue a row onto a torn line
                with open(p, "rb") as g:
                    g.seek(-1, os.SEEK_END)
                    if g.read(1) != b"\n":
                        f.write(b"\n")
            f.write((json.dumps(row, default=str) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())

    # decisions
    def prefix(self, cid: str) -> list[bool]:
        out, i, r = [], 0, self.rows[cid]
        while i in r:
            out.append(bool(r[i]["success"]))
            i += 1
        return out

    def decision(self, cid: str) -> dict | None:
        return sequential_decision(self.prefix(cid), self.plan["sprt"]) if self.conds[cid]["sprt"] else None

    def used(self, cid: str) -> dict[int, bool]:
        c = self.conds[cid]
        if c["sprt"]:
            n = self.decision(cid)["n_used"]
            return {i: bool(self.rows[cid][i]["success"]) for i in range(n)}
        return {i: bool(r["success"]) for i, r in self.rows[cid].items() if i < c["n_max"]}

    def finished(self, cid: str) -> bool:
        c = self.conds[cid]
        if c.get("_failed"):
            return True
        if c["sprt"]:
            return self.decision(cid)["stopped"]
        return all(i in self.rows[cid] for i in range(c["n_max"]))

    def _selected(self, cid: str) -> bool:
        if not self.only:
            return True
        c = self.conds[cid]
        return cid in self.only or _arm_dir(c["arm"]) in self.only or c["arm"] in self.only

    def next_index(self, cid: str) -> int | None:
        c = self.conds[cid]
        if not self._selected(cid) or self.finished(cid):
            return None
        for i in range(c["n_max"]):
            if i not in self.rows[cid] and i not in self.inflight[cid]:
                return i
        return None

    def next(self) -> tuple[str, int] | None:
        best = None
        for pos, cid in enumerate(self.order):
            i = self.next_index(cid)
            if i is None:
                continue
            key = (len(self.rows[cid]) + len(self.inflight[cid]), pos)
            if best is None or key < best[0]:
                best = (key, cid, i)
        if best is None:
            return None
        _k, cid, i = best
        self.inflight[cid].add(i)
        return cid, i

    def record(self, cid: str, row: dict) -> dict:
        i = int(row["ep"])
        self.inflight[cid].discard(i)
        if i in self.rows[cid]:
            self.dups += 1
            return self.rows[cid][i]
        dec_before = self.decision(cid)
        row = {"type": "episode", "cond": cid, "plan_sha256": self.sha, **row}
        # informative only (summary.json recomputes from the prefix): finished after this condition had already stopped
        row["after_stop"] = bool(dec_before is not None and dec_before["stopped"] and i >= dec_before["n_used"])
        self._append(self.path(cid), row)
        self.rows[cid][i] = row
        return row

    def error(self, cid: str, i: int, err: str):
        self.inflight[cid].discard(i)
        k = (cid, i)
        self.errors[k] = self.errors.get(k, 0) + 1
        self._append(self.out / "errors.jsonl", {"cond": cid, "ep": i, "n": self.errors[k], "error": err[-4000:],
                                                 "t": time.strftime("%Y-%m-%d %H:%M:%S")})
        if self.errors[k] >= self.MAX_ERRORS:  # stop this condition rather than skip an index (would bias the prefix)
            self.conds[cid]["_failed"] = f"episode {i} failed {self.errors[k]}x"
            self.log(f"[ablate] {cid} STOPPED: {self.conds[cid]['_failed']}")

    # summaries
    def summary(self, cid: str) -> dict:
        c = self.conds[cid]
        used = self.used(cid)
        rows = [self.rows[cid][i] for i in sorted(used)]
        n, k = len(used), sum(used.values())
        kinds: dict[str, int] = {}
        for r in rows:
            if not r["success"]:
                kinds[str(r.get("fail_kind"))] = kinds.get(str(r.get("fail_kind")), 0) + 1
        s = {"cond": cid, "arm": c["arm"], "kind": c["kind"], "latents": c["latents"], "dir": c["dir"], "plan_sha256": self.sha,
             "n": n, "successes": k, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n),
             "fail_kind_counts": dict(sorted(kinds.items())),
             "n_logged": len(self.rows[cid]), "n_logged_not_used": len(self.rows[cid]) - n, "inflight": sorted(self.inflight[cid]),
             "n_max": c["n_max"], "schedule_position": self.order.index(cid), "failed": c.get("_failed"),
             "finished": self.finished(cid)}
        s["sprt"] = self.decision(cid) if c["sprt"] else {"none": "fixed n, no SPRT" if c["kind"] == "baseline" else "sanity check, no SPRT"}
        s["sequential"] = self.sequential(cid)
        if cid != "baseline" and "baseline" in self.conds:
            base = self.used("baseline")
            bn, bk = len(base), sum(base.values())
            s["vs_baseline"] = {"paired_shared_seeds": paired_delta(used, base),
                                "unpaired": {"baseline_n": bn, "baseline_rate": round(bk / bn, 4) if bn else None,
                                             "delta": round(k / n - bk / bn, 4) if n and bn else None}}
        es = [r.get("edit_stats") for r in rows if r.get("edit_stats")]
        if es and c["latents"] is not None:
            track = es[0].get("track_latents")
            nst = sum(e["n_tok_stat"] for e in es)
            fired = np.sum([e["fired_stat"] for e in es], 0)
            s["edit"] = {"track_latents": track, "live_fire_frac_stat_tokens": [round(float(x) / max(1, nst), 5) for x in fired],
                         "mean_rel_delta": round(float(np.mean([e["mean_rel_delta"] for e in es])), 6),
                         "max_abs_change": max(e["max_abs_change"] for e in es), "n_calls": int(sum(e["n_calls"] for e in es))}
        if c["kind"] == "recon_only":
            b = self.rows.get("baseline", {})
            cmp = [(r, b[i]) for i, r in self.rows[cid].items() if i in b]
            s["recon_check"] = {"n_compared_with_baseline": len(cmp),
                                "n_identical_actions": sum(r.get("actions_sha256") == q.get("actions_sha256") for r, q in cmp),
                                "n_same_success": sum(bool(r["success"]) == bool(q["success"]) for r, q in cmp),
                                "hook_max_abs_change": max((r["edit_stats"]["max_abs_change"] for r in self.rows[cid].values()
                                                            if r.get("edit_stats")), default=None)}
        return s

    def sequential(self, cid: str) -> dict | None:
        """Exact early-stopping-valid interval and p-values for an SPRT condition that has stopped (None otherwise: a
        running or failed condition is not at a stopping point of the design, and fixed-n conditions keep Wilson)."""
        c = self.conds[cid]
        if not c["sprt"] or c.get("_failed"):
            return None
        d = self.decision(cid)
        if not d["stopped"]:
            return None
        return sequential_inference(d["n_used"], sum(self.used(cid).values()), self.plan["sprt"])

    def write_summaries(self, extra: dict | None = None) -> dict:
        per = {}
        for cid in self.order:
            s = self.summary(cid)
            _atomic_json(self.out / self.conds[cid]["dir"] / "summary.json", s)
            per[cid] = s
        top = {"plan_sha256": self.sha, "version": ABLATE_VERSION, "unvalidated": True, "real": False,
               "updated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "schedule_order": self.order,
               "episodes_logged": sum(len(r) for r in self.rows.values()), "duplicates_ignored": self.dups,
               "table": [{k: per[c].get(k) for k in ("cond", "n", "successes", "rate", "wilson95")}
                         | {"verdict": (per[c]["sprt"] or {}).get("verdict"),
                            "delta_paired": ((per[c].get("vs_baseline") or {}).get("paired_shared_seeds") or {}).get("delta"),
                            "n_shared": ((per[c].get("vs_baseline") or {}).get("paired_shared_seeds") or {}).get("n_shared")}
                         | table_sequential_columns(per[c].get("sequential"))
                         for c in self.order],
               **(extra or {})}
        top["arms"] = arms_block(top["table"], self.conds, self.plan["sprt"])
        _atomic_json(self.out / "summary.json", top)
        return top


def table_sequential_columns(seq: dict | None) -> dict:
    """The columns the top-level table carries next to wilson95."""
    if not seq:
        return {"seq95": None, "p_seq_rise": None, "p_seq_fall": None}
    return {"seq95": seq["ci95"], "p_seq_rise": seq["p_vs_p0"]["rise"], "p_seq_fall": seq["p_vs_p0"]["fall"]}


def arms_block(table: list[dict], conds: dict[str, dict], sprt: dict, alpha: float = 0.05) -> dict | None:
    """(b) + (c) from the top-level table rows, once every candidate and control has stopped (else None)."""
    rows = {r["cond"]: r for r in table}
    need = [c for c, d in conds.items() if d["kind"] in ("candidate", "control")]
    if any(rows.get(c, {}).get("seq95") is None for c in need):
        return None
    per = {c: {"n": rows[c]["n"], "successes": rows[c]["successes"], "verdict": rows[c]["verdict"],
               "sequential": {"p_vs_p0": {"rise": rows[c]["p_seq_rise"], "fall": rows[c]["p_seq_fall"]}}} for c in need}
    return arm_inference(per, {c: conds[c] for c in need}, alpha) | {
        "sequential_method": "exact, under the plan's SPRT stopping rule; seq95 = equal-tailed 95% interval, p_seq_* vs p0",
        "p0": float(sprt["p0"]), "sets_verdicts": False}


def threshold_crossings(top: dict, alpha: float = 0.05) -> dict:
    """What, in a finished summary, would bear on a rise finding (research rules 1-3) and what is merely below alpha.

    "rise" lists anything pointing toward a finding: an SPRT rise verdict on a candidate, a Holm-adjusted sequential rise
    p < alpha, a permutation rise p <= alpha, or a candidate beating every matched control. "below_alpha" lists every
    other p-value under alpha (falls), which do not bear on real=true. Nothing here changes a verdict."""
    rise, below = [], []
    arms = top.get("arms") or {}
    for c, d in (arms.get("candidates") or {}).items():
        if d["verdict"] == "rise":
            rise.append(f"{c}: SPRT verdict rise")
        if d["holm"]["p_seq_rise"] < alpha:
            rise.append(f"{c}: Holm sequential rise p {d['holm']['p_seq_rise']}")
        if d["vs_controls"]["p_perm_rise"] <= alpha:
            rise.append(f"{c}: permutation rise p {d['vs_controls']['p_perm_rise']}")
        if d["vs_controls"]["beats_every_control_rise"]:
            rise.append(f"{c}: beats every matched control (rise direction)")
        for fam, v in d["holm"].items():
            if "fall" in fam and v < alpha:
                below.append(f"{c}: Holm {fam} {v}")
    for r in top.get("table", []):
        for k in ("p_seq_rise", "p_seq_fall"):
            if r.get(k) is not None and r[k] < alpha:
                (rise if ("cand" in r["cond"] and k == "p_seq_rise") else below).append(f"{r['cond']}: {k} {r[k]} (verdict {r['verdict']})")
    return {"alpha": alpha, "rise": rise, "below_alpha": below}


def _atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


# --------------------------------------------------------------------------- runner (Colab / box mock)
def make_edit(cid: str, cond: dict, sae) -> Edit | None:
    """`sae`: the trained SAESubset, or {dictionary: SAESubset} when the plan has random-weights SAE conditions."""
    if cond["latents"] is None:
        return None  # baseline: hook leaves the block output untouched
    name = cond_dictionary(cond)
    s = sae[name] if isinstance(sae, dict) else sae
    if not isinstance(sae, dict) and name != TRAINED:
        raise ValueError(f"condition {cid!r} edits {name}, but only the trained SAE was loaded")
    sel = tuple(s.pos(cond["latents"]))
    track = sel if sel else tuple(range(len(s.latent_ids)))  # recon-only: firing stats for every subset latent
    return Edit(cid, sel, track, name)


def load_dictionaries(plan: dict, sae_path, dictionaries: dict | None = None, norm_stats=None) -> dict:
    """{dictionary: SAESubset} for every dictionary the plan's conditions edit, each restricted to its latents.

    The trained SAE comes from `sae_path` (checked against the plan's subset or full sha256, as before). An untrained
    "random:<seed>" SAE comes from dictionaries[name], checked against plan["sae"]["dictionaries"][name]["subset_sha256"]."""
    by_dict: dict[str, set] = {}
    for c in plan["conditions"]:
        if c["latents"] is None:
            continue
        by_dict.setdefault(cond_dictionary(c), set()).update(int(l) for l in c["latents"])
    by_dict.setdefault(TRAINED, set())
    declared = (plan["sae"].get("dictionaries") or {})
    out = {}
    for name, lat in sorted(by_dict.items()):
        if name == TRAINED:
            path = sae_path
            allowed = {plan["sae"].get("subset_sha256"), plan["sae"].get("full_sha256")} - {None}
            if TRAINED in declared:
                allowed.add(declared[TRAINED].get("subset_sha256"))
                allowed.discard(None)
        else:
            if not dictionaries or name not in dictionaries:
                raise ValueError(f"plan edits {name}; pass its subset file in dictionaries={{{name!r}: path}}")
            if name not in declared:
                raise ValueError(f"plan edits {name} but plan['sae']['dictionaries'] does not declare it")
            path = dictionaries[name]
            allowed = {declared[name]["subset_sha256"]}
        sha = sha256_file(path)
        if plan["sae"].get("check_sha256", True) and sha not in allowed:
            raise ValueError(f"SAE file {path} sha256 {sha} does not match the plan for {name}")
        out[name] = load_sae(path, sorted(lat) or None, norm_stats if name == TRAINED else None)
        if plan["sae"].get("d_in") and out[name].d_in != int(plan["sae"]["d_in"]):
            raise ValueError(f"SAE {name} d_in {out[name].d_in} != plan {plan['sae']['d_in']}")
    return out


def run_ablation(ctx, plan: dict, *, sae_path, norm_stats=None, only: list[str] | None = None,
                 dictionaries: dict | None = None) -> dict:
    """ctx: physint_inproc_v9.Ctx with the live policy / processors (ctx.run = output dir). Resumable.

    dictionaries: {"random:<seed>": subset file} for plans with random-weights SAE conditions."""
    import physint_inproc_v8 as v8  # noqa: PLC0415
    import physint_inproc_v9 as v9  # noqa: PLC0415

    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan schema {plan.get('schema')!r} != {PLAN_SCHEMA}")
    out = Path(ctx.run)
    log = ctx.log
    saes = load_dictionaries(plan, sae_path, dictionaries, norm_stats)
    hk = plan["hook"]
    if int(hk["layer"]) != int(ctx.layer):
        raise ValueError(f"plan hook layer {hk['layer']} != ctx.layer {ctx.layer}")
    abl = LatentAblator(saes, denoise_steps=hk["denoise_steps"], tokens=hk["tokens"], stat_tokens=hk.get("stat_tokens"))
    st = AblationState(plan, out, only=only, log=log)
    task = plan["task"]
    key = ("plus", int(task["index"]))
    n_init = int(task["n_init_states"])
    if os.environ.get("GROOT_ABLATE_CHECK_NINIT", "1") == "1":
        got = v9.worker_query_n_init(key)
        if got != n_init:
            raise RuntimeError(f"task {task} has {got} init states, plan says {n_init}")
    base = int(plan["seeds"]["base"])
    edits = {cid: make_edit(cid, c, saes) for cid, c in st.conds.items()}
    track_ids = {cid: ([saes[e.dictionary].latent_ids[j] for j in e.track] if e is not None else None)
                 for cid, e in edits.items()}
    t0 = time.time()
    n_new = [0]

    def next_job():
        nx = st.next()
        if nx is None:
            return None
        cid, i = nx
        return v9.Job(key=key, seed=base + i, init_state_id=i % n_init, meta={"cond": cid, "ep": i, "edit": edits[cid]})

    def on_done(job, ep):
        cid, i = job.meta["cond"], int(job.meta["ep"])
        fr = v9.label_episode(ep, ep.task_name)
        acts = np.ascontiguousarray(np.asarray(ep.actions, np.float32))
        es = getattr(ep, "edit_stats", None)
        if es is not None:
            es = dict(es, track_latents=track_ids[cid])
        row = {"ep": i, "seed": job.seed, "init_state_id": job.init_state_id, "success": bool(ep.success),
               "fail_kind": fr["fail_kind"], "fail_t": fr["fail_t"], **{k: fr["extra"].get(k) for k in
               ("fail_kind_active", "fail_t_active", "fail_subkind", "fail_pot", "fail_rule", "pot0_lift_t", "pot0_place_t")},
               "n_steps": int(ep.n_steps), "n_predict": int(ep.timing.get("n_predict", 0)), "seconds": ep.seconds, "task_name": ep.task_name,
               "latents": st.conds[cid]["latents"], "dictionary": cond_dictionary(st.conds[cid]), "edit_stats": es,
               "actions_sha256": hashlib.sha256(acts.tobytes()).hexdigest(), "first_action": ep.first_action,
               "amp_mode": ctx.amp_mode, "runner": v9.V9_VERSION, "ablate": ABLATE_VERSION,
               "t": time.strftime("%Y-%m-%d %H:%M:%S")}
        simd = out / st.conds[cid]["dir"] / "sim"
        simd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(simd / f"ep-{i:05d}.npz", actions=acts, **{k: v for k, v in ep.sim.items()})
        row = st.record(cid, row)
        s = st.summary(cid)
        st.write_summaries({"pool_episodes_this_session": n_new[0] + 1, "session_wall_s": round(time.time() - t0, 1)})
        dec = s["sprt"].get("verdict") if isinstance(s["sprt"], dict) else None
        pd = (s.get("vs_baseline") or {}).get("paired_shared_seeds") or {}
        print(f"EPISODE ablate cond={cid} ep={i} seed={job.seed} success={ep.success} fail_kind={fr['fail_kind']} after_stop={row['after_stop']} "
              f"n={s['n']} k={s['successes']} rate={s['rate']} sprt={dec} delta_paired={pd.get('delta')} (n_shared={pd.get('n_shared')}) "
              f"n_steps={ep.n_steps} s={ep.seconds}", flush=True)
        n_new[0] += 1
        if n_new[0] % int(getattr(ctx, "snapshot_every", 10)) == 0:
            v8.write_snapshot(ctx, f"ablate_n{n_new[0]}")

    def on_error(job, err):
        log(f"[ablate] ERROR {job.meta} {err[:500]}")
        st.error(job.meta["cond"], int(job.meta["ep"]), err)

    def keep_going():
        if ctx.hours_left() < 0.25:
            log("[ablate] stop: soft cap")
            return False
        return True

    log(f"[ablate] start plan_sha={st.sha[:12]} out={out} conditions={len(st.order)} only={only} "
        f"resumed_episodes={sum(len(r) for r in st.rows.values())} denoise_steps={hk['denoise_steps']} tokens={hk['tokens']}")
    stats = v9.run_pool(ctx, [], next_job=next_job, editor=abl, on_done=on_done, on_error=on_error, capture=False,
                        keep_going=keep_going, label="ablate")
    top = st.write_summaries({"pool": stats.as_dict(), "session_wall_s": round(time.time() - t0, 1),
                              "all_finished": all(st.finished(c) for c in st.order if st._selected(c))})
    v8.write_snapshot(ctx, "ablate_final")
    log(f"[ablate] done {json.dumps(top['table'], default=str)[:3000]}")
    return top
