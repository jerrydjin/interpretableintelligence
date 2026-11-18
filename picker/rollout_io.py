"""rollout_io: the ONLY file in the candidate picker that knows robot-policy run-folder formats.

If Jeff's capture writes something different, fix it here (aliases / one reader); pick_candidates.py only sees the
canonical objects below.

Canonical objects
-----------------
RolloutRun(run_dir)
  .episodes : dict of np arrays, one entry per episode (row order = internal episode index e = 0..E-1)
      episode_idx int64 (id used in the acts shards), episode_id str, task_id str, seed int64,
      success bool, fail_t float (NaN if unknown/success), n_steps int64
  .d_model, .sae (PolicySAE or None), .warnings (list[str])
  .chunks(max_tokens, episodes=None) -> iterator of Chunk, rows grouped so no (episode, step) spans two chunks
  .actions() -> (e int64 [S], step int64 [S], action float32 [S, A]) or None

Chunk: x float32 torch [n, D] or None, f float32 torch [n, M] (SAE codes), e int64 np [n] (internal episode index),
       step int64 np [n], tok int64 np [n]

Formats read (proposed capture format first; see PICKER.md):
  episodes : manifest.json["episodes"] | episodes.jsonl | episodes.json | rollouts/episodes.parquet (baseline rows) |
             episodes.parquet ; column names via EP_ALIASES
  acts     : acts/shard-*.safetensors with acts [N,D] + per-row episode/step/token columns (ROW_ALIASES); this is
             V-JEPA schema v0 with clip_idx->episode, t->step
  codes    : features/feature_acts.parquet (sparse SAE codes, V-JEPA features.py layout) when there are no acts shards
             (no offline ablation possible without raw acts unless the downstream fn only needs codes)
  actions  : actions.safetensors | actions/shard-*.safetensors (episode/step/action) | an 'action' column in acts shards
  sae      : sae/weights.safetensors + sae/config.json (sae.py layout: W_enc [D,M], b_enc, W_dec [M,D], b_dec, threshold,
             norm_mu, norm_scale, optional pos_mean indexed by token position)
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, torch

ROW_ALIASES = dict(episode=("episode_idx", "ep_idx", "episode", "clip_idx", "clip_id", "ep"),
                   step=("step", "timestep", "t_step", "env_step", "t", "frame"),
                   tok=("tok", "token", "token_idx", "pos", "position"),
                   acts=("acts", "activations", "resid", "x", "hidden"),
                   action=("action", "actions", "act_out"))
EP_ALIASES = dict(episode_idx=("episode_idx", "ep_idx", "clip_id", "clip_idx", "episode", "idx"),
                  episode_id=("episode_id", "name", "id", "uid"),
                  task_id=("task_id", "task", "task_name"),
                  seed=("seed",),
                  success=("success", "succeeded", "is_success", "done_success"),
                  fail_t=("fail_t", "failure_step", "fail_step", "failure_t", "failure_timestep", "t_fail", "fail_timestep"),
                  n_steps=("n_steps", "steps", "length", "T", "episode_len"),
                  split=("split",))


def _pick(keys, aliases):
    for a in aliases:
        if a in keys: return a
    return None


class PolicySAE:
    """sae.py / sae_lib.py layout. pos_mean (if present) is indexed by token position `tok` here."""
    def __init__(self, sae_dir: Path):
        from safetensors.torch import load_file
        w = load_file(str(sae_dir / "weights.safetensors"))
        cfg_p = sae_dir / "config.json"; self.cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        self.W_enc, self.b_enc, self.W_dec, self.b_dec = (w["W_enc"].float(), w["b_enc"].float(), w["W_dec"].float(), w["b_dec"].float())
        self.thr = float(w["threshold"]) if "threshold" in w else 0.0
        self.mu = w["norm_mu"].float() if "norm_mu" in w else torch.zeros(self.W_dec.shape[1])
        s = w["norm_scale"].float().reshape(-1) if "norm_scale" in w else torch.ones(1)
        if s.numel() not in (1, w["W_enc"].shape[0]):
            raise ValueError(f"{sae_dir}: norm_scale has {s.numel()} values; expected 1 or d_in={w['W_enc'].shape[0]}")
        # One value (v9 SAEs) -> float; d_in values (ii retrain-sae --norm channel) -> per-channel [D] tensor.
        self.s = float(s[0]) if s.numel() == 1 else s
        self.pm = w["pos_mean"].float() if "pos_mean" in w else None
        self.M, self.D = self.W_dec.shape

    def _norm(self, x, tok):
        if self.pm is not None: x = x - self.pm[torch.as_tensor(tok).long() % self.pm.shape[0]]
        return (x - self.mu) / self.s

    def encode(self, x, tok=None, cols=None):
        """raw x [n,D] -> codes [n,M] (or [n,len(cols)]). JumpReLU at the BatchTopK inference threshold (per feature,
        so encoding a column subset is exact)."""
        W, b = (self.W_enc, self.b_enc) if cols is None else (self.W_enc[:, cols], self.b_enc[cols])
        z = torch.relu((self._norm(x, tok) - self.b_dec) @ W + b); return z * (z > self.thr)

    def raw_dir(self, fids):
        """raw-space decoder contribution per unit activation: x_edit = x - f_i * raw_dir[i] (error + pos_mean untouched)."""
        return self.W_dec[torch.as_tensor(fids).long()] * self.s


def _read_table(path: Path) -> dict:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pydict()
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        obj = json.loads(path.read_text()); rows = obj["episodes"] if isinstance(obj, dict) else obj
    keys = sorted({k for r in rows for k in r})
    return {k: [r.get(k) for r in rows] for k in keys}


class RolloutRun:
    def __init__(self, run_dir, fail_t_scale: float = 1.0):
        self.dir = Path(run_dir); self.warnings = []
        self.man = json.loads((self.dir / "manifest.json").read_text()) if (self.dir / "manifest.json").exists() else {}
        self.shards = sorted((self.dir / "acts").glob("shard-*.safetensors"))
        self.codes_parquet = self.dir / "features" / "feature_acts.parquet"
        self.sae = PolicySAE(self.dir / "sae") if (self.dir / "sae" / "weights.safetensors").exists() else None
        if not self.shards and not self.codes_parquet.exists():
            raise FileNotFoundError(f"{self.dir}: no acts/shard-*.safetensors and no features/feature_acts.parquet")
        if self.shards and self.sae is None:
            raise FileNotFoundError(f"{self.dir}: acts shards but no sae/weights.safetensors (train the SAE first)")
        self.d_model = int(self.man.get("d_model") or (self.sae.D if self.sae else 0))
        self._row_keys = self._shard_keys()
        self.episodes = self._load_episodes(fail_t_scale)
        self._e_of = {int(v): i for i, v in enumerate(self.episodes["episode_idx"])}

    # ---------- episodes ----------
    def _episode_table(self):
        for src in ("manifest:episodes", "episodes.jsonl", "episodes.json", "rollouts/episodes.parquet", "episodes.parquet"):
            if src == "manifest:episodes":
                if isinstance(self.man.get("episodes"), list) and self.man["episodes"]:
                    rows = self.man["episodes"]; keys = sorted({k for r in rows for k in r})
                    return src, {k: [r.get(k) for r in rows] for k in keys}
                continue
            p = self.dir / src
            if p.exists():
                t = _read_table(p)
                if "condition" in t:  # physint PR#5 rollouts table: keep the unedited policy only
                    keep = [i for i, c in enumerate(t["condition"]) if c == "baseline"]
                    t = {k: [v[i] for i in keep] for k, v in t.items()}
                    self.warnings.append(f"{src}: using {len(keep)} baseline rows only")
                return src, t
        raise FileNotFoundError(f"{self.dir}: no episode metadata (manifest.json 'episodes', episodes.jsonl/json/parquet)")

    def _load_episodes(self, fail_t_scale):
        src, t = self._episode_table(); keys = set(t); n = len(next(iter(t.values())))
        col = {k: _pick(keys, a) for k, a in EP_ALIASES.items()}
        if col["success"] is None: raise KeyError(f"{src}: no success column (looked for {EP_ALIASES['success']})")
        ep = {}
        ep["episode_idx"] = np.array(t[col["episode_idx"]] if col["episode_idx"] else range(n), dtype=np.int64)
        ep["episode_id"] = np.array([str(v) for v in (t[col["episode_id"]] if col["episode_id"] else ep["episode_idx"])])
        ep["task_id"] = np.array([str(v) for v in (t[col["task_id"]] if col["task_id"] else ["?"] * n)])
        ep["seed"] = np.array([(-1 if v is None else int(v)) for v in (t[col["seed"]] if col["seed"] else [None] * n)], dtype=np.int64)
        ep["success"] = np.array([bool(v) for v in t[col["success"]]])
        ft = t[col["fail_t"]] if col["fail_t"] else [None] * n
        ep["fail_t"] = np.array([np.nan if (v is None or (isinstance(v, float) and math.isnan(v)) or int(v) < 0) else float(v) / fail_t_scale
                                 for v in ft], dtype=np.float64)
        if col["fail_t"] is None: self.warnings.append(f"{src}: no failure-timestep column; failed episodes anchored at their last step")
        ns = t[col["n_steps"]] if col["n_steps"] else [None] * n
        ep["n_steps"] = np.array([-1 if v is None else int(v) for v in ns], dtype=np.int64)
        self.episode_source = src
        return ep

    def fill_n_steps(self, max_step_by_e):
        """called by the picker after the index pass: n_steps = captured steps when metadata lacks it or disagrees.
        (n_steps in metadata may be simulator steps; captured steps are policy-call steps. fail_t must be in captured steps.)"""
        ns = self.episodes["n_steps"]; cap = max_step_by_e + 1
        bad = (ns != cap) & (ns > 0)
        if bad.any(): self.warnings.append(f"{int(bad.sum())} episodes: metadata n_steps != captured steps (e.g. {int(ns[bad][0])} vs {int(cap[bad][0])}); using captured steps. Check fail_t units (--fail-t-scale).")
        self.episodes["n_steps"] = cap

    # ---------- rows ----------
    def _shard_keys(self):
        if not self.shards: return {}
        from safetensors import safe_open
        with safe_open(str(self.shards[0]), framework="np") as fh: keys = set(fh.keys())
        k = {r: _pick(keys, a) for r, a in ROW_ALIASES.items()}
        if k["episode"] is None or k["step"] is None or k["acts"] is None:
            raise KeyError(f"{self.shards[0]}: need acts + episode + step columns, have {sorted(keys)}")
        return k

    def index(self):
        """cheap pass over index columns only -> (episode_idx int64 [N], step int64 [N])"""
        if not self.shards:
            import pyarrow.parquet as pq
            t = pq.read_table(self.codes_parquet, columns=None).to_pydict()
            ek = _pick(set(t), ROW_ALIASES["episode"]); sk = _pick(set(t), ROW_ALIASES["step"])
            return np.asarray(t[ek], np.int64), np.asarray(t[sk], np.int64)
        from safetensors import safe_open
        E, S = [], []
        for sh in self.shards:
            with safe_open(str(sh), framework="np") as fh:
                E.append(fh.get_tensor(self._row_keys["episode"]).astype(np.int64)); S.append(fh.get_tensor(self._row_keys["step"]).astype(np.int64))
        return np.concatenate(E), np.concatenate(S)

    def e_of(self, episode_idx):
        return np.array([self._e_of.get(int(v), -1) for v in episode_idx], dtype=np.int64) if len(episode_idx) < 64 else \
            _vec_map(self._e_of, episode_idx)

    def chunks(self, max_tokens=4096, episodes=None, cols=None, need_x=True):
        """yield Chunk dicts grouped on (episode, step) boundaries. episodes: optional set of internal indices to keep.
        cols: optional list of SAE feature ids to encode (f has len(cols) columns, exact for JumpReLU)."""
        if not self.shards:
            yield from self._parquet_chunks(max_tokens, episodes, cols); return
        from safetensors.numpy import load_file
        k = self._row_keys; carry = None
        keep = None if episodes is None else np.zeros(len(self.episodes["success"]), bool)
        if keep is not None: keep[list(episodes)] = True
        for si, sh in enumerate(self.shards):
            d = load_file(str(sh))
            e = self.e_of(d[k["episode"]].astype(np.int64)); st = d[k["step"]].astype(np.int64)
            tok = d[k["tok"]].astype(np.int64) if k["tok"] else _within_group_rank(e, st)
            m = e >= 0
            if keep is not None: m &= keep[np.maximum(e, 0)]
            x = d[k["acts"]][m]; e, st, tok = e[m], st[m], tok[m]
            o = np.lexsort((tok, st, e)); x, e, st, tok = x[o], e[o], st[o], tok[o]
            if carry is not None:
                x = np.concatenate([carry[0], x]); e = np.concatenate([carry[1], e]); st = np.concatenate([carry[2], st]); tok = np.concatenate([carry[3], tok])
                o = np.lexsort((tok, st, e)); x, e, st, tok = x[o], e[o], st[o], tok[o]
            last = si == len(self.shards) - 1
            if not last and len(e):  # hold back the final (e, step) group: it may continue in the next shard
                cut = np.searchsorted(e * (1 << 32) + st, e[-1] * (1 << 32) + st[-1])
                carry = (x[cut:], e[cut:], st[cut:], tok[cut:]); x, e, st, tok = x[:cut], e[:cut], st[:cut], tok[:cut]
            else:
                carry = None
            yield from self._emit(x, e, st, tok, max_tokens, cols, need_x)

    def _emit(self, x, e, st, tok, max_tokens, cols, need_x):
        if not len(e): return
        key = e * (1 << 32) + st
        bounds = np.flatnonzero(np.diff(key)) + 1; starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(key)]])
        i = 0
        while i < len(starts):
            j = i
            while j + 1 < len(starts) and ends[j + 1] - starts[i] <= max_tokens: j += 1
            a, b = starts[i], ends[j]
            xt = torch.from_numpy(np.ascontiguousarray(x[a:b]).astype(np.float32))
            f = self.sae.encode(xt, tok[a:b], cols)
            yield dict(x=xt if need_x else None, f=f, e=e[a:b], step=st[a:b], tok=tok[a:b])
            i = j + 1

    def _parquet_chunks(self, max_tokens, episodes, cols):
        import pyarrow.parquet as pq
        t = pq.read_table(self.codes_parquet).to_pydict(); ks = set(t)
        ek, sk, tk = _pick(ks, ROW_ALIASES["episode"]), _pick(ks, ROW_ALIASES["step"]), _pick(ks, ROW_ALIASES["tok"])
        e = self.e_of(np.asarray(t[ek], np.int64)); st = np.asarray(t[sk], np.int64)
        tok = np.asarray(t[tk], np.int64) if tk else np.zeros_like(st)
        fid = np.asarray(t["feature_id"], np.int64); val = np.asarray(t["value"], np.float32)
        M = self.sae.M if self.sae else int(fid.max()) + 1
        m = e >= 0
        if episodes is not None: m &= np.isin(e, list(episodes))
        e, st, tok, fid, val = e[m], st[m], tok[m], fid[m], val[m]
        o = np.lexsort((tok, st, e)); e, st, tok, fid, val = e[o], st[o], tok[o], fid[o], val[o]
        tkey = (e * (1 << 20) + st) * (1 << 12) + tok
        utok, inv = np.unique(tkey, return_inverse=True)
        te, tst, ttok = utok >> 32, (utok >> 12) & ((1 << 20) - 1), utok & ((1 << 12) - 1)
        skey = te * (1 << 32) + tst
        bounds = np.flatnonzero(np.diff(skey)) + 1; starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(skey)]])
        self.warnings.append("codes read from feature_acts.parquet: tokens with zero active features are invisible (firing rates are per active token)")
        i = 0
        while i < len(starts):
            j = i
            while j + 1 < len(starts) and ends[j + 1] - starts[i] <= max_tokens: j += 1
            a, b = starts[i], ends[j]; rm = (inv >= a) & (inv < b)
            F = torch.zeros(b - a, M); F[torch.from_numpy(inv[rm] - a), torch.from_numpy(fid[rm])] = torch.from_numpy(val[rm])
            if cols is not None: F = F[:, cols]
            yield dict(x=None, f=F, e=te[a:b], step=tst[a:b], tok=ttok[a:b])
            i = j + 1

    def actions(self):
        """recorded per-step actions -> (e, step, action [S,A]) or None. First matching source wins."""
        from safetensors.numpy import load_file
        srcs = [self.dir / "actions.safetensors"] + sorted((self.dir / "actions").glob("shard-*.safetensors"))
        parts = [load_file(str(p)) for p in srcs if p.exists()]
        if not parts and self.shards and self._row_keys.get("action"):
            parts = [{k: v for k, v in load_file(str(sh)).items() if k != self._row_keys["acts"]} for sh in self.shards]
        if not parts: return None
        ks = set(parts[0]); ek, sk, ak = _pick(ks, ROW_ALIASES["episode"]), _pick(ks, ROW_ALIASES["step"]), _pick(ks, ROW_ALIASES["action"])
        e = self.e_of(np.concatenate([p[ek] for p in parts]).astype(np.int64)); st = np.concatenate([p[sk] for p in parts]).astype(np.int64)
        a = np.concatenate([p[ak] for p in parts]).astype(np.float32)
        a = a.reshape(len(a), -1)
        key = e * (1 << 32) + st; _, first = np.unique(key, return_index=True)  # one action per (episode, step)
        m = e[first] >= 0; first = first[m]
        return e[first], st[first], a[first]


def _vec_map(d, arr):
    arr = np.asarray(arr, np.int64); u, inv = np.unique(arr, return_inverse=True)
    return np.array([d.get(int(v), -1) for v in u], np.int64)[inv]


def _within_group_rank(e, st):
    o = np.lexsort((st, e)); key = (e * (1 << 32) + st)[o]
    r = np.arange(len(key)) - np.maximum.accumulate(np.where(np.r_[True, key[1:] != key[:-1]], np.arange(len(key)), 0))
    out = np.empty_like(r); out[o] = r; return out
