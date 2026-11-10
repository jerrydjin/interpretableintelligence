"""Candidate picker for robot-policy failure causes (Andrej). Reads policy run folders through rollout_io.py only.

  nice -n 19 env OMP_NUM_THREADS=2 .venv/bin/python -u pick_candidates.py /workspace/runs/groot-* /workspace/runs/pi05-* [flags]
  -> <run>/picker/{candidates.json, contrast.jsonl, offline.jsonl, config.json}

1 FAILURE CONTRAST (discovery episodes). Per captured step s, per feature i: a_i(s) = mean SAE activation over the step's
  tokens, r_i(s) = fraction of tokens firing. Each episode gets one anchor: failed = fail_t (no fail_t -> last step);
  successful = matched (--match time: a failure anchor drawn from the pooled failure anchors, clipped to the episode;
  phase: the same fraction of its own length; end: its last step). Window = the W captured steps before the anchor.
  X[e,i] = window mean of a_i (minus the label-free per-time-bin mean over all episodes if --detrend, default on).
  Effect d = (mean_fail - mean_succ) / pooled SD (Cohen's d; d_fire is the same on r_i). p = two-sided episode-level
  permutation of success labels (adaptive Besag-Clifford: >= --min-perms for every feature, a feature stops after
  --h exceedances, strong ones go to --max-perms). BH-FDR over the tested features -> q. fr_pct = percentile of |d|
  among the 50 tested features nearest in log token firing rate.
2 EXCLUSIONS. |corr(a_i(s), step)| > --pos-r over all steps (time-index features), token firing rate > --dense, or
  token-position eta^2 > --pos-r^2 (token-index features); dead or firing in < --min-eps discovery windows -> untested.
3 OFFLINE PRUNE (held-out episodes). Top --top tested features by (q, p, -|d|). Edit x -> x - a_i(x) W_dec[i] s at every
  token (decoder contribution; SAE error and pos_mean untouched). Metric = action divergence ||pi(x_edit) - pi(x)||^2 /
  total action variance, averaged over the held-out window steps of the group where the feature is MORE active
  (failed windows if d>0). pi = --downstream module:fn (real action head; fn(acts[B,n_tok,D] float32, meta)->[B,A]) or,
  if absent, a ridge probe from token-mean activations to the recorded actions fitted on discovery episodes (a proxy:
  reported with its held-out R^2). d_loss vs recorded actions is reported too. Each feature vs --n-random random
  features matched on token firing rate (drawn from the 3*n nearest); pct = 100*(#ctrl < x + 0.5 ties)/n; keep >= --keep-pct
  AND sensitivity pct (divergence per unit squared activation, i.e. does the head read this direction at all) >= --min-sens-pct.
  (Synthetic check: a planted pre-failure feature the head ignores passes the raw pct gate under the probe proxy, 90th pct,
  but has sensitivity pct 0, so it is dropped.)
4 OUTPUT candidates.json: top --n-final that pass q <= --fdr, fr_pct >= --fr-pct and offline pct >= --keep-pct, with
  examples and matched random control ids; plus 'exploratory' (best of the rest, never mixed into candidates).
"""
from __future__ import annotations
import argparse, glob, hashlib, importlib, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_io import RolloutRun

torch.set_grad_enabled(False)


def args_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None, help="output dir (default <run>/picker)")
    ap.add_argument("--W", type=int, default=10, help="window: captured steps before the anchor")
    ap.add_argument("--match", choices=["time", "phase", "end"], default="time")
    ap.add_argument("--no-detrend", dest="detrend", action="store_false")
    ap.add_argument("--bin", type=int, default=0, help="time-bin width in captured steps for detrend (0 = W)")
    ap.add_argument("--heldout-frac", type=float, default=0.3, help="episodes held out for the offline prune (stratified)")
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--min-eps", type=int, default=5, help="feature must fire in >= this many discovery windows")
    ap.add_argument("--min-perms", type=int, default=2000)
    ap.add_argument("--max-perms", type=int, default=200000)
    ap.add_argument("--h", type=int, default=20, help="Besag-Clifford: stop a feature after h exceedances")
    ap.add_argument("--pos-r", type=float, default=0.8)
    ap.add_argument("--dense", type=float, default=0.5)
    ap.add_argument("--fr-pct", type=float, default=90.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--keep-pct", type=float, default=90.0)
    ap.add_argument("--n-final", type=int, default=5)
    ap.add_argument("--min-sens-pct", type=float, default=10.0, help="offline gate 2: action-change per unit activation must beat >= this pct of controls")
    ap.add_argument("--downstream", default=None, help="module:function action head (see docstring)")
    ap.add_argument("--fail-t-scale", type=float, default=1.0, help="divide metadata fail_t by this (sim steps per captured step)")
    ap.add_argument("--chunk-tokens", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap


def log(a, *m):
    if not a.quiet: print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)


def stable_u(s, seed):  # deterministic uniform in [0,1) from a string
    return int(hashlib.sha1(f"{seed}:{s}".encode()).hexdigest()[:12], 16) / 16 ** 12


def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out


def cohen_d(X, y):
    """X [n, m], y bool [n] -> d [m] (pooled SD); 0 where SD == 0"""
    a, b = X[y], X[~y]; n1, n0 = len(a), len(b)
    v = ((a.var(0, ddof=1) * (n1 - 1) if n1 > 1 else 0) + (b.var(0, ddof=1) * (n0 - 1) if n0 > 1 else 0)) / max(n1 + n0 - 2, 1)
    sd = np.sqrt(v); return np.where(sd > 0, (a.mean(0) - b.mean(0)) / np.where(sd > 0, sd, 1), 0.0)


def perm_test(X, y, a, rng):
    """two-sided episode-level label permutation of |d|. Every feature gets >= min_perms; it keeps going (to max_perms) until
    it has h exceedances (Besag & Clifford-style adaptive stop). p = (exceed + 1) / (perms + 1). returns p, n_perms, |d|."""
    n, m = X.shape; n1 = int(y.sum()); n0 = n - n1
    Xt = torch.from_numpy(X).double(); X2 = Xt ** 2
    tot, tot2 = Xt.sum(0), X2.sum(0)
    d_obs = np.abs(cohen_d(X, y)); dobs = torch.from_numpy(d_obs)
    exceed = np.zeros(m, np.int64); done = np.zeros(m, np.int64); active = np.arange(m)
    while len(active):
        B = 1000 if done.max() < 20000 else 5000
        idx = torch.from_numpy(np.argsort(rng.random((B, n)), axis=1)[:, :n1])
        P = torch.zeros(B, n, dtype=torch.float64).scatter_(1, idx, 1.0)
        Xa, X2a = Xt[:, active], X2[:, active]
        s1 = P @ Xa; q1 = P @ X2a; s0 = tot[active] - s1; q0 = tot2[active] - q1
        m1, m0 = s1 / n1, s0 / n0
        v = ((q1 - n1 * m1 ** 2) + (q0 - n0 * m0 ** 2)) / max(n - 2, 1)
        sd = v.clamp_min(0).sqrt(); d = torch.where(sd > 0, (m1 - m0) / torch.where(sd > 0, sd, torch.ones_like(sd)), torch.zeros_like(sd))
        exceed[active] += (d.abs() >= dobs[active] - 1e-12).sum(0).numpy(); done[active] += B
        stop = (done[active] >= a.min_perms) & ((exceed[active] >= a.h) | (done[active] >= a.max_perms))
        active = active[~stop]
    return (exceed + 1) / (done + 1), done, d_obs



def nearest_by_rate(fid, pool, logfr, k):
    pool = pool[pool != fid]; o = np.argsort(np.abs(logfr[pool] - logfr[fid]), kind="stable"); return pool[o[:k]]


def pick(run_dir, a):
    t0 = time.time(); torch.set_num_threads(a.threads); rng = np.random.default_rng(a.seed)
    R = RolloutRun(run_dir, a.fail_t_scale); ep = R.episodes; E = len(ep["success"])
    out = Path(a.out) if a.out else R.dir / "picker"; out.mkdir(parents=True, exist_ok=True)
    warn = R.warnings
    # ---------- index pass ----------
    eidx, st = R.index(); e = R.e_of(eidx); ok = e >= 0
    if (~ok).any(): warn.append(f"{int((~ok).sum())} activation rows belong to episodes missing from metadata (ignored)")
    maxst = np.full(E, -1, np.int64); np.maximum.at(maxst, e[ok], st[ok])
    if (maxst < 0).any(): warn.append(f"{int((maxst < 0).sum())} episodes have no captured activations (dropped)")
    R.fill_n_steps(maxst)
    has = maxst >= 0; T = ep["n_steps"]
    succ = ep["success"]; fail = ~succ
    nfail_meta = int((fail & has).sum())
    log(a, f"{R.dir.name}: {int(has.sum())} episodes with activations, {nfail_meta} failed, {int((succ & has).sum())} successful; episodes from {R.episode_source}")
    if nfail_meta < 4 or (succ & has).sum() < 4: raise SystemExit("need >= 4 failed and >= 4 successful episodes")
    # ---------- anchors + windows (metadata only, before any activation is read) ----------
    ft = ep["fail_t"].copy(); no_ft = fail & np.isnan(ft)
    if no_ft.any(): warn.append(f"{int(no_ft.sum())} failed episodes lack fail_t -> anchored at their last captured step (timeout?)")
    ft[no_ft] = T[no_ft]
    ft = np.clip(ft, 1, np.maximum(T, 1))
    timeout_like = fail & (ft >= T - 1)
    if fail.any() and timeout_like[fail & has].mean() > 0.5:
        warn.append(f"{timeout_like[fail & has].mean():.0%} of failures are anchored at the episode end (timeouts): the window then contrasts "
                    "'end of a timed-out episode' with successes, not a failure moment. Ask for an informative failure step (drop, stall, wrong grasp).")
    anchor = np.zeros(E); clipped = np.zeros(E, bool)
    pool_t = ft[fail & has]; pool_ph = (ft / np.maximum(T, 1))[fail & has]
    for i in np.flatnonzero(succ & has):
        u = stable_u(ep["episode_id"][i], a.seed)
        if a.match == "time":
            v = float(pool_t[int(u * len(pool_t))]); clipped[i] = v > T[i]; anchor[i] = min(v, T[i])
        elif a.match == "phase":
            anchor[i] = round(float(pool_ph[int(u * len(pool_ph))]) * T[i])
        else:
            anchor[i] = T[i]
    anchor[fail] = ft[fail]; anchor = np.maximum(anchor.round().astype(np.int64), 1)
    if a.match == "time" and clipped[succ & has].mean() > 0.3:
        warn.append(f"{clipped[succ & has].mean():.0%} of successful episodes ended before their matched failure time and were clipped to their "
                    "last steps; consider --match phase.")
    wlo = np.maximum(anchor - a.W, 0); whi = anchor  # window = [wlo, whi)
    nwin = np.maximum(whi - wlo, 0)
    # ---------- split (stratified, stable by episode id) ----------
    u = np.array([stable_u("split:" + s, a.seed) for s in ep["episode_id"]])
    held = np.zeros(E, bool)
    for g in (succ, fail):
        ids = np.flatnonzero(g & has); k = int(round(a.heldout_frac * len(ids)))
        held[ids[np.argsort(u[ids])[:k]]] = True
    disc = has & ~held
    if a.heldout_frac <= 0: held = has.copy(); warn.append("--heldout-frac 0: offline prune uses the discovery episodes (circular)")
    # ---------- pass 1: every token through the SAE ----------
    M, D = R.sae.M, R.sae.D; binw = a.bin or a.W; NB = int(T[has].max() // binw) + 1
    win_a = torch.zeros(E, M, dtype=torch.float64); win_r = torch.zeros(E, M, dtype=torch.float64)
    bin_a = torch.zeros(NB, M, dtype=torch.float64); bin_r = torch.zeros(NB, M, dtype=torch.float64); bin_n = torch.zeros(NB, dtype=torch.float64)
    win_bins = np.zeros((E, NB))
    Ss = torch.zeros(M, dtype=torch.float64); Ss2 = torch.zeros(M, dtype=torch.float64); Sst = torch.zeros(M, dtype=torch.float64)
    St = St2 = Nst = 0.0
    tok_fire = torch.zeros(M, dtype=torch.float64); tok_sum = torch.zeros(M, dtype=torch.float64); tok_sq = torch.zeros(M, dtype=torch.float64); ntok = 0
    tokpos_sum = {}; tokpos_n = {}
    xbar_rows = []; xbar_keys = []; seen = 0
    tlog = time.time()
    for c in R.chunks(a.chunk_tokens):
        x, f, ce, cs, ct = c["x"], c["f"], c["e"], c["step"], c["tok"]
        key = ce * (1 << 32) + cs; uk, inv = np.unique(key, return_inverse=True); ns = len(uk)
        inv_t = torch.from_numpy(inv); cnt = torch.bincount(inv_t, minlength=ns).double()
        S = torch.zeros(ns, M, dtype=torch.float64).index_add_(0, inv_t, f.double()) / cnt[:, None]
        Fr = torch.zeros(ns, M, dtype=torch.float64).index_add_(0, inv_t, (f > 0).double()) / cnt[:, None]
        se, sst = uk >> 32, uk & ((1 << 32) - 1)
        inw = (sst >= wlo[se]) & (sst < whi[se]); wgt = torch.from_numpy(np.where(inw, 1.0 / np.maximum(nwin[se], 1), 0.0))
        win_a.index_add_(0, torch.from_numpy(se), S * wgt[:, None]); win_r.index_add_(0, torch.from_numpy(se), Fr * wgt[:, None])
        b = sst // binw; bt = torch.from_numpy(b)
        bin_a.index_add_(0, bt, S); bin_r.index_add_(0, bt, Fr); bin_n.index_add_(0, bt, torch.ones(ns, dtype=torch.float64))
        np.add.at(win_bins, (se[inw], b[inw]), 1.0 / np.maximum(nwin[se[inw]], 1))
        tt = torch.from_numpy(sst.astype(np.float64))
        Ss += S.sum(0); Ss2 += (S ** 2).sum(0); Sst += (S * tt[:, None]).sum(0); St += float(tt.sum()); St2 += float((tt ** 2).sum()); Nst += ns
        tok_fire += (f > 0).double().sum(0); tok_sum += f.double().sum(0); tok_sq += (f.double() ** 2).sum(0); ntok += len(f)
        for tv in np.unique(ct):  # token-position means (for token-index features); bounded by #positions
            mm = torch.from_numpy(ct == tv)
            tokpos_sum[int(tv)] = tokpos_sum.get(int(tv), 0) + (f[mm] > 0).double().sum(0); tokpos_n[int(tv)] = tokpos_n.get(int(tv), 0) + int(mm.sum())
        if x is not None:
            xbar_rows.append((torch.zeros(ns, D).index_add_(0, inv_t, x) / cnt[:, None].float()).numpy()); xbar_keys.append(uk)
        seen += ns
        if time.time() - tlog > 30: log(a, f"  pass 1: {seen} steps, {ntok} tokens"); tlog = time.time()
    if seen != len(np.unique(eidx[ok] * (1 << 32) + st[ok])):
        warn.append(f"step groups emitted ({seen}) != unique (episode, step) pairs: a step's tokens are split across shards non-contiguously")
    log(a, f"pass 1 done: {seen} steps, {ntok} tokens, {time.time() - t0:.0f}s")
    # ---------- feature stats ----------
    freq = (tok_fire / ntok).numpy(); logfr = np.log10(np.maximum(freq, 1e-9))
    ms, mt = (Ss / Nst).numpy(), St / Nst
    var_s = (Ss2 / Nst).numpy() - ms ** 2; var_t = St2 / Nst - mt ** 2
    r_time = np.where(var_s > 1e-15, ((Sst / Nst).numpy() - ms * mt) / np.sqrt(np.maximum(var_s, 1e-30) * max(var_t, 1e-30)), 0.0)
    tk = sorted(tokpos_sum); tokm = np.stack([(tokpos_sum[k] / tokpos_n[k]).numpy() for k in tk]) if len(tk) > 1 else None
    if tokm is not None:  # eta^2 of the firing indicator on token position: between-position var / total (Bernoulli) var
        w = np.array([tokpos_n[k] for k in tk], float); gm = freq
        between = ((tokm - gm) ** 2 * w[:, None]).sum(0) / w.sum(); total = gm * (1 - gm)
        tok_eta2 = np.where(total > 1e-15, between / np.maximum(total, 1e-30), 0.0)
    else:
        tok_eta2 = np.zeros(M)
    mu_a = (bin_a / bin_n.clamp_min(1)[:, None]).numpy(); mu_r = (bin_r / bin_n.clamp_min(1)[:, None]).numpy()
    Xa, Xr = win_a.numpy(), win_r.numpy()
    if a.detrend: Xa = Xa - win_bins @ mu_a; Xr = Xr - win_bins @ mu_r
    fires_eps = (win_r.numpy()[disc] > 0).sum(0)
    alive = tok_fire.numpy() > 0
    excl = np.full(M, "", dtype=object)
    excl[~alive] = "dead"
    excl[(excl == "") & (np.abs(r_time) > a.pos_r)] = "time_index"
    excl[(excl == "") & (freq > a.dense)] = "dense"
    excl[(excl == "") & (tok_eta2 > a.pos_r ** 2)] = "token_position"
    excl[(excl == "") & (fires_eps < a.min_eps)] = "too_rare_in_windows"
    tested = np.flatnonzero(excl == "")
    log(a, f"features: {M} total, {int(alive.sum())} alive, tested {len(tested)}; excluded " +
        json.dumps({k: int((excl == k).sum()) for k in ("dead", "time_index", "dense", "token_position", "too_rare_in_windows")}))
    y = fail[disc]
    Xd = Xa[disc][:, tested]
    p, nperm, absd = perm_test(Xd, y, a, rng)
    d = cohen_d(Xd, y); d_fire = cohen_d(Xr[disc][:, tested], y); q = bh(p)
    fr_pct = np.zeros(len(tested))
    for j, fid in enumerate(tested):
        nb = nearest_by_rate(fid, tested, logfr, 50); pos = np.searchsorted(tested, nb); o = absd[pos]
        fr_pct[j] = 100 * ((o < absd[j]).sum() + 0.5 * (o == absd[j]).sum()) / max(len(o), 1)
    mean_f = Xa[disc & fail][:, tested].mean(0); mean_s = Xa[disc & succ][:, tested].mean(0)
    log(a, f"contrast: {len(tested)} features x {int(disc.sum())} discovery episodes, perms min {int(nperm.min())} max {int(nperm.max())}, "
           f"min p {p.min():.2e}, #q<={a.fdr}: {int((q <= a.fdr).sum())}  [{time.time() - t0:.0f}s]")
    with open(out / "contrast.jsonl", "w") as fh:
        for j in np.lexsort((-absd, p)):
            fid = int(tested[j])
            fh.write(json.dumps(dict(feature_id=fid, d=round(float(d[j]), 4), d_fire=round(float(d_fire[j]), 4), p=float(p[j]), q=float(q[j]), n_perms=int(nperm[j]),
                                     fr_pct=round(float(fr_pct[j]), 1), firing_rate=float(freq[fid]), r_time=round(float(r_time[fid]), 3), tok_eta2=round(float(tok_eta2[fid]), 3),
                                     mean_fail=float(mean_f[j]), mean_succ=float(mean_s[j]))) + "\n")
    # ---------- offline prune ----------
    order = np.lexsort((-absd, p, q)); sel_j = order[:a.top]; sel = tested[sel_j]
    pool = np.flatnonzero((excl == "") | (excl == "too_rare_in_windows"))
    pool = np.setdiff1d(pool, sel); used = set(); ctrl = {}
    for fid in sel:
        near = nearest_by_rate(fid, np.setdiff1d(pool, list(used)) if len(pool) - len(used) >= 3 * a.n_random else pool, logfr, 3 * a.n_random)
        ch = rng.choice(near, size=min(a.n_random, len(near)), replace=False); ctrl[int(fid)] = [int(v) for v in ch]; used.update(ch.tolist())
    cols = np.unique(np.concatenate([sel, np.array([v for vs in ctrl.values() for v in vs], np.int64)])).astype(np.int64)
    ci = {int(f): i for i, f in enumerate(cols)}
    # pass 2: selected columns for every step (examples) + downstream on held-out window steps
    ds = None
    if a.downstream:
        mod, fn = a.downstream.split(":"); sys.path.insert(0, str(Path.cwd())); ds = getattr(importlib.import_module(mod), fn)
    act = R.actions()
    A_rec = {}
    if act is not None:
        for ee, ss, av in zip(*act): A_rec[(int(ee), int(ss))] = av
    dirs = R.sae.raw_dir(cols)  # [C, D]
    step_keys, step_S = [], []
    ds_div = {}; ds_dl = {}; ds_n = {"fail": 0, "succ": 0}
    colst = torch.from_numpy(cols)
    for c in R.chunks(a.chunk_tokens, cols=colst, need_x=ds is not None):
        f, ce, cs = c["f"], c["e"], c["step"]
        key = ce * (1 << 32) + cs; uk, inv = np.unique(key, return_inverse=True); inv_t = torch.from_numpy(inv)
        cnt = torch.bincount(inv_t, minlength=len(uk)).double()
        step_S.append((torch.zeros(len(uk), len(cols), dtype=torch.float64).index_add_(0, inv_t, f.double()) / cnt[:, None]).numpy()); step_keys.append(uk)
        if ds is None: continue
        for g in range(len(uk)):
            ee, ss = int(uk[g] >> 32), int(uk[g] & ((1 << 32) - 1))
            if not held[ee] or not (wlo[ee] <= ss < whi[ee]): continue
            rows = torch.from_numpy(inv == g); x = c["x"][rows]; fc = f[rows]  # [n_tok, D], [n_tok, C]
            grp = "fail" if fail[ee] else "succ"; ds_n[grp] += 1
            meta = dict(run=str(R.dir), episode_id=str(ep["episode_id"][ee]), task_id=str(ep["task_id"][ee]), step=ss)
            base = torch.as_tensor(ds(x[None], meta)).float().reshape(1, -1)
            arec = A_rec.get((ee, ss))
            for k0 in range(0, len(cols), 16):
                kk = range(k0, min(k0 + 16, len(cols)))
                Xe = torch.stack([x - fc[:, k:k + 1] * dirs[k][None] for k in kk])
                out_e = torch.as_tensor(ds(Xe, meta)).float().reshape(len(kk), -1)
                dv = ((out_e - base) ** 2).sum(1).numpy()
                for k, v in zip(kk, dv): ds_div.setdefault((int(cols[k]), grp), []).append(float(v))
                if arec is not None:
                    ar = torch.from_numpy(arec).float()[None]
                    dl = (((out_e - ar) ** 2).sum(1) - ((base - ar) ** 2).sum(1)).numpy()
                    for k, v in zip(kk, dl): ds_dl.setdefault((int(cols[k]), grp), []).append(float(v))
    SK = np.concatenate(step_keys); SS = np.concatenate(step_S); se_all = SK >> 32; ss_all = SK & ((1 << 32) - 1)
    inwin = (ss_all >= wlo[se_all]) & (ss_all < whi[se_all])
    hw_f = held[se_all] & inwin & fail[se_all]; hw_s = held[se_all] & inwin & succ[se_all]
    off = {}; metric = None; probe_info = None
    if ds is not None:
        avar = None
        if act is not None: avar = float(np.var(act[2], axis=0).sum())
        nrm = avar if avar and avar > 0 else 1.0
        for fid in cols:
            off[int(fid)] = {g: (float(np.mean(ds_div.get((int(fid), g), [np.nan]))) / nrm) for g in ("fail", "succ")}
            off[int(fid)].update({f"dloss_{g}": (float(np.mean(ds_dl[(int(fid), g)])) / nrm if (int(fid), g) in ds_dl else None) for g in ("fail", "succ")})
        metric = f"downstream {a.downstream}: ||pi(x_edit)-pi(x)||^2 / total action variance ({'recorded' if avar else 'unnormalised'}), held-out window steps (fail n={ds_n['fail']}, succ n={ds_n['succ']})"
    elif act is not None and xbar_rows:
        XK = np.concatenate(xbar_keys); XB = np.concatenate(xbar_rows).astype(np.float64)
        ae, ast, av = act; amap = {int(k): i for i, k in enumerate(ae * (1 << 32) + ast)}
        ia = np.array([amap.get(int(k), -1) for k in XK]); okr = ia >= 0
        XK, XB, Y = XK[okr], XB[okr], av[ia[okr]].astype(np.float64)
        xe = XK >> 32; tr = disc[xe]; te = held[xe]
        mx, my = XB[tr].mean(0), Y[tr].mean(0); Xc, Yc = XB[tr] - mx, Y[tr] - my
        G = Xc.T @ Xc; lam = 1e-2 * np.trace(G) / G.shape[0]
        Wp = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc)
        pred_te = (XB[te] - mx) @ Wp + my; avar = float(np.var(Y[tr], axis=0).sum())
        r2 = 1 - float(((pred_te - Y[te]) ** 2).sum(1).mean()) / max(float(((Y[te] - Y[tr].mean(0)) ** 2).sum(1).mean()), 1e-12)
        probe_info = dict(kind="ridge token-mean acts -> recorded action", heldout_r2=round(r2, 4), lam=lam, n_train_steps=int(tr.sum()), n_test_steps=int(te.sum()))
        if r2 < 0.3: warn.append(f"action probe held-out R^2 = {r2:.2f}: the offline prune is weak; prefer --downstream with the real action head")
        V = (dirs.double().numpy() @ Wp)  # [C, A] action change per unit activation
        vn = (V ** 2).sum(1)
        pmap = {int(k): i for i, k in enumerate(XK)}; pidx = np.array([pmap.get(int(k), -1) for k in SK])
        for g, msk in (("fail", hw_f), ("succ", hw_s)):
            Sg = SS[msk]; div = (Sg ** 2) * vn[None] / max(avar, 1e-12)
            pi_ = pidx[msk]; okp = pi_ >= 0
            for k, fid in enumerate(cols):
                off.setdefault(int(fid), {})[g] = float(div[:, k].mean()) if len(div) else float("nan")
                if okp.any():
                    pr = (XB[pi_[okp]] - mx) @ Wp + my; yy = Y[pi_[okp]]
                    pe = pr - Sg[okp, k:k + 1] * V[k][None]
                    off[int(fid)][f"dloss_{g}"] = float((((pe - yy) ** 2).sum(1) - ((pr - yy) ** 2).sum(1)).mean()) / max(avar, 1e-12)
        metric = f"linear action probe proxy (held-out R^2 {r2:.2f}): ||dpred||^2 / total action variance on held-out window steps (fail n={int(hw_f.sum())}, succ n={int(hw_s.sum())})"
    else:
        warn.append("no --downstream and no recorded actions: offline prune SKIPPED (offline_pct null); candidates gated on contrast only")
    rows_off = []
    for j, fid in zip(sel_j, sel):
        rec = dict(feature_id=int(fid), d=float(d[j]), q=float(q[j]), controls=ctrl[int(fid)])
        if off:
            g = "fail" if d[j] > 0 else "succ"; x0 = off[int(fid)][g]; cs_ = np.array([off[c][g] for c in ctrl[int(fid)]])
            pct = 100 * ((cs_ < x0).sum() + 0.5 * (np.isclose(cs_, x0)).sum()) / len(cs_) if len(cs_) else None
            msk = hw_f if g == "fail" else hw_s
            def sens(f_):  # divergence per unit squared activation on the same steps: does the head READ this direction?
                m2 = float((SS[msk, ci[f_]] ** 2).mean()) if msk.any() else 0.0; return off[f_][g] / m2 if m2 > 0 else float("nan")
            s0 = sens(int(fid)); sc = np.array([sens(c) for c in ctrl[int(fid)]]); sc = sc[np.isfinite(sc)]
            spct = 100 * ((sc < s0).sum() + 0.5 * np.isclose(sc, s0).sum()) / len(sc) if len(sc) and np.isfinite(s0) else None
            rec.update(sens=s0, sens_pct=spct)
            rec.update(group=g, div=x0, ctrl_div=[float(v) for v in cs_], pct=pct, dloss=off[int(fid)].get(f"dloss_{g}"),
                       div_other=off[int(fid)]["succ" if g == "fail" else "fail"])
        rows_off.append(rec)
    with open(out / "offline.jsonl", "w") as fh:
        for r_ in rows_off: fh.write(json.dumps(r_) + "\n")
    # ---------- examples + final ----------
    def examples(fid, k=5):
        col = ci[int(fid)]; m = inwin & fail[se_all]; idx = np.flatnonzero(m)
        top = idx[np.argsort(-SS[idx, col])[:k]]
        return [dict(episode_id=str(ep["episode_id"][se_all[i]]), task_id=str(ep["task_id"][se_all[i]]), seed=int(ep["seed"][se_all[i]]),
                     step=int(ss_all[i]), fail_t=int(anchor[se_all[i]]), value=round(float(SS[i, col]), 4), heldout=bool(held[se_all[i]])) for i in top if SS[i, col] > 0]
    finals, explo = [], []
    for rec, j in zip(rows_off, sel_j):
        fid = rec["feature_id"]
        gates = dict(fdr=bool(q[j] <= a.fdr), fr_matched=bool(fr_pct[j] >= a.fr_pct), offline=(None if rec.get("pct") is None else bool(rec["pct"] >= a.keep_pct and (rec.get("sens_pct") is None or rec["sens_pct"] >= a.min_sens_pct))))
        item = dict(feature_id=fid, direction=("more active before failure" if d[j] > 0 else "less active before failure"),
                    contrast=dict(d=round(float(d[j]), 3), d_fire=round(float(d_fire[j]), 3), p=float(p[j]), q=float(q[j]), n_perms=int(nperm[j]),
                                  fr_matched_pct=round(float(fr_pct[j]), 1), mean_fail=float(mean_f[j]), mean_succ=float(mean_s[j]), r_time=round(float(r_time[fid]), 3)),
                    firing_rate=float(freq[fid]),
                    offline=(dict(pct_vs_random=rec.get("pct"), sensitivity_pct_vs_random=rec.get("sens_pct"), div=rec.get("div"), dloss=rec.get("dloss"), group=rec.get("group"),
                                  div_other_group=rec.get("div_other"), random_ctrl_div=rec.get("ctrl_div")) if off else None),
                    matched_random_controls=rec["controls"], gates=gates, examples=examples(fid))
        (finals if all(v is not False for v in gates.values()) and gates["fdr"] else explo).append(item)
    finals = finals[:a.n_final]
    res = dict(schema="picker-v1", run=str(R.dir), created=time.strftime("%Y-%m-%d %H:%M %Z"),
               policy=R.man.get("model") or R.man.get("policy"), layer=R.man.get("layer"), sae=dict(M=M, D=D, cfg=R.sae.cfg.get("arch")),
               episodes=dict(total=E, with_acts=int(has.sum()), failed=int((fail & has).sum()), successful=int((succ & has).sum()),
                             discovery=int(disc.sum()), discovery_failed=int((disc & fail).sum()), heldout=int((held & has).sum()) if a.heldout_frac > 0 else 0,
                             tasks=sorted(set(ep["task_id"][has].tolist())), fail_t_timeouts=int(timeout_like[fail & has].sum()), success_clipped=int(clipped[succ & has].sum())),
               features=dict(total=M, alive=int(alive.sum()), tested=int(len(tested)), excluded={k: int((excl == k).sum()) for k in ("dead", "time_index", "dense", "token_position", "too_rare_in_windows")},
                             n_q_le_fdr=int((q <= a.fdr).sum())),
               offline_metric=metric, probe=probe_info,
               candidates=finals, exploratory=explo[:max(0, a.n_final * 2)],
               candidate_ids=[c["feature_id"] for c in finals],
               matched_random_control_ids={str(c["feature_id"]): c["matched_random_controls"] for c in finals},
               runner_hint=(f"physint rollouts --features {','.join(str(c['feature_id']) for c in finals)} --layer {R.man.get('layer')} --kind ablate"
                            " (the runner draws its own controls; ours are firing-rate matched: matched_random_control_ids)") if finals else None,
               caveats=["Candidates were selected on these same rollouts (discovery split); the held-out offline prune is on a small split. Validate with FRESH seeds in the rollout batch.",
                        "Contrast is correlational: a feature that fires before failure may be a consequence of the failing state (e.g. 'gripper empty') rather than its cause."],
               warnings=warn, params=vars(a), seconds=round(time.time() - t0, 1))
    (out / "candidates.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "config.json").write_text(json.dumps(dict(params=vars(a), rollout_io_episode_source=R.episode_source), indent=1, default=str))
    log(a, f"wrote {out / 'candidates.json'}: candidates {res['candidate_ids']} ({len(explo)} exploratory)  [{time.time() - t0:.0f}s]")
    for w_ in warn: log(a, "WARN", w_)
    return res


def main(argv=None):
    a = args_parser().parse_args(argv)
    runs = [p for r in a.runs for p in (sorted(glob.glob(r)) or [r])]
    out = []
    for r in runs:
        if not Path(r).is_dir(): print("skip (not a dir):", r); continue
        out.append(pick(r, a))
    return out


if __name__ == "__main__":
    main()
