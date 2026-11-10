"""Fake policy run folders in the proposed capture format (rollout_io canonical), for validating pick_candidates.py.
Ground truth: orthonormal SAE dictionary, so the SAE recovers the codes exactly (noise below threshold).
  planted feature PLANT (if --plant > 0): fires on a fraction `plant` of tokens in the W_p steps before fail_t of failed episodes
  feature TIMEF: activation grows with step (|r| > 0.8 -> must be excluded as time-index)
  feature DENSEF: fires on 70% of tokens (must be excluded as dense)
  feature TOKF: fires only at token position 0 (must be excluded as token-position)
  feature DRIFT: firing prob rises moderately with step (|r| ~ 0.3-0.6), a time confound when failures are timeouts
  failures: half 'mid' (fail_t ~ U(30, H-10), episode runs to H), half timeouts (fail_t = None -> end)
Actions: a = tanh(xbar_n @ Wh) with Wh drawn so every feature has comparable action effect; the planted feature is not special in Wh.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from safetensors.numpy import save_file as save_np
from safetensors.torch import save_file

PLANT, TIMEF, DENSEF, TOKF, DRIFT = 7, 3, 5, 11, 9

def make(out, n_ep=100, M=256, D=256, n_tok=8, H=120, A=7, plant=0.5, Wp=8, p_succ=0.5, seed=0, timeout_frac=0.5, k_base=3):
    rng = np.random.default_rng(seed); out = Path(out); (out / "acts").mkdir(parents=True, exist_ok=True); (out / "sae").mkdir(exist_ok=True)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D))); W_dec = Q[:M].astype(np.float32)  # [M, D] orthonormal rows
    mu = rng.standard_normal(D).astype(np.float32) * 2; s = 10.0
    base = np.exp(rng.uniform(np.log(0.003), np.log(0.08), M)); base[DENSEF] = 0.7; base[TOKF] = 0.0; base[TIMEF] = 0.3; base[PLANT] = 0.02
    base *= k_base / max(base.sum() / M * M, 1) * M / M if False else 1.0
    Wh = (rng.standard_normal((D, A)) / np.sqrt(D)).astype(np.float32) * 3
    eps, A_e, A_s, A_a = [], [], [], []; rows = dict(acts=[], episode_idx=[], step=[], tok=[])
    for e in range(n_ep):
        succ = rng.random() < p_succ
        if succ: T = int(rng.integers(50, H - 10)); ft = None
        elif rng.random() < timeout_frac: T = H; ft = None
        else: T = H; ft = int(rng.integers(30, H - 10))
        anchor = ft if ft is not None else T
        tt = np.arange(T)[:, None, None]; tokk = np.arange(n_tok)[None, :, None]
        pr = np.broadcast_to(base[None, None, :], (T, n_tok, M)).copy()
        pr[:, :, TOKF] = 0.0; pr[:, 0, TOKF] = 0.9
        pr[:, :, DRIFT] = 0.01 + 0.15 * (np.arange(T) / H)[:, None]
        if not succ and plant > 0:
            lo = max(anchor - Wp, 0); pr[lo:anchor, :, PLANT] = plant
        fire = rng.random((T, n_tok, M)) < pr
        val = rng.exponential(1.0, (T, n_tok, M)) + 0.3
        val[:, :, TIMEF] = 0.3 + 3.0 * (np.arange(T) / H)[:, None]; fire[:, :, TIMEF] = True
        codes = (fire * val).astype(np.float32)
        x = (codes @ W_dec) * s + mu + rng.standard_normal((T, n_tok, D)).astype(np.float32) * 0.05 * s
        xn = (x - mu) / s
        act = np.tanh(xn.mean(1) @ Wh) + rng.standard_normal((T, A)).astype(np.float32) * 0.01
        rows["acts"].append(x.reshape(-1, D).astype(np.float16)); rows["episode_idx"].append(np.full(T * n_tok, e, np.int32))
        rows["step"].append(np.repeat(np.arange(T, dtype=np.int32), n_tok)); rows["tok"].append(np.tile(np.arange(n_tok, dtype=np.int16), T))
        A_e.append(np.full(T, e, np.int32)); A_s.append(np.arange(T, dtype=np.int32)); A_a.append(act.astype(np.float32))
        eps.append(dict(episode_idx=e, episode_id=f"synth-{seed}-{e:03d}", task_id="libero_plus/synth_task", seed=1000 + e, success=bool(succ),
                        fail_t=ft, n_steps=T))
    cat = {k: np.concatenate(v) for k, v in rows.items()}; n = len(cat["step"]); sh = (n + 1) // 2
    for i, a0 in enumerate(range(0, n, sh)):  # 2 shards; a step can straddle the boundary (tests carry)
        save_np({k: np.ascontiguousarray(v[a0:a0 + sh]) for k, v in cat.items()}, str(out / "acts" / f"shard-{i:05d}.safetensors"))
    save_np(dict(episode_idx=np.concatenate(A_e), step=np.concatenate(A_s), action=np.concatenate(A_a)), str(out / "actions.safetensors"))
    W = torch.from_numpy(W_dec)
    save_file(dict(W_enc=W.T.contiguous(), b_enc=torch.zeros(M), W_dec=W.contiguous(), b_dec=torch.zeros(D), threshold=torch.tensor(0.15),
                   norm_mu=torch.from_numpy(mu), norm_scale=torch.tensor([s])), str(out / "sae" / "weights.safetensors"))
    (out / "sae" / "config.json").write_text(json.dumps(dict(arch="synthetic-orthonormal", d_in=D, d_sae=M)))
    (out / "manifest.json").write_text(json.dumps(dict(format="rollcap-v0", run_id=out.name, model=dict(adapter="synthetic", name="fake-policy"),
                                                       layer=0, d_model=D, n_tok=n_tok, episodes=eps,
                                                       truth=dict(plant=PLANT if plant > 0 else None, time=TIMEF, dense=DENSEF, tok=TOKF, drift=DRIFT)), indent=0))
    np.save(out / "sae" / "_synth_Wh.npy", Wh)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--plant", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-ep", type=int, default=100); a = ap.parse_args(); make(a.out, n_ep=a.n_ep, plant=a.plant, seed=a.seed)
