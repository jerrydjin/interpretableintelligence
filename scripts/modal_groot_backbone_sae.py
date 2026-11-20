"""Random-weights SAE check for the task 1028 backbone lever (Nov 19 2026).

The class-mean phase direction at GR00T's VLM backbone output (``backbone_features``, the tokens the action head
cross-attends to) flipped pot order in closed loop (8/25 vs 0/25). A supervised direction has no dictionary, so
research rule 4 cannot apply to it. This script asks the rule-4 question offline: does a feature of an SAE trained
on backbone_features reproduce that lever better than the same picker run on untrained SAEs?

    modal run scripts/modal_groot_backbone_sae.py --smoke
    II_MODAL_HOURS=0.5 modal run --detach scripts/modal_groot_backbone_sae.py   # then --fetch

Pre-registered (written before the full run):

- Data: every window of the recorded clips (the steer run's split.json): all backbone tokens of train-clip windows
  train the SAE; labels are the block-5 probe's phase (as in modal_groot_phase_backbone.py).
- SAE: BatchTopK with AuxK, 4096 latents, k = 32, channel normalisation (``retrain_sae.train``). Untrained controls:
  ``random_sae.untrained_weights`` with the five rule-4 seeds, threshold set for the same k on the same train tokens.
- Picker (same for every SAE): a window's feature value is its latent's mean activation over the window's tokens.
  Among latents active in >= 1% of train windows, pick the one with the highest train-window AUROC for phase 1.
- Steering: add the pick's decoder direction in raw units (``W_dec[j] * norm_scale``), rescaled to 3x the class-mean
  difference's norm, to every backbone token of the held-out phase-0 windows (up to 3 per clip, as the backbone gate);
  phase shift = change in the held-out-trained action classifier's score. 19 random directions at the same norm.
- Pass: the trained pick's mean phase shift is above every untrained pick's and every random direction's, and its
  held-out AUROC is at least every untrained pick's. Exploration only: offline, never a finding.

Retry (Nov 20 2026, ``--extra``): the first run's SAE saw only 18k train tokens. The retry adds the backbone tokens
of 100 unedited closed-loop episodes on unused seeds (``modal_groot_steer_backbone.py --capture``, every other policy
call) to the SAE's training data only, and trains 6000 steps. Labels, the picker's train windows, the held-out windows,
the steered windows and the pass rule are unchanged.

    modal run --detach scripts/modal_groot_steer_backbone.py --capture
    II_MODAL_HOURS=0.5 modal run --detach scripts/modal_groot_backbone_sae.py --extra   # then --fetch --extra
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/repo/scripts")
from modal_groot_phase import CLIPS, RUN, SAE, _capture_labels  # noqa: E402
from modal_groot_steer import CAPTURE_DIR, GPU, VOL, WORK, _prepare, image, retrain_volume, volume  # noqa: E402

HOURS = float(os.environ.get("II_MODAL_HOURS", "0.5"))
OUT_BASE = f"{VOL}/work/backbone-sae"
OUT_EXTRA = f"{VOL}/work/backbone-sae-extra"
EXTRA = f"{WORK}/capture-backbone"  # retry (Nov 20): closed-loop episodes on seeds 90000+ (modal_groot_steer_backbone.py --capture)
SEEDS = [2217175575, 1703964116, 2684811898, 1837130788, 3864071313]  # the rule-4 untrained-SAE seeds
SCALE = 3.0
MIN_ACTIVE = 0.01
app = modal.App("ii-groot-backbone-sae", image=image.pip_install("scikit-learn"))


@app.function(volumes={VOL: volume, "/retrain": retrain_volume}, gpu=GPU, timeout=int(HOURS * 3600), cpu=4, memory=32768)
def backbone_sae(smoke: bool = False, controls: int = 19, n_latents: int = 4096, k: int = 32, steps: int = 4000,
                 extra: bool = False) -> dict:
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    _prepare()
    from interpretableintelligence.adapters.groot import random_sae, retrain_sae
    from interpretableintelligence.artifacts import read_json
    from interpretableintelligence.steer import Steerer

    dev = torch.device("cuda")
    X, y, split = _capture_labels()
    tr5 = split == "train"
    sae5 = load_file(SAE)
    mu5, sc5 = sae5["norm_mu"].float().numpy(), sae5["norm_scale"].float().reshape(-1).numpy()
    clf5 = LogisticRegression(C=0.05, max_iter=3000).fit((X[tr5] - mu5) / sc5, y[tr5])
    token_idx = np.asarray(read_json(Path(CAPTURE_DIR) / "manifest.json")["token_indices"])

    steerer = Steerer(Path(RUN), Path(CLIPS))
    adapter, backend = steerer.adapter, steerer.adapter.backend
    sp = read_json(Path(RUN) / "split.json")
    heldout, train_clips = sorted(map(str, sp["heldout"])), sorted(map(str, sp["train"]))
    if smoke:
        heldout, train_clips = heldout[:2], train_clips[:2]

    rows, toks = [], []
    for cid in train_clips + heldout:
        clip = steerer._clip(cid)
        for window in adapter.windows(clip):
            with torch.no_grad():
                with adapter.hook(steerer.ctx.layer, mode="capture") as handle:
                    adapter.encode(clip, window)
                a = handle.captured[0].detach().float().reshape(-1, handle.captured[0].shape[-1]).cpu().numpy()
                base, _ = adapter.predict_masked(clip, window)
                bb = adapter._context(clip, window)["backbone_features"].detach()[0].to(torch.float16).cpu()
            a = a[token_idx] if a.shape[0] == adapter.grid().w else a
            logit5 = float(((a.mean(0) - mu5) / sc5) @ clf5.coef_[0] + clf5.intercept_[0])
            rows.append({"clip": cid, "window": int(window.index), "label": int(logit5 > 0), "heldout": cid in heldout,
                         "base": base.detach().float().cpu().numpy()})
            toks.append(bb)
        print(f"captured {cid}: {len(rows)} windows", flush=True)
    extra_tok = None
    if extra:  # SAE training tokens only: never labelled, picked on, or steered
        tok_dir = Path(EXTRA + ("-smoke" if smoke else "")) / "baseline" / "tokens"
        files = sorted(tok_dir.glob("ep-*.npy"))
        if not files:
            raise SystemExit(f"no captured tokens in {tok_dir}: run modal_groot_steer_backbone.py --capture first")
        extra_tok = torch.cat([torch.from_numpy(np.load(f)).reshape(-1, 2048) for f in files])
        print(f"extra: {len(files)} episodes, {extra_tok.shape[0]} tokens", flush=True)

    lab = np.array([r["label"] for r in rows]); is_tr = np.array([not r["heldout"] for r in rows]); te = ~is_tr
    n_tok = np.array([t.shape[0] for t in toks])
    win_of_tok = np.repeat(np.arange(len(rows)), n_tok)
    T = torch.cat(toks).to(dev)  # [N, 2048] float16
    B = torch.stack([t.float().mean(0) for t in toks]).numpy()
    dom = B[is_tr][lab[is_tr] == 1].mean(0) - B[is_tr][lab[is_tr] == 0].mean(0)
    norm = float(np.linalg.norm(dom))
    tr_tok = torch.from_numpy(is_tr[win_of_tok]).to(dev)
    train_split = retrain_sae.Split(T[tr_tok], torch.zeros(int(tr_tok.sum()), dtype=torch.long, device=dev),
                                    torch.from_numpy(win_of_tok).to(dev)[tr_tok])
    if extra_tok is not None:
        xt = extra_tok.to(dev, torch.float16)
        train_split = retrain_sae.Split(torch.cat([train_split.acts, xt]), torch.zeros(train_split.n + xt.shape[0], dtype=torch.long, device=dev),
                                        torch.cat([train_split.episode, torch.full((xt.shape[0],), -1, dtype=torch.long, device=dev)]))
    held_split = retrain_sae.Split(T[~tr_tok], torch.zeros(int((~tr_tok).sum()), dtype=torch.long, device=dev),
                                   torch.from_numpy(win_of_tok).to(dev)[~tr_tok])
    print(f"tokens: train {train_split.n}, held-out {held_split.n}; windows {len(rows)}, phase1 {lab.mean():.2f}", flush=True)

    mu, scale = retrain_sae.norm_stats(train_split, None, "channel")
    config = retrain_sae.RetrainConfig(n_latents=n_latents, k=k, steps=200 if smoke else steps, norm="channel")
    trained, curve = retrain_sae.train(train_split, mu, scale, config, log=lambda s: None)
    saes = {"trained": trained}
    for seed in SEEDS:
        w = random_sae.untrained_weights(trained, seed)
        w = {key: v.to(dev) for key, v in w.items()}
        thr, _ = random_sae.batchtopk_threshold(w, train_split.chunks, k)
        w["threshold"] = torch.tensor(thr)
        saes[f"random:{seed}"] = {key: v.cpu() for key, v in w.items()}

    def window_feats(w: dict) -> np.ndarray:
        """[windows, latents] mean JumpReLU activation over each window's tokens."""
        wd = {key: v.float().to(dev) for key, v in w.items()}
        thr = float(wd["threshold"].reshape(-1)[0])
        out = torch.zeros(len(rows), n_latents, device=dev)
        wot = torch.from_numpy(win_of_tok).to(dev)
        for s in range(0, T.shape[0], 8192):
            pre = random_sae.pre_activations(wd, T[s:s + 8192])
            out.index_add_(0, wot[s:s + 8192], pre * (pre > thr))
        return (out / torch.from_numpy(n_tok).to(dev)[:, None].float()).cpu().numpy()

    evals = {"trained_heldout": retrain_sae.evaluate(trained, held_split)} if not smoke else {}
    picks = {}
    for name, w in saes.items():
        F = window_feats(w)
        active = (F[is_tr] > 0).mean(0) >= MIN_ACTIVE
        best, best_auc = None, -1.0
        for j in np.flatnonzero(active):
            if F[is_tr, j].std() == 0:
                continue
            a = roc_auc_score(lab[is_tr], F[is_tr, j])
            if a > best_auc:
                best, best_auc = int(j), float(a)
        raw = (w["W_dec"][best].float() * w["norm_scale"].float().reshape(-1)).numpy() if best is not None else None
        picks[name] = {
            "latent": best, "train_auroc": best_auc,
            "heldout_auroc": float(roc_auc_score(lab[te], F[te, best])) if best is not None and len(set(lab[te])) == 2 else None,
            "n_active_latents": int(active.sum()),
            "cos_with_dom": float(raw @ dom / (np.linalg.norm(raw) * norm)) if raw is not None else None,
            "vec": None if raw is None else raw / np.linalg.norm(raw) * norm * SCALE,
        }
        print(name, {key: v for key, v in picks[name].items() if key != "vec"}, flush=True)

    A = np.stack([r["base"][:16].reshape(-1) for r in rows])
    aclf = LogisticRegression(C=0.1, max_iter=5000).fit(A[is_tr], lab[is_tr])
    per_clip: dict[str, list] = {}
    for i, r in enumerate(rows):
        if r["heldout"] and r["label"] == 0:
            per_clip.setdefault(r["clip"], []).append(r)
    chosen = []
    for clip_rows in per_clip.values():
        idx = np.unique(np.linspace(0, len(clip_rows) - 1, min(3, len(clip_rows))).round().astype(int))
        chosen += [clip_rows[i] for i in idx]

    arms = {"dom_x3": SCALE * dom}
    arms.update({f"pick:{n}": p["vec"] for n, p in picks.items() if p["vec"] is not None})
    rng = np.random.default_rng(19)
    ctrl = rng.standard_normal((controls, B.shape[1])).astype(np.float32)
    ctrl /= np.linalg.norm(ctrl, axis=1, keepdims=True)
    names = list(arms) + [f"rand{c}" for c in range(controls)]
    vecs = list(arms.values()) + [SCALE * norm * c for c in ctrl]
    shift = {n: [] for n in names}
    for r in chosen:
        clip = steerer._clip(r["clip"])
        window = next(w for w in adapter.windows(clip) if int(w.index) == r["window"])
        ctx = adapter._context(clip, window)
        feats = ctx["backbone_features"]
        preds = []
        for s in range(0, len(vecs), 8):
            chunk = vecs[s:s + 8]
            delta = torch.tensor(np.stack(chunk), dtype=feats.dtype, device=feats.device)[:, None, :]
            edited = dict(ctx)
            edited["backbone_features"] = feats.expand(len(chunk), -1, -1) + delta
            noise = adapter._noise(clip, window).expand(len(chunk), -1, -1)
            with torch.no_grad():
                preds.append(backend.actions(edited, noise).float().cpu().numpy())
        preds = np.concatenate(preds)
        base = r["base"]; s0 = float(aclf.decision_function(base[:16].reshape(1, -1))[0])
        for n, p in zip(names, preds):
            shift[n].append(float(aclf.decision_function(p.reshape(base.shape)[:16].reshape(1, -1))[0]) - s0)

    mean = {n: float(np.mean(v)) if v else 0.0 for n, v in shift.items()}
    rand_max = max(mean[f"rand{c}"] for c in range(controls))
    untrained = [n for n in picks if n != "trained"]
    t = picks["trained"]
    passed = (
        "pick:trained" in mean
        and all(mean["pick:trained"] > mean.get(f"pick:{n}", -np.inf) for n in untrained)
        and mean["pick:trained"] > rand_max
        and t["heldout_auroc"] is not None
        and all(t["heldout_auroc"] >= (picks[n]["heldout_auroc"] or 0) for n in untrained)
    )
    out = {
        "windows": len(rows), "train_tokens": train_split.n, "heldout_tokens": held_split.n, "phase1_frac": float(lab.mean()),
        "sae": {"n_latents": n_latents, "k": k, "steps": config.steps, "norm": "channel", "final_fvu": curve[-1]["fvu"] if curve else None,
                "dead_at_end": curve[-1]["dead"] if curve else None, **{key: v for key, v in evals.get("trained_heldout", {}).items()
                                                                    if isinstance(v, (int, float, str, type(None)))}},
        "dom_norm": norm, "scale": SCALE, "steered_windows": len(chosen), "controls": controls,
        "picks": {n: {key: v for key, v in p.items() if key != "vec"} for n, p in picks.items()},
        "phase_shift": {n: mean[n] for n in arms},
        "windows_toward_phase1": {n: float(np.mean(np.array(shift[n]) > 0)) if shift[n] else None for n in arms},
        "random_direction_shift_max": rand_max, "random_direction_shift_mean": float(np.mean([mean[f"rand{c}"] for c in range(controls)])),
        "pass": bool(passed), "finding": False,
        "note": "offline exploration; a pass means the trained SAE beats untrained SAEs on this lever, not a rollout finding",
    }
    out["extra_clips"] = bool(extra)
    if not smoke:
        outdir = OUT_EXTRA if extra else OUT_BASE
        Path(outdir).mkdir(parents=True, exist_ok=True)
        save_file({key: v.contiguous() for key, v in trained.items()}, f"{outdir}/sae-trained.safetensors")
        np.save(f"{outdir}/picks.npy", np.stack([p["vec"] for p in picks.values()]))
        Path(outdir, "result.json").write_text(json.dumps(out, indent=1) + "\n")
        volume.commit()
    return out


@app.function(volumes={VOL: volume}, timeout=300)
def fetch_remote(extra: bool = False) -> dict:
    path = Path(OUT_EXTRA if extra else OUT_BASE, "result.json")
    out = json.loads(path.read_text()) if path.is_file() else {"missing": str(path)}
    out["extra_captured_episodes"] = len(list(Path(EXTRA, "baseline", "tokens").glob("ep-*.npy")))
    return out


@app.local_entrypoint()
def main(smoke: bool = False, fetch: bool = False, extra: bool = False, out: str = "backbone-sae.json") -> None:
    if not smoke and not fetch:  # spawn under --detach so a closed launcher cannot cancel it
        print("spawned", backbone_sae.spawn(False, 19, 4096, 32, 6000 if extra else 4000, extra).object_id, "- fetch with --fetch")
        return
    result = fetch_remote.remote(extra) if fetch else backbone_sae.remote(True, 3, 4096, 32, 4000, extra)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
