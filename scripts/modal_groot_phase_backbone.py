"""The task 1028 pot-phase check moved to the VLM backbone output (Nov 19 2026): is the phase (moving the first pot vs
the second) readable from the backbone features the action head cross-attends to, and does steering along it move
GR00T's planned actions toward second-pot actions more than matched random directions? Exploration on held-out
recorded clips, never a finding.

    modal run scripts/modal_groot_phase_backbone.py --smoke
    II_MODAL_HOURS=0.5 modal run scripts/modal_groot_phase_backbone.py

Labels: the recorded clips have no object poses, so each window's phase comes from the block-5 probe of
modal_groot_phase.py (capture held-out AUROC 0.998). Edits add one vector to every backbone token
(``context["backbone_features"]``); the backbone is not re-run.
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
from modal_groot_steer import CAPTURE_DIR, GPU, VOL, _prepare, image, retrain_volume, volume  # noqa: E402

HOURS = float(os.environ.get("II_MODAL_HOURS", "0.5"))
OUT = f"{VOL}/results-phase"
app = modal.App("ii-groot-phase-backbone", image=image.pip_install("scikit-learn"))


@app.function(volumes={VOL: volume, "/retrain": retrain_volume}, gpu=GPU, timeout=int(HOURS * 3600), cpu=4, memory=24576)
def backbone_gate(smoke: bool = False, controls: int = 19, save_dom: str = "") -> dict:
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    _prepare()
    from interpretableintelligence.artifacts import read_json
    from interpretableintelligence.steer import Steerer

    # block-5 phase probe on the capture (the labeller)
    X, y, split = _capture_labels()
    tr = split == "train"
    sae = load_file(SAE)
    mu, scale = sae["norm_mu"].float().numpy(), sae["norm_scale"].float().reshape(-1).numpy()
    clf5 = LogisticRegression(C=0.05, max_iter=3000).fit((X[tr] - mu) / scale, y[tr])
    token_idx = np.asarray(read_json(Path(CAPTURE_DIR) / "manifest.json")["token_indices"])

    steerer = Steerer(Path(RUN), Path(CLIPS))
    adapter, backend = steerer.adapter, steerer.adapter.backend
    sp = read_json(Path(RUN) / "split.json")
    heldout, train_clips = sorted(map(str, sp["heldout"])), sorted(map(str, sp["train"]))
    if smoke:
        heldout, train_clips = heldout[:2], train_clips[:2]

    rows = []
    for cid in train_clips + heldout:
        clip = steerer._clip(cid)
        for window in adapter.windows(clip):
            with torch.no_grad():
                with adapter.hook(steerer.ctx.layer, mode="capture") as handle:
                    adapter.encode(clip, window)
                a = handle.captured[0].detach().float().reshape(-1, handle.captured[0].shape[-1]).cpu().numpy()
                base, _ = adapter.predict_masked(clip, window)
                bb = adapter._context(clip, window)["backbone_features"].detach().float()
            a = a[token_idx] if a.shape[0] == adapter.grid().w else a
            logit5 = float(((a.mean(0) - mu) / scale) @ clf5.coef_[0] + clf5.intercept_[0])
            rows.append({"clip": cid, "window": int(window.index), "label": int(logit5 > 0), "heldout": cid in heldout,
                         "bb": bb[0].mean(0).cpu().numpy(), "n_tok": int(bb.shape[1]), "base": base.detach().float().cpu().numpy()})
    lab = np.array([r["label"] for r in rows]); is_tr = np.array([not r["heldout"] for r in rows])
    B = np.stack([r["bb"] for r in rows]); win = np.array([r["window"] for r in rows])
    if save_dom:  # the rollout test's fixed vector: class-mean difference over every recorded window (train + held-out)
        vec = (B[lab == 1].mean(0) - B[lab == 0].mean(0)).astype(np.float32)
        Path(save_dom).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_dom, vec)
        volume.commit()
        return {"saved": save_dom, "norm": float(np.linalg.norm(vec)), "windows": len(rows), "phase1_frac": float(lab.mean()),
                "cos_with_train_only_dom": None}
    bmu, bsd = B[is_tr].mean(0), B[is_tr].std(0) + 1e-6
    probe = LogisticRegression(C=0.05, max_iter=5000).fit((B[is_tr] - bmu) / bsd, lab[is_tr])
    score = ((B - bmu) / bsd) @ probe.coef_[0]
    te = ~is_tr

    def auc(mask, s):
        return float(roc_auc_score(lab[mask], s[mask])) if len(set(lab[mask])) == 2 else None

    mid = te & (win >= 8) & (win <= 14)  # policy calls 8-14 (env steps 128-224): both phases occur here
    dom = B[is_tr][lab[is_tr] == 1].mean(0) - B[is_tr][lab[is_tr] == 0].mean(0)
    norm = float(np.linalg.norm(dom))
    pdir = probe.coef_[0] / bsd
    arms = {"dom": dom, "probe_dir_at_dom_norm": pdir / np.linalg.norm(pdir) * norm, "minus_dom": -dom,
            "dom_x3": 3 * dom}
    rng = np.random.default_rng(19)
    ctrl = rng.standard_normal((controls, B.shape[1])).astype(np.float32)
    ctrl /= np.linalg.norm(ctrl, axis=1, keepdims=True)

    A = np.stack([r["base"][:16].reshape(-1) for r in rows])
    aclf = LogisticRegression(C=0.1, max_iter=5000).fit(A[is_tr], lab[is_tr])

    per_clip: dict[str, list] = {}
    for r in rows:
        if r["heldout"] and r["label"] == 0:
            per_clip.setdefault(r["clip"], []).append(r)
    chosen = []
    for clip_rows in per_clip.values():
        idx = np.unique(np.linspace(0, len(clip_rows) - 1, min(3, len(clip_rows))).round().astype(int))
        chosen += [clip_rows[i] for i in idx]

    shift = {n: [] for n in arms}; rel = {n: [] for n in arms}
    cshift = [[] for _ in range(controls)]; crel = [[] for _ in range(controls)]
    cshift3 = [[] for _ in range(controls)]; crel3 = [[] for _ in range(controls)]
    for k, r in enumerate(chosen):
        clip = steerer._clip(r["clip"])
        window = next(w for w in adapter.windows(clip) if int(w.index) == r["window"])
        ctx = adapter._context(clip, window)
        feats = ctx["backbone_features"]
        vecs = [arms[n] for n in arms] + [3 * norm * c for c in ctrl] + [norm * c for c in ctrl]
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
        for j, p in enumerate(preds):
            p = p.reshape(base.shape)
            sh = float(aclf.decision_function(p[:16].reshape(1, -1))[0]) - s0
            ch = float(np.linalg.norm(p - base) / max(np.linalg.norm(base), 1e-12))
            if j < len(arms):
                n = list(arms)[j]; shift[n].append(sh); rel[n].append(ch)
            elif j < len(arms) + controls:
                c = j - len(arms); cshift3[c].append(sh); crel3[c].append(ch)
            else:
                c = j - len(arms) - controls; cshift[c].append(sh); crel[c].append(ch)
        print(f"window {k + 1}/{len(chosen)}", flush=True)

    out = {"windows": len(rows), "phase1_frac": float(lab.mean()), "backbone_tokens": rows[0]["n_tok"] if rows else None,
           "backbone_dim": int(B.shape[1]), "probe_heldout_auroc": auc(te, score), "probe_heldout_auroc_mid": auc(mid, score),
           "time_only_auroc_mid": auc(mid, win.astype(float)), "mid_n": int(mid.sum()),
           "dom_heldout_auroc": auc(te, B @ dom), "action_classifier_heldout_auroc": auc(te, aclf.decision_function(A)),
           "steered_windows": len(chosen), "controls": controls, "dom_norm": norm,
           "mean_bb_norm": float(np.linalg.norm(B, axis=1).mean()), "arms": {}}
    for n in arms:
        f = float(np.mean(shift[n])) if shift[n] else 0.0
        three = n == "dom_x3"
        cs = [float(np.mean(c)) for c in (cshift3 if three else cshift)] if shift[n] else []
        cr = [float(np.mean(c)) for c in (crel3 if three else crel)] if shift[n] else []
        out["arms"][n] = {"phase_shift": f, "control_shift_max": max(cs) if cs else None, "control_shift_mean": float(np.mean(cs)) if cs else None,
                          "beats_all_controls": bool(cs) and f > max(cs), "p_mc": (1 + sum(c >= f for c in cs)) / (1 + len(cs)) if cs else None,
                          "windows_toward_phase1": float(np.mean(np.array(shift[n]) > 0)) if shift[n] else None,
                          "rel_action_change": float(np.mean(rel[n])) if rel[n] else None,
                          "control_rel_change_mean": float(np.mean(cr)) if cr else None}
    if not smoke:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        Path(OUT, "backbone-gate.json").write_text(json.dumps(out, indent=1) + "\n")
        volume.commit()
    return out


@app.local_entrypoint()
def main(smoke: bool = False, controls: int = 19, out: str = "backbone-gate.json", save_dom: str = "") -> None:
    result = backbone_gate.remote(smoke, 3 if smoke else controls, save_dom)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
