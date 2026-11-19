"""Offline gate for the task 1028 "steer a named behaviour" plan (Nov 19 2026): does a supervised direction for the
pot phase (moving the first pot vs the second) at DiT block 5 move GR00T's planned actions toward the other phase,
more than matched random directions? Exploration on held-out recorded clips, never a finding.

    modal run scripts/modal_groot_phase.py --smoke      # one held-out clip, 3 controls
    II_MODAL_HOURS=0.75 modal run --detach scripts/modal_groot_phase.py

Phase label per policy call (capture sim traces): 1 once pot 1 (moka_pot_2, always moved first) has been lifted and
rests in the stove region with the gripper open, else 0. The recorded clips have no object poses, so their windows
are labelled by a logistic probe fitted on the capture's train episodes (held-out AUROC 0.998).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/repo/scripts")  # in the container this file runs from /root; the repo copy has modal_groot_steer
from modal_groot_steer import CAPTURE_DIR, GPU, VOL, WORK, _prepare, image, retrain_volume, volume  # noqa: E402

HOURS = float(os.environ.get("II_MODAL_HOURS", "0.75"))
RUN, CLIPS = f"{WORK}/run-rescaled", f"{WORK}/clips-run"
SAE = "/retrain/runs/auxk-8192-chnorm/sae/weights.safetensors"
OUT = f"{VOL}/results-phase"
STOVE_LO, STOVE_HI = (-0.1292, -0.2819), (0.0349, -0.1047)
app = modal.App("ii-groot-phase", image=image.pip_install("scikit-learn"))


def _capture_labels():
    import numpy as np
    from safetensors.numpy import load_file

    root = Path(CAPTURE_DIR)
    lo, hi = np.array(STOVE_LO), np.array(STOVE_HI)
    X, y, split = [], [], []
    for line in (root / "episodes.jsonl").read_text().splitlines():
        e = json.loads(line)
        z = np.load(root / "sim" / f"{e['episode_id']}.npz", allow_pickle=True)
        P, gw = z["obj_pos"], z["gripper_width"]
        zr = P[:, 1, 2] - P[0, 1, 2]
        done = np.zeros(len(P), bool)
        if (zr > 0.02).any():
            lift = int(np.argmax(zr > 0.02))
            inreg = ((P[:, 1, :2] >= lo) & (P[:, 1, :2] <= hi)).all(1)
            c = np.where((np.arange(len(P)) > lift) & inreg & (gw >= 0.078))[0]
            if len(c):
                done[c[0]:] = True
        a = load_file(str(root / "acts" / ("shard-%05d.safetensors" % e["episode_idx"])))
        keep = a["tok"] >= 0
        acts, step = a["acts"][keep].astype(np.float32), a["step"][keep]
        for s in np.unique(step):
            if s < len(P):
                X.append(acts[step == s].mean(0)); y.append(int(done[s])); split.append(e["split"])
    return np.array(X), np.array(y), np.array(split)


@app.function(volumes={VOL: volume, "/retrain": retrain_volume}, gpu=GPU, timeout=int(HOURS * 3600), cpu=4, memory=24576)
def phase_gate(smoke: bool = False, controls: int = 19) -> dict:
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    _prepare()
    from interpretableintelligence.artifacts import read_json
    from interpretableintelligence.steer import Steerer
    from interpretableintelligence.steer_scorecard import _delta_edit, _predict_many

    X, y, split = _capture_labels()
    tr = split == "train"
    sae = load_file(SAE)
    mu, scale = sae["norm_mu"].float().numpy(), sae["norm_scale"].float().reshape(-1).numpy()
    Xn = (X - mu) / scale
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(Xn[tr], y[tr])
    w_raw = clf.coef_[0] / scale  # logit = w_raw . x + const
    logit = X @ w_raw
    gap = float(logit[tr][y[tr] == 1].mean() - logit[tr][y[tr] == 0].mean())
    probe_dir = gap * w_raw / float(w_raw @ w_raw)  # shifts the logit by the class-mean gap
    dom = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
    dec = sae["W_dec"].float().numpy()[3008]  # trained latent best aligned with dom (cos 0.62, gate 1a)
    dec = dec * scale  # decoder rows live in normalised space
    dec = dec / np.linalg.norm(dec) * np.linalg.norm(dom)
    # every arm at the same norm (the class-mean difference), so one set of random-direction controls serves all
    norm = float(np.linalg.norm(dom))
    arms = {"dom": dom, "probe_dir_at_dom_norm": probe_dir / np.linalg.norm(probe_dir) * norm, "sae3008_at_dom_norm": dec,
            "minus_dom": -dom}
    rng = np.random.default_rng(19)
    ctrl_dirs = rng.standard_normal((controls, X.shape[1])).astype(np.float32)
    ctrl_dirs /= np.linalg.norm(ctrl_dirs, axis=1, keepdims=True)
    token_idx = read_json(Path(CAPTURE_DIR) / "manifest.json").get("token_indices")

    steerer = Steerer(Path(RUN), Path(CLIPS))
    adapter = steerer.adapter
    heldout = sorted(str(c) for c in read_json(Path(RUN) / "split.json")["heldout"])
    train_clips = sorted(str(c) for c in read_json(Path(RUN) / "split.json")["train"])
    if smoke:
        heldout, train_clips = heldout[:1], train_clips[:1]

    def window_pass(clip, window, edits):
        with torch.no_grad():
            with adapter.hook(steerer.ctx.layer, mode="capture") as handle:
                adapter.encode(clip, window)
            acts = handle.captured[0].detach().float()
            base, _ = adapter.predict_masked(clip, window)
            preds = _predict_many(adapter, clip, window, edits) if edits else []
        a = acts.reshape(-1, acts.shape[-1]).cpu().numpy()
        if token_idx is not None and a.shape[0] == adapter.grid().w:
            a = a[np.asarray(token_idx)]
        return acts.shape, float((a.mean(0) - mu) / scale @ clf.coef_[0] + clf.intercept_[0]), base.detach().float().cpu().numpy(), [p.detach().float().cpu().numpy() for p in preds]

    # 1. baseline predictions + probe phase on every window; train clips fit the action-space phase classifier
    rows = []
    for cid in train_clips + heldout:
        clip = steerer._clip(cid)
        for window in adapter.windows(clip):
            shape, lg, base, _ = window_pass(clip, window, [])
            rows.append({"clip": cid, "window": int(window.index), "logit": lg,
                         "base": base, "shape": shape, "heldout": cid in heldout})
    lab = np.array([int(r["logit"] > 0) for r in rows])
    A = np.stack([r["base"][:16].reshape(-1) for r in rows])
    is_tr = np.array([not r["heldout"] for r in rows])
    aclf = LogisticRegression(C=0.1, max_iter=5000).fit(A[is_tr], lab[is_tr]) if len(set(lab[is_tr])) == 2 else None
    action_auroc = float(roc_auc_score(lab[~is_tr], aclf.decision_function(A[~is_tr]))) if aclf is not None and len(set(lab[~is_tr])) == 2 else None

    # 2. steer phase-0 held-out windows toward phase 1
    effects = {name: {"feature": [], "controls": [[] for _ in range(controls)]} for name in arms}
    rel = {name: {"feature": [], "controls": [[] for _ in range(controls)]} for name in arms}
    n_win = 0
    per_clip: dict[str, list] = {}
    for r in rows:
        if r["heldout"] and r["logit"] <= 0:
            per_clip.setdefault(r["clip"], []).append(r)
    chosen = []
    for clip_rows in per_clip.values():  # at most 3 phase-0 windows per clip, spread over the phase, to bound GPU time
        idx = np.unique(np.linspace(0, len(clip_rows) - 1, min(3, len(clip_rows))).round().astype(int))
        chosen += [clip_rows[i] for i in idx]
    for r in chosen:
        clip = steerer._clip(r["clip"])
        window = next(w for w in adapter.windows(clip) if int(w.index) == r["window"])
        edits, keys = [], []
        for name, vec in arms.items():
            v = torch.tensor(vec, dtype=torch.float32)
            edits.append(_delta_edit(v.expand(*r["shape"][:-1], -1).clone())); keys.append((name, -1))
        for c in range(controls):
            cv = torch.tensor(ctrl_dirs[c] * norm, dtype=torch.float32)
            edits.append(_delta_edit(cv.expand(*r["shape"][:-1], -1).clone())); keys.append(("__control__", c))
        _, _, base, preds = window_pass(clip, window, edits)
        b16 = base[:16].reshape(1, -1)
        s0 = float(aclf.decision_function(b16)[0]) if aclf is not None else 0.0
        for (name, c), p in zip(keys, preds):
            p = p.reshape(base.shape)
            shift = (float(aclf.decision_function(p[:16].reshape(1, -1))[0]) - s0) if aclf is not None else 0.0
            change = float(np.linalg.norm(p - base) / max(np.linalg.norm(base), 1e-12))
            for target in (arms if name == "__control__" else [name]):
                (effects[target]["feature"] if c < 0 else effects[target]["controls"][c]).append(shift)
                (rel[target]["feature"] if c < 0 else rel[target]["controls"][c]).append(change)
        n_win += 1
        print(f"window {n_win}: {r['clip']}#{r['window']}", flush=True)

    win_idx = np.array([r["window"] for r in rows]); lg_all = np.array([r["logit"] for r in rows])
    summary = {"recorded_probe_vs_window_index_auroc": float(roc_auc_score(lab, win_idx)) if len(set(lab)) == 2 else None,
               "probe_heldout_auroc_capture": float(roc_auc_score(y[~tr], Xn[~tr] @ clf.coef_[0])),
               "recorded_windows": len(rows), "recorded_phase1_frac": float(lab.mean()),
               "action_classifier_heldout_auroc": action_auroc, "steered_windows": n_win, "controls": controls,
               "norms": {k: float(np.linalg.norm(v)) for k, v in arms.items()}, "mean_act_norm": float(np.linalg.norm(X, axis=1).mean()),
               "arms": {}}
    for name in arms:
        f = float(np.mean(effects[name]["feature"])) if n_win else 0.0
        cs = [float(np.mean(c)) for c in effects[name]["controls"]] if n_win else []
        fr = float(np.mean(rel[name]["feature"])) if n_win else 0.0
        crs = [float(np.mean(c)) for c in rel[name]["controls"]] if n_win else []
        frac_toward = float(np.mean(np.array(effects[name]["feature"]) > 0)) if n_win else 0.0
        summary["arms"][name] = {"phase_shift": f, "control_phase_shift_max": max(cs) if cs else None,
                                 "control_phase_shift_mean": float(np.mean(cs)) if cs else None,
                                 "beats_all_controls": bool(cs) and f > max(cs), "p_mc": (1 + sum(c >= f for c in cs)) / (1 + len(cs)) if cs else None,
                                 "windows_toward_phase1": frac_toward, "rel_action_change": fr,
                                 "control_rel_change_mean": float(np.mean(crs)) if crs else None}
    if not smoke:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        Path(OUT, "phase-gate.json").write_text(json.dumps(summary, indent=1) + "\n")
        volume.commit()
    return summary


@app.local_entrypoint()
def main(smoke: bool = False, controls: int = 19, out: str = "phase-gate.json") -> None:
    result = phase_gate.remote(smoke, 3 if smoke else controls)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
