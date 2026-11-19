"""Gate 0 of the task 1028 phase-steering plan (Nov 19 2026): does any behaviour vary in the capture, and is the pot phase
linearly readable at block 5? CPU only. Results: project folder groot-phase-steer/gate0.json."""
import modal
vol = modal.Volume.from_name("ii-groot-steer")
app = modal.App("ii-gate0", image=modal.Image.debian_slim().pip_install("numpy", "safetensors", "scikit-learn"))
ROOT = "/vol/work/groot-1028-v9-n100"

@app.function(volumes={"/vol": vol}, timeout=1200, cpu=4, memory=16384)
def gate0():
    import json, numpy as np
    from safetensors.numpy import load_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    eps = [json.loads(l) for l in open(f"{ROOT}/episodes.jsonl")]
    LO, HI = np.array([-0.1292, -0.2819]), np.array([0.0349, -0.1047])
    out = {"episodes": len(eps)}
    first_lift = {"pot0": 0, "pot1": 0, "none": 0}
    X, y, grp, split, tsec = [], [], [], [], []
    regrasp = []
    for e in eps:
        z = np.load(f"{ROOT}/sim/{e['episode_id']}.npz", allow_pickle=True)
        P, eef, gw = z["obj_pos"], z["eef_pos"], z["gripper_width"]
        zr = P[:, :, 2] - P[0, :, 2]
        lift = [np.argmax(zr[:, k] > 0.02) if (zr[:, k] > 0.02).any() else 10**9 for k in (0, 1)]
        first_lift["none" if min(lift) == 10**9 else ("pot0" if lift[0] < lift[1] else "pot1")] += 1
        regrasp.append(len(e.get("pot0_close_attempts") or []))
        # phase label per env step: 1 once pot 1 has been lifted and is resting in the stove region with the gripper open
        inreg = ((P[:, 1, :2] >= LO) & (P[:, 1, :2] <= HI)).all(1)
        done1 = np.zeros(len(P), bool)
        if lift[1] < 10**9:
            cand = np.where((np.arange(len(P)) > lift[1]) & inreg & (gw >= 0.078))[0]
            if len(cand): done1[cand[0]:] = True
        a = load_file(f"{ROOT}/acts/{e['episode_id']}.safetensors".replace(e['episode_id'], "shard-%05d" % e["episode_idx"]))
        acts, step, tok = a["acts"].astype(np.float32), a["step"], a["tok"]
        keep = tok >= 0
        acts, step, tok = acts[keep], step[keep], tok[keep]
        for s in np.unique(step):
            if s >= len(P): continue
            m = step == s
            X.append(acts[m].mean(0)); y.append(int(done1[s])); grp.append(e["episode_idx"]); split.append(e["split"]); tsec.append(s)
    X, y, grp, split, tsec = map(np.asarray, (X, y, grp, split, tsec))
    tr, te = split == "train", split == "heldout"
    out.update(first_lift=first_lift, chunks=int(len(y)), frac_phase_pot0=float(y.mean()),
               episodes_reaching_pot0_phase=int(len({g for g, v in zip(grp, y) if v})),
               regrasp_hist={int(k): int(v) for k, v in zip(*np.unique(regrasp, return_counts=True))},
               fail_kinds={str(k): int(v) for k, v in zip(*np.unique([str(e.get("fail_kind")) for e in eps], return_counts=True))})
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(Z[tr], y[tr])
    out["probe_auroc_heldout"] = float(roc_auc_score(y[te], clf.decision_function(Z[te])))
    # control: time alone
    out["time_only_auroc_heldout"] = float(roc_auc_score(y[te], tsec[te]))
    # diff-of-means direction (train) projected on heldout
    dm = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
    out["dom_auroc_heldout"] = float(roc_auc_score(y[te], X[te] @ dm))
    out["dom_norm"] = float(np.linalg.norm(dm)); out["act_norm_mean"] = float(np.linalg.norm(X, axis=1).mean())
    # within the same time window (steps 160-320), is the phase still decodable beyond time?
    w = te & (tsec >= 160) & (tsec <= 320)
    if len(set(y[w])) == 2:
        out["probe_auroc_heldout_mid"] = float(roc_auc_score(y[w], clf.decision_function(Z[w])))
        out["time_only_auroc_mid"] = float(roc_auc_score(y[w], tsec[w]))
        out["mid_n"] = int(w.sum())
    np.save("/vol/work/gate0-dom.npy", dm.astype(np.float32)); vol.commit()
    return out

@app.local_entrypoint()
def main():
    import json
    r = gate0.remote()
    print(json.dumps(r, indent=1))
    open("gate0.json", "w").write(json.dumps(r, indent=1))
