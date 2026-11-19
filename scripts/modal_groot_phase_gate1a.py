"""Gate 1a of the task 1028 phase-steering plan (Nov 19 2026): best single latent for the pot phase in the rescaled SAE
vs 5 untrained SAEs (exp 2 seeds), and decoder alignment with the phase directions. CPU only. Results: gate1a.json."""
import modal
REPO_SRC = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix() + "/src/interpretableintelligence/adapters/groot/random_sae.py"
vol = modal.Volume.from_name("ii-groot-steer")
rvol = modal.Volume.from_name("ii-groot-sae-retrain")
image = (modal.Image.debian_slim().pip_install("numpy", "safetensors", "scikit-learn")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
         .add_local_file(REPO_SRC, "/pkg/random_sae.py"))
app = modal.App("ii-gate1a", image=image)
ROOT = "/vol/work/groot-1028-v9-n100"
SEEDS = [2217175575, 1703964116, 2684811898, 1837130788, 3864071313]

@app.function(volumes={"/vol": vol, "/retrain": rvol}, timeout=3000, cpu=8, memory=32768)
def gate1a():
    import json, sys, numpy as np, torch
    sys.path.insert(0, "/pkg")
    import random_sae as rs
    from pathlib import Path
    from safetensors.numpy import load_file
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    torch.set_num_threads(8)
    eps = [json.loads(l) for l in open(f"{ROOT}/episodes.jsonl")]
    LO, HI = np.array([-0.1292, -0.2819]), np.array([0.0349, -0.1047])
    toks, y, split, tsec, chunk_of = [], [], [], [], []
    ci = 0
    for e in eps:
        z = np.load(f"{ROOT}/sim/{e['episode_id']}.npz", allow_pickle=True)
        P, gw = z["obj_pos"], z["gripper_width"]
        zr = P[:, 1, 2] - P[0, 1, 2]
        done1 = np.zeros(len(P), bool)
        if (zr > 0.02).any():
            l1 = int(np.argmax(zr > 0.02))
            inreg = ((P[:, 1, :2] >= LO) & (P[:, 1, :2] <= HI)).all(1)
            c = np.where((np.arange(len(P)) > l1) & inreg & (gw >= 0.078))[0]
            if len(c): done1[c[0]:] = True
        a = load_file(f"{ROOT}/acts/shard-%05d.safetensors" % e["episode_idx"])
        acts, step, tok = a["acts"], a["step"], a["tok"]
        keep = tok >= 0
        acts, step = acts[keep], step[keep]
        for s in np.unique(step):
            if s >= len(P): continue
            m = step == s
            toks.append(acts[m]); y.append(int(done1[s])); split.append(e["split"]); tsec.append(int(s)); chunk_of.append(np.full(m.sum(), ci)); ci += 1
    T = torch.from_numpy(np.concatenate(toks).astype(np.float32)); cid = np.concatenate(chunk_of)
    y, split, tsec = np.array(y), np.array(split), np.array(tsec)
    tr, te = split == "train", split == "heldout"
    mid = te & (tsec >= 160) & (tsec <= 320)
    trained = rs.load_tensors(Path("/retrain/runs/auxk-8192-chnorm/sae/weights.safetensors"))
    cfg = json.loads(Path("/retrain/runs/auxk-8192-chnorm/sae/config.json").read_text())
    k = int(cfg.get("k") or cfg.get("train", {}).get("k") or 32)
    tok_tr = T[np.isin(cid, np.where(tr)[0])]
    def chunk_feats(w):
        th = float(w["threshold"].float().reshape(-1)[0])
        F = torch.zeros(ci, w["W_enc"].shape[1]); cnt = torch.zeros(ci)
        idx = torch.from_numpy(cid)
        for s in range(0, len(T), 8192):
            pre = rs.pre_activations(w, T[s:s+8192]); act = pre * (pre > th)
            F.index_add_(0, idx[s:s+8192], act); cnt.index_add_(0, idx[s:s+8192], torch.ones(len(act)))
        return (F / cnt[:, None]).numpy()
    # probe direction in the SAE's normalised space
    Xn = ((T - trained["norm_mu"]) / trained["norm_scale"].float().reshape(-1)).numpy()
    Xc = np.zeros((ci, Xn.shape[1]), np.float32); np.add.at(Xc, cid, Xn); Xc /= np.bincount(cid)[:, None]
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(Xc[tr], y[tr])
    wdir = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    dom = Xc[tr][y[tr] == 1].mean(0) - Xc[tr][y[tr] == 0].mean(0); dom /= np.linalg.norm(dom)
    res = {"k": k, "chunks": int(ci), "probe_normspace_auroc_heldout": float(roc_auc_score(y[te], clf.decision_function(Xc[te]))),
           "dom_normspace_auroc_heldout": float(roc_auc_score(y[te], Xc[te] @ dom)),
           "dom_normspace_auroc_mid": float(roc_auc_score(y[mid], Xc[mid] @ dom)), "time_auroc_mid": float(roc_auc_score(y[mid], tsec[mid])),
           "dicts": {}}
    np.save("/vol/work/gate1a-dom-normspace.npy", dom.astype(np.float32)); np.save("/vol/work/gate1a-probe-normspace.npy", wdir.astype(np.float32))
    dicts = [("trained", trained)]
    for sd in SEEDS:
        w = rs.untrained_weights(trained, sd)
        th, _ = rs.batchtopk_threshold(w, lambda: (tok_tr[i:i+8192] for i in range(0, len(tok_tr), 8192)), k)
        w["threshold"] = torch.tensor(th); dicts.append((f"random:{sd}", w))
    for name, w in dicts:
        F = chunk_feats(w)
        alive = F[tr].max(0) > 0
        a_tr = np.full(F.shape[1], 0.5)
        for j in np.where(alive)[0]:
            a_tr[j] = roc_auc_score(y[tr], F[tr, j])
        score = np.abs(a_tr - 0.5); top = np.argsort(-score)[:5]
        out = []
        for j in top:
            sgn = 1 if a_tr[j] >= 0.5 else -1
            out.append({"latent": int(j), "train_auroc": float(a_tr[j]), "heldout_auroc": float(roc_auc_score(y[te], sgn * F[te, j])),
                        "mid_auroc": float(roc_auc_score(y[mid], sgn * F[mid, j])), "fire_rate": float((F[:, j] > 0).mean())})
        D = w["W_dec"].float().numpy(); Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
        cos_dom = Dn @ dom; cos_probe = Dn @ wdir
        res["dicts"][name] = {"alive_train": int(alive.sum()), "top": out,
                              "best_heldout_of_train_top1": out[0]["heldout_auroc"],
                              "max_abs_cos_dom": float(np.abs(cos_dom).max()), "argmax_cos_dom": int(np.abs(cos_dom).argmax()),
                              "max_abs_cos_probe": float(np.abs(cos_probe).max())}
        print(name, json.dumps(res["dicts"][name]["top"][:2]), res["dicts"][name]["max_abs_cos_dom"], flush=True)
    vol.commit()
    return res

@app.local_entrypoint()
def main():
    import json
    r = gate1a.remote()
    open("gate1a.json", "w").write(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "dicts"}, indent=1))
    for n, d in r["dicts"].items():
        print(n, d["alive_train"], d["top"][0], d["max_abs_cos_dom"], d["max_abs_cos_probe"])
