"""Validation of pick_candidates.py on synthetic rollouts. Writes picker/synth_results.json.
  nice -n 19 env OMP_NUM_THREADS=2 .venv/bin/python -u picker/synth_test.py [--null-seeds 30]"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import synth_rollouts as S
import pick_candidates as P

ap = argparse.ArgumentParser(); ap.add_argument("--null-seeds", type=int, default=30); ap.add_argument("--plant-seeds", type=int, default=5)
ap.add_argument("--root", default="/tmp/synth_picker"); a = ap.parse_args()
root = Path(a.root); res = dict(planted=[], weak=[], downstream=None, null={}, adapter=None)

def run(d, *extra):
    r = P.main([str(d), "--quiet", *extra])[0]
    con = [json.loads(l) for l in open(Path(d) / "picker" / "contrast.jsonl")]
    rank = next((i + 1 for i, c in enumerate(con) if c["feature_id"] == S.PLANT), None)
    return r, con, rank

t0 = time.time()
for sd in range(a.plant_seeds):  # planted, strong-ish (fires on 50% of tokens in 8 steps before failure)
    d = S.make(root / f"plant-{sd}", plant=0.5, seed=100 + sd); r, con, rank = run(d)
    excl = r["features"]["excluded"]
    res["planted"].append(dict(seed=100 + sd, rank=rank, q=con[rank - 1]["q"] if rank else None, candidates=r["candidate_ids"], excluded=excl,
                               offline_pct=(r["candidates"][0]["offline"]["pct_vs_random"] if r["candidates"] else None)))
    print("planted", res["planted"][-1], flush=True); shutil.rmtree(d)
for sd in range(a.plant_seeds):  # weak plant: 15% of tokens
    d = S.make(root / f"weak-{sd}", plant=0.15, seed=200 + sd); r, con, rank = run(d)
    res["weak"].append(dict(seed=200 + sd, rank=rank, q=con[rank - 1]["q"] if rank else None, candidates=r["candidate_ids"]))
    print("weak", res["weak"][-1], flush=True); shutil.rmtree(d)
d = S.make(root / "ds", plant=0.5, seed=300); r, con, rank = run(d, "--downstream", "picker.synth_head:act")
res["downstream"] = dict(rank=rank, candidates=r["candidate_ids"], metric=r["offline_metric"],
                         offline=(r["candidates"][0]["offline"] if r["candidates"] else None))
print("downstream", res["downstream"], flush=True); shutil.rmtree(d)
for mode, extra in (("time+detrend", []), ("time,no-detrend", ["--no-detrend"]), ("phase+detrend", ["--match", "phase"])):
    flags, finals, min_q, drift_flag = 0, 0, [], 0
    for sd in range(a.null_seeds):
        d = S.make(root / f"null-{sd}", plant=0.0, seed=1000 + sd); r, con, _ = run(d, *extra)
        nflag = r["features"]["n_q_le_fdr"]; flags += nflag > 0; finals += len(r["candidate_ids"]) > 0
        drift_flag += any(c["feature_id"] == S.DRIFT and c["q"] <= 0.1 for c in con)
        min_q.append(min(c["q"] for c in con)); shutil.rmtree(d)
    res["null"][mode] = dict(n=a.null_seeds, runs_with_any_q_le_0p1=flags, runs_with_final_candidate=finals, drift_feature_flagged=drift_flag,
                             min_q_median=float(np.median(min_q)))
    print("null", mode, res["null"][mode], f"[{time.time() - t0:.0f}s]", flush=True)
res["seconds"] = round(time.time() - t0)
(Path(__file__).parent / "synth_results.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["null"], indent=1))
