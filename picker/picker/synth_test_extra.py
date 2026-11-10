"""Extra synthetic checks: (a) planted feature with NO action effect must fail the offline gate; (b) power at 50 episodes / weak plant."""
import json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import synth_rollouts as S, pick_candidates as P
from safetensors.numpy import load_file, save_file
root = Path("/tmp/synth_picker_x"); out = {}
# (a) inert plant: rebuild actions with Wh orthogonal to the planted feature's direction (in normalised space)
d = S.make(root / "inert", plant=0.5, seed=400)
from safetensors.torch import load_file as lt
w = lt(str(d / "sae" / "weights.safetensors")); Wd = w["W_dec"].numpy(); mu = w["norm_mu"].numpy(); s = float(w["norm_scale"][0])
Wh = np.load(d / "sae" / "_synth_Wh.npy"); u = Wd[S.PLANT] / np.linalg.norm(Wd[S.PLANT]); Wh = Wh - np.outer(u, u @ Wh); np.save(d / "sae" / "_synth_Wh.npy", Wh)
acts = {}; 
for sh in sorted((d / "acts").glob("*.safetensors")):
    z = load_file(str(sh))
    for e, st, x in zip(z["episode_idx"], z["step"], z["acts"]): acts.setdefault((int(e), int(st)), []).append(x.astype(np.float32))
keys = sorted(acts); A = np.stack([np.tanh(((np.stack(acts[k]) - mu) / s).mean(0) @ Wh) for k in keys]).astype(np.float32)
save_file(dict(episode_idx=np.array([k[0] for k in keys], np.int32), step=np.array([k[1] for k in keys], np.int32), action=A), str(d / "actions.safetensors"))
for tag, extra in (("probe", []), ("downstream", ["--downstream", "picker.synth_head:act"])):
    r = P.main([str(d), "--quiet", *extra])[0]
    o = next(json.loads(l) for l in open(d / "picker" / "offline.jsonl") if json.loads(l)["feature_id"] == S.PLANT)
    out[f"inert_plant_{tag}"] = dict(candidates=r["candidate_ids"], plant_offline_pct=o.get("pct"), plant_q=o["q"], plant_div=o.get("div"), plant_sens_pct=o.get("sens_pct"),
                                     exploratory=[x["feature_id"] for x in r["exploratory"][:3]])
    print(tag, out[f"inert_plant_{tag}"], flush=True)
shutil.rmtree(d)
# (b) power: 50 episodes, plant on 5% / 15% of tokens
for plant in (0.05, 0.15):
    ranks, hits = [], 0
    for sd in range(5):
        d = S.make(root / f"pw-{sd}", plant=plant, seed=500 + sd, n_ep=50); r = P.main([str(d), "--quiet"])[0]
        con = [json.loads(l) for l in open(d / "picker" / "contrast.jsonl")]
        ranks.append(next((i + 1 for i, c in enumerate(con) if c["feature_id"] == S.PLANT), None)); hits += S.PLANT in r["candidate_ids"]; shutil.rmtree(d)
    out[f"power_50ep_plant{plant}"] = dict(ranks=ranks, recovered_as_candidate=f"{hits}/5"); print(plant, out[f"power_50ep_plant{plant}"], flush=True)
(Path(__file__).parent / "synth_results_extra.json").write_text(json.dumps(out, indent=1, default=float))
