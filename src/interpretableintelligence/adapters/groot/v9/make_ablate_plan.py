"""Write the default ablation plan + SAE subset for the 1028 rise test (box, CPU, no GPU).

  .venv313/bin/python v9/make_ablate_plan.py [--out /workspace/runs/groot-1028-v9-ablate]
Reads (never modifies) runs/groot-1028-v9-n100/{candidates_final.json, sae/, manifest.json}.
Writes <out>/ablate_plan.json and <out>/sae_subset.safetensors (exact rows/cols of sae/weights.safetensors).
"""
import argparse, json, sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / "v8")]
import ablate_v9 as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/workspace/runs/groot-1028-v9-n100")
ap.add_argument("--out", default="/workspace/runs/groot-1028-v9-ablate")
ap.add_argument("--picker-commit", default="146ee8a")
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
man = json.loads((src / "manifest.json").read_text())
cfg = json.loads((src / "sae/config.json").read_text())
cf = json.loads((src / "candidates_final.json").read_text())
ctrl = {int(k): v for k, v in cf["matched_random_controls"].items()}
lat = sorted({720, 7081, 5852, *ctrl[720], *ctrl[7081], *ctrl[5852]})
full, ns = src / "sae/weights.safetensors", src / "sae/norm_stats.safetensors"
sub_sha = A.write_subset(full, lat, out / "sae_subset.safetensors", ns)
sae_meta = {"source": "embedded subset (or a path via GROOT_ABLATE_SAE)", "subset_file": "sae_subset.safetensors",
            "subset_sha256": sub_sha, "full": str(full), "full_sha256": A.sha256_file(full),
            "norm_stats": str(ns), "norm_stats_sha256": A.sha256_file(ns), "latent_ids": lat,
            "d_in": cfg["d_in"], "n_latents": cfg["n_latents"], "k": cfg["k"], "arch": cfg["arch"],
            "threshold": cfg["inference_threshold"], "norm_scale": cfg["norm"]["norm_scale"],
            "normalisation": cfg["norm"]["method"], "config_hook": cfg["hook"]}
task = {"suite": "plus", "index": int(man["task_index"]), "id": int(man["task_id"]), "name": man["task_name"],
        "n_init_states": int(man["n_init_states"])}
plan = A.default_plan(src / "candidates_final.json", sae_meta=sae_meta, token_indices=man["token_indices"], task=task,
                      picker_commit=a.picker_commit)
(out / "ablate_plan.json").write_text(json.dumps(plan, indent=1))
print("plan", out / "ablate_plan.json", "sha", A.plan_sha(plan)[:16], "conditions", len(plan["conditions"]),
      "| subset", len(lat), "latents", sub_sha[:16])
