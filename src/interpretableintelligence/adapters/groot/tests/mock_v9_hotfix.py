"""Box test: exec v9_hotfix_cell.py on a simulated LIVE v8 runtime (stand-in policy, v8 ctx set up, a v8
probe.json with 5 eps, and a stale v8 capture wrapper left on the policy). /content -> /tmp/content9h."""
import os, sys, shutil, json
from pathlib import Path

if __name__ == "__main__":
    os.environ.update({"MUJOCO_GL": "egl", "GROOT_MAX_STEPS": "20", "GROOT_PROBE_N": "3", "GROOT_CAPTURE_N": "3",
                       "GROOT_N_ENVS": "3"})
    sys.path.insert(0, "/workspace/groot-colab/tests")
    from v9_standin import StandIn, CK
    import physint_inproc_v8 as v8
    root = Path("/tmp/content9h"); shutil.rmtree(root, ignore_errors=True); root.mkdir()
    pol = StandIn()
    V8 = v8.Ctx(run=root / "physint/groot/colab-t4-libero-plus-v8", run_id="colab-t4-libero-plus-v8", policy=pol,
                ckpt=CK, layer=5, plus_dir=Path("/workspace/groot-colab/LIBERO-plus"), snapshot_tar=root / "physint_snapshot.tar")
    v8.setup(V8, device="cpu")
    v8.atomic_json(V8.run / "probe/probe.json", {"instances": {"613": {"index": 613, "id": 614, "name": "x", "results": [
        {"ep": k, "seed": 10000 + k, "init_state_id": k, "success": k < 3, "n_steps": 5, "s": 1} for k in range(5)]}}})
    v8.atomic_json(V8.run / "smoke/smoke.json", {"ok": True})
    v8.FinalDenoiseCapture(pol, 5).install()  # stale wrapper, as if a v8 capture cell was interrupted badly
    src = open("/workspace/groot-colab/v9_hotfix_cell.py").read().replace("/content", str(root))
    g = {"__name__": "__main__", "policy": pol, "CKPT": CK, "FAMILY": "groot", "LAYER": 5,
         "PLUS_DIR": "/workspace/groot-colab/LIBERO-plus", "V8": V8, "v8": v8}
    exec(compile(src, "v9_hotfix_cell.py", "exec"), g)
    run9 = root / "physint/groot/colab-l4-libero-plus-v9"
    pr = json.loads((run9 / "probe/probe.json").read_text())
    assert {k: len(v["results"]) for k, v in pr["instances"].items()} == {"613": 3, "1027": 3}, pr["instances"].keys()
    assert all(r.get("runner") == "v9.0" for v in pr["instances"].values() for r in v["results"])  # v8 eps not imported
    man = json.loads((run9 / "manifest.json").read_text())
    assert man["run_id"] == "colab-l4-libero-plus-v9" and len(man["episodes"]) == 3 and man["runner"]["parallel_envs"] == 3
    assert (run9 / "acts/shard-00002.safetensors").is_file() and (run9 / "smoke/smoke.json").is_file()
    assert "predict_action_chunk" not in pol.__dict__
    print("MOCK V9 HOTFIX OK", g["CHOSEN"]["id"], g["CHOSEN"]["wilson95"], g["CAPTURE_SUMMARY"], g["V9"].pool_stats["capture"]["episodes_per_hour"])
