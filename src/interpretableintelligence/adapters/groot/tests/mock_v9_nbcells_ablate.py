"""Box test: notebook v9 cells 6..9 in ablation mode (GROOT_ABLATE_PLAN set), stand-in policy, real env 1028.
Cells 7-9 must run the ablation (no probe / capture). /content -> /tmp/content9n."""
import os, sys, json, shutil
from pathlib import Path

if __name__ == "__main__":
    os.environ.update({"MUJOCO_GL": "egl", "GROOT_MAX_STEPS": "20", "GROOT_DIAG_GATE": "0", "GROOT_N_ENVS": "3"})
    sys.path[:0] = ["/workspace/groot-colab/tests", "/workspace/groot-colab/v9", "/workspace/groot-colab/v8"]
    from v9_standin import StandIn, CK
    from mock_v9_ablate import make_mock
    root = Path("/tmp/content9n"); shutil.rmtree(root, ignore_errors=True); root.mkdir()
    plan_p, sae_p, mp = make_mock(root)
    os.environ.update({"GROOT_ABLATE_PLAN": str(plan_p), "GROOT_ABLATE_SAE": str(sae_p), "GROOT_ABLATE_ONLY": "baseline,arm-A"})
    nb = json.load(open("/workspace/groot-colab/groot_colab_nodrive_v9.ipynb"))
    run = root / "physint/groot/colab-l4-libero-plus-v9"
    g = {"__name__": "__main__", "policy": StandIn(), "CKPT": CK, "FAMILY": "groot", "LAYER": 5,
         "PLUS_DIR": "/workspace/groot-colab/LIBERO-plus", "RUN": run, "RUN_ID": "colab-l4-libero-plus-v9",
         "SNAPSHOT_TAR": root / "physint_snapshot.tar", "TASK": {"probe_episodes": 2, "capture_target_rollouts": 2}}
    for i in (6, 7, 8, 9, 10, 11):
        src = "".join(nb["cells"][i]["source"]).replace("/content", str(root))
        if i == 6:
            src = src.replace("V8_SETUP = v8.setup(V8)", "V8_SETUP = v8.setup(V8, device='cpu')")
        exec(compile(src, f"cell{i}", "exec"), g)
    assert g["CHOSEN"] is None and "CAPTURE_SUMMARY" not in g and not (run / "probe/probe.json").is_file()
    top = g["ABLATE_SUMMARY"]
    n = {r["cond"]: r["n"] for r in top["table"]}
    assert n == {"baseline": 4, "recon-only": 2, "A-cand-720": 3, "A-ctrl01-7774": 3, "B-cand-7081+5852": 0, "B-ctrl01-1159+7339": 0}, n
    assert top["all_finished"] and (root / "ablate_snapshot.tar").is_file()
    print("V9 NB CELLS ABLATE OK", n)
