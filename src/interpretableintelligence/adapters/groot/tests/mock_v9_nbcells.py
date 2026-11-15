"""Box test: exec notebook v9 cells 6 / 6b / 6c / 7 / 8 / 9 with the stand-in policy (/content -> /tmp/content9)."""
from box_paths import COLAB  # noqa: E402 (box roots; see box_paths.py)
import os, sys, json, shutil
from pathlib import Path

if __name__ == "__main__":
    os.environ.update({"MUJOCO_GL": "egl", "GROOT_MAX_STEPS": "20", "GROOT_DIAG_GATE": "0", "GROOT_N_ENVS": "2"})
    sys.path.insert(0, f"{COLAB}/tests")
    from v9_standin import StandIn, CK
    shutil.rmtree("/tmp/content9", ignore_errors=True); os.makedirs("/tmp/content9")
    nb = json.load(open(f"{COLAB}/groot_colab_nodrive_v9.ipynb"))
    run = Path("/tmp/content9/physint/groot/colab-l4-libero-plus-v9")
    g = {"__name__": "__main__", "policy": StandIn(), "CKPT": CK, "FAMILY": "groot", "LAYER": 5,
         "PLUS_DIR": f"{COLAB}/LIBERO-plus", "RUN": run, "RUN_ID": "colab-l4-libero-plus-v9",
         "SNAPSHOT_TAR": Path("/tmp/content9/physint_snapshot.tar"), "TASK": {"probe_episodes": 2, "capture_target_rollouts": 2}}
    for i in (6, 7, 8, 9, 10, 11):
        src = "".join(nb["cells"][i]["source"]).replace("/content", "/tmp/content9")
        if i == 6:
            src = src.replace("V8_SETUP = v8.setup(V8)", "V8_SETUP = v8.setup(V8, device='cpu')")
        exec(compile(src, f"cell{i}", "exec"), g)
    assert g["SMOKE"]["ok"] and g["CAPTURE_SUMMARY"]["n_done"] == 2
    fs = json.loads((run / "final_summary.json").read_text())
    assert fs["runner"] == "v9.0" and "capture" in fs["pool_stats"]
    print("V9 NB CELLS OK", g["CHOSEN"]["id"], g["CHOSEN"]["wilson95"], g["CAPTURE_SUMMARY"])
