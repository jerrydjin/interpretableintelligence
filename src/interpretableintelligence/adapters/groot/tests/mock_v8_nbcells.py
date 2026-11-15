from box_paths import COLAB  # noqa: E402 (box roots; see box_paths.py)
def _skip_if_unavailable():
    """Sim script: skip unless LIBERO and the groot-colab checkout are present."""
    import importlib.util
    from pathlib import Path

    reasons = []
    if importlib.util.find_spec("libero") is None:
        reasons.append("LIBERO")
    if importlib.util.find_spec("lerobot") is None:
        reasons.append("lerobot")
    if not Path(f"{COLAB}/LIBERO-plus").exists():
        reasons.append("LIBERO-plus checkout")
    if not reasons:
        return
    import pytest

    pytest.skip("unavailable: " + ", ".join(reasons), allow_module_level=True)


_skip_if_unavailable()

"""Box test: exec notebook v8 cells 6 / 6b / 6c with a MOCK policy (/content -> /tmp/content)."""
import os, sys, json, shutil
from pathlib import Path
os.environ.update({"MUJOCO_GL": "egl", "GROOT_MAX_STEPS": "20", "GROOT_DIAG_GATE": "0"})
sys.path.insert(0, "/tmp/wandstub")
shutil.rmtree("/tmp/content", ignore_errors=True); os.makedirs("/tmp/content")
exec(open(f"{COLAB}/tests/mock_diag_run.py").read().split("g = {")[0])
nb = json.load(open(f"{COLAB}/groot_colab_nodrive_v8.ipynb"))
pol = Mock()
for n, m in pol.named_modules():
    if hasattr(m, "inv_freq"): before = m.inv_freq.clone()
g = {"__name__": "__main__", "policy": pol, "CKPT": CK, "FAMILY": "groot", "LAYER": 5, "PLUS_DIR": f"{COLAB}/LIBERO-plus",
     "RUN": Path("/tmp/content/physint/groot/colab-t4-libero-plus-v8"), "RUN_ID": "colab-t4-libero-plus-v8",
     "SNAPSHOT_TAR": Path("/tmp/content/physint_snapshot.tar"), "TASK": {"probe_episodes": 1, "capture_target_rollouts": 1}}
for i in (6, 7, 8):
    src = "".join(nb["cells"][i]["source"]).replace("/content", "/tmp/content")
    if i == 6:
        src = src.replace("V8_SETUP = v8.setup(V8)", "V8_SETUP = v8.setup(V8, device='cpu')")
    exec(compile(src, f"cell{i}", "exec"), g)
assert g["V8_SETUP"]["rope"][0]["repaired"] and g["SMOKE"]["ok"]
print("NB CELLS OK", g["SMOKE"]["timing"])
