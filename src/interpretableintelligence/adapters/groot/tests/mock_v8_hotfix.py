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

"""Box test: exec v8_hotfix_cell.py (as on a live v7 runtime) with a MOCK policy; /content -> /tmp/content."""
import os, sys, shutil
os.environ.update({"MUJOCO_GL": "egl", "GROOT_MAX_STEPS": "20", "GROOT_PROBE_N": "1", "GROOT_CAPTURE_N": "1", "GROOT_DIAG_GATE": "0"})
sys.path.insert(0, "/tmp/wandstub")
shutil.rmtree("/tmp/content", ignore_errors=True); os.makedirs("/tmp/content")
exec(open(f"{COLAB}/tests/mock_diag_run.py").read().split("g = {")[0])
src = open(f"{COLAB}/v8_hotfix_cell.py").read().replace("/content", "/tmp/content")
g = {"__name__": "__main__", "policy": Mock(), "CKPT": CK, "FAMILY": "groot", "LAYER": 5,
     "PLUS_DIR": f"{COLAB}/LIBERO-plus"}
exec(compile(src, "v8_hotfix_cell.py", "exec"), g)
assert g["DIAG"]["rope_before"][0]["max_abs_diff"] > 0.1 and g["V8_SETUP"]["rope"][0]["repaired"] is False
assert g["SMOKE"]["timing"]["n_predict"] == 2 and g["CAPTURE_SUMMARY"]["n_done"] == 1
print("MOCK HOTFIX OK", g["CHOSEN"])
