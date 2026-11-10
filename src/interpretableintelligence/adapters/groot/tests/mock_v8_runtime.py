def _skip_if_unavailable():
    """Sim script: skip unless LIBERO and the groot-colab checkout are present."""
    import importlib.util
    from pathlib import Path

    reasons = []
    if importlib.util.find_spec("libero") is None:
        reasons.append("LIBERO")
    if importlib.util.find_spec("lerobot") is None:
        reasons.append("lerobot")
    if not Path("/workspace/groot-colab/LIBERO-plus").exists():
        reasons.append("LIBERO-plus checkout")
    if not reasons:
        return
    import pytest

    pytest.skip("unavailable: " + ", ".join(reasons), allow_module_level=True)


_skip_if_unavailable()

"""Box test: v8 setup -> smoke -> capture(1 ep) with a MOCK policy, GROOT_MAX_STEPS=40."""
import os, sys, json, shutil
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["GROOT_MAX_STEPS"] = "40"
sys.path.insert(0, "/tmp/wandstub"); sys.path.insert(0, "/workspace/groot-colab/tests")
from pathlib import Path
exec(open("/workspace/groot-colab/tests/mock_diag_run.py").read().split("g = {")[0])  # reuse Mock + v8 import
pol = Mock()
run = Path("/tmp/mock_v8_run"); shutil.rmtree(run, ignore_errors=True)
ctx = v8.Ctx(run=run, run_id="colab-t4-libero-plus-v8", policy=pol, ckpt=CK, layer=5,
             plus_dir=Path("/workspace/groot-colab/LIBERO-plus"), snapshot_tar=Path("/tmp/mock_v8_snapshot.tar"), capture_n=1)
rep = v8.setup(ctx, device="cpu")
assert rep["rope"][0]["repaired"] and rep["rope"][0]["max_abs_diff_before"] > 0.1
assert v8.repair_rope(pol)[0]["repaired"] is False  # idempotent
sm = v8.smoke(ctx)
print("SMOKE timing", sm["timing"], "n_chunks", sm["n_chunks"])
assert sm["timing"]["n_predict"] == 3 == sm["n_chunks"], sm["timing"]
v8.atomic_json(run / "probe" / "chosen.json", {"index": 613, "id": 614, "name": "x", "sr": 0.5, "episodes_done": 1})
summ = v8.capture(ctx)
man = json.loads((run / "manifest.json").read_text())
from safetensors.numpy import load_file
a = load_file(str(run / "acts" / "shard-00000.safetensors"))
print("capture", summ, "acts", a["acts"].shape, sorted(set(a["step"].tolist())), man["hook_target"], man["v7"])
assert man["hook_target"].endswith("transformer_blocks.5") and sorted(set(a["step"].tolist())) == [0, 16, 32]
print("MOCK V8 RUNTIME OK")
