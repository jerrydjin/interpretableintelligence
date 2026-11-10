def _skip_if_unavailable():
    """GPU script: skip unless CUDA, lerobot, and the GR00T checkpoint are present."""
    import glob
    import importlib.util
    from pathlib import Path

    reasons = []
    try:
        import torch
        if not torch.cuda.is_available():
            reasons.append("GPU")
    except ImportError:
        reasons.append("GPU")
    if importlib.util.find_spec("lerobot") is None:
        reasons.append("lerobot")
    if not glob.glob(
        "/home/box/.cache/huggingface/hub/models--nvidia--gr00t17-lerobot-libero_10-640/snapshots/*"
    ):
        reasons.append("GR00T checkpoint")
    if not Path("/workspace/groot-colab/v8/low_ram_policy_load.v7_buggy.py").is_file():
        reasons.append("v7 loader fixture")
    if not reasons:
        return
    import pytest

    pytest.skip("unavailable: " + ", ".join(reasons), allow_module_level=True)


_skip_if_unavailable()

import sys, torch
sys.path.insert(0, "/workspace/groot-colab")
import low_ram_policy_load as L
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config, _tie_unused_qwen_lm_head
from lerobot.policies.groot.modeling_groot import GrootPolicy
import glob
ck = glob.glob("/home/box/.cache/huggingface/hub/models--nvidia--gr00t17-lerobot-libero_10-640/snapshots/*")[0]
config = GrootConfig.from_pretrained(ck)
fixed = L.fix_groot_base_model_path(config, override="nvidia/GR00T-N1.7-3B")
config.device = "meta"
base_cfg = GR00TN17Config.from_pretrained(fixed)
print("base num_inference_timesteps", getattr(base_cfg,"num_inference_timesteps",None), "action_horizon", getattr(base_cfg,"action_horizon",None))
def _create_empty(self):
    m = GR00TN17(base_cfg, transformers_loading_kwargs={"trust_remote_code": True}, load_backbone_weights=False)
    return m
GrootPolicy._create_groot_model = _create_empty
pol = L._construct_on_meta(lambda: GrootPolicy(config))
mods = {n: m for n, m in pol.named_modules() if hasattr(m, "inv_freq")}
rebuilt = L.materialize_computed_buffers(pol, "cpu")
print(rebuilt)
for n, m in mods.items():
    ref = type(m)(m.config)  # real CPU init
    a, b = m.inv_freq, ref.inv_freq
    print(n, type(m).__name__, a.shape, b.shape, "maxabs", (a.float()-b.float()).abs().max().item() if a.shape==b.shape else "SHAPE MISMATCH",
          "attn_scaling", getattr(m,"attention_scaling",None), getattr(ref,"attention_scaling",None),
          "theta", m.config.rope_parameters)
print("queue steps", pol._action_queue_steps, "n_action_steps", config.n_action_steps)
ah = pol._groot_model.action_head
print("ah num_inference_timesteps", ah.num_inference_timesteps, "action_horizon", ah.action_horizon)
print("policy cfg num_inference_timesteps", config.num_inference_timesteps)
for n, m in mods.items():
    ref = type(m)(m.config)
    assert torch.equal(m.inv_freq.float(), ref.inv_freq.float()), n
import importlib.util
spec = importlib.util.spec_from_file_location("lrl_v7", "/workspace/groot-colab/v8/low_ram_policy_load.v7_buggy.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
pol2 = L._construct_on_meta(lambda: GrootPolicy(config))
old.materialize_computed_buffers(pol2, "cpu")
lm = dict(pol2.named_modules())["_groot_model.backbone.model.model.language_model.rotary_emb"]
d = float((lm.inv_freq - type(lm)(lm.config).inv_freq).abs().max())
assert d > 0.1, d
print("v7 loader text-RoPE max|d|", d, "-> repair:", L.repair_rotary_buffers(pol2)[-1])
# Mismatched theta must raise (verify the repaired-then-corrupted buffer, do not swallow).
n = lm.inv_freq.numel()
saved = lm.inv_freq.detach().clone()
lm.inv_freq = torch.nn.Buffer(1.0 / (10000.0 ** (torch.arange(0, n, dtype=torch.float32) / n)), persistent=False)
try:
    L.verify_rotary_inv_freq(lm, "language_model.rotary_emb")
except RuntimeError as e:
    print("mismatch raise:", e)
    assert "expected theta=" in str(e) and "language_model.rotary_emb" in str(e)
else:
    raise AssertionError("verify_rotary_inv_freq did not raise on theta mismatch")
# Repair of an already-loaded (now-corrupted) policy must succeed and re-verify.
rep = L.repair_rotary_buffers(pol2)
assert any(r.get("repaired") and r["module"].endswith("language_model.rotary_emb") for r in rep)
L.verify_rotary_inv_freq(lm, "language_model.rotary_emb")
print("ROPE REBUILD CHECK OK")
