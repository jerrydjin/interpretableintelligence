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

"""Box test: run v8_diag_cell.py against a MOCK GR00T policy (real checkpoint processors, real LIBERO env)."""
import os, sys, json, glob, collections, types
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/tmp/wandstub")  # box has no ImageMagick; LIBERO-plus only needs it for blur perturbations
sys.path.insert(0, "/workspace/groot-colab/v8")
import torch
import inproc_v8 as v8
from lerobot.policies.groot.configuration_groot import GrootConfig
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

CK = "nvidia/gr00t17-lerobot-libero_10-640"
cfg = GrootConfig.from_pretrained(glob.glob("/home/box/.cache/huggingface/hub/models--nvidia--gr00t17-lerobot-libero_10-640/snapshots/*")[0])
cfg.device = "cpu"
tcfg = AutoConfig.from_pretrained("Qwen/Qwen3-VL-2B-Instruct").text_config

class Blk(torch.nn.Module):
    def __init__(s): super().__init__(); s.l = torch.nn.Linear(8, 8)
    def forward(s, x): return s.l(x)

class Mock(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.config = cfg
        s._action_queue_steps = 16
        g = torch.nn.Module(); s._groot_model = g
        g.backbone = torch.nn.Module(); g.backbone.model = torch.nn.Module(); g.backbone.model.model = torch.nn.Module()
        lm = torch.nn.Module(); g.backbone.model.model.language_model = lm
        lm.layers = torch.nn.ModuleList([Blk() for _ in range(2)])
        lm.rotary_emb = Qwen3VLTextRotaryEmbedding(tcfg)
        n = lm.rotary_emb.inv_freq.numel()  # corrupt like the v7 loader did (theta 1e4)
        lm.rotary_emb.inv_freq = torch.nn.Buffer(1.0 / (10000.0 ** (torch.arange(0, n).float() / n)), persistent=False)
        ah = torch.nn.Module(); g.action_head = ah
        ah.config = types.SimpleNamespace(action_horizon=40); ah.action_dim = 132; ah.num_inference_timesteps = 4
        ah.action_horizon = 40
        ah.model = torch.nn.Module(); ah.model.transformer_blocks = torch.nn.ModuleList([Blk() for _ in range(32)])
        s.reset()
    def reset(s): s._action_queue = collections.deque([], maxlen=16)
    def predict_action_chunk(s, batch, noise=None):
        x = torch.zeros(1, 41, 8)
        for l in s._groot_model.backbone.model.model.language_model.layers: x = l(x)
        for _ in range(4): x = s._groot_model.action_head.model.transformer_blocks[5](x)
        st = batch["state"][:, :, :7].float()  # deterministic fake actions in normalized space
        return torch.tanh(st.repeat(1, 16, 1) * 0.3)
    @torch.no_grad()
    def select_action(s, batch, noise=None):
        if len(s._action_queue) == 0:
            a = s.predict_action_chunk(batch, noise=noise)
            s._action_queue.extend(a[:, :16].transpose(0, 1))
        return s._action_queue.popleft()

g = {"policy": Mock(), "CKPT": CK, "PLUS_DIR": "/workspace/groot-colab/LIBERO-plus", "v8": v8, "LAYER": 5,
     "RUN": "/tmp/diag_mock_run"}
g["__name__"] = "__main__"
code = open("/workspace/groot-colab/v8_diag_cell.py").read()
exec(compile(code, "v8_diag_cell.py", "exec"), g)
d = json.load(open("/tmp/diag_mock_run/smoke/diag_v8.json"))
assert d["rope_before"][0]["max_abs_diff"] > 0.1, d["rope_before"]
assert d["rope_repair"][0]["repaired"]
assert "fp16_rope_fixed" in d["open_loop"] and "bf16_autocast" in d["open_loop"]
assert d["queue_ok"] is True, d.get("timing")
assert d["live_vs_ds"]["images"]["observation.images.image"]["corr_vs_ds"] > d["live_vs_ds"]["images"]["observation.images.image"]["corr_vs_ds_rot180"]
print("MOCK DIAG TEST OK")

# oracle: return the normalized GT chunk -> open-loop must PASS with ~0 MAE
st = g["_ACT_STATS"]; mn, mx = torch.tensor(st["min"][:7]), torch.tensor(st["max"][:7])
samples = g["_samples"]; it = iter(samples)
def oracle(batch, noise=None):
    s = next(it)
    gt = torch.tensor(s["gt"])
    return (2 * (gt - mn) / (mx - mn) - 1)[None]
g["_P"].predict_action_chunk = oracle
r = g["open_loop"](g["_task"], samples, "oracle")
g["_P"].__dict__.pop("predict_action_chunk")
assert r["pass"] and max(r["mae"]) < 1e-4, r["mae"]
print("ORACLE PASS OK")
