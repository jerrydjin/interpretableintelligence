"""Stand-in GR00T policy for v9 box tests: real checkpoint processors + real LIBERO env, tiny fake model.

Like tests/mock_diag_run.Mock, but predict_action_chunk is batch-aware, depends on state AND image
pixels (so wrong collation order would show up), and draws its flow-matching noise through
lerobot.policies.groot.groot_n1_7.sample_noise exactly like the real GR00T action head.
"""
from box_paths import COLAB, groot_snapshot  # noqa: E402 (box roots; see box_paths.py)
import os, sys, glob, collections, types
os.environ.setdefault("MUJOCO_GL", "egl")
for p in ("/tmp/wandstub", f"{COLAB}/v8", f"{COLAB}/v9"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
import lerobot.policies.groot.groot_n1_7 as gm
from lerobot.policies.groot.configuration_groot import GrootConfig
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

CK = "nvidia/gr00t17-lerobot-libero_10-640"
cfg = GrootConfig.from_pretrained(groot_snapshot())
cfg.device = "cpu"
tcfg = AutoConfig.from_pretrained("Qwen/Qwen3-VL-2B-Instruct").text_config


class Blk(torch.nn.Module):
    def __init__(s, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        s.l = torch.nn.Linear(8, 8)
        with torch.no_grad():
            s.l.weight.copy_(torch.randn(8, 8, generator=g) * 0.3); s.l.bias.copy_(torch.randn(8, generator=g) * 0.1)
    def forward(s, x): return torch.tanh(s.l(x))


class StandIn(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.config = cfg
        s._action_queue_steps = 16
        g = torch.nn.Module(); s._groot_model = g
        g.backbone = torch.nn.Module(); g.backbone.model = torch.nn.Module(); g.backbone.model.model = torch.nn.Module()
        lm = torch.nn.Module(); g.backbone.model.model.language_model = lm
        lm.layers = torch.nn.ModuleList([Blk(i) for i in range(2)])
        lm.rotary_emb = Qwen3VLTextRotaryEmbedding(tcfg)  # correct theta (v8 setup verifies it)
        ah = torch.nn.Module(); g.action_head = ah
        ah.config = types.SimpleNamespace(action_horizon=40); ah.action_dim = 132; ah.num_inference_timesteps = 4
        ah.action_horizon = 40
        ah.model = torch.nn.Module(); ah.model.transformer_blocks = torch.nn.ModuleList([Blk(100 + i) for i in range(32)])
        s.reset()

    def reset(s): s._action_queue = collections.deque([], maxlen=16)

    def predict_action_chunk(s, batch, noise=None):
        st = batch["state"][:, :, :8].float()                       # [B,1,8]
        B = st.shape[0]
        pv = batch["pixel_values"].float().reshape(B, -1)            # per-env patches, env order
        img = torch.stack([pv[:, ::7].mean(1), pv[:, 3::11].std(1)], 1)  # [B,2]
        nz = gm.sample_noise((B, 40, 132), st.device, dtype=torch.float32)  # same call site as GR00T
        x = st.expand(B, 41, 8) + 0.1 * torch.linspace(0, 1, 41)[None, :, None]
        for l in s._groot_model.backbone.model.model.language_model.layers: x = l(x)
        blk = s._groot_model.action_head.model.transformer_blocks[5]
        for i in range(4):
            x = blk(x + 0.2 * torch.cat([nz[:, :1, :8], nz[:, :, :8]], 1) * (1 - i / 4))
        base = torch.tanh(st[:, :, :7] * 0.3 + 0.5 * img[:, None, :1])   # [B,1,7]
        act = torch.tanh(base.repeat(1, 40, 1) + 0.4 * nz[:, :, :7] + 0.2 * x[:, 1:, :7])
        return act[:, :16]

    @torch.no_grad()
    def select_action(s, batch, noise=None):
        if len(s._action_queue) == 0:
            a = s.predict_action_chunk(batch, noise=noise)
            s._action_queue.extend(a[:, :16].transpose(0, 1))
        return s._action_queue.popleft()
