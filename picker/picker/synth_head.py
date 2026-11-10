"""--downstream picker.synth_head:act for synthetic runs: the true action head tanh(mean_tok((x-mu)/s) @ Wh)."""
import functools, numpy as np, torch
from safetensors.torch import load_file
from pathlib import Path

@functools.lru_cache(None)
def _load(run):
    w = load_file(str(Path(run) / "sae" / "weights.safetensors")); Wh = torch.from_numpy(np.load(Path(run) / "sae" / "_synth_Wh.npy"))
    return w["norm_mu"], float(w["norm_scale"][0]), Wh

def act(x, meta):  # x [B, n_tok, D]
    mu, s, Wh = _load(meta["run"]); return torch.tanh(((x - mu) / s).mean(1) @ Wh)
