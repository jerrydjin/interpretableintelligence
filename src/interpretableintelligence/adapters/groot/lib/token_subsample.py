"""Fixed token-index subsample shared across every step and episode."""
from __future__ import annotations
import numpy as np

def choose_token_indices(n_tok: int, k: int = 32) -> list[int]:
    """Evenly spaced unique indices in [0, n_tok). Deterministic; no RNG."""
    n_tok = int(n_tok); k = int(k)
    if n_tok <= 0 or k <= 0:
        return []
    k = min(k, n_tok)
    if k == n_tok:
        return list(range(n_tok))
    raw = np.linspace(0, n_tok - 1, k)
    idx = np.unique(np.round(raw).astype(np.int64))
    if len(idx) < k:
        have = set(idx.tolist())
        for j in range(n_tok):
            if len(have) >= k:
                break
            have.add(j)
        idx = np.array(sorted(have)[:k], dtype=np.int64)
    return [int(x) for x in idx.tolist()]


def subsample_tokens(x, token_indices: list[int], keep_mean: bool = True):
    """x: [n_tok, D] or [B, n_tok, D] -> rows [n_keep(+mean), D], tok indices (mean=-1)."""
    import torch
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim == 3:
        x = x[0]
    # x: [n_tok, D]
    n_tok = x.shape[0]
    idx = [i for i in token_indices if 0 <= i < n_tok]
    parts = [x[idx]] if idx else [x.new_zeros((0, x.shape[-1]))]
    toks = list(idx)
    if keep_mean:
        parts.append(x.mean(dim=0, keepdim=True))
        toks.append(-1)
    out = torch.cat(parts, dim=0)
    return out, toks
