import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from token_subsample import choose_token_indices, subsample_tokens

def test_fixed_even():
    a = choose_token_indices(256, 32)
    b = choose_token_indices(256, 32)
    assert a == b
    assert len(a) == 32
    assert a[0] == 0 and a[-1] == 255
    assert sorted(a) == a
    assert len(set(a)) == 32

def test_small_n():
    assert choose_token_indices(10, 32) == list(range(10))

def test_subsample_mean():
    x = torch.randn(100, 16)
    idx = choose_token_indices(100, 8)
    y, toks = subsample_tokens(x, idx, keep_mean=True)
    assert y.shape == (9, 16)  # 8 + mean
    assert toks[-1] == -1
    assert toks[:-1] == idx

if __name__ == "__main__":
    test_fixed_even(); test_small_n(); test_subsample_mean()
    print("ALL_TOKEN_TESTS_PASSED")
