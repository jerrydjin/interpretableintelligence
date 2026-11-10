"""v9's RoPE check is the repo's v8 function, not a second copy."""

from __future__ import annotations

import sys
from pathlib import Path


def test_v9_verify_rotary_inv_freq_is_v8():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "v8"))
    import interpretableintelligence.adapters.groot.v9 as v9
    import inproc_v8 as v8

    assert v9.verify_rotary_inv_freq is v8.verify_rotary_inv_freq
