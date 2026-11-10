from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.adapters.base import Clip
from interpretableintelligence.adapters.fake import FakeAdapter, cue_volume, render_clip


def test_painted_cues_roundtrip():
    frames = render_clip(4, seed=0)
    volume = cue_volume(frames)
    peak = np.float32(1.0 + 0.01 * (4 % 17))
    assert volume.shape[0] == 6
    for cue in range(6):
        assert int((volume[cue] > 0).sum()) == 4
        assert np.isclose(volume[cue].max(), peak)


def test_zero_hook_raises_predictor_loss():
    adapter = FakeAdapter(layer=16, device=torch.device("cpu"))
    adapter.load(random_init=False)
    frames = render_clip(0, seed=0)
    clip = Clip(clip_id="clip-000", path=Path("clip-000.npy"), frames=frames, fps=8.0, sha256="0" * 64)
    window = adapter.windows(clip)[0]
    clean, target = adapter.predict_masked(clip, window)
    with adapter.hook(16, mode="zero"):
        zeroed, target_zero = adapter.predict_masked(clip, window)
    clean_loss = float(((clean - target) ** 2).mean())
    zero_loss = float(((zeroed - target_zero) ** 2).mean())
    assert zero_loss > clean_loss * 5
    with adapter.hook(16, mode="capture") as handle:
        encoded = adapter.encode(clip, window)
    assert len(handle.captured) == 1
    assert torch.allclose(handle.captured[0], encoded)
