"""CLIP concept labels reach vjepa2 (dense maps) and groot (one score per window)."""

from __future__ import annotations

import numpy as np

from interpretableintelligence.adapters.base import Window
from interpretableintelligence.labelers import WindowConceptSource
from interpretableintelligence.pipeline import concept_labelling, sae_config_for


def test_which_adapters_get_concept_labels():
    assert concept_labelling("hf:facebook/dinov2-small") == "dense"
    assert concept_labelling("vjepa2") == "dense"
    assert concept_labelling("vjepa2", tiny=True) is None
    assert concept_labelling("groot:nvidia/gr00t17-lerobot-libero_10-640") == "window"
    assert concept_labelling("groot:tiny") is None
    assert concept_labelling("fake") is None


def test_wide_dictionaries_are_topk():
    assert sae_config_for("vjepa2", 64, seed=0).activation == "relu"
    wide = sae_config_for("groot:x", 64, seed=0, requested=512)
    assert wide.activation == "topk" and wide.n_features == 512


class _TwoWindows:
    def windows(self, clip):
        return [Window(index=i, start_frame=i * 4, frames=clip.frames[i * 4 : i * 4 + 4]) for i in range(2)]


def test_window_concepts_follow_the_token_window(monkeypatch, tmp_path):
    frames = np.zeros((8, 4, 4, 3), dtype=np.uint8)
    frames[4:] = 255
    np.save(tmp_path / "c.npy", frames)
    meta = [{"clip_id": "c", "path": str(tmp_path / "c.npy"), "fps": 8.0, "sha256": "x"}]
    source = WindowConceptSource(_TwoWindows(), ["dark", "bright"])
    monkeypatch.setattr(
        WindowConceptSource,
        "frame_scores",
        lambda self, f: np.stack([[1.0 - f[i].mean() / 255, f[i].mean() / 255] for i in range(len(f))]),
    )
    zeros = np.zeros(4, dtype=np.int64)
    rows = source.rows(meta, zeros, zeros, zeros, zeros, window=np.array([0, 0, 1, 1]))
    assert rows.tolist() == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
