"""Pluggable feature labellers.

The fake adapter uses cue metadata read back from the clip arrays.
``LLMLabeller`` is the seam for a later VLM; it does not make network calls.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from physint.adapters.base import Clip
from physint.adapters.fake import CUE_NAMES, cue_volume
from physint.stats import pearson
from physint.thresholds import LABEL_ASSIGN_MIN_CORR, LABEL_HOLD_MIN_CORR


class Labeller(Protocol):
    name: str

    def label_feature(
        self,
        activation: np.ndarray,
        train_mask: np.ndarray,
        heldout_mask: np.ndarray,
    ) -> dict:
        ...


def cue_patterns(clips: list[Clip], tokens: list) -> dict[str, np.ndarray]:
    """Cue strength at each captured token, aligned with activation rows.

    The fake adapter uses one window per clip whose ``t`` is the clip grid time.
    """
    volumes = {clip.clip_id: cue_volume(clip.frames) for clip in clips}
    patterns = {name: np.zeros(len(tokens), dtype=np.float64) for name in CUE_NAMES}
    for index, token in enumerate(tokens):
        volume = volumes[token.clip_id]
        for cue_index, name in enumerate(CUE_NAMES):
            patterns[name][index] = float(volume[cue_index, token.t, token.h, token.w])
    return patterns


class FakeCueLabeller:
    """Deterministic labels from planted cue channels. No network."""

    name = "fake-cue-v1"

    def __init__(self, patterns: dict[str, np.ndarray]) -> None:
        self.patterns = patterns

    def label_feature(
        self,
        activation: np.ndarray,
        train_mask: np.ndarray,
        heldout_mask: np.ndarray,
    ) -> dict:
        act = np.asarray(activation, dtype=np.float64)
        best_name = ""
        best_r = LABEL_ASSIGN_MIN_CORR
        for name, pattern in self.patterns.items():
            corr = pearson(act[train_mask], pattern[train_mask])
            if corr > best_r:
                best_r = corr
                best_name = name
        if not best_name:
            return {
                "label": "",
                "label_model": self.name,
                "label_holds_heldout": False,
            }
        held = pearson(act[heldout_mask], self.patterns[best_name][heldout_mask])
        return {
            "label": best_name,
            "label_model": self.name,
            "label_holds_heldout": bool(held >= LABEL_HOLD_MIN_CORR),
        }


class NullLabeller:
    """No label. Used when the clip has no planted cue channels."""

    name = "none"

    def label_feature(
        self,
        activation: np.ndarray,
        train_mask: np.ndarray,
        heldout_mask: np.ndarray,
    ) -> dict:
        del activation, train_mask, heldout_mask
        return {
            "label": "",
            "label_model": self.name,
            "label_holds_heldout": False,
        }


class LLMLabeller:
    """Stub for a future VLM/LLM labeller. Not called on the fake path."""

    name = "llm-stub"

    def label_feature(
        self,
        activation: np.ndarray,
        train_mask: np.ndarray,
        heldout_mask: np.ndarray,
    ) -> dict:
        raise NotImplementedError(
            "LLM/VLM labelling is a later lane and does not make network calls here."
        )
