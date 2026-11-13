"""Pluggable feature labellers.

The fake adapter uses cue metadata read back from the clip arrays.
``LLMLabeller`` is the seam for a later VLM; it does not make network calls.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from interpretableintelligence.adapters.base import Clip
from interpretableintelligence.adapters.fake import CUE_NAMES, cue_volume
from interpretableintelligence.stats import pearson
from interpretableintelligence.thresholds import LABEL_ASSIGN_MIN_CORR, LABEL_HOLD_MIN_CORR


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


# --------------------------------------------------------------------------
# CLIP concept labeller for real video (hf adapter)

CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
CLIP_LABEL_MODEL = f"clip-concepts-v1:{CLIP_MODEL_ID}"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_PROMPT = "a photo of {}."

# Everyday scene content a vision model is likely to carve up: objects, parts
# of a robot or person, surfaces, textures, and layout. ``--concepts`` replaces it.
DEFAULT_CONCEPTS: tuple[str, ...] = (
    "a robot arm", "a robot gripper", "a human hand", "a person", "a face",
    "a table", "a countertop", "a shelf", "a cabinet", "a drawer", "a sink", "a stove",
    "a cup", "a mug", "a bowl", "a plate", "a pot", "a pan", "a bottle", "a can", "a jar",
    "a box", "a bag", "a towel", "a cloth", "a marker", "a pen", "a spoon", "a fork", "a knife",
    "a fruit", "a vegetable", "food", "a toy", "a block", "a ball", "a book", "paper",
    "a laptop", "a screen", "a keyboard", "a phone", "a lamp", "a chair", "a door", "a window",
    "a wall", "the floor", "the ceiling", "a carpet", "wood", "metal", "plastic", "glass", "fabric",
    "a white surface", "a dark surface", "a bright light", "a shadow", "a reflection",
    "an edge", "a corner", "stripes", "a grid pattern", "text", "a logo",
    "the sky", "grass", "a tree", "a road", "a car", "a building", "water",
    "blurry motion", "an empty background", "clutter",
)


def load_concepts(path=None) -> list[str]:
    """Concepts from a text file (one per line, ``#`` comments), else the built-in list."""
    if path is None:
        return list(DEFAULT_CONCEPTS)
    from pathlib import Path

    names: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text and text not in names:
            names.append(text)
    if len(names) < 2:
        raise ValueError(f"{path} must list at least two concepts, one per line")
    return names


def clip_text_embeddings(model, input_ids, attention_mask):
    """Projected CLIP text embeddings. ``get_text_features`` changed return type in transformers 5."""
    pooled = model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    return model.text_projection(pooled)


class ClipConceptSource:
    """Per-token CLIP similarity to each concept, aligned with the run's token grid.

    Each sampled frame goes through CLIP once. Dense patch embeddings use the
    MaskCLIP trick: the last block's attention is replaced by its value path,
    so each patch keeps its own content instead of the image summary. The
    similarity maps are resized to the model's ``h x w`` grid (frames are
    squashed to a square exactly as the hf adapter does) and averaged over each
    tubelet. A feature is labelled with the concept whose map it tracks best
    on train clips, and the label holds only if that correlation survives on
    held-out clips (the same gate the fake cue labeller uses).
    """

    name = CLIP_LABEL_MODEL

    def __init__(self, adapter, names: list[str], device=None) -> None:
        import torch

        self.adapter = adapter
        self.names = list(names)
        self.device = device or torch.device("cpu")
        self._model = None
        self._text = None
        self._index: int | None = None
        self._maps: np.ndarray | None = None

    def rows(self, meta, clip_index, t, h, w) -> np.ndarray:
        out = np.zeros((int(clip_index.shape[0]), len(self.names)), dtype=np.float64)
        for chosen in np.unique(clip_index):
            mask = clip_index == chosen
            maps = self._clip_maps(int(chosen), meta[int(chosen)])
            out[mask] = np.transpose(maps[:, t[mask], h[mask], w[mask]])
        return out

    def _clip_maps(self, clip_index: int, item: dict) -> np.ndarray:
        if self._index != clip_index or self._maps is None:
            from interpretableintelligence.clips import load_clip

            clip = load_clip(item)
            try:
                window = self.adapter.windows(clip)[0]
                self._maps = self.concept_maps(window.frames)
            finally:
                del clip
            self._index = clip_index
        return self._maps

    def concept_maps(self, frames: np.ndarray) -> np.ndarray:
        """``[n_concepts, t, h, w]`` similarity maps for one window's frames."""
        import torch
        import torch.nn.functional as F

        grid = self.adapter.grid()
        sims = self._dense_similarity(frames)  # [frames, concepts, P, P]
        sims = F.interpolate(sims, size=(grid.h, grid.w), mode="bilinear", align_corners=False)
        n_frames = int(sims.shape[0])
        if n_frames == grid.t * grid.tubelet:
            sims = sims.reshape(grid.t, grid.tubelet, *sims.shape[1:]).mean(dim=1)
        elif n_frames != grid.t:
            raise RuntimeError(f"{n_frames} frames do not fit a grid with t={grid.t}, tubelet={grid.tubelet}")
        return sims.permute(1, 0, 2, 3).contiguous().detach().cpu().numpy().astype(np.float64)

    def _dense_similarity(self, frames: np.ndarray):
        import torch
        import torch.nn.functional as F

        model, text = self._ensure()
        pixels = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device).float()
        if float(pixels.max()) > 1.5:
            pixels = pixels / 255.0
        pixels = pixels.permute(0, 3, 1, 2)
        if pixels.shape[1] == 1:
            pixels = pixels.expand(-1, 3, -1, -1)
        size = int(model.config.vision_config.image_size)
        pixels = F.interpolate(pixels, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        pixels = (pixels - pixels.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)) / pixels.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        with torch.no_grad():
            vision = model.vision_model
            hidden = vision(pixel_values=pixels, output_hidden_states=True).hidden_states[-2]
            last = vision.encoder.layers[-1]
            values = last.self_attn.out_proj(last.self_attn.v_proj(last.layer_norm1(hidden)))
            hidden = hidden + values
            hidden = hidden + last.mlp(last.layer_norm2(hidden))
            patches = model.visual_projection(vision.post_layernorm(hidden[:, 1:]))
            patches = F.normalize(patches, dim=-1)
            side = int(round(patches.shape[1] ** 0.5))
            sims = patches @ text.T  # [frames, P*P, concepts]
        return sims.reshape(sims.shape[0], side, side, -1).permute(0, 3, 1, 2)

    def _ensure(self):
        if self._model is None:
            import torch
            import torch.nn.functional as F
            from transformers import CLIPModel, CLIPTokenizer

            model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device).eval()
            tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
            prompts = [CLIP_PROMPT.format(name) for name in self.names]
            with torch.no_grad():
                tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
                text = clip_text_embeddings(model, tokens["input_ids"], tokens["attention_mask"])
            self._model = model
            self._text = F.normalize(text, dim=-1)
        return self._model, self._text
