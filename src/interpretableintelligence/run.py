"""Backwards-compatible entry for the resumable pipeline, plus ``interpretableintelligence rank``.

``execute_run`` writes a finished scoreboard and returns. The CLI serves the
viewer unless ``--no-serve`` is set; call :func:`interpretableintelligence.pipeline.execute_pipeline`
directly when a script needs the report or the server.

``execute_rank`` scores every SAE feature with one attribution pass and writes
the shortlist. It does not ablate.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.adapters import get_adapter
from interpretableintelligence.artifacts import read_json, write_attribution, write_json
from interpretableintelligence.attribute import (
    attribution_by_clip,
    candidates_document,
    check_objective,
    summarize_attribution,
)
from interpretableintelligence.clips import load_clips
from interpretableintelligence.device import pick_device
from interpretableintelligence.labelers import FakeCueLabeller, cue_patterns
from interpretableintelligence.pipeline import execute_pipeline
from interpretableintelligence.sae import SAEConfig, load_sae, save_sae, train_sae
from interpretableintelligence.score import capture_activations, default_objective, check_extra_effects, extra_effect_rows as _extra_effect_rows
from interpretableintelligence.split import assign_split
from interpretableintelligence.thresholds import (
    DEFAULT_HELDOUT_FRAC,
    DEFAULT_LAYER,
    DEFAULT_PREFILTER_K,
    DEFAULT_SEED,
    MIN_CLIPS,
)


def execute_run(
    adapter_name: str,
    clips_dir: Path,
    out_dir: Path,
    layer: int = DEFAULT_LAYER,
    seed: int = DEFAULT_SEED,
    heldout_frac: float = DEFAULT_HELDOUT_FRAC,
    tiny: bool = False,
    prefilter: int | None = None,
    extra_effects: tuple[str, ...] = (),
    log=print,
) -> dict:
    """Run capture, SAE, features, and score. Does not start the viewer."""
    if prefilter is not None and int(prefilter) < 1:
        raise ValueError(f"prefilter K must be >= 1, got {prefilter}")
    modes = check_extra_effects(extra_effects)
    result = execute_pipeline(
        adapter_name=adapter_name,
        clips_dir=clips_dir,
        out_dir=out_dir,
        layer=layer,
        seed=seed,
        heldout_frac=heldout_frac,
        tiny=tiny,
        prefilter=prefilter,
        extra_effects=modes,
        score=True,
        serve=False,
        open_browser=False,
        log=log,
    )
    scoreboard = result["scoreboard"]
    if scoreboard is None:
        raise RuntimeError("scoring did not produce a scoreboard")
    return scoreboard


def execute_rank(
    adapter_name: str,
    clips_dir: Path,
    out_dir: Path,
    layer: int = DEFAULT_LAYER,
    seed: int = DEFAULT_SEED,
    heldout_frac: float = DEFAULT_HELDOUT_FRAC,
    prefilter: int = DEFAULT_PREFILTER_K,
    from_run: Path | None = None,
    objective: str | None = None,
    log=print,
) -> dict:
    """Score every feature with one attribution pass and write the top-K shortlist.

    Does not ablate. The shortlist is not evidence: every pick still has to be
    confirmed by a real ablation. Pass ``from_run`` to reuse an SAE that a full
    ``interpretableintelligence run`` already trained; otherwise an SAE is trained on the train
    split.
    """
    if objective is not None:
        objective = check_objective(objective)
    if prefilter < 1:
        raise ValueError(f"prefilter K must be >= 1, got {prefilter}")
    torch.set_num_threads(1)
    device = pick_device()
    clips = load_clips(Path(clips_dir))
    if not clips:
        raise ValueError(f"no clips in {clips_dir}")
    if from_run is None and len(clips) < MIN_CLIPS:
        raise ValueError(f"need at least {MIN_CLIPS} clips, found {len(clips)} in {clips_dir}")
    if not 0.0 < heldout_frac < 1.0:
        raise ValueError(f"heldout_frac must be in (0, 1), got {heldout_frac}")

    saved_split = None
    if from_run is not None:
        split_path = Path(from_run) / "split.json"
        if split_path.is_file():
            saved_split = read_json(split_path)
            seed = int(saved_split["seed"])
            heldout_frac = float(saved_split["heldout_frac"])

    split = assign_split([clip.clip_id for clip in clips], seed, heldout_frac)
    if saved_split is not None and (
        list(saved_split.get("train", [])) != split.train or list(saved_split.get("heldout", [])) != split.heldout
    ):
        raise ValueError(
            f"{from_run} split does not match these clips at seed {seed} heldout_frac {heldout_frac}"
        )
    membership = split.membership()
    adapter_cls = get_adapter(adapter_name)
    adapter = adapter_cls(layer=layer, device=device)
    if objective is None:
        objective = default_objective(adapter)
    log(
        f"interpretableintelligence rank  device={device}  adapter={adapter_name}  "
        f"clips={len(clips)}  prefilter={prefilter}  objective={objective}"
    )
    adapter.load(random_init=False)
    tokens, acts = capture_activations(adapter, clips, membership)
    if acts.shape[1] != adapter.d_model:
        raise RuntimeError(f"captured d_model {acts.shape[1]} != adapter {adapter.d_model}")

    train_rows = np.array([token.split == "train" for token in tokens])
    held_rows = ~train_rows
    if from_run is None:
        config = SAEConfig(d_model=int(acts.shape[1]), seed=seed)
        log("training SAE on train clips")
        sae = train_sae(acts[torch.from_numpy(train_rows)], config, device, log=log)
    else:
        sae, config = load_sae(Path(from_run) / "sae", device)
        if config.d_model != int(acts.shape[1]):
            raise RuntimeError(
                f"SAE d_model {config.d_model} != captured activations {acts.shape[1]}"
            )

    started = time.perf_counter()
    by_clip = attribution_by_clip(adapter, sae, clips, objective=objective)
    attribution_seconds = time.perf_counter() - started
    attr_rows, mean_scores = summarize_attribution(by_clip)
    labels = _labels_for(adapter, clips, tokens, sae, acts, train_rows, held_rows, device)
    document = candidates_document(
        mean_scores,
        prefilter,
        objective=objective,
        labels=labels,
        seconds={"attribution": attribution_seconds, "ablation": None, "random_controls": None},
    )

    out = Path(out_dir)
    (out / "features").mkdir(parents=True, exist_ok=True)
    if from_run is None:
        (out / "sae").mkdir(parents=True, exist_ok=True)
        save_sae(sae, config, out / "sae")
    write_attribution(out / "features" / "attribution.parquet", attr_rows, objective=objective)
    write_json(out / "features" / "candidates.json", document)
    shown = ", ".join(
        f"{row['feature_id']}:{row['label'] or '-'}" for row in document["ranked"][: document["prefilter_k"]]
    )
    log(
        f"objective={objective}  attribution {attribution_seconds:.3f}s  shortlist=[{shown}]  "
        f"wrote {out / 'features' / 'attribution.parquet'}"
    )
    return document


def _labels_for(adapter, clips, tokens, sae, acts, train_rows, held_rows, device) -> list[str]:
    if adapter.name != "fake":
        return [""] * int(sae.n_features)
    with torch.no_grad():
        encoded = sae.encode(acts.to(device)).detach().cpu().numpy()
    labeller = FakeCueLabeller(cue_patterns(clips, tokens))
    labels = []
    for feature_id in range(encoded.shape[1]):
        labelled = labeller.label_feature(encoded[:, feature_id], train_rows, held_rows)
        labels.append(labelled["label"])
    return labels
