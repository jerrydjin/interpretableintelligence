"""capture → train SAE → label → score → run folder."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import torch

from physint import __version__
from physint.adapters import get_adapter
from physint.artifacts import (
    read_json,
    write_activation_shards,
    write_attribution,
    write_clip_effects,
    write_feature_acts,
    write_json,
    write_jsonl,
)
from physint.attribute import attribution_by_clip, candidates_document, rank_feature_ids, summarize_attribution
from physint.clips import load_clips
from physint.device import pick_device
from physint.labelers import FakeCueLabeller, cue_patterns
from physint.sae import SAEConfig, load_sae, save_sae, train_sae
from physint.score import (
    Token,
    ablate_edit,
    absent_in_random_init,
    capture_activations,
    clean_predictions,
    cosine_distance,
    dead_fraction,
    mean_cosine_shift,
    estimate_effects,
    explained_variance,
    mean_predictor_loss,
    random_percentile,
    splice_edit,
    top_examples,
)
from physint.split import assign_split
from physint.stats import feature_is_real, is_usable, loss_recovered
from physint.thresholds import (
    DEAD_ACT_EPS,
    DEFAULT_HELDOUT_FRAC,
    DEFAULT_LAYER,
    DEFAULT_PREFILTER_K,
    DEFAULT_SEED,
    MIN_CLIPS,
    N_RANDOM_FEATURES,
    TOP_K_ABLATION_PER_CLIP,
)


def execute_run(
    adapter_name: str,
    clips_dir: Path,
    out_dir: Path,
    layer: int = DEFAULT_LAYER,
    seed: int = DEFAULT_SEED,
    heldout_frac: float = DEFAULT_HELDOUT_FRAC,
    prefilter: int | None = None,
    log=print,
) -> dict:
    started = time.perf_counter()
    # Tiny matmuls, and a stable reference scoreboard.
    torch.set_num_threads(1)
    device = pick_device()
    clips = load_clips(Path(clips_dir))
    if len(clips) < MIN_CLIPS:
        raise ValueError(f"need at least {MIN_CLIPS} clips, found {len(clips)} in {clips_dir}")
    if not 0.0 < heldout_frac < 1.0:
        raise ValueError(f"heldout_frac must be in (0, 1), got {heldout_frac}")

    split = assign_split([clip.clip_id for clip in clips], seed, heldout_frac)
    membership = split.membership()
    log(
        f"physint {__version__}  device={device}  adapter={adapter_name}  "
        f"clips={len(clips)}  train={len(split.train)}  heldout={len(split.heldout)}"
    )

    adapter_cls = get_adapter(adapter_name)
    adapter = adapter_cls(layer=layer, device=device)
    adapter.load(random_init=False)
    if adapter.name != "fake":
        raise ValueError("only the fake labeller is implemented in this lane")

    tokens, acts = capture_activations(adapter, clips, membership)
    if acts.shape[1] != adapter.d_model:
        raise RuntimeError(f"captured d_model {acts.shape[1]} != adapter {adapter.d_model}")

    train_rows = np.array([token.split == "train" for token in tokens])
    held_rows = ~train_rows
    config = SAEConfig(d_model=int(acts.shape[1]), seed=seed)
    log("training SAE on train clips")
    sae = train_sae(acts[torch.from_numpy(train_rows)], config, device, log=log)

    held_x = acts[torch.from_numpy(held_rows)]
    with torch.no_grad():
        features = sae.encode(acts.to(device)).detach().cpu()
        recon = sae.reconstruct(held_x.to(device)).detach().cpu()
    ev = explained_variance(held_x, recon)
    dead = dead_fraction(features)

    heldout_clips = [clip for clip in clips if membership[clip.clip_id] == "heldout"]
    loss_clean = mean_predictor_loss(adapter, heldout_clips)
    loss_zero = mean_predictor_loss(adapter, heldout_clips, zero=True)
    loss_sae = mean_predictor_loss(adapter, heldout_clips, edit=splice_edit(sae))
    recovered = loss_recovered(loss_zero, loss_sae, loss_clean)
    usable = is_usable(recovered)
    log(
        f"ev_heldout={ev:.4f}  loss_clean={loss_clean:.6f}  loss_zero={loss_zero:.6f}  "
        f"loss_sae={loss_sae:.6f}  loss_recovered={recovered:.4f}  dead_frac={dead:.3f}  usable={usable}"
    )

    log("labelling and attributing")
    labeller = FakeCueLabeller(cue_patterns(clips, tokens))
    clean = clean_predictions(adapter, clips)
    estimate_rows, rank_scores, by_clip, attribution_seconds = _collect_estimates(
        adapter, sae, clips, features.shape[1]
    )
    attr_rows, mean_scores = summarize_attribution(by_clip)
    ranked_ids = rank_feature_ids(mean_scores)
    if prefilter is not None and prefilter < 1:
        raise ValueError(f"prefilter K must be >= 1, got {prefilter}")
    limit = None if prefilter is None else min(int(prefilter), int(features.shape[1]))
    selected = None if limit is None else set(ranked_ids[:limit])
    subset = selected is not None and len(selected) < int(features.shape[1])
    if subset:
        log(f"prefilter K={limit}  candidates={ranked_ids[:limit]}  (ablations only on these features)")
    else:
        log("scoring every feature; attribution shortlist is written alongside the full ablations")

    log("random-init control")
    control = adapter_cls(layer=layer, device=device)
    control.load(random_init=True)
    control_tokens, control_acts = capture_activations(control, clips, membership)
    if not _same_layout(tokens, control_tokens):
        raise RuntimeError("random-init activations are not aligned with the run")
    control_sae = train_sae(control_acts[torch.from_numpy(train_rows)], config, device, log=log)
    with torch.no_grad():
        control_features = control_sae.encode(control_acts.to(device)).detach().cpu().numpy()

    feature_np = features.numpy()
    n_features = feature_np.shape[1]
    records: list[dict] = []
    jsonl: list[dict] = []
    ablation_rows: dict[str, list] = {key: [] for key in ("feature_id", "clip_id", "window", "t", "effect", "effect_kind")}
    top_k = min(TOP_K_ABLATION_PER_CLIP, n_features)
    ablated: set[tuple[str, int]] = set()
    if subset:
        assert selected is not None
        for clip in clips:
            for feature_id in selected:
                ablated.add((clip.clip_id, int(feature_id)))
    else:
        for clip in clips:
            order = torch.argsort(rank_scores[clip.clip_id], descending=True)[:top_k]
            for feature_id in order.tolist():
                ablated.add((clip.clip_id, int(feature_id)))

    ablation_seconds = 0.0
    control_seconds = 0.0
    for feature_id in range(n_features):
        values = feature_np[:, feature_id]
        top, distinct = top_examples(values, tokens)
        labelled = labeller.label_feature(values, train_rows, held_rows)
        measure = selected is None or feature_id in selected
        if measure:
            started_ablation = time.perf_counter()
            effect = _heldout_ablation(adapter, sae, heldout_clips, feature_id, clean)
            ablation_seconds += time.perf_counter() - started_ablation
            started_controls = time.perf_counter()
            percentile = random_percentile(
                adapter=adapter,
                sae=sae,
                feature_id=feature_id,
                real_effect=effect,
                tokens=tokens,
                features=feature_np,
                heldout_clips=heldout_clips,
                clean=clean,
                seed=seed,
                n_random=N_RANDOM_FEATURES,
            )
            control_seconds += time.perf_counter() - started_controls
            n_random = N_RANDOM_FEATURES
        else:
            effect = None
            percentile = None
            n_random = 0
        absent = absent_in_random_init(values, control_features)
        real = feature_is_real(
            usable=usable,
            distinct_clips_top20=distinct,
            label_holds_heldout=labelled["label_holds_heldout"],
            random_percentile=percentile,
            absent_in_random_init=absent,
        )
        record = {
            "id": feature_id,
            "label": labelled["label"],
            "distinct_clips_top20": distinct,
            "label_holds_heldout": labelled["label_holds_heldout"],
            "ablation_effect": effect,
            "random_percentile": percentile,
            "n_random": n_random,
            "absent_in_random_init": absent,
            "real": real,
        }
        records.append(record)
        jsonl.append(
            {
                "id": feature_id,
                "label": labelled["label"],
                "label_model": labelled["label_model"],
                "top": top,
                "distinct_clips_top20": distinct,
                "label_holds_heldout": labelled["label_holds_heldout"],
                "ablation_effect": effect,
                "random_percentile": percentile,
                "n_random": n_random,
                "absent_in_random_init": absent,
                "real": real,
            }
        )
        pct_text = f"{percentile:6.1f}" if percentile is not None else "     -"
        abl_text = f"{effect:.5f}" if effect is not None else "skipped"
        log(
            f"feature {feature_id:02d}  label={labelled['label'] or '-':<8}  real={str(real).lower():<5}  "
            f"distinct={distinct:<3}  holds={labelled['label_holds_heldout']}  "
            f"pct={pct_text}  absent={absent}  abl={abl_text}"
        )
        started_rows = time.perf_counter()
        _append_feature_ablations(
            ablation_rows,
            adapter,
            sae,
            clips,
            feature_id,
            clean,
            ablated,
        )
        ablation_seconds += time.perf_counter() - started_rows

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "acts").mkdir(parents=True)
    (out / "sae").mkdir(parents=True)
    (out / "features").mkdir(parents=True)

    model_sha = adapter.model_sha()
    manifest = {
        "run_id": out.name,
        "tool_version": __version__,
        "model": {
            "adapter": adapter.name,
            "name": adapter.name,
            "sha": model_sha,
            "random_init": False,
        },
        "layer": int(layer),
        "d_model": int(acts.shape[1]),
        "grid": adapter.grid().as_dict(),
        "split": {"seed": split.seed, "heldout_frac": split.heldout_frac, "sha": split.sha},
        "clips": [
            {
                "clip_id": clip.clip_id,
                "sha256": clip.sha256,
                "fps": clip.fps,
                "n_frames": clip.n_frames,
                "split": membership[clip.clip_id],
            }
            for clip in clips
        ],
    }
    scoreboard = {
        "run": {
            "model_sha": model_sha,
            "layer": int(layer),
            "split_sha": split.sha,
            "seed": int(seed),
            "n_train_clips": len(split.train),
            "n_heldout_clips": len(split.heldout),
        },
        "sae": {
            "ev_heldout": ev,
            "loss_recovered": recovered,
            "dead_frac": dead,
            "usable": usable,
        },
        "features": records,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "split.json", split.as_dict())
    write_json(out / "scoreboard.json", scoreboard)
    write_activation_shards(out / "acts", acts.numpy(), tokens)
    save_sae(sae, config, out / "sae")
    write_jsonl(out / "features" / "features.jsonl", jsonl)
    write_feature_acts(out / "features" / "feature_acts.parquet", _sparse_feature_rows(feature_np, tokens))
    write_attribution(out / "features" / "attribution.parquet", attr_rows)
    reported_k = limit if limit is not None else min(DEFAULT_PREFILTER_K, n_features)
    labels = [row["label"] for row in records]
    seconds = {
        "attribution": attribution_seconds,
        "ablation": ablation_seconds,
        "random_controls": control_seconds,
    }
    write_json(
        out / "features" / "candidates.json",
        candidates_document(mean_scores, reported_k, labels=labels, seconds=seconds),
    )
    effect_rows = {
        key: estimate_rows[key] + ablation_rows[key]
        for key in ("feature_id", "clip_id", "window", "t", "effect", "effect_kind")
    }
    write_clip_effects(out / "features" / "clip_effects.parquet", effect_rows)

    real_labels = [row["label"] for row in records if row["real"]]
    elapsed = time.perf_counter() - started
    log(
        f"attribution {attribution_seconds:.3f}s  ablation {ablation_seconds:.3f}s  "
        f"random_controls {control_seconds:.3f}s"
    )
    log(f"wrote {out}  real={len(real_labels)} {real_labels}  {elapsed:.1f}s")
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
    log=print,
) -> dict:
    """Score every feature with one attribution pass and write the top-K shortlist.

    Does not ablate. Pass ``from_run`` to reuse an SAE that a full ``physint run``
    already trained; otherwise an SAE is trained on the train split, matching
    ``execute_run``.
    """
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
    log(
        f"physint rank  device={device}  adapter={adapter_name}  "
        f"clips={len(clips)}  prefilter={prefilter}"
    )

    adapter_cls = get_adapter(adapter_name)
    adapter = adapter_cls(layer=layer, device=device)
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
    by_clip = attribution_by_clip(adapter, sae, clips)
    attribution_seconds = time.perf_counter() - started
    attr_rows, mean_scores = summarize_attribution(by_clip)
    labels = _labels_for(adapter, clips, tokens, sae, acts, train_rows, held_rows, device)
    document = candidates_document(
        mean_scores,
        prefilter,
        labels=labels,
        seconds={"attribution": attribution_seconds, "ablation": None, "random_controls": None},
    )

    out = Path(out_dir)
    (out / "features").mkdir(parents=True, exist_ok=True)
    if from_run is None:
        (out / "sae").mkdir(parents=True, exist_ok=True)
        save_sae(sae, config, out / "sae")
    write_attribution(out / "features" / "attribution.parquet", attr_rows)
    write_json(out / "features" / "candidates.json", document)
    shown = ", ".join(
        f"{row['feature_id']}:{row['label'] or '-'}" for row in document["ranked"][: document["prefilter_k"]]
    )
    log(
        f"attribution {attribution_seconds:.3f}s  candidates=[{shown}]  "
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


def _collect_estimates(adapter, sae, clips, n_features: int):
    rows = {key: [] for key in ("feature_id", "clip_id", "window", "t", "effect", "effect_kind")}
    scores = {clip.clip_id: torch.zeros(n_features) for clip in clips}
    by_clip = {clip.clip_id: np.zeros(n_features, dtype=np.float64) for clip in clips}
    attribution_seconds = 0.0
    for clip in clips:
        for window in adapter.windows(clip):
            started = time.perf_counter()
            effect = estimate_effects(adapter, sae, clip, window)
            attribution_seconds += time.perf_counter() - started
            effect = effect.detach().cpu()
            scores[clip.clip_id] += effect.abs().mean(dim=(0, 1, 2))
            flat = effect.numpy().reshape(-1, n_features).astype(np.float64)
            by_clip[clip.clip_id] += flat.sum(axis=0)
            frame = effect.mean(dim=(1, 2))
            for t in range(frame.shape[0]):
                for feature_id in range(n_features):
                    rows["feature_id"].append(feature_id)
                    rows["clip_id"].append(clip.clip_id)
                    rows["window"].append(window.index)
                    rows["t"].append(t)
                    rows["effect"].append(float(frame[t, feature_id]))
                    rows["effect_kind"].append("estimate")
    return rows, scores, by_clip, attribution_seconds


def _heldout_ablation(adapter, sae, heldout_clips, feature_id: int, clean) -> float:
    return mean_cosine_shift(adapter, heldout_clips, ablate_edit(sae, feature_id), clean)


def _append_feature_ablations(rows, adapter, sae, clips, feature_id: int, clean, wanted: set[tuple[str, int]]) -> None:
    targets = [clip for clip in clips if (clip.clip_id, feature_id) in wanted]
    if not targets:
        return
    edit = ablate_edit(sae, feature_id)
    grid_t = adapter.grid().t
    with torch.no_grad():
        for clip in targets:
            for window in adapter.windows(clip):
                edited, _target = adapter.predict_masked(clip, window, edit=edit)
                baseline = clean[(clip.clip_id, window.index)]
                dist = cosine_distance(baseline, edited)
                coords = adapter.mask(window).nonzero(as_tuple=False)
                for t in range(grid_t):
                    selected = coords[:, 0] == t
                    value = float(dist[selected].mean()) if int(selected.sum()) else 0.0
                    rows["feature_id"].append(feature_id)
                    rows["clip_id"].append(clip.clip_id)
                    rows["window"].append(window.index)
                    rows["t"].append(t)
                    rows["effect"].append(value)
                    rows["effect_kind"].append("ablation")


def _sparse_feature_rows(features: np.ndarray, tokens: list[Token]) -> dict[str, list]:
    rows = {key: [] for key in ("feature_id", "clip_id", "window", "t", "h", "w", "value")}
    token_index, feature_index = np.nonzero(features > DEAD_ACT_EPS)
    for row, feature_id in zip(token_index.tolist(), feature_index.tolist(), strict=True):
        token = tokens[row]
        rows["feature_id"].append(int(feature_id))
        rows["clip_id"].append(token.clip_id)
        rows["window"].append(int(token.window))
        rows["t"].append(int(token.t))
        rows["h"].append(int(token.h))
        rows["w"].append(int(token.w))
        rows["value"].append(float(features[row, feature_id]))
    return rows


def _same_layout(left: list[Token], right: list[Token]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if (a.clip_id, a.window, a.t, a.h, a.w, a.split) != (b.clip_id, b.window, b.t, b.h, b.w, b.split):
            return False
    return True
