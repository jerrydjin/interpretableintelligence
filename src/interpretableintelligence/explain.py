"""Explain a new clip with a finished run.

The cited answer contains only features whose scoreboard record has ``real``
true, and only numbers measured by ablation (zero the feature, measure the
predictor change). Estimates rank which features to ablate. They are not
evidence. Features that are not real, and features that only have an estimate,
are listed under ``unverified``.

Every payload carries ``run_id``, ``model_sha``, and ``layer``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.adapters import get_adapter
from interpretableintelligence.adapters.base import Clip
from interpretableintelligence.artifacts import read_json, read_jsonl
from interpretableintelligence.clips import file_sha256
from interpretableintelligence.device import pick_device
from interpretableintelligence.sae import load_sae
from interpretableintelligence.score import ablate_edit, cosine_distance, estimate_effects
from interpretableintelligence.stats import is_usable
from interpretableintelligence.thresholds import DEAD_ACT_EPS, TOP_K_ABLATION_PER_CLIP, USABLE_LOSS_RECOVERED

EFFECT_ABLATION = "ablation"
EFFECT_ESTIMATE = "estimate"


class ExplainError(Exception):
    """The clip cannot be explained. ``run_id``, ``model_sha``, and ``layer`` are filled when known."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        model_sha: str | None = None,
        layer: int | None = None,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.model_sha = model_sha
        self.layer = layer

    def as_dict(self) -> dict:
        return {
            "error": str(self),
            "run_id": self.run_id,
            "model_sha": self.model_sha,
            "layer": self.layer,
        }


class SAEUnusable(ExplainError):
    """The run's scoreboard does not mark the SAE usable. Nothing is cited."""


class RunNotFound(ExplainError):
    """No run folder with this id under the runs directory."""


class FeatureNotFound(ExplainError):
    """The run has no feature with this id."""


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    run_id: str
    model_sha: str
    layer: int
    adapter_name: str
    manifest: dict
    scoreboard: dict
    features_by_id: dict[int, dict]


def cites_feature(*, real: bool, effect_kind: str) -> bool:
    """True only for a real scoreboard feature measured by ablation."""
    return real is True and effect_kind == EFFECT_ABLATION


def _unscored(manifest: dict, run_dir: Path) -> dict:
    """In-memory stand-in for a missing scoreboard: every feature unverified. Never written."""
    config = run_dir / "sae" / "config.json"
    if not config.is_file():
        raise ExplainError(f"run folder {run_dir} has no scoreboard.json and no sae/config.json", run_id=run_dir.name)
    n_features = int(read_json(config)["n_features"])
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    return {
        "run": {"model_sha": model.get("sha"), "layer": manifest.get("layer")},
        "features": [{"id": index, "real": False} for index in range(n_features)],
    }


def load_run(run_dir: Path, allow_unscored: bool = False) -> RunContext:
    """Read the manifest and scoreboard. Does not decide whether the SAE may be cited.

    ``allow_unscored`` (the playground only) accepts a run stopped before scoring, with every
    feature unverified. ``explain`` never passes it, because it cites only scored features.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RunNotFound(f"run folder not found: {run_dir}", run_id=run_dir.name)
    manifest_path = run_dir / "manifest.json"
    scoreboard_path = run_dir / "scoreboard.json"
    unscored = allow_unscored and manifest_path.is_file() and not scoreboard_path.is_file()
    if not manifest_path.is_file() or not (scoreboard_path.is_file() or unscored):
        raise ExplainError(
            f"run folder {run_dir} is missing manifest.json or scoreboard.json",
            run_id=run_dir.name,
        )
    manifest = read_json(manifest_path)
    scoreboard = _unscored(manifest, run_dir) if unscored else read_json(scoreboard_path)
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ExplainError(f"{manifest_path} has no run_id", run_id=run_dir.name)
    if run_id != run_dir.name:
        raise ExplainError(
            f"manifest.run_id {run_id!r} does not match the folder name {run_dir.name!r}",
            run_id=run_id,
        )
    model = manifest.get("model")
    block = scoreboard.get("run")
    if not isinstance(model, dict) or not isinstance(block, dict):
        raise ExplainError(f"run {run_id} is missing model or scoreboard.run", run_id=run_id)
    model_sha = model.get("sha")
    if not isinstance(model_sha, str) or model_sha != block.get("model_sha"):
        raise ExplainError(
            f"run {run_id} model sha does not match between manifest and scoreboard",
            run_id=run_id,
            model_sha=model_sha if isinstance(model_sha, str) else None,
        )
    layer = manifest.get("layer")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer != block.get("layer"):
        raise ExplainError(
            f"run {run_id} layer does not match between manifest and scoreboard",
            run_id=run_id,
            model_sha=model_sha,
        )
    adapter_name = model.get("adapter")
    if not isinstance(adapter_name, str) or not adapter_name:
        raise ExplainError(f"run {run_id} has no adapter name", run_id=run_id, model_sha=model_sha, layer=layer)
    features = scoreboard.get("features")
    if not isinstance(features, list) or not features:
        raise ExplainError(f"run {run_id} scoreboard has no features", run_id=run_id, model_sha=model_sha, layer=layer)
    by_id: dict[int, dict] = {}
    for record in features:
        if not isinstance(record, dict) or not isinstance(record.get("id"), int) or isinstance(record.get("id"), bool):
            raise ExplainError(
                f"run {run_id} has a scoreboard feature without an integer id",
                run_id=run_id,
                model_sha=model_sha,
                layer=layer,
            )
        feature_id = int(record["id"])
        if feature_id in by_id:
            raise ExplainError(
                f"run {run_id} lists feature {feature_id} more than once",
                run_id=run_id,
                model_sha=model_sha,
                layer=layer,
            )
        by_id[feature_id] = record
    expected = list(range(len(features)))
    if sorted(by_id) != expected:
        raise ExplainError(
            f"run {run_id} feature ids must be 0..{len(features) - 1}",
            run_id=run_id,
            model_sha=model_sha,
            layer=layer,
        )
    return RunContext(
        run_dir=run_dir,
        run_id=run_id,
        model_sha=model_sha,
        layer=int(layer),
        adapter_name=adapter_name,
        manifest=manifest,
        scoreboard=scoreboard,
        features_by_id=by_id,
    )


def resolve_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Resolve ``run_id`` to a child of ``runs_dir``. Rejects paths that escape the directory."""
    if not isinstance(run_id, str) or not run_id or run_id.strip() != run_id:
        raise ExplainError("run_id must be the run folder name")
    if "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ExplainError(f"run_id must be a single folder name, got {run_id!r}")
    root = Path(runs_dir).resolve()
    if not root.is_dir():
        raise ExplainError(f"runs directory not found: {runs_dir}")
    path = (root / run_id).resolve()
    if path.parent != root:
        raise ExplainError(f"run_id escapes the runs directory: {run_id!r}", run_id=run_id)
    if not path.is_dir():
        raise RunNotFound(f"run not found: {run_id}", run_id=run_id)
    return path


def list_runs(runs_dir: Path) -> dict:
    """Summaries of run folders under ``runs_dir``. Each entry carries the trace fields."""
    root = Path(runs_dir)
    if not root.is_dir():
        raise ExplainError(f"runs directory not found: {runs_dir}")
    runs = []
    for path in sorted(child for child in root.iterdir() if child.is_dir()):
        if not (path / "manifest.json").is_file() or not (path / "scoreboard.json").is_file():
            continue
        ctx = load_run(path)
        sae = ctx.scoreboard.get("sae")
        usable = isinstance(sae, dict) and sae.get("usable") is True
        runs.append(
            {
                "run_id": ctx.run_id,
                "model_sha": ctx.model_sha,
                "layer": ctx.layer,
                "usable": usable,
                "adapter": ctx.adapter_name,
            }
        )
    return {"runs": runs}


def get_feature(run_dir: Path, feature_id: int) -> dict:
    """One feature record, reconciled with the scoreboard, plus the trace fields."""
    if isinstance(feature_id, bool) or not isinstance(feature_id, int):
        raise FeatureNotFound(f"feature id must be an integer, got {feature_id!r}")
    ctx = load_run(run_dir)
    record = ctx.features_by_id.get(feature_id)
    if record is None:
        raise FeatureNotFound(
            f"run {ctx.run_id} has no feature {feature_id}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    path = ctx.run_dir / "features" / "features.jsonl"
    if not path.is_file():
        raise ExplainError(
            f"run {ctx.run_id} is missing features/features.jsonl",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    rows = [row for row in read_jsonl(path) if row.get("id") == feature_id]
    if len(rows) != 1:
        raise ExplainError(
            f"run {ctx.run_id} features.jsonl does not contain exactly one record for feature {feature_id}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    feature = dict(rows[0])
    for key in (
        "label",
        "distinct_clips_top20",
        "label_holds_heldout",
        "ablation_effect",
        "random_percentile",
        "n_random",
        "absent_in_random_init",
        "real",
    ):
        feature[key] = record[key]
    feature["ablation_effect_kind"] = EFFECT_ABLATION
    return {
        "run_id": ctx.run_id,
        "model_sha": ctx.model_sha,
        "layer": ctx.layer,
        "feature": feature,
    }


def load_clip_file(path: Path, clip_id: str | None = None, fps: float = 8.0) -> Clip:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"clip not found: {path}")
    sha = file_sha256(path)
    try:
        frames = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ExplainError(f"could not read {path} as a numpy array: {exc}") from exc
    frames = np.ascontiguousarray(frames, dtype=np.float32)
    if frames.ndim != 4:
        raise ExplainError(
            f"clip must be a float array shaped [frames, height, width, channels], got {tuple(frames.shape)}"
        )
    name = clip_id if clip_id else path.stem
    if not name:
        raise ExplainError(f"clip {path} has an empty id")
    return Clip(clip_id=name, path=path, frames=frames, fps=float(fps), sha256=sha)


def explain_clip(run_dir: Path, clip_path: Path, *, clip_id: str | None = None) -> dict:
    """Run the run's adapter on ``clip_path`` and return the ranked ablation answer."""
    ctx = load_run(run_dir)
    _refuse_if_unusable(ctx)
    torch.set_num_threads(1)
    device = pick_device()
    try:
        sae, config = load_sae(ctx.run_dir / "sae", device)
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        raise ExplainError(
            f"refusing to explain: SAE for run {ctx.run_id} could not be loaded ({exc})",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        ) from exc
    if config.activation not in ("relu", "topk"):
        raise ExplainError(
            f"refusing to explain: SAE for run {ctx.run_id} has activation {config.activation!r}, expected relu or topk",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    if config.d_model != ctx.manifest.get("d_model") or config.n_features != len(ctx.features_by_id):
        raise ExplainError(
            f"refusing to explain: SAE for run {ctx.run_id} does not match the manifest "
            f"(d_model {config.d_model} vs {ctx.manifest.get('d_model')}, "
            f"n_features {config.n_features} vs {len(ctx.features_by_id)})",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    if ctx.manifest["model"].get("random_init") is not False:
        raise ExplainError(
            f"refusing to explain: run {ctx.run_id} is not a trained model (random_init must be false)",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    try:
        adapter_cls = get_adapter(ctx.adapter_name)
    except ValueError as exc:
        raise ExplainError(str(exc), run_id=ctx.run_id, model_sha=ctx.model_sha, layer=ctx.layer) from exc
    adapter = adapter_cls(layer=ctx.layer, device=device)
    adapter.load(random_init=False)
    loaded_sha = adapter.model_sha()
    if loaded_sha != ctx.model_sha:
        raise ExplainError(
            f"refusing to explain: loaded {ctx.adapter_name} weights sha {loaded_sha} "
            f"does not match run {ctx.run_id} model sha {ctx.model_sha}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    clip = load_clip_file(clip_path, clip_id=clip_id)
    try:
        windows = adapter.windows(clip)
    except ValueError as exc:
        raise ExplainError(
            f"clip does not match the {ctx.adapter_name} adapter for run {ctx.run_id}: {exc}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        ) from exc
    if not windows:
        raise ExplainError(
            f"adapter {ctx.adapter_name} produced no windows for clip {clip.clip_id}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )

    captured: list[tuple] = []
    scores = torch.zeros(config.n_features)
    for window in windows:
        features = _capture_features(adapter, sae, clip, window, ctx)
        effect = estimate_effects(adapter, sae, clip, window).detach().cpu()
        if tuple(effect.shape[:3]) != tuple(features.shape[:3]) or effect.shape[-1] != config.n_features:
            raise ExplainError(
                f"estimate grid {tuple(effect.shape)} does not match activations {tuple(features.shape)}",
                run_id=ctx.run_id,
                model_sha=ctx.model_sha,
                layer=ctx.layer,
            )
        scores += effect.abs().mean(dim=(0, 1, 2))
        captured.append((window, features, effect))

    order = sorted(range(config.n_features), key=lambda index: (-float(scores[index]), index))
    chosen = set(order[: min(TOP_K_ABLATION_PER_CLIP, config.n_features)])
    answer_rows: list[dict] = []
    unverified: list[dict] = []
    for window, features, effect in captured:
        ablations = {
            feature_id: _ablation_by_time(adapter, sae, clip, window, feature_id) for feature_id in sorted(chosen)
        }
        frame_estimate = effect.mean(dim=(1, 2))
        for feature_id, record in ctx.features_by_id.items():
            real_flag = record.get("real")
            label = record.get("label") if isinstance(record.get("label"), str) else ""
            fired = _fired_times(features, feature_id)
            if feature_id in chosen:
                measured = ablations[feature_id]
                kind = EFFECT_ABLATION
            else:
                measured = frame_estimate[:, feature_id]
                kind = EFFECT_ESTIMATE
            for t in fired:
                value = float(measured[t])
                if not math.isfinite(value):
                    raise ExplainError(
                        f"non-finite {kind} effect for feature {feature_id} at t={t}",
                        run_id=ctx.run_id,
                        model_sha=ctx.model_sha,
                        layer=ctx.layer,
                    )
                if cites_feature(real=real_flag is True, effect_kind=kind):
                    answer_rows.append(
                        {
                            "window": int(window.index),
                            "t": int(t),
                            "feature_id": feature_id,
                            "label": label,
                            "effect": value,
                            "effect_kind": EFFECT_ABLATION,
                            "real": True,
                        }
                    )
                else:
                    unverified.append(
                        {
                            "feature_id": feature_id,
                            "label": label,
                            "window": int(window.index),
                            "t": int(t),
                            "effect": value,
                            "effect_kind": kind,
                            "real": real_flag if isinstance(real_flag, bool) else False,
                            "reason": _unverified_reason(real=real_flag is True, effect_kind=kind),
                        }
                    )

    return {
        "run_id": ctx.run_id,
        "model_sha": ctx.model_sha,
        "layer": ctx.layer,
        "clip_id": clip.clip_id,
        "clip_sha256": clip.sha256,
        "answer": _group_answer(answer_rows),
        "unverified": sorted(unverified, key=lambda row: (row["window"], row["t"], row["feature_id"], row["effect_kind"])),
    }


def _refuse_if_unusable(ctx: RunContext) -> None:
    sae = ctx.scoreboard.get("sae")
    if not isinstance(sae, dict):
        raise SAEUnusable(
            f"refusing to explain: run {ctx.run_id} has no SAE block on the scoreboard. No features were cited.",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    recovered = sae.get("loss_recovered")
    loss_sae = sae.get("loss_sae")
    loss_visible_mean = sae.get("loss_visible_mean")
    numbers_ok = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (recovered, loss_sae, loss_visible_mean)
    )
    floor_ok = numbers_ok and is_usable(float(recovered), float(loss_sae), float(loss_visible_mean))
    if sae.get("usable") is True and floor_ok:
        return
    raise SAEUnusable(
        f"refusing to explain: the SAE for run {ctx.run_id} is not usable "
        f"(usable={sae.get('usable')!r}, loss_recovered={recovered!r}, "
        f"loss_sae={loss_sae!r}, loss_visible_mean={loss_visible_mean!r}; "
        f"need usable true, loss_recovered >= {USABLE_LOSS_RECOVERED:.2f}, "
        "and loss_sae < loss_visible_mean). "
        "No features were cited.",
        run_id=ctx.run_id,
        model_sha=ctx.model_sha,
        layer=ctx.layer,
    )


def _capture_features(adapter, sae, clip: Clip, window, ctx: RunContext) -> torch.Tensor:
    """Capture the run's layer and encode it with the run's SAE. Returns [t, h, w, n_features] on CPU."""
    with adapter.hook(ctx.layer, mode="capture") as handle:
        encoded = adapter.encode(clip, window)
    if len(handle.captured) != 1:
        raise ExplainError(
            f"expected one captured tensor for {clip.clip_id} window {window.index}, got {len(handle.captured)}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    acts = handle.captured[0]
    if not torch.equal(acts, encoded):
        raise ExplainError(
            f"captured layer does not match encode() for {clip.clip_id} window {window.index}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    if acts.ndim != 4:
        raise ExplainError(
            f"activations must be [t, h, w, d], got {tuple(acts.shape)}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    grid = adapter.grid()
    if tuple(acts.shape[:3]) != (grid.t, grid.h, grid.w):
        raise ExplainError(
            f"activation grid {tuple(acts.shape[:3])} does not match the adapter grid {(grid.t, grid.h, grid.w)}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    if acts.shape[-1] != sae.d_model:
        raise ExplainError(
            f"activation width {acts.shape[-1]} does not match the SAE d_model {sae.d_model}",
            run_id=ctx.run_id,
            model_sha=ctx.model_sha,
            layer=ctx.layer,
        )
    flat = acts.detach().to(dtype=torch.float32, device=sae.W_enc.device).reshape(-1, sae.d_model)
    with torch.no_grad():
        features = sae.encode(flat).reshape(grid.t, grid.h, grid.w, sae.n_features)
    return features.detach().cpu()


def _fired_times(features: torch.Tensor, feature_id: int) -> list[int]:
    peak = features[..., feature_id].amax(dim=(1, 2))
    return [index for index, value in enumerate(peak.tolist()) if float(value) > DEAD_ACT_EPS]


def _ablation_by_time(adapter, sae, clip: Clip, window, feature_id: int) -> list[float]:
    """Mean cosine distance of the masked predictor output after the feature is zeroed, one value per frame."""
    edit = ablate_edit(sae, feature_id)
    with torch.no_grad():
        baseline, _target = adapter.predict_masked(clip, window)
        edited, _target = adapter.predict_masked(clip, window, edit=edit)
        dist = cosine_distance(baseline, edited).detach().cpu()
    coords = adapter.mask(window).detach().cpu().nonzero(as_tuple=False)
    values: list[float] = []
    for t in range(adapter.grid().t):
        selected = coords[:, 0] == t
        values.append(float(dist[selected].mean()) if int(selected.sum()) else 0.0)
    return values


def _unverified_reason(*, real: bool, effect_kind: str) -> str:
    if effect_kind != EFFECT_ABLATION:
        return "estimate only"
    if real is not True:
        return "scoreboard real is false"
    return "not cited"


def _group_answer(rows: list[dict]) -> list[dict]:
    """Groups of (window, t), strongest ablation first. Features inside a time are ranked the same way."""
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["window"], row["t"]), []).append(row)

    def group_key(item: tuple[tuple[int, int], list[dict]]) -> tuple[float, int, int]:
        (window, t), features = item
        best = max(feature["effect"] for feature in features)
        return (-best, window, t)

    answer = []
    for (window, t), features in sorted(grouped.items(), key=group_key):
        ranked = sorted(features, key=lambda feature: (-feature["effect"], feature["feature_id"]))
        answer.append(
            {
                "window": window,
                "t": t,
                "features": [
                    {
                        "feature_id": feature["feature_id"],
                        "label": feature["label"],
                        "effect": feature["effect"],
                        "effect_kind": EFFECT_ABLATION,
                        "real": True,
                    }
                    for feature in ranked
                ],
            }
        )
    return answer
