"""Export one run as a static site anyone can open from a link.

``ii export runs/<id> --out site/`` writes the viewer plus every file it would
otherwise ask the local server for. Nothing in the output calls a server, and
every path is relative, so the folder works from any static host or subpath
(GitHub Pages, Netlify, an S3 bucket, ``python -m http.server``).

The playground needs the model, which a static page cannot run. The export
therefore precomputes a fixed menu of edits (remove, scale, clamp, add at a few
strengths) for the strongest features on a handful of clips, and the page looks
each one up instead of posting to ``/api/steer``. Like the live playground this
is exploration only: nothing here touches the run folder or the scoreboard.
"""

from __future__ import annotations

import functools
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from interpretableintelligence.view import RunModel, load_run

STATIC = Path(__file__).resolve().parent / "static"
MARKER = ".ii-export"

# The strengths the static playground offers, per edit. Each list holds the live
# viewer's default for that edit (app.js STEER_KINDS) so the page opens on a
# precomputed value.
STEER_PRESETS: dict[str, list[float]] = {
    "ablate": [0.0],
    "scale": [0.0, 0.5, 2.0, 3.0, 5.0, 10.0],
    "clamp": [-1.0, 0.0, 1.0, 3.0, 5.0],
    "add": [-1.0, 1.0, 3.0, 5.0],
}
DEFAULT_STEER_CLIPS = 12
DEFAULT_STEER_FEATURES = 6
SHIFT_DIGITS = 5
# Heatmap cells in a slim export keep this many significant digits: the page draws them as
# colours and block heights, and full floats double the size of every clip's heat.json.
HEAT_DIGITS = 3


@dataclass
class ExportSummary:
    out_dir: Path
    clips: int
    features: int
    steer_clips: list[str] = field(default_factory=list)
    steer_results: int = 0
    bytes_total: int = 0


def export_run(
    run_dir: Path,
    out_dir: Path,
    clips_dir: Path | None = None,
    steer: bool = True,
    steer_clips: int = DEFAULT_STEER_CLIPS,
    steer_features: int = DEFAULT_STEER_FEATURES,
    title: str | None = None,
    max_clips: int | None = None,
    log: Callable[[str], None] = lambda _message: None,
) -> ExportSummary:
    """Write a self-contained static copy of the viewer for ``run_dir`` into ``out_dir``.

    ``max_clips`` keeps only that many clips, the ones the playground and the top clips of
    real and candidate features use first, so a run over many clips fits a free static host.
    """
    run_dir = Path(run_dir)
    out = Path(out_dir)
    if max_clips is not None and max_clips < 1:
        raise ValueError(f"max_clips must be >= 1, got {max_clips}")
    _prepare_out(out)
    model = load_run(run_dir, None if clips_dir is None else Path(clips_dir))
    keep = clip_priority(model)[:max_clips] if max_clips is not None else None
    data = out / "data"
    shown = len(keep) if keep is not None else len(model.run_payload["clips"])
    log(f"writing viewer data for {shown} clips")
    write_viewer_data(model, data, clip_ids=keep, slim=True)
    summary = ExportSummary(out_dir=out, clips=shown, features=len(model.run_payload["features"]))
    if steer:
        chosen, results = _precompute_steering(
            model, run_dir, clips_dir, data / "steer", steer_clips, steer_features, log, only=keep
        )
        summary.steer_clips = chosen
        summary.steer_results = results
        if chosen:
            run_payload = json.loads((data / "run.json").read_text(encoding="utf-8"))
            run_payload["steer"] = {
                "enabled": True,
                "static": True,
                "sae_usable": _sae_usable(model),
                "presets": STEER_PRESETS,
                "clips": chosen,
            }
            write_json(data / "run.json", run_payload)
    _copy_static(out / "static")
    run_id = str(model.run_payload.get("run_id") or run_dir.name)
    (out / "index.html").write_text(_shell(title or f"{run_id} · interpretableintelligence", model), encoding="utf-8")
    (out / MARKER).write_text(json.dumps({"run_id": run_id}) + "\n", encoding="utf-8")
    summary.bytes_total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    return summary


def write_viewer_data(
    model: RunModel,
    data: Path,
    indent: int | None = None,
    clip_ids: list[str] | None = None,
    slim: bool = False,
) -> None:
    """Every read the viewer makes, as files under ``data`` (``app.js`` ``apiUrl`` maps the paths).

    ``indent`` keeps the committed ``demo-site/`` diff-friendly; ``ii export`` writes compact JSON.
    ``clip_ids`` keeps only those clips (``run.json`` gains ``export.clips_total``, and feature
    pages list only top clips that are kept). ``slim`` writes only what a page can reach: the
    heatmaps of the features a clip frame lists or a feature page links to (cells rounded to
    ``HEAT_DIGITS`` significant digits), and for a clip that plays as video, only the frame
    pictures used as thumbnails (frame 0 and the model's frames).
    """
    data.mkdir(parents=True)
    write_json = functools.partial(_write_json, indent=indent)
    kept = None if clip_ids is None else {str(clip_id) for clip_id in clip_ids}
    clips = [clip for clip in model.run_payload["clips"] if kept is None or str(clip["clip_id"]) in kept]
    run_payload = model.run_payload
    if kept is not None:
        run_payload = {**run_payload, "clips": clips, "export": {"clips_total": len(model.run_payload["clips"])}}
    write_json(data / "run.json", run_payload)
    feature_ids = [int(feature["id"]) for feature in model.run_payload["features"]]
    feature_dir = data / "features"
    feature_dir.mkdir()
    linked: dict[str, dict[int, set[int]]] = {}
    for feature_id in feature_ids:
        detail = model.feature(feature_id)
        if detail is None:
            continue
        if kept is not None:
            detail = {**detail, "top_clips": [top for top in detail.get("top_clips", []) if str(top["clip_id"]) in kept]}
        for top in detail.get("top_clips", []):
            if top.get("frame") is not None:
                linked.setdefault(str(top["clip_id"]), {}).setdefault(int(top["frame"]), set()).add(feature_id)
        write_json(feature_dir / f"{feature_id}.json", detail)
        viz = model.viz_file(feature_id)
        if viz is not None:
            shutil.copy2(viz, feature_dir / f"{feature_id}.png")
    for clip in clips:
        clip_id = str(clip["clip_id"])
        payload = model.clips[clip_id]
        clip_dir = data / "clips" / clip_id
        clip_dir.mkdir(parents=True)
        write_json(data / "clips" / f"{clip_id}.json", payload)
        frames = clip_dir / "frames"
        frames.mkdir()
        n_frames = int(payload["n_frames"])
        heatmaps = model.frame_heatmaps(clip_id, feature_ids)
        video = payload.get("media") == "video" and payload.get("pixels") == "ok"
        pack: dict[str, list] = {}
        for frame in range(n_frames):
            row = payload["frames"][frame] if frame < len(payload["frames"]) else {}
            if not (slim and video) or frame == 0 or row.get("t") is not None:
                png = model.frame_png(clip_id, frame)
                if png:
                    (frames / f"{frame}.png").write_bytes(png)
            wanted = None
            if slim:
                wanted = {int(feature["id"]) for feature in row.get("features", [])}
                wanted |= linked.get(clip_id, {}).get(frame, set())
            for feature_id, grid in (heatmaps[frame] if frame < len(heatmaps) else {}).items():
                if wanted is None or feature_id in wanted:
                    pack[f"{frame}:{feature_id}"] = _round_grid(grid) if slim else grid
        write_json(clip_dir / "heat.json", pack)
        media = model.media_file(clip_id)
        if media is not None:
            path, _kind = media
            shutil.copy2(path, clip_dir / "media.mp4")


def _round_grid(grid: list[list[float]]) -> list[list[float]]:
    return [[float(f"{value:.{HEAT_DIGITS}g}") if value else 0 for value in row] for row in grid]


def _write_json(path: Path, payload, indent: int | None = None) -> None:
    separators = None if indent is not None else (",", ":")
    path.write_text(json.dumps(payload, allow_nan=False, indent=indent, separators=separators), encoding="utf-8")


write_json = _write_json


def steer_key(window: int, feature_id: int, kind: str, value: float) -> str:
    """Lookup key shared with ``app.js`` (``steerKey``). Values print like JavaScript numbers."""
    number = float(value)
    text = str(int(number)) if number.is_integer() else repr(number)
    return f"{int(window)}:{int(feature_id)}:{kind}:{text}"


def clip_priority(model: RunModel) -> list[str]:
    """Every clip id, the ones a visitor is most likely to open first.

    Clips that are a top example of a scoreboard-real or candidate feature come first,
    then the rest in run order. Clips without pixels go last.
    """
    priority: list[str] = []
    for feature in model.run_payload["features"]:
        if feature.get("badge") != "real" and not feature.get("candidate"):
            continue
        for top in (model.feature(int(feature["id"])) or {}).get("top_clips", [])[:2]:
            if top["clip_id"] not in priority:
                priority.append(top["clip_id"])
    ordered = priority + [str(clip["clip_id"]) for clip in model.run_payload["clips"] if str(clip["clip_id"]) not in priority]
    ok = {str(clip["clip_id"]) for clip in model.run_payload["clips"] if clip.get("pixels", "ok") == "ok"}
    return [clip_id for clip_id in ordered if clip_id in ok] + [clip_id for clip_id in ordered if clip_id not in ok]


def pick_steer_targets(
    model: RunModel, n_clips: int, n_features: int, only: list[str] | None = None
) -> list[tuple[str, int, list[int]]]:
    """``(clip_id, window, feature_ids)`` to precompute, strongest first.

    Clips come in ``clip_priority`` order, so the features a visitor is most likely to
    open can be steered; ``only`` limits them to an exported subset. Within each window,
    the features that fire hardest there are taken, which is what the clip page lists first.
    """
    ordered = clip_priority(model)
    if only is not None:
        allowed = {str(clip_id) for clip_id in only}
        ordered = [clip_id for clip_id in ordered if clip_id in allowed]
    targets: list[tuple[str, int, list[int]]] = []
    for clip_id in ordered:
        payload = model.clips.get(clip_id)
        if payload is None:
            continue
        loaded = model.loaded.get(clip_id)
        if loaded is None or loaded.status != "ok":
            continue
        peak: dict[int, dict[int, float]] = {}
        for frame in payload["frames"]:
            window = frame.get("window")
            if window is None or frame.get("t") is None:
                continue
            for feature in frame["features"]:
                best = peak.setdefault(int(window), {})
                best[int(feature["id"])] = max(best.get(int(feature["id"]), 0.0), float(feature["activation"]))
        added = False
        for window in sorted(peak):
            ranked = sorted(peak[window].items(), key=lambda item: (-item[1], item[0]))
            ids = [feature_id for feature_id, activation in ranked if activation > 0][:n_features]
            if ids:
                targets.append((clip_id, window, ids))
                added = True
        if added and len({target[0] for target in targets}) >= n_clips:
            break
    return targets


def _precompute_steering(
    model, run_dir, clips_dir, dest: Path, n_clips: int, n_features: int, log, only: list[str] | None = None
) -> tuple[list[str], int]:
    if n_clips <= 0 or n_features <= 0:
        return [], 0
    from interpretableintelligence.steer import Steerer

    targets = pick_steer_targets(model, n_clips, n_features, only=only)
    if not targets:
        log("no clip has pixels and active features; the playground is left off")
        return [], 0
    log(f"loading the model and SAE to precompute the playground on {len({t[0] for t in targets})} clips")
    steerer = Steerer(run_dir, None if clips_dir is None else Path(clips_dir))
    dest.mkdir(parents=True)
    packs: dict[str, dict[str, dict]] = {}
    total = 0
    for clip_id, window, ids in targets:
        edits = [(feature_id, kind, value) for feature_id in ids for kind, values in STEER_PRESETS.items() for value in values]
        results = steerer.sweep(clip_id, window, edits)
        pack = packs.setdefault(clip_id, {})
        for (feature_id, kind, value), result in zip(edits, results):
            pack[steer_key(window, feature_id, kind, value)] = _slim(result)
        total += len(results)
        log(f"  {clip_id} window {window}: {len(ids)} features, {len(results)} edits")
    for clip_id, pack in packs.items():
        write_json(dest / f"{clip_id}.json", pack)
    return list(packs), total


def _slim(result: dict) -> dict:
    """Drop what the page never reads and round the shift grid; keeps exports small."""
    slim = {key: value for key, value in result.items() if key != "activation"}
    slim["shift"] = [
        [[None if value is None else round(value, SHIFT_DIGITS) for value in row] for row in plane] for plane in result["shift"]
    ]
    return slim


def _sae_usable(model: RunModel) -> bool:
    sae = model.run_payload.get("sae")
    return isinstance(sae, dict) and sae.get("usable") is True


def _prepare_out(out: Path) -> None:
    if out.exists():
        if not out.is_dir():
            raise ValueError(f"{out} exists and is not a directory")
        if any(out.iterdir()) and not (out / MARKER).is_file():
            raise ValueError(f"{out} is not empty and was not written by ii export; pick another --out")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def _copy_static(dest: Path) -> None:
    dest.mkdir(parents=True)
    for name in ("guide.js", "app.js"):
        shutil.copy2(STATIC / name, dest / name)
    # style.css points at /static/fonts for the local server; relative paths let the
    # export live under any subpath.
    css = (STATIC / "style.css").read_text(encoding="utf-8").replace("/static/fonts/", "fonts/")
    (dest / "style.css").write_text(css, encoding="utf-8")
    fonts = STATIC / "fonts"
    if fonts.is_dir():
        shutil.copytree(fonts, dest / "fonts")


def _shell(title: str, model: RunModel) -> str:
    """The viewer's own index.html with relative paths, the data root, and share-card tags."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace('<html lang="en">', '<html lang="en" data-ii-root="data">', 1)
    html = html.replace('href="/static/', 'href="static/').replace('src="/static/', 'src="static/')
    html = html.replace("<title>interpretableintelligence</title>", f"<title>{escape(title)}</title>", 1)
    model_info = model.run_payload.get("model") or {}
    name = model_info.get("name") or model_info.get("adapter") or "a model"
    real = sum(1 for feature in model.run_payload["features"] if feature.get("badge") == "real")
    description = (
        f"Explore the features a sparse autoencoder found inside {name}: where each one fires, "
        f"what removing it does, and {real} that passed every check. No install needed."
    )
    meta = "\n".join(
        f'  <meta {attr}="{key}" content="{escape(value)}">'
        for attr, key, value in (
            ("name", "description", description),
            ("property", "og:type", "website"),
            ("property", "og:title", title),
            ("property", "og:description", description),
            ("name", "twitter:card", "summary"),
            ("name", "twitter:title", title),
            ("name", "twitter:description", description),
        )
    )
    return html.replace("</title>\n", f"</title>\n{meta}\n", 1)
