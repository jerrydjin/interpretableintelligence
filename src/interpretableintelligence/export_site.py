"""Write the static demo site.

The fake section is the clip viewer, with every request turned into a file.
The other sections are pages rendered from the vendored measurements.
"""

from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

from interpretableintelligence.adapters.fake import generate_clips
from interpretableintelligence.pipeline import execute_pipeline
from interpretableintelligence.view import load_run

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
CAUSAL = ("stripe", "pulse", "corner")
DECOYS = ("flicker", "band", "speckle")


def export_site(
    out_dir: Path,
    vjepa_dir: Path | None = None,
    picker_dir: Path | None = None,
    work_dir: Path | None = None,
) -> Path:
    """Regenerate ``out_dir`` from a fresh fake run plus the vendored measurements."""
    out = Path(out_dir)
    vjepa = Path(vjepa_dir) if vjepa_dir else REPO / "demo-data" / "vjepa-slim"
    picker = Path(picker_dir) if picker_dir else REPO / "demo-data" / "picker"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _copy_static(out / "static")
    (out / "favicon.ico").write_bytes(_favicon_png())
    work = Path(work_dir) if work_dir else out / ".work"
    work.mkdir(parents=True, exist_ok=True)
    clips = work / "clips"
    run = work / "run"
    generate_clips(clips, n=30, seed=0)
    execute_pipeline(
        adapter_name="fake",
        clips_dir=clips,
        out_dir=run,
        layer=16,
        seed=0,
        heldout_frac=0.2,
        score=True,
        serve=False,
        open_browser=False,
        log=lambda _message: None,
    )
    model = load_run(run, clips)
    _export_viewer(model, out / "fake", _fake_sentence(run / "scoreboard.json"))
    _write_home(out / "index.html")
    _write_picker(out / "picker", picker)
    _write_vjepa(out / "vjepa", vjepa)
    _write_groot(out / "groot" / "index.html")
    shutil.rmtree(work, ignore_errors=True)
    return out


def _copy_static(dest: Path) -> None:
    dest.mkdir(parents=True)
    for name in ("app.js", "style.css", "demo.css"):
        source = STATIC / name
        if source.is_file():
            shutil.copy2(source, dest / name)
    fonts = STATIC / "fonts"
    if fonts.is_dir():
        shutil.copytree(fonts, dest / "fonts")


def _export_viewer(model, dest: Path, sentence: str) -> None:
    data = dest / "data"
    data.mkdir(parents=True)
    _write_json(data / "run.json", model.run_payload)
    feature_ids = [int(feature["id"]) for feature in model.run_payload["features"]]
    feature_dir = data / "features"
    feature_dir.mkdir()
    for feature_id in feature_ids:
        detail = model.feature(feature_id)
        if detail is None:
            continue
        _write_json(feature_dir / f"{feature_id}.json", detail)
        viz = model.viz_file(feature_id)
        if viz is not None:
            shutil.copy2(viz, feature_dir / f"{feature_id}.png")
    for clip in model.run_payload["clips"]:
        clip_id = str(clip["clip_id"])
        payload = model.clips[clip_id]
        clip_dir = data / "clips" / clip_id
        clip_dir.mkdir(parents=True)
        _write_json(data / "clips" / f"{clip_id}.json", payload)
        frames = clip_dir / "frames"
        frames.mkdir()
        n_frames = int(payload["n_frames"])
        pack: dict[str, list] = {}
        for frame in range(n_frames):
            png = model.frame_png(clip_id, frame)
            if png:
                (frames / f"{frame}.png").write_bytes(png)
            for feature_id in feature_ids:
                grid = model.heatmap(clip_id, frame, feature_id)
                if grid and any(value > 0 for row in grid for value in row):
                    pack[f"{frame}:{feature_id}"] = grid
        _write_json(clip_dir / "heat.json", pack)
        media = model.media_file(clip_id)
        if media is not None:
            path, _kind = media
            shutil.copy2(path, clip_dir / "media.mp4")
    (dest / "index.html").write_text(_viewer_page(sentence), encoding="utf-8")


def _fake_sentence(scoreboard_path: Path) -> str:
    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    rows = scoreboard["features"]
    real = [row["label"] for row in rows if row.get("real") is True]
    if set(real) == set(CAUSAL):
        real = [name for name in CAUSAL if name in real]
    decoys = [name for name in DECOYS if any(row.get("label") == name and row.get("real") is not True for row in rows)]
    real_text = ", ".join(real) if real else "none"
    decoy_text = ", ".join(decoys) if decoys else "none"
    return f"Scoreboard: {real_text} are real. {decoy_text} are not."


def _viewer_page(sentence: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en" data-ii-root="/fake/data">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fake demo · interpretableintelligence</title>
  <link rel="stylesheet" href="/static/style.css">
  <link rel="stylesheet" href="/static/demo.css">
  <link rel="icon" href="/favicon.ico">
</head>
<body>
  <header class="top" data-testid="run-header">
    <div class="brand-row">
      <a class="wordmark" href="/">interpretableintelligence</a>
      <span id="run-id" class="run-id" data-testid="run-id"></span>
      <nav class="nav">{_nav("fake")}</nav>
    </div>
    <div class="meta">
      <span id="run-model" data-testid="run-model"></span>
      <span id="run-layer" data-testid="run-layer"></span>
      <span id="split-sizes" data-testid="split-sizes"></span>
      <span id="sae-summary" data-testid="sae-summary"></span>
    </div>
    <p class="score-line" data-testid="score-line">{escape(sentence)}</p>
  </header>
  <main id="app"></main>
  <script src="/static/app.js"></script>
</body>
</html>
"""


def _nav(current: str) -> str:
    items = (("fake", "/fake/", "Fake"), ("picker", "/picker/", "Picker"), ("vjepa", "/vjepa/", "V-JEPA 2"), ("groot", "/groot/", "GR00T"))
    parts = []
    for key, href, label in items:
        attrs = f' href="{href}"'
        if key == current:
            attrs += ' aria-current="page"'
        parts.append(f"<a{attrs}>{label}</a>")
    return "".join(parts)


def _page(title: str, current: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} · interpretableintelligence</title>
  <link rel="stylesheet" href="/static/style.css">
  <link rel="stylesheet" href="/static/demo.css">
  <link rel="icon" href="/favicon.ico">
</head>
<body>
  <header class="top">
    <div class="brand-row">
      <a class="wordmark" href="/">interpretableintelligence</a>
      <nav class="nav">{_nav(current)}</nav>
    </div>
  </header>
  <main class="prose">{body}</main>
</body>
</html>
"""


def _write_home(path: Path) -> None:
    body = """
    <h1>Demo</h1>
    <p>Four sections. The fake run is a working viewer. The other three are the measurements on disk, or a section that has no validated run.</p>
    <ul class="section-list">
      <li><a href="/fake/">Fake demo</a> — 30 synthetic clips, layer 16, seed 0, held-out fraction 0.2. The scoreboard is the source of <span class="badge badge-real">real</span>.</li>
      <li><a href="/picker/">Candidate picker</a> — synthetic rollouts. Counts from the rerun of the picker scripts.</li>
      <li><a href="/vjepa/">V-JEPA 2 on DROID</a> — held-out <code>droid_ep068.mp4</code>, window 0, then feature 270.</li>
      <li><a href="/groot/">GR00T N1.7 on LIBERO</a> — <span class="badge badge-unverified">Not yet validated</span></li>
    </ul>
    """
    path.write_text(_page("Demo", "", body), encoding="utf-8")


def _write_picker(dest: Path, picker: Path) -> None:
    dest.mkdir(parents=True)
    results = json.loads((picker / "synth_results.json").read_text(encoding="utf-8"))
    extra = json.loads((picker / "synth_results_extra.json").read_text(encoding="utf-8"))
    shutil.copy2(picker / "synth_results.json", dest / "synth_results.json")
    shutil.copy2(picker / "synth_results_extra.json", dest / "synth_results_extra.json")
    null = results["null"]
    rows = []
    labels = (
        ("time+detrend", "time + detrend"),
        ("time,no-detrend", "time, no detrend"),
        ("phase+detrend", "phase + detrend"),
    )
    for key, label in labels:
        block = null[key]
        rows.append(
            f"<tr><td>{escape(label)}</td><td>{int(block['runs_with_final_candidate'])}/{int(block['n'])}</td></tr>"
        )
    strong = extra["power_50ep_plant0.15"]["recovered_as_candidate"]
    weak = extra["power_50ep_plant0.05"]["recovered_as_candidate"]
    final_bits = []
    for key, label in labels:
        block = null[key]
        final_bits.append(f"{int(block['runs_with_final_candidate'])}/{int(block['n'])} ({label})")
    body = f"""
    <h1>Candidate picker on synthetic rollouts</h1>
    <p>Null runs with a final candidate, from <code>picker/synth_test.py --null-seeds 30</code>.</p>
    <table class="score-table" data-testid="null-counts">
      <thead><tr><th>Match</th><th>Final candidate</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p>Planted feature recovered as a candidate, from <code>picker/synth_test_extra.py</code>, 50 episodes per run, 5 seeds. <code>plant</code> is the fraction of tokens that fire in the window before failure.</p>
    <table class="score-table" data-testid="plant-counts">
      <thead><tr><th>Plant fraction</th><th>Recovered</th></tr></thead>
      <tbody>
        <tr><td>0.15</td><td>{escape(str(strong))}</td></tr>
        <tr><td>0.05</td><td>{escape(str(weak))}</td></tr>
      </tbody>
    </table>
    <p>Rerun of those two scripts. Final candidates: {escape('; '.join(final_bits))}. Recovered: {escape(str(strong))} at plant 0.15 and {escape(str(weak))} at plant 0.05.</p>
    """
    (dest / "index.html").write_text(_page("Candidate picker", "picker", body), encoding="utf-8")


def _write_vjepa(dest: Path, vjepa: Path) -> None:
    dest.mkdir(parents=True)
    scoreboard = json.loads((vjepa / "scoreboard.json").read_text(encoding="utf-8"))
    split = json.loads((vjepa / "split.json").read_text(encoding="utf-8"))
    heldout = json.loads((vjepa / "heldout_loss.json").read_text(encoding="utf-8"))
    random_meta = json.loads((vjepa / "random_init" / "meta.json").read_text(encoding="utf-8"))
    random_cands = json.loads((vjepa / "random_init" / "candidates.json").read_text(encoding="utf-8"))
    features = scoreboard["features"]
    feature = next(row for row in features if int(row["id"]) == 270)
    status = feature["status"]
    window = next(
        row
        for row in heldout["windows"]
        if row.get("clip_name") == "droid_ep068.mp4" and int(row.get("window")) == 0
    )
    media = dest / "media"
    viz = dest / "viz"
    media.mkdir()
    viz.mkdir()
    for name in ("droid_ep067.mp4", "droid_ep068.mp4"):
        shutil.copy2(vjepa / name, media / name)
    picture = vjepa / "features" / "viz" / "feature_00270.png"
    shutil.copy2(picture, viz / "feature_00270.png")
    for path in sorted((vjepa / "features" / "viz").glob("feature_*.png")):
        if path.name != "feature_00270.png":
            shutil.copy2(path, viz / path.name)
    width, height = _video_size(media / "droid_ep068.mp4")
    run = scoreboard["run"]
    sae = scoreboard["sae"]
    split_map = split.get("clips") if isinstance(split.get("clips"), dict) else split
    ep068_split = split_map["droid_ep068.mp4"]
    ep067_split = split_map["droid_ep067.mp4"]
    above = [row for row in features if _num(row.get("random_percentile")) is not None and float(row["random_percentile"]) >= 95]
    above_text = ", ".join(f"{int(row['id'])} at {row['random_percentile']}" for row in above) or "none"
    real_ids = [int(row["id"]) for row in features if row.get("real") is True]
    n_random_cands = len(random_cands["candidates"])
    firing = int(random_meta["features_firing_random"])
    rows = []
    for row in features:
        rows.append(
            "<tr>"
            f"<td>{int(row['id'])}</td>"
            f"<td>{escape(str(row.get('status')))}</td>"
            f"<td>{escape(_fmt_num(row.get('random_percentile')))}</td>"
            f"<td>{escape(str(row.get('real')))}</td>"
            "</tr>"
        )
    body = f"""
    <h1>V-JEPA 2 on DROID</h1>
    <p class="meta-line">{escape(str(run['model']))} · layer {int(run['layer'])} · run {escape(str(run['run_id']))} · SAE usable {escape(str(sae.get('usable')))} · loss_recovered {escape(_fmt_num(sae.get('loss_recovered')))}</p>
    <p>Opened on held-out <code>droid_ep068.mp4</code> window 0. <code>split.json</code> marks that file <code>{escape(str(ep068_split))}</code> and <code>droid_ep067.mp4</code> <code>{escape(str(ep067_split))}</code>.</p>
    <div class="clip-row">
      <figure>
        <div class="stage" style="aspect-ratio: {width} / {height}">
          <video src="media/droid_ep068.mp4" controls playsinline data-testid="ep068"></video>
        </div>
        <figcaption>droid_ep068.mp4 · window {int(window['window'])} · source frames {int(window['frame_start'])}–{int(window['frame_end'])} · {escape(str(window.get('task_text') or ''))}</figcaption>
      </figure>
      <article class="feature-card" data-testid="feature-270" data-status="{escape(str(status))}">
        <p class="kicker">Feature 270</p>
        <div class="feature-title"><h2>{escape(str(feature.get('label') or 'feature 270'))}</h2> <span class="badge badge-status" data-testid="status-270">{escape(str(status))}</span></div>
        <img class="viz" alt="Feature 270 activations" src="viz/feature_00270.png">
        <dl class="score">
          <div><dt>random_percentile</dt><dd>{escape(_fmt_num(feature.get('random_percentile')))}</dd></div>
          <div><dt>n_random</dt><dd>{escape(str(feature.get('n_random')))}</dd></div>
          <div><dt>ablation_effect</dt><dd>{escape(_fmt_num(feature.get('ablation_effect')))}</dd></div>
          <div><dt>real</dt><dd>{escape(str(feature.get('real')))}</dd></div>
        </dl>
      </article>
    </div>
    <p>The other clip in this bundle, <code>droid_ep067.mp4</code>, is the train split above.</p>
    <div class="stage" style="aspect-ratio: {width} / {height}">
      <video src="media/droid_ep067.mp4" controls playsinline data-testid="ep067"></video>
    </div>
    <h2>Ten candidates</h2>
    <p>Scoreboard threshold for the random-control gate is <code>{escape(str(scoreboard['thresholds']['random_percentile']))}</code>. Candidates at or above 95: {escape(above_text)}. <code>real</code> is true for {escape(str(real_ids) if real_ids else 'no feature')}.</p>
    <table class="score-table" data-testid="candidates">
      <thead><tr><th>Feature</th><th>Status</th><th>random_percentile</th><th>real</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p>The random-init SAE still fires {firing} features, and <code>random_init/candidates.json</code> still lists {n_random_cands} candidates. Nothing here is claimed real.</p>
    """
    (dest / "index.html").write_text(_page("V-JEPA 2", "vjepa", body), encoding="utf-8")


def _write_groot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = """
    <h1>GR00T N1.7 on LIBERO</h1>
    <p><span class="badge badge-unverified" data-testid="groot-status">Not yet validated</span></p>
    """
    path.write_text(_page("GR00T N1.7", "groot", body), encoding="utf-8")


def _favicon_png() -> bytes:
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


def _video_size(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return int(stream.width), int(stream.height)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt_num(value) -> str:
    number = _num(value)
    if number is None:
        return str(value)
    return format(number, ".12g")
