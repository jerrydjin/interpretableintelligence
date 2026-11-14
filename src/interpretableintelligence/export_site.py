"""Write the static demo site.

The fake section is the clip viewer, with every request turned into a file.
The other sections are pages rendered from the vendored measurements.
"""

from __future__ import annotations

import json
import re
import shutil
from html import escape
from pathlib import Path

from interpretableintelligence.adapters.fake import generate_clips
from interpretableintelligence.export_run import write_viewer_data
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
    groot_dir: Path | None = None,
    work_dir: Path | None = None,
    randinit_dir: Path | None = None,
    atp_dir: Path | None = None,
) -> Path:
    """Regenerate ``out_dir`` from a fresh fake run plus the vendored measurements."""
    out = Path(out_dir)
    vjepa = Path(vjepa_dir) if vjepa_dir else REPO / "demo-data" / "vjepa-slim"
    picker = Path(picker_dir) if picker_dir else REPO / "demo-data" / "picker"
    groot = Path(groot_dir) if groot_dir else REPO / "demo-data" / "groot"
    randinit = Path(randinit_dir) if randinit_dir else REPO / "demo-data" / "vjepa-randinit"
    atp = Path(atp_dir) if atp_dir else REPO / "demo-data" / "vjepa-atp"
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
    groot_summary = _load_groot_summary(groot)
    groot_status = _groot_status(groot_summary)
    checks = _groot_checks(groot)
    random_control = _random_control(randinit)
    atp_result = _atp(atp)
    _write_product(out / "index.html", groot_summary, groot_status)
    _write_results(out / "results" / "index.html", groot_summary, groot_status, checks, random_control, atp_result)
    _write_picker(out / "picker", picker)
    _write_vjepa(out / "vjepa", vjepa, random_control, atp_result)
    _write_groot(out / "groot", groot / "groot-1028-ablate-summary.json", groot_summary, groot_status, checks)
    shutil.rmtree(work, ignore_errors=True)
    return out


def _copy_static(dest: Path) -> None:
    dest.mkdir(parents=True)
    for name in ("guide.js", "app.js", "style.css", "demo.css"):
        source = STATIC / name
        if source.is_file():
            shutil.copy2(source, dest / name)
    fonts = STATIC / "fonts"
    if fonts.is_dir():
        shutil.copytree(fonts, dest / "fonts")


def _export_viewer(model, dest: Path, sentence: str) -> None:
    write_viewer_data(model, dest / "data", indent=1)
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
    """The viewer's own index.html, rooted at /fake/data, with the demo bar above it."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace('<html lang="en">', '<html lang="en" data-ii-root="/fake/data">', 1)
    html = html.replace(
        "<title>interpretableintelligence</title>",
        "<title>Pipeline demo (synthetic) · interpretableintelligence</title>",
        1,
    )
    html = html.replace(
        '<link rel="stylesheet" href="/static/style.css">',
        '<link rel="stylesheet" href="/static/style.css">\n'
        '  <link rel="stylesheet" href="/static/demo.css">\n'
        '  <link rel="icon" href="/favicon.ico">',
        1,
    )
    html = html.replace("<body>\n", f"<body>\n{_demo_bar('fake')}\n", 1)
    score = (
        '  <p class="score-line" data-testid="synthetic-banner"><span class="badge badge-unverified">Synthetic</span> '
        "Pipeline demo on 30 generated clips with planted features. It shows the pipeline end to end; "
        "it is not a result on a real model.</p>\n"
        f'  <p class="score-line" data-testid="score-line">{escape(sentence)}</p>\n'
    )
    return html.replace("  </header>\n", "  </header>\n" + score, 1)


def _demo_bar(current: str) -> str:
    return (
        '  <nav class="demo-bar" aria-label="Demo sections">'
        f'<a class="demo-home" href="/">Home</a>{_nav(current)}</nav>'
    )


def _wordmark() -> str:
    """The viewer's logo and wordmark, pointed at the demo home."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<a class="wordmark".*?</a>', html, re.S)
    if not match:
        return '<a class="wordmark" href="/">interpretable intelligence</a>'
    return match.group(0).replace('href="#/"', 'href="/"', 1)


def _nav(current: str) -> str:
    items = (
        ("results", "/results/", "Results"),
        ("groot", "/groot/", "GR00T"),
        ("vjepa", "/vjepa/", "V-JEPA 2"),
        ("picker", "/picker/", "Picker"),
        ("fake", "/fake/", "Pipeline demo"),
    )
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
  <meta name="theme-color" content="#faf9f5">
  <title>{escape(title)} · interpretableintelligence</title>
  <link rel="stylesheet" href="/static/style.css">
  <link rel="stylesheet" href="/static/demo.css">
  <link rel="icon" href="/favicon.ico">
</head>
<body>
  <header class="top demo-top">
    <div class="brand-row">
      {_wordmark()}
      <nav class="nav">{_nav(current)}</nav>
    </div>
  </header>
  <main class="prose">{body}</main>
</body>
</html>
"""


def _write_product(path: Path, summary: dict, status: str) -> None:
    commands = dict(_ii_commands())
    known = set(commands)
    headline = escape(_control_headline(summary))
    useful = ("try", "run", "view", "rollouts", "validate")
    command_items = "".join(
        f"<li>{_cmd(name, known)} — {escape(commands[name])}</li>" for name in useful
    )
    episodes = _episode_total(summary)
    body = f"""
    <h1>Find out why your robot policy failed, and whether you're fooling yourself.</h1>
    <p class="lede">For teams training robot policies and world models who need to know why a rollout went wrong, not just that it did.</p>
    <p><a class="button" href="#quickstart" data-testid="cta">Run `ii` on your policy.</a></p>
    <h2>What you get today</h2>
    <p><code>ii</code> is free and local. {_cmd("run", known)} captures a layer, trains an SAE and scores features on the adapters that ship (fake, vjepa2, hf, groot); {_cmd("view", known)} opens the run; {_cmd("validate", known)} checks it. {_cmd("rollouts", known)} compares baseline and intervened task success with matched random-feature controls, on the synthetic reacher today; GR00T joins once it lands.</p>
    <h2>Proof</h2>
    <p data-testid="control-headline">{headline}</p>
    <p data-testid="proof">Run with the repo's GR00T ablation runner (<a href="{RUNNER_URL}"><code>ablate_v9.py</code></a>), {episodes} rollouts. Label: <span class="badge badge-unverified">{escape(status)}</span>. <a href="/results/">Results</a>.</p>
    <p data-testid="vjepa-proof"><a href="/results/">On V-JEPA 2, loss-based scores couldn't tell a trained SAE from a random one — so that control is now the bar.</a></p>
    <h2>What's coming</h2>
    <ul class="coming" data-testid="coming">
      <li><span class="badge badge-unverified">Coming</span> {_coming_cmd("import", known)} (bring the rollout logs you already have).</li>
      <li><span class="badge badge-unverified">Coming</span> The random-weights SAE control (a feature has to beat an untrained SAE before it counts).</li>
      <li><span class="badge badge-unverified">Coming</span> Closed-loop rollouts at scale.</li>
    </ul>
    <h2 id="quickstart">Quickstart</h2>
    <pre class="quickstart"><code>uv sync
uv run ii try</code></pre>
    <p>{_cmd("try", known)} writes practice clips and runs the pipeline on the fake model.</p>
    <ul class="commands" data-testid="commands">{command_items}</ul>
    <p><a href="{README_URL}">The rest of the commands are in the README.</a></p>
    <h2>Synthetic demos</h2>
    <ul class="section-list" data-testid="synthetic-sections">
      <li><a href="/picker/"><span class="section-name">Candidate picker</span><span class="section-text"><span class="badge badge-unverified">Synthetic</span> The picker run on generated rollouts with a known answer. Not a result on a real model.</span></a></li>
      <li><a href="/fake/"><span class="section-name">Pipeline demo</span><span class="section-text"><span class="badge badge-unverified">Synthetic</span> Generated clips with planted features. Shows the viewer end to end. Not a result on a real model.</span></a></li>
    </ul>
    """
    path.write_text(_page("ii", "", body), encoding="utf-8")


def _write_results(path: Path, summary: dict, groot_status: str, checks: dict, control: dict, atp: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headline = escape(_control_headline(summary))
    body = f"""
    <h1>Results</h1>
    <p class="lede"><b>What we measured on real robot models, and the controls that decide what counts.</b> Every number on these pages is read from a result file in the repo. A feature is called real only when it passes the scoreboard rule. The synthetic demos are at the bottom and labelled.</p>
    <ul class="section-list" data-testid="real-sections">
      <li><a href="/groot/"><span class="section-name">GR00T N1.7 on LIBERO</span><span class="section-text">{headline} <span class="badge badge-unverified">{escape(groot_status)}</span></span></a></li>
      <li><a href="/groot/#checks"><span class="section-name">Method checks</span><span class="section-text">{escape(_checks_summary(checks))}</span></a></li>
      <li><a href="/vjepa/#random-control"><span class="section-name">V-JEPA 2 on DROID</span><span class="section-text">{escape(_control_summary(control))}</span></a></li>
      <li><a href="/vjepa/#atp"><span class="section-name">Attribution vs ablation</span><span class="section-text">{escape(_atp_summary(atp))}</span></a></li>
    </ul>
    <h2>Synthetic demos</h2>
    <ul class="section-list" data-testid="synthetic-sections">
      <li><a href="/picker/"><span class="section-name">Candidate picker</span><span class="section-text"><span class="badge badge-unverified">Synthetic</span> The picker run on generated rollouts with a known answer.</span></a></li>
      <li><a href="/fake/"><span class="section-name">Pipeline demo</span><span class="section-text"><span class="badge badge-unverified">Synthetic</span> 30 generated clips with planted features, layer 16, seed 0, held-out fraction 0.2. Shows the viewer end to end.</span></a></li>
    </ul>
    """
    path.write_text(_page("Results", "results", body), encoding="utf-8")


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
    <p><span class="badge badge-unverified">Synthetic</span> A test of the picker itself on generated rollouts where the answer is known. It is not a result on a real model.</p>
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


def _write_vjepa(dest: Path, vjepa: Path, control: dict, atp: dict) -> None:
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
{_control_html(control)}
{_atp_html(atp)}
    <h2>The run on display</h2>
    <p>Opened on held-out <code>droid_ep068.mp4</code> window 0. <code>split.json</code> marks that file <code>{escape(str(ep068_split))}</code> and <code>droid_ep067.mp4</code> <code>{escape(str(ep067_split))}</code>.</p>
    <div class="clip-row">
      <figure>
        <div class="stage" style="aspect-ratio: {width} / {height}">
          <video src="media/droid_ep068.mp4" controls playsinline data-testid="ep068"></video>
        </div>
        <figcaption>droid_ep068.mp4 · window {int(window['window'])} · source frames {int(window['frame_start'])}–{int(window['frame_end'])} · {escape(str(window.get('task_text') or ''))}</figcaption>
      </figure>
      <article class="demo-card" data-testid="feature-270" data-status="{escape(str(status))}">
        <p class="kicker">Feature 270</p>
        <div class="demo-title"><h2>{escape(str(feature.get('label') or 'feature 270'))}</h2> <span class="badge badge-status" data-testid="status-270">{escape(str(status))}</span></div>
        <img class="viz" alt="Feature 270 activations" src="viz/feature_00270.png">
        <dl class="demo-score">
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


def _write_groot(dest: Path, summary_path: Path, summary: dict, status: str, checks: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(summary_path, dest / summary_path.name)
    baseline = _one_row(summary, lambda cond: cond == "baseline")
    candidates = [row for row in summary["table"] if "-cand-" in str(row["cond"])]
    candidate_html = "\n".join(
        f'    <p data-testid="groot-candidate">{escape(_candidate_text(row))}</p>' for row in candidates
    )
    body = f"""
    <h1>GR00T N1.7 on LIBERO</h1>
    <p data-testid="control-headline">{escape(_control_headline(summary))}</p>
    <p data-testid="ablation-runner">{_runner_html(summary)}</p>
    <p><span class="badge badge-unverified" data-testid="groot-status">{escape(status)}</span></p>
    <p data-testid="groot-result">{escape(_groot_result_sentence(summary))}</p>
    <p data-testid="groot-baseline">{escape(_baseline_line(baseline))}</p>
{candidate_html}
    <p data-testid="groot-controls">{escape(_controls_line(summary))}</p>
{_checks_html(checks)}
{_conditions_html(summary)}
    """
    (dest / "index.html").write_text(_page("GR00T N1.7", "groot", body), encoding="utf-8")


def _load_groot_summary(groot: Path) -> dict:
    path = groot / "groot-1028-ablate-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(summary.get("table"), list):
        raise ValueError(f"{path} has no ablation table")
    return summary


def _groot_status(summary: dict) -> str:
    _require_no_effect(summary)
    return "Tested: no causal effect found"


def _groot_result_sentence(summary: dict) -> str:
    _require_no_effect(summary)
    real = "true" if summary["real"] is True else "false"
    return (
        "The test was whether switching a feature off raises success. "
        f"Neither candidate did, so both are real={real}."
    )


def _require_no_effect(summary: dict) -> None:
    candidates = [row for row in summary["table"] if "-cand-" in str(row["cond"])]
    if len(candidates) != 2:
        raise ValueError(f"expected two candidate rows, found {len(candidates)}")
    if summary.get("all_finished") is not True:
        raise ValueError("ablation summary is not finished")
    if summary.get("real") is not False:
        raise ValueError("ablation summary real flag is not false")
    if any(row.get("verdict") == "rise" for row in candidates):
        raise ValueError("a candidate rose")


def _one_row(summary: dict, match) -> dict:
    rows = [row for row in summary["table"] if match(str(row["cond"]))]
    if len(rows) != 1:
        raise ValueError(f"expected one matching row, found {len(rows)}")
    return rows[0]


def _baseline_line(row: dict) -> str:
    successes, n = int(row["successes"]), int(row["n"])
    low, high = row["wilson95"]
    return (
        f"Baseline: {successes}/{n} ({_percent(successes, n)}%), "
        f"Wilson 95% interval [{_bound(low)}, {_bound(high)}]."
    )


def _candidate_text(row: dict) -> str:
    ident = str(row["cond"]).split("-", 2)[-1]
    successes, n = int(row["successes"]), int(row["n"])
    verdict = row["verdict"]
    if "+" in ident:
        return f"The {ident} pair: {successes}/{n} ({_percent(successes, n)}%), verdict {verdict}."
    return f"Feature {ident}: {successes}/{n}, verdict {verdict}."


def _episode_total(summary: dict) -> int:
    rollouts = summary.get("episodes_logged")
    if isinstance(rollouts, bool) or not isinstance(rollouts, int):
        raise ValueError("ablation summary has no episode total")
    return rollouts


def _single_feature_candidate(summary: dict) -> dict:
    rows = [
        row
        for row in summary["table"]
        if "-cand-" in str(row["cond"]) and "+" not in str(row["cond"])
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one single-feature candidate, found {len(rows)}")
    return rows[0]


def _control_headline(summary: dict) -> str:
    """The GR00T worked example. Every count is read from the ablation summary."""
    baseline = _one_row(summary, lambda cond: cond == "baseline")
    single = _single_feature_candidate(summary)
    feature = str(single["cond"]).split("-")[-1]
    controls = _control_rows(summary)
    fell = sum(1 for row in controls if row.get("verdict") == "fall")
    return (
        f"Feature {feature} tracked failures, and turning it off dropped success "
        f"from {int(baseline['successes'])}/{int(baseline['n'])} "
        f"to {int(single['successes'])}/{int(single['n'])}. "
        f"Turning off random features caused falls too in {fell} of {len(controls)} runs. "
        "`ii` makes you run that control before calling a feature real."
    )


RUNNER_PATH = "src/interpretableintelligence/adapters/groot/v9/ablate_v9.py"
RUNNER_URL = "https://github.com/jerrydjin/interpretableintelligence/blob/main/" + RUNNER_PATH
README_URL = "https://github.com/jerrydjin/interpretableintelligence/blob/main/README.md"


def _runner_html(summary: dict) -> str:
    episodes = _episode_total(summary)
    return (
        f"{episodes} rollouts. The matched random-feature controls that caught this result "
        "live in the repo's ablation runner, "
        f'<a href="{RUNNER_URL}"><code>{escape(RUNNER_PATH)}</code></a>, '
        "not through an <code>ii</code> command. "
        "That is the check <code>ii</code> is built around, until GR00T ablation is wired into <code>ii</code>."
    )


def _ii_commands() -> list[tuple[str, str]]:
    """Subcommands `ii --help` actually registers, with a number-free first line."""
    from typer.main import get_command

    from interpretableintelligence.cli import app

    lines = []
    for name, command in get_command(app).commands.items():
        blurb = (command.help or "").strip().splitlines()[0].strip()
        if name == "validate":
            # The docstring's first line cites exit code 0. Keep result numbers out of this list.
            blurb = "Validate a run folder against the schema."
        if name == "rollouts":
            # The command only ships the fake reacher. Don't let the list read as a GR00T run.
            blurb = "Compare baseline and intervened task success on the synthetic reacher. Resume skips episodes already written."
        if any(character.isdigit() for character in blurb):
            raise ValueError(f"ii {name} blurb contains a digit: {blurb}")
        lines.append((name, blurb))
    if not lines:
        raise ValueError("ii has no commands")
    return lines


def _cmd(name: str, known: set[str]) -> str:
    if name not in known:
        raise ValueError(f"ii {name} is not a command on main")
    return f'<code data-cmd="{escape(name)}">ii {escape(name)}</code>'


def _coming_cmd(name: str, known: set[str]) -> str:
    if name in known:
        raise ValueError(f"ii {name} is already a command; do not mark it coming")
    return f'<code data-cmd="{escape(name)}">ii {escape(name)}</code>'


def _controls_line(summary: dict) -> str:
    controls = _control_rows(summary)
    fell = sum(1 for row in controls if row.get("verdict") == "fall")
    return (
        f"{fell} of the {len(controls)} random controls also fell, "
        "so a fall is not specific to these features."
    )


def _control_rows(summary: dict) -> list:
    controls = [row for row in summary["table"] if "-ctrl" in str(row["cond"])]
    if not controls:
        raise ValueError("ablation summary has no random controls")
    return controls


def _percent(successes: int, n: int) -> int:
    if n <= 0:
        raise ValueError("ablation row has no episodes")
    return round(100 * successes / n)


def _bound(value: float) -> str:
    return f"{float(value):.4f}"


V9_NOTES = REPO / "src" / "interpretableintelligence" / "adapters" / "groot" / "v9" / "V9_NOTES.md"


def _groot_checks(groot: Path) -> dict:
    """Method checks behind the GR00T null, each read from a vendored file."""
    recon = json.loads((groot / "recon-only-summary.json").read_text(encoding="utf-8"))
    check = recon["recon_check"]
    if recon.get("kind") != "recon_only" or recon.get("latents") != []:
        raise ValueError("recon-only summary edits latents")
    sae = json.loads((groot / "sae-heldout-eval.json").read_text(encoding="utf-8"))
    heldout = sae["heldout"]
    notes = V9_NOTES.read_text(encoding="utf-8")
    match = re.search(r"labelled pot 1's normal placement on the stove as a `drop` in (\d+) of (\d+) cases", notes)
    if not match:
        raise ValueError(f"{V9_NOTES} no longer states the pot-1 mislabel count")
    return {
        "recon_identical": int(check["n_identical_actions"]),
        "recon_compared": int(check["n_compared_with_baseline"]),
        "recon_same_success": int(check["n_same_success"]),
        "recon_max_change": float(check["hook_max_abs_change"]),
        "ev_heldout": float(heldout["ev_raw"]),
        "ev_episodes": int(heldout["n_episodes"]),
        "ev_tokens": int(heldout["n_tokens"]),
        "episodes": int(sae["rollout_io_load"]["episodes"]),
        "successes": int(sae["rollout_io_load"]["successes"]),
        "loss_recovered": sae.get("loss_recovered"),
        "mislabel": int(match.group(1)),
        "mislabel_of": int(match.group(2)),
    }


def _checks_summary(checks: dict) -> str:
    return (
        f"Reconstruction-only rollouts identical to baseline in {checks['recon_identical']} of {checks['recon_compared']}. "
        f"SAE held-out explained variance {checks['ev_heldout']:.3f}. Failure onset re-derived from pot 0."
    )


def _checks_html(checks: dict) -> str:
    loss = "not measured" if checks["loss_recovered"] is None else _fmt_num(checks["loss_recovered"])
    return f"""    <h2 id="checks">Method checks</h2>
    <p data-testid="check-recon">Reconstruction only: the hook replaced block 5 with the SAE reconstruction and edited no latent. Actions were identical to baseline in {checks['recon_identical']} of {checks['recon_compared']} shared-seed episodes, success matched in {checks['recon_same_success']} of {checks['recon_compared']}, and the largest change at the hook was {checks['recon_max_change']:g}. So the hook itself does not move the policy.</p>
    <p data-testid="check-ev">SAE: BatchTopK, k=32, 8192 latents, trained on block 5 of the {checks['episodes']}-episode capture ({checks['successes']} successes). Held-out explained variance {checks['ev_heldout']:.3f} on {checks['ev_episodes']} held-out episodes ({checks['ev_tokens']} tokens). loss_recovered: {escape(loss)}.</p>
    <p data-testid="check-pot0">Failure labels: the first labeller judged a drop on whichever object was nearest the gripper, and marked pot 1's normal placement on the stove as a drop in {checks['mislabel']} of {checks['mislabel_of']} cases. Failure onset was re-derived from pot 0 (rule <code>pot0-v1</code>) before the picker chose candidates.</p>
    <p>Sources: <code>demo-data/groot/recon-only-summary.json</code>, <code>demo-data/groot/sae-heldout-eval.json</code>, <code>adapters/groot/v9/V9_NOTES.md</code>.</p>"""


def _conditions_html(summary: dict) -> str:
    rows = []
    for row in summary["table"]:
        low, high = row["wilson95"]
        delta = row.get("delta_paired")
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['cond']))}</td>"
            f"<td>{int(row['successes'])}/{int(row['n'])}</td>"
            f"<td>[{_bound(low)}, {_bound(high)}]</td>"
            f"<td>{escape('' if delta is None else f'{float(delta):+.3f}')}</td>"
            f"<td>{escape(str(row.get('verdict') or ''))}</td>"
            "</tr>"
        )
    return f"""    <h2 id="conditions">Every condition</h2>
    <p>Conditions share seeds. Paired change is against baseline on the shared seeds. From <code>demo-data/groot/groot-1028-ablate-summary.json</code>.</p>
    <table class="score-table" data-testid="groot-conditions">
      <thead><tr><th>Condition</th><th>Success</th><th>Wilson 95%</th><th>Paired change</th><th>Verdict</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _random_control(randinit: Path) -> dict:
    trained = json.loads((randinit / "score-trained.json").read_text(encoding="utf-8"))
    random = json.loads((randinit / "score-randinit.json").read_text(encoding="utf-8"))
    return {"trained": trained, "random": random}


def _control_summary(control: dict) -> str:
    trained, random = control["trained"], control["random"]
    return (
        f"An SAE on a random-weights V-JEPA 2 scored higher on the loss-based checks than the trained model "
        f"(loss_recovered {random['loss_recovered']:.4f} vs {trained['loss_recovered']:.4f}). "
        "Those scores cannot tell trained from random. The random-weights SAE control is not in the tool yet."
    )


def _control_html(control: dict) -> str:
    trained, random = control["trained"], control["random"]
    rows = [
        ("Held-out explained variance", f"{trained['ev_heldout']:.3f}", f"{random['ev_heldout']:.3f}"),
        ("loss_recovered", f"{trained['loss_recovered']:.4f}", f"{random['loss_recovered']:.4f}"),
        ("L0 (held-out)", f"{trained['l0_heldout']:.1f}", f"{random['l0_heldout']:.1f}"),
        ("Beats the context-mean floor", str(trained["beats_floor"]), str(random["beats_floor"])),
        ("Held-out windows scored", str(int(trained["n_windows"])), str(int(random["n_windows"]))),
    ]
    body = "".join(f"<tr><td>{escape(a)}</td><td>{escape(b)}</td><td>{escape(c)}</td></tr>" for a, b, c in rows)
    return f"""    <h2 id="random-control">Random-weights control</h2>
    <p data-testid="random-control">The same SAE recipe, trained on a randomly initialised V-JEPA 2 (same DROID clips, layer 16), reconstructs its activations better than the trained model's SAE does: explained variance {random['ev_heldout']:.3f} vs {trained['ev_heldout']:.3f}, loss_recovered {random['loss_recovered']:.4f} vs {trained['loss_recovered']:.4f}. On the random model, zeroing layer 16 lowers the loss (L_zero {random['L_zero']:.3f} vs L_clean {random['L_clean']:.3f}), so its loss_recovered measures nothing, yet it still reads near 1. Only the context-mean floor check told the two apart. Explained variance and loss_recovered cannot separate a trained dictionary from a random one, so we closed this track. The random-weights SAE control is not in the tool yet.</p>
    <table class="score-table" data-testid="random-control-table">
      <thead><tr><th></th><th>Trained V-JEPA 2</th><th>Random weights</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
    <p>Sources: <code>demo-data/vjepa-randinit/score-trained.json</code> (run droid100-L16-posmean) and <code>score-randinit.json</code> (run droid100-L16-randinit).</p>"""


def _atp(atp: Path) -> dict:
    summary = json.loads((atp / "summary.json").read_text(encoding="utf-8"))
    corr = summary["correlations"]["|dL1| vs |AtP_dL1|"]["per_feature_mean"]
    base = summary["baseline_edit_size"]
    timing = summary["timing"]
    recall = summary["topk_recall"]["by |dL1| (mean over windows of |dL1|) vs mean |AtP|"]
    return {
        "n_features": int(summary["n_features"]),
        "spearman": float(corr["spearman"]),
        "n": int(corr["n"]),
        "edit_size": float(base["spearman_feat_absdL1_vs_act_sum"]),
        "speedup_all": float(timing["speedup_if_ranking_all_8192"]),
        "speedup_tested": float(timing["speedup_tested_features_only"]),
        "top10": int(recall["in_atp_top10"]),
        "windows": int(summary["n_feature_windows"]),
    }


def _atp_summary(atp: dict) -> str:
    return (
        f"On {atp['n_features']} V-JEPA 2 features, attribution ranks features the way measured ablation does "
        f"(Spearman {atp['spearman']:.3f}) at about {_round_speedup(atp['speedup_all'])}x less compute across all 8192 latents. "
        f"Edit size alone gets {atp['edit_size']:.2f}."
    )


def _atp_html(atp: dict) -> str:
    return f"""    <h2 id="atp">Attribution vs ablation</h2>
    <p data-testid="atp">Attribution patching (one forward and backward pass per window, covering all 8192 latents) against measured ablation of {atp['n_features']} features ({atp['windows']} feature-windows). Spearman between |attribution| and |ablation effect|, per feature: <b>{atp['spearman']:.3f}</b>. {atp['top10']} of ablation's top 10 are in attribution's top 10.</p>
    <p data-testid="atp-speed">Ranking all 8192 latents this way takes about {_round_speedup(atp['speedup_all'])}x less compute than ablating each one (projected from measured per-window times). On only the {atp['n_features']} features tested the saving is {atp['speedup_tested']:.1f}x.</p>
    <p data-testid="atp-caveat">Caveat: ranking by edit size alone (summed activation) already gives Spearman {atp['edit_size']:.2f}. Attribution is a prefilter for what to ablate; it does not replace ablation. Source: <code>demo-data/vjepa-atp/summary.json</code> (run droid100-L16).</p>"""


def _round_speedup(value: float) -> str:
    return f"{int(round(value, -2)):,}"


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
