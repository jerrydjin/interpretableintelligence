"""Write the static site: the front door, the run-it-yourself page and the blog.

The live playground (``demo-site/live/``) and the host config are hand-kept, so a
regeneration leaves them as they are. Every page here shares one stylesheet
(``static/site.css``) and one header, the same ones the playground uses.
Every number on the blog is read from a vendored result file.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from html import escape
from pathlib import Path

from interpretableintelligence import site_explore

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
LIVE_KEEP = ("live", "vercel.json")
GITHUB_URL = "https://github.com/jerrydjin/interpretableintelligence"
RUNNER_PATH = "src/interpretableintelligence/adapters/groot/v9/ablate_v9.py"
RUNNER_URL = GITHUB_URL + "/blob/main/" + RUNNER_PATH
README_URL = GITHUB_URL + "/blob/main/README.md"
POST_SLUG = "groot-1028"
POST_TITLE = "A feature tracked our robot's failures. The control said it wasn't the cause."
POST_DATE = "November 2026"
VJEPA_RUN = REPO / "demo-data" / "vjepa-slim"
V9_NOTES = REPO / "src" / "interpretableintelligence" / "adapters" / "groot" / "v9" / "V9_NOTES.md"

# The site has four sections. Every page belongs to one, and the nav marks it.
SECTIONS = (
    ("explore", "/explore/", "Explore"),
    ("play", "/live/", "Playground"),
    ("model", "/start/", "Your model"),
    ("blog", "/blog/", "Research"),
)
SECTION_OF = {"explore": "explore", "feature": "explore", "live": "play", "start": "model", "blog": "blog", "post": "blog"}
# Old addresses and where they live now. The pages behind them were cut.
REDIRECTS = {
    "droid": "/live/",
    "fake": "/start/",
    "demos": "/",
    "picker": "/blog/",
    "results": "/blog/",
    "vjepa": "/explore/",
    "groot": f"/blog/{POST_SLUG}/",
}
MARK_SVG = (
    '<svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">'
    '<path d="M4 7.5 7.5 5.5 11 7.5 7.5 9.5Z" fill="#e8582a"/><path d="M4 7.5 7.5 9.5V20L4 18Z" fill="#c8400f"/>'
    '<path d="M11 7.5 7.5 9.5V20L11 18Z" fill="#e04a14"/><path d="M11 3 14.5 1 18 3 14.5 5Z" fill="#141413"/>'
    '<path d="M11 3 14.5 5V20L11 18Z" fill="#141413"/><path d="M18 3 14.5 5V20L18 18Z" fill="#45423b"/></svg>'
)


def export_site(out_dir: Path, groot_dir: Path | None = None, vjepa_dir: Path | None = None) -> Path:
    """Regenerate ``out_dir`` from the vendored V-JEPA 2 and GR00T results, keeping the live playground."""
    out = Path(out_dir)
    groot = Path(groot_dir) if groot_dir else REPO / "demo-data" / "groot"
    run = site_explore.load_run(Path(vjepa_dir) if vjepa_dir else VJEPA_RUN)
    summary = _load_groot_summary(groot)
    checks = _groot_checks(groot)
    keep_dir = Path(tempfile.mkdtemp(prefix="ii-site-keep-"))
    kept = [name for name in LIVE_KEEP if (out / name).exists()]
    for name in kept:
        source = out / name
        (shutil.copytree if source.is_dir() else shutil.copy2)(source, keep_dir / name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in kept:
        source = keep_dir / name
        (shutil.copytree if source.is_dir() else shutil.copy2)(source, out / name)
    shutil.rmtree(keep_dir, ignore_errors=True)
    _copy_static(out / "static")
    (out / "favicon.ico").write_bytes(_favicon_png())
    _write_home(out / "index.html", summary, run)
    _write_explore(out / "explore", run)
    _write_start(out / "start" / "index.html")
    _write_blog_index(out / "blog" / "index.html", summary)
    _write_post(out / "blog" / POST_SLUG, groot / "groot-1028-ablate-summary.json", summary, checks)
    for old, target in REDIRECTS.items():
        _redirect(out / old / "index.html", target)
    return out


def _copy_static(dest: Path) -> None:
    dest.mkdir(parents=True)
    shutil.copy2(STATIC / "site.css", dest / "site.css")
    fonts = STATIC / "fonts"
    if fonts.is_dir():
        shutil.copytree(fonts, dest / "fonts")


def site_header(current: str = "", extra: str = "") -> str:
    """The one header every page uses, the playground included."""
    return (
        '<header class="bar">\n'
        f'    <a class="mark" href="/" aria-label="interpretable intelligence home">{MARK_SVG}<span>interpretable intelligence</span></a>\n'
        f'    <nav class="site-nav" aria-label="Site sections">{_nav(current)}</nav>\n'
        f"{extra}"
        "  </header>"
    )


def _nav(current: str) -> str:
    section = SECTION_OF.get(current, "")
    parts = []
    for key, href, label in SECTIONS:
        attrs = f' href="{href}"'
        if key == section:
            attrs += ' aria-current="page"'
        parts.append(f"<a{attrs}>{label}</a>")
    parts.append(f'<a href="{GITHUB_URL}">GitHub</a>')
    return "".join(parts)


def _page(title: str, current: str, body: str, description: str = "", wide: bool = False) -> str:
    meta = f'\n  <meta name="description" content="{escape(description)}">' if description else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#faf9f5">
  <title>{escape(title)}</title>{meta}
  <link rel="stylesheet" href="/static/site.css">
  <link rel="icon" href="/favicon.ico">
</head>
<body>
  {site_header(current)}
  <main class="prose{" prose-wide" if wide else ""}">{body}</main>
  <footer class="foot{" prose-wide" if wide else ""}"><p>ii is free, local and open on <a href="{GITHUB_URL}">GitHub</a>.</p></footer>
</body>
</html>
"""


def _redirect(path: Path, target: str) -> None:
    """An old address whose page was cut: a static page that forwards to where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={target}">
  <link rel="canonical" href="{target}">
  <title>Moved · interpretable intelligence</title>
</head>
<body><p>This page moved to <a href="{target}">{target}</a>.</p></body>
</html>
""",
        encoding="utf-8",
    )


# The home page's picture: one of the playground pack's sample pictures (demo-site/live/, hand-kept).
HERO_PACK = "clip-b16-words"
HERO_SAMPLE = "01-coco-14475"


def _write_home(path: Path, summary: dict, run: dict) -> None:
    """The front door: ask a model what it sees, then find out why. Then the ways in."""
    del run
    picture = f"/live/packs/{HERO_PACK}/samples/{HERO_SAMPLE}.jpg"
    body = f"""
    <section class="hero hero-split">
      <div>
        <h1>Ask a vision model what it sees, then find out why.</h1>
        <p class="lede">Pick a picture and the model says what it sees, in words. ii shows you the features inside it that make it see that, and lets you switch them off to check.</p>
        <p class="actions"><a class="button" href="/live/" data-testid="cta">Open the playground</a><a class="button button-quiet" href="/start/">Run it on your model</a></p>
      </div>
      <figure class="stage">
        <a href="/live/#p={HERO_SAMPLE}" data-testid="hero-picture"><img src="{picture}" alt="A picture from the playground" class="stage-picture"></a>
        <a class="stage-feature" href="/live/#p={HERO_SAMPLE}"><span>What does it see? Open this picture in the playground</span></a>
      </figure>
    </section>
    <ul class="doors" data-testid="doors">
      <li><a class="card" href="/live/"><span class="card-title">Playground</span><span class="card-text">Your own pictures, in your browser. See what the model sees, which features make it see that, and switch them off.</span></a></li>
      <li><a class="card" href="/start/"><span class="card-title">Your model</span><span class="card-text">The same tools on any Hugging Face vision or video model, on your laptop.</span></a></li>
      <li><a class="card" href="/explore/"><span class="card-title">V-JEPA 2</span><span class="card-text">A robot's world model: its scored features, with where each fires and which checks it passed.</span></a></li>
    </ul>
    <section>
      <h2>Research</h2>
      <a class="card" href="/blog/{POST_SLUG}/" data-testid="post-link">
        <span class="card-title">{escape(POST_TITLE)}</span>
        <span class="card-text">{escape(_post_teaser(summary))}</span>
      </a>
    </section>
    """
    path.write_text(
        _page("interpretable intelligence", "", body, "Ask a vision model what it sees, then find the features that make it see that and switch them off.", wide=True),
        encoding="utf-8",
    )


def _write_explore(dest: Path, run: dict) -> None:
    """Explore: a list of the scored V-JEPA 2 features and a page for each."""
    site_explore.copy_media(run["dir"], dest / "media")
    (dest / "index.html").write_text(
        _page("Explore · interpretable intelligence", "explore", site_explore.index_body(run)), encoding="utf-8"
    )
    for feature in run["features"]:
        page = dest / site_explore.RUN_SLUG / str(feature["id"]) / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(
            _page(f"{feature['name']} · V-JEPA 2 · interpretable intelligence", "feature", site_explore.feature_body(run, feature)),
            encoding="utf-8",
        )


def _write_start(path: Path) -> None:
    """Your model: install, the practice run, your model and clips, and the commands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    commands = dict(_ii_commands())
    known = set(commands)
    useful = ("run", "view", "rollouts", "validate")
    command_items = "".join(
        f"<li>{_cmd(name, known)}<span>{escape(commands[name])}</span></li>" for name in useful
    )
    body = f"""
    <h1>Run it on your own model</h1>
    <p class="lede">Free, on your laptop. No GPU, no account.</p>
    <h2>Install</h2>
    <p>You need git and <a href="https://docs.astral.sh/uv/">uv</a>.</p>
    <pre class="code"><code>git clone {GITHUB_URL}
cd interpretableintelligence
uv sync</code></pre>
    <h2>Try the practice model</h2>
    <p>{_cmd("try", known)} runs on a small model with three features planted on purpose, so you can check what it finds against the truth. About 15 seconds.</p>
    <pre class="code"><code>uv run ii try</code></pre>
    <h2>Your model</h2>
    <p>Any Hugging Face vision or video model. No clips? <code>uv run ii get-clips --out clips/droid</code> downloads 100 short robot clips.</p>
    <pre class="code"><code>uv run ii run hf --model facebook/dinov2-small \\
  --clips clips/droid --out runs/mine --no-score --steer</code></pre>
    <h2>Score it</h2>
    <p>Run it again without <code>--no-score</code> and every feature gets <a href="/explore/">the four checks</a>.</p>
    <h2>Commands</h2>
    <ul class="commands" data-testid="commands">{command_items}</ul>
    <p class="fine"><a href="{README_URL}">All commands</a></p>
    """
    path.write_text(_page("Your model · interpretable intelligence", "start", body), encoding="utf-8")


def _write_blog_index(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"""
    <h1>Research</h1>
    <ul class="rows" data-testid="posts">
      <li><a href="/blog/{POST_SLUG}/"><span class="row-name">{escape(POST_TITLE)}</span><span class="meta">{POST_DATE}</span></a></li>
    </ul>
    """
    path.write_text(_page("Research · interpretable intelligence", "blog", body), encoding="utf-8")


def _post_teaser(summary: dict) -> str:
    baseline = _one_row(summary, lambda cond: cond == "baseline")
    single = _single_feature_candidate(summary)
    return (
        f"Switching off feature {_latent_name(str(single['cond']))} in NVIDIA's GR00T dropped success "
        f"from {_fraction(baseline)} to {_fraction(single)}. Random features did that too. "
        "How ii kept us from calling it a finding."
    )


def _write_post(dest: Path, summary_path: Path, summary: dict, checks: dict) -> None:
    """How we used ii on GR00T task 1028. Every number is read from the vendored result files."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(summary_path, dest / summary_path.name)
    _require_no_effect(summary)
    identity = _groot_identity()
    spec = _sprt_spec()
    baseline = _one_row(summary, lambda cond: cond == "baseline")
    single = _single_feature_candidate(summary)
    pair = _one_row(summary, lambda cond: "-cand-" in cond and "+" in cond)
    single_name = _latent_name(str(single["cond"]))
    pair_name = _latent_name(str(pair["cond"]))
    matched = _matched_controls(summary, str(single["cond"]))
    matched_fell = sum(1 for row in matched if row.get("verdict") == "fall")
    controls = _control_rows(summary)
    fell = sum(1 for row in controls if row.get("verdict") == "fall")
    episodes = _episode_total(summary)
    real = "true" if summary["real"] is True else "false"
    body = f"""
    <article class="post">
    <p class="back"><a href="/blog/">Research</a></p>
    <h1>{escape(POST_TITLE)}</h1>
    <p class="lede">We pointed <code>ii</code> at {escape(identity["name"])}, NVIDIA's robot policy, on LIBERO-plus task {escape(identity["task"])}, a kitchen task it completes about half the time. Does one learned feature inside it make it fail?</p>

    <h2>What we did</h2>
    <ol>
      <li>Recorded {escape(spec["capture_n"])} episodes ({escape(spec["capture_successes"])} successes) and captured transformer block {escape(spec["layer"])} of the action model on its final denoise step.</li>
      <li>Trained a sparse dictionary on it: 8,192 features, 32 active at a time, held-out explained variance {checks["ev_heldout"]:.3f}.</li>
      <li>Picked the features that fire more in the moments before a failure. Two stood out: feature {escape(single_name)} and the pair {escape(pair_name)}.</li>
      <li>Ran the policy again with each candidate switched off, on shared seeds, with a stop rule that ends an arm as soon as the answer is clear.</li>
    </ol>
    <p>Before any of that, a sanity check: putting the dictionary's reconstruction in place of the real activations, with nothing switched off, left the actions identical to the baseline in {checks["recon_identical"]} of {checks["recon_compared"]} episodes. So the hook itself does not move the policy.</p>

    <h2>It looked like the answer</h2>
    <p data-testid="post-headline">Switching feature {escape(single_name)} off dropped success from {_fraction(baseline)} to {_fraction(single)}. The stop rule called it a fall after {int(single["n"])} episodes.</p>

    <h2>Then the control</h2>
    <p data-testid="post-controls">Next to every candidate we switched off random features matched to it. For feature {escape(single_name)}, {matched_fell} of {len(matched)} matched random features also made the policy fail. Across both candidates, {fell} of {len(controls)} random controls fell.</p>
    <p>Breaking a policy is easy. Many edits make it worse, so a drop when you remove a feature only says the policy uses that part of its activations. If feature {escape(single_name)} caused failures, switching it off should have raised success. That was the test, and <span data-testid="post-verdict">neither candidate passed it</span>. Both stay <code>real={real}</code>.</p>

    <h2>What we changed in ii</h2>
    <p>The result shaped the tool. A feature now counts as a finding on a robot policy only when all of these hold on shared seeds:</p>
    <ul>
      <li>The stop rule calls a rise in success, not a fall.</li>
      <li>The rise over baseline is significant (McNemar p &lt; 0.05).</li>
      <li>It beats every matched random-feature control.</li>
      <li>It beats an untrained dictionary run through the same picker, so the feature has to do better than what random weights would produce.</li>
    </ul>
    <p>{_cmd("rollouts", set(dict(_ii_commands())))} with <code>--pick</code> runs all four. The ablation in this post ran on the repo's earlier runner (<a href="{RUNNER_URL}"><code>ablate_v9.py</code></a>), before the untrained-dictionary control shipped.</p>

    <details class="appendix" data-testid="appendix">
      <summary>Every number behind this post</summary>
      <p>{episodes} rollouts in total. {escape(_baseline_line(baseline))} {escape(_candidate_text(single))} {escape(_candidate_text(pair))}</p>
      <p data-testid="stop-rule" data-stop-rule="{escape(spec["stop_rule"])}">{escape(_stop_rule_text(spec))}</p>
{_conditions_html(summary)}
      <p>SAE: BatchTopK, k=32, 8192 latents, trained on block {escape(spec["layer"])} of the {checks["episodes"]}-episode capture. Held-out explained variance {checks["ev_heldout"]:.3f} on {checks["ev_episodes"]} episodes ({checks["ev_tokens"]} tokens). loss_recovered: {"not measured" if checks["loss_recovered"] is None else _fmt_num(checks["loss_recovered"])}. Failure labels: the first labeller marked pot 1's normal placement as a drop in {checks["mislabel"]} of {checks["mislabel_of"]} cases, so failure onset was re-derived from pot 0 before candidates were picked.</p>
      <p>Sources: <a href="{summary_path.name}"><code>{summary_path.name}</code></a>, <code>demo-data/groot/recon-only-summary.json</code>, <code>demo-data/groot/sae-heldout-eval.json</code>.</p>
    </details>
    </article>
    """
    (dest / "index.html").write_text(
        _page(f"{POST_TITLE} · interpretable intelligence", "post", body, _post_teaser(summary)), encoding="utf-8"
    )


def _latent_name(cond: str) -> str:
    return cond.split("-", 2)[-1]


def _fraction(row: dict) -> str:
    return f"{int(row['successes'])}/{int(row['n'])}"


def _matched_controls(summary: dict, cond: str) -> list:
    """Controls paired with this candidate. A-cand matches A-ctrl; B-cand matches B-ctrl."""
    prefix = str(cond).split("-", 1)[0] + "-ctrl"
    return [row for row in _control_rows(summary) if str(row["cond"]).startswith(prefix)]


def _sprt_spec() -> dict:
    """Stop-rule parameters, copied from the literals in ablate_v9.default_plan."""
    source = (REPO / RUNNER_PATH).read_text(encoding="utf-8")
    match = re.search(
        r'"p0": (?P<p0>\d+(?:\.\d+)?), "p1_rise": (?P<p1_rise>\d+(?:\.\d+)?), '
        r'"p1_fall": (?P<p1_fall>\d+(?:\.\d+)?), "alpha": (?P<alpha>\d+(?:\.\d+)?), '
        r'"beta": (?P<beta>\d+(?:\.\d+)?), "cap": cap,\s*'
        r'"stop_rule": "(?P<stop_rule>[^"]+)"',
        source,
    )
    cap = re.search(r"cap: int = (\d+)", source)
    note = re.search(r'"p0_note": "(\d+(?:\.\d+)?) = capture success rate (\d+)/(\d+)', source)
    layer = re.search(r'"layer": (\d+)', source)
    if not match or not cap or not note or not layer:
        raise ValueError(f"{RUNNER_PATH} no longer states the stop rule")
    if note.group(1) != match.group("p0"):
        raise ValueError("p0 and its capture-rate note disagree")
    spec = match.groupdict()
    spec["cap"] = cap.group(1)
    spec["capture_successes"] = note.group(2)
    spec["capture_n"] = note.group(3)
    spec["layer"] = layer.group(1)
    return spec


def _stop_rule_text(spec: dict) -> str:
    return (
        f"Stop rule: Sobel–Wald. Stop when the rise test or the fall test accepts H1, "
        f"or both accept H0, or at {spec['cap']} episodes. "
        f"Rise compares {spec['p0']} with {spec['p1_rise']}. "
        f"Fall compares {spec['p0']} with {spec['p1_fall']}. "
        f"Alpha is {spec['alpha']}. Beta is {spec['beta']}. "
        f"{spec['p0']} is the capture success rate, {spec['capture_successes']}/{spec['capture_n']}. "
        "The verdict is rise when the rise test accepts H1, fall when the fall test accepts H1, "
        "and no_effect when both accept H0. At the cap with no decision, the verdict is cap_undecided."
    )


def _groot_identity() -> dict:
    notes = (REPO / "data" / "groot-1028.md").read_text(encoding="utf-8")
    task = re.search(r"task \*\*(\d+)\*\*", notes)
    checkpoint = re.search(r"`(nvidia/gr00t17-lerobot-libero_10-640)`", notes)
    adapter = (REPO / "src" / "interpretableintelligence" / "adapters" / "groot" / "adapter.py").read_text(encoding="utf-8")
    name = "GR00T N1.7"
    if not task or not checkpoint or name not in adapter:
        raise ValueError("GR00T task identity is not in the repo")
    return {"task": task.group(1), "checkpoint": checkpoint.group(1), "name": name}


def _load_groot_summary(groot: Path) -> dict:
    path = groot / "groot-1028-ablate-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(summary.get("table"), list):
        raise ValueError(f"{path} has no ablation table")
    return summary


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


def _ii_commands() -> list[tuple[str, str]]:
    """Subcommands `ii --help` actually registers, with the first line of their help."""
    from typer.main import get_command

    from interpretableintelligence.cli import app

    lines = []
    for name, command in get_command(app).commands.items():
        blurb = (command.help or "").strip().splitlines()[0].strip()
        if name == "validate":
            # The docstring's first line cites exit code 0. Keep result numbers out of this list.
            blurb = "Validate a run folder against the schema."
        if name == "rollouts":
            blurb = "Switch features off inside a policy and compare task success against matched random controls."
        lines.append((name, blurb))
    if not lines:
        raise ValueError("ii has no commands")
    return lines


def _cmd(name: str, known: set[str]) -> str:
    if name not in known:
        raise ValueError(f"ii {name} is not a command on main")
    return f'<code data-cmd="{escape(name)}">ii {escape(name)}</code>'


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
            f"<td>{_seq_cell(row)}</td>"
            f"<td>{escape('' if delta is None else f'{float(delta):+.3f}')}</td>"
            f"<td>{escape(str(row.get('verdict') or ''))}</td>"
            "</tr>"
        )
    return f"""    <h2 id="conditions">Every condition</h2>
    <p>Conditions share seeds. Paired change is against baseline on the shared seeds. From <code>demo-data/groot/groot-1028-ablate-summary.json</code>.</p>
    <p data-testid="groot-sequential-note">The Wilson interval assumes the number of episodes was fixed in advance. Each SPRT condition stopped as soon as its boundary was crossed, so the sequential column is the exact 95% interval under that stopping rule, valid however early the run stopped. It is usually wider, and it is the one to read for the ablated conditions.</p>
{_arms_html(summary)}
    <table class="score-table" data-testid="groot-conditions">
      <thead><tr><th>Condition</th><th>Success</th><th>Wilson 95%</th><th>Sequential 95%</th><th>Paired change</th><th>Verdict</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _seq_cell(row: dict) -> str:
    seq = row.get("seq95")
    if seq is None:
        return "fixed n"
    return f"[{_bound(seq[0])}, {_bound(seq[1])}]"


def _arms_html(summary: dict) -> str:
    """Per candidate: matched controls that also fell, the permutation-style p-value, and Holm across candidates."""
    arms = summary.get("arms")
    if not arms:
        return ""
    lines = []
    for cond, d in arms["candidates"].items():
        vs, holm = d["vs_controls"], d["holm"]
        ident = str(cond).split("-", 2)[-1]
        name = f"The {ident} pair" if "+" in ident else f"Feature {ident}"
        lines.append(
            f'    <p data-testid="groot-arm">{escape(name)}: {vs["controls_fell"]} of {vs["n_controls"]} matched random '
            f"controls also fell, and {vs['n_controls_at_least_as_bad']} of {vs['n_controls']} did at least as badly "
            f"(permutation p = {vs['p_perm_fall']:.3f}, Holm {holm['p_perm_fall']:.3f}). Against p0 = {arms['p0']:g} "
            f"the sequential fall p is {d['p_seq']['fall']:.3f} (Holm {holm['p_seq_fall']:.3f}) and the rise p is "
            f"{d['p_seq']['rise']:.3f} (Holm {holm['p_seq_rise']:.3f}). A drop below p0 that random features also "
            "produce is not specific to this feature.</p>"
        )
    k = next(iter(arms["candidates"].values()))["vs_controls"]
    lines.append(
        f"    <p>With {k['n_controls']} controls per arm the permutation p-value cannot go below "
        f"{k['min_attainable_p']:.3f}. Holm is across the {len(arms['candidates'])} candidates. None of these numbers sets a verdict.</p>"
    )
    return "\n".join(lines)


def _favicon_png() -> bytes:
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


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
