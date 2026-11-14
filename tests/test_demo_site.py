"""The static demo site only points at files that exist."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "demo-site"
SKIP_PREFIXES = ("#", "mailto:", "data:", "javascript:")


class _Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src", "poster"} and value:
                self.refs.append(value)


def _html_refs(path: Path) -> list[str]:
    parser = _Refs()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.refs


def _css_refs(text: str) -> list[str]:
    return [match.group(1) for match in re.finditer(r"""url\((?:['"]?)([^'")]+)(?:['"]?)\)""", text)]


def _resolve(page: Path, ref: str) -> Path | None:
    if ref.startswith(SKIP_PREFIXES) or ref.startswith(("http://", "https://")):
        return None
    if ref.startswith("/"):
        return ROOT / ref.lstrip("/")
    return (page.parent / ref).resolve()


def test_demo_site_references_exist():
    assert ROOT.is_dir(), "demo-site/ is missing; run make demo-site"
    missing: list[str] = []
    pages = list(ROOT.rglob("*.html")) + list(ROOT.rglob("*.css"))
    assert pages, "demo-site has no pages"
    for page in pages:
        refs = _html_refs(page) if page.suffix == ".html" else _css_refs(page.read_text(encoding="utf-8"))
        for ref in refs:
            target = _resolve(page, ref.split("#", 1)[0])
            if target is None:
                continue
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                missing.append(f"{page.relative_to(ROOT)} -> {ref}")
    assert missing == []


def test_demo_site_does_not_call_the_viewer_api():
    text = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.rglob("*.html"))
    assert "/api/" not in text
    assert "localhost" not in text
    assert "127.0.0.1" not in text


def test_fake_scoreboard_real_labels():
    run = json.loads((ROOT / "fake" / "data" / "run.json").read_text(encoding="utf-8"))
    real = sorted(feature["label"] for feature in run["features"] if feature.get("badge") == "real")
    assert real == ["corner", "pulse", "stripe"]
    labels = {feature["label"]: feature.get("badge") for feature in run["features"]}
    for name in ("flicker", "band", "speckle"):
        assert labels[name] != "real"
    page = (ROOT / "fake" / "index.html").read_text(encoding="utf-8")
    assert "stripe" in page and "pulse" in page and "corner" in page


def test_vjepa_status_is_the_scoreboard_string():
    scoreboard = json.loads(
        (Path(__file__).resolve().parents[1] / "demo-data" / "vjepa-slim" / "scoreboard.json").read_text(
            encoding="utf-8"
        )
    )
    status = next(row["status"] for row in scoreboard["features"] if row["id"] == 270)
    page = (ROOT / "vjepa" / "index.html").read_text(encoding="utf-8")
    assert status in page


def test_groot_page_reads_the_ablation_summary():
    repo = Path(__file__).resolve().parents[1]
    summary_path = repo / "demo-data" / "groot" / "groot-1028-ablate-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    page = (ROOT / "groot" / "index.html").read_text(encoding="utf-8")
    built = ROOT / "groot" / "groot-1028-ablate-summary.json"
    assert built.read_bytes() == summary_path.read_bytes()
    controls = [row for row in summary["table"] if "-ctrl" in row["cond"]]
    headline = _control_headline(summary)
    assert headline in page
    assert page.index(headline) < page.index("Tested: no causal effect found")
    assert f"{summary['episodes_logged']} rollouts." in page
    assert "ablate_v9.py" in page
    assert "not through an ii command" in _text(page)
    assert "the check ii is built around" in _text(page)
    assert 'data-testid="groot-status">Tested: no causal effect found<' in page
    assert "Not yet validated" not in page
    assert "Not yet validated" not in (ROOT / "index.html").read_text(encoding="utf-8")
    assert "Not yet validated" not in (ROOT / "results" / "index.html").read_text(encoding="utf-8")
    baseline = next(row for row in summary["table"] if row["cond"] == "baseline")
    low, high = baseline["wilson95"]
    assert (
        f"Baseline: {baseline['successes']}/{baseline['n']} "
        f"({round(100 * baseline['successes'] / baseline['n'])}%), "
        f"Wilson 95% interval [{low:.4f}, {high:.4f}]."
    ) in page
    single = next(row for row in summary["table"] if row["cond"] == "A-cand-720")
    pair = next(row for row in summary["table"] if row["cond"] == "B-cand-7081+5852")
    assert f"Feature 720: {single['successes']}/{single['n']}, verdict {single['verdict']}." in page
    assert (
        f"The 7081+5852 pair: {pair['successes']}/{pair['n']} "
        f"({round(100 * pair['successes'] / pair['n'])}%), verdict {pair['verdict']}."
    ) in page
    fell = sum(row["verdict"] == "fall" for row in controls)
    assert (
        f"{fell} of the {len(controls)} random controls also fell, "
        "so a fall is not specific to these features."
    ) in page
    assert summary["real"] is False
    assert "real=false" in page
    assert "The test was whether switching a feature off raises success." in page


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _nav(page: str) -> str:
    return re.search(r'<nav class="nav">(.*?)</nav>', page).group(1)


def test_results_put_real_results_above_the_synthetic_demos():
    page = (ROOT / "results" / "index.html").read_text(encoding="utf-8")
    real = page.index('data-testid="real-sections"')
    synthetic = page.index('data-testid="synthetic-sections"')
    assert real < synthetic
    main = page[page.index("<main"):]
    assert main.index('href="/groot/"') < main.index('data-testid="synthetic-sections"')
    assert page[synthetic:].count('href="/fake/"') == main.count('href="/fake/"') == 1
    assert page[synthetic:].count('href="/picker/"') == main.count('href="/picker/"') == 1
    nav = _nav(page)
    assert nav.index("/results/") < nav.index("/groot/") < nav.index("/vjepa/") < nav.index("/picker/") < nav.index("/fake/")
    assert 'aria-current="page"' in nav and "/results/" in nav
    headline = _control_headline(_ablation_summary())
    assert headline in page
    assert page.index(headline) < page.index("Tested: no causal effect found")
    fake = (ROOT / "fake" / "index.html").read_text(encoding="utf-8")
    assert 'data-testid="synthetic-banner"' in fake
    assert "not a result on a real model" in fake
    assert 'href="/">Home</a>' in fake
    assert 'href="/results/"' in fake


def _ablation_summary() -> dict:
    path = _repo() / "demo-data" / "groot" / "groot-1028-ablate-summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _control_headline(summary: dict) -> str:
    baseline = next(row for row in summary["table"] if row["cond"] == "baseline")
    single = next(row for row in summary["table"] if row["cond"] == "A-cand-720")
    feature = str(single["cond"]).split("-")[-1]
    controls = [row for row in summary["table"] if "-ctrl" in row["cond"]]
    fell = sum(row["verdict"] == "fall" for row in controls)
    return (
        f"Feature {feature} tracked failures, and turning it off dropped success "
        f"from {int(baseline['successes'])}/{int(baseline['n'])} "
        f"to {int(single['successes'])}/{int(single['n'])}. "
        f"Turning off random features caused falls too in {fell} of {len(controls)} runs. "
        "`ii` makes you run that control before calling a feature real."
    )


def _coming_and_rest(html: str) -> tuple[str, str]:
    match = re.search(r'(<ul\b[^>]*data-testid="coming"[^>]*>.*?</ul>)', html, re.S)
    assert match, "product page has no coming block"
    coming = match.group(1)
    return coming, html.replace(coming, "", 1)


def _cli_names() -> set[str]:
    from typer.main import get_command

    from interpretableintelligence.cli import app

    return set(get_command(app).commands)


def _visible_numbers(html: str) -> list[str]:
    """Measurement digits in the product copy. Model names are not measurements."""
    main = html[html.index("<main"):html.index("</main>")]
    without_code = re.sub(r"<code\b[^>]*>.*?</code>", "", main, flags=re.S)
    text = _text(without_code)
    for name in ("V-JEPA 2", "GR00T", "vjepa2"):
        text = text.replace(name, " ")
    return re.findall(r"\d+(?:\.\d+)?", text)


def test_product_page_is_the_front_door():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    text = _text(page)
    assert "Find out why your robot policy failed, and whether you're fooling yourself." in text
    assert (
        "For teams training robot policies and world models who need to know why a rollout went wrong, not just that it did."
        in text
    )
    assert "Run `ii` on your policy." in text
    assert 'href="#quickstart"' in page
    assert "free and local" in text
    assert "open source" not in text.lower()
    assert "mailto:" not in page
    assert not re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    assert "contact" not in text.lower()
    assert "$" not in text and "price" not in text.lower()
    summary = _ablation_summary()
    headline = _control_headline(summary)
    assert page.count(headline) == 1
    assert page.index(headline) < page.index("Tested: no causal effect found")
    assert "ablate_v9.py" in page
    assert f"{summary['episodes_logged']} rollouts." in text
    assert (
        "On V-JEPA 2, loss-based scores couldn't tell a trained SAE from a random one — so that control is now the bar."
        in text
    )
    today = page.split("What you get today", 1)[1].split("<h2", 1)[0]
    assert "GR00T" not in today
    assert (
        "ii rollouts compares baseline and intervened task success with matched random-feature controls "
        "and early stopping, on the bundled fake policy."
    ) in _text(today)
    assert "on the synthetic reacher. Resume skips episodes already written." in text
    assert "ii runs the" not in text
    assert "not through an ii command" not in text
    assert "did not run the GR00T episodes" not in text
    assert "wired into" not in text
    assert "nothing is called real" not in text.lower()
    assert "not a command on main" not in text.lower()
    assert "not in the tool yet" not in text.lower()
    assert "causes failure" not in text.lower()
    assert page.index('id="quickstart"') < page.index('data-testid="synthetic-sections"')
    assert page.index("What you get today") < page.index('data-testid="coming"')
    assert page.index('data-testid="coming"') < page.index('id="quickstart"')
    main = page[page.index("<main"):]
    synthetic = page.index('data-testid="synthetic-sections"')
    assert main.index('href="/results/"') < main.index('data-testid="synthetic-sections"')
    assert page[synthetic:].count('href="/fake/"') == main.count('href="/fake/"') == 1
    assert page[synthetic:].count('href="/picker/"') == main.count('href="/picker/"') == 1
    assert main.count('href="/results/"') == 2
    nav = _nav(page)
    assert nav.index("/results/") < nav.index("/groot/") < nav.index("/vjepa/") < nav.index("/picker/") < nav.index("/fake/")
    quick = re.search(r'<h2 id="quickstart">.*?</pre>', page, re.S).group(0)
    assert "uv sync" in quick and "uv run ii try" in quick
    assert "README.md" in page
    single = next(row for row in summary["table"] if "-cand-" in row["cond"] and "+" not in row["cond"])
    controls = [row for row in summary["table"] if "-ctrl" in row["cond"]]
    fell = sum(row["verdict"] == "fall" for row in controls)
    baseline = next(row for row in summary["table"] if row["cond"] == "baseline")
    assert _visible_numbers(page) == [
        str(single["cond"]).split("-")[-1],
        str(int(baseline["successes"])),
        str(int(baseline["n"])),
        str(int(single["successes"])),
        str(int(single["n"])),
        str(fell),
        str(len(controls)),
        str(summary["episodes_logged"]),
    ]


def test_product_page_commands_exist_or_are_coming():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    coming, rest = _coming_and_rest(page)
    cli = _cli_names()
    useful = {"try", "run", "view", "rollouts", "validate"}
    named = set(re.findall(r'data-cmd="([^"]+)"', rest))
    coming_named = set(re.findall(r'data-cmd="([^"]+)"', coming))
    assert named == useful
    assert named <= cli
    assert coming_named == {"import"}
    assert coming_named.isdisjoint(cli)

    def command_tokens(fragment: str) -> set[str]:
        tokens = set(re.findall(r"\bii\s+([a-z][a-z0-9-]*)", _text(fragment)))
        return {token for token in tokens if token in cli or token in {"import", "probe"} or "-" in token}

    assert command_tokens(rest) == useful
    assert command_tokens(coming) == {"import"}
    assert "random-weights SAE control" in _text(coming)
    assert "a feature has to beat an untrained SAE before it counts" in _text(coming)
    assert "Closed-loop rollouts at scale" in coming
    assert "Live GR00T rollouts through" in _text(coming) and re.search(r"through <code>ii</code>\.", coming)
    assert "ii import" not in _text(rest)
    assert "ii probe" not in page


def test_groot_method_checks_come_from_the_vendored_files():
    groot = _repo() / "demo-data" / "groot"
    recon = json.loads((groot / "recon-only-summary.json").read_text(encoding="utf-8"))["recon_check"]
    sae = json.loads((groot / "sae-heldout-eval.json").read_text(encoding="utf-8"))
    text = _text((ROOT / "groot" / "index.html").read_text(encoding="utf-8"))
    assert (
        f"Actions were identical to baseline in {recon['n_identical_actions']} of "
        f"{recon['n_compared_with_baseline']} shared-seed episodes"
    ) in text
    assert recon["n_identical_actions"] == recon["n_compared_with_baseline"] == 8
    assert f"Held-out explained variance {sae['heldout']['ev_raw']:.3f}" in text
    assert sae["loss_recovered"] is None and "loss_recovered: not measured" in text
    notes = (_repo() / "src/interpretableintelligence/adapters/groot/v9/V9_NOTES.md").read_text(encoding="utf-8")
    assert "in 37 of 40 cases" in notes and "in 37 of 40 cases" in text
    summary = json.loads((groot / "groot-1028-ablate-summary.json").read_text(encoding="utf-8"))
    for row in summary["table"]:
        assert f"{row['cond']} {row['successes']}/{row['n']}" in text
    lower = text.lower()
    assert "no causal effect found" in lower
    assert "failure cause" not in lower
    assert "causes failure" not in lower
    assert "the cause" not in lower
    assert "caused the failure" not in lower


def test_vjepa_random_control_and_attribution_come_from_the_vendored_files():
    repo = _repo()
    trained = json.loads((repo / "demo-data/vjepa-randinit/score-trained.json").read_text(encoding="utf-8"))
    random = json.loads((repo / "demo-data/vjepa-randinit/score-randinit.json").read_text(encoding="utf-8"))
    atp = json.loads((repo / "demo-data/vjepa-atp/summary.json").read_text(encoding="utf-8"))
    text = _text((ROOT / "vjepa" / "index.html").read_text(encoding="utf-8"))
    assert (
        f"loss_recovered {random['loss_recovered']:.4f} vs {trained['loss_recovered']:.4f}" in text
    )
    assert random["loss_recovered"] > trained["loss_recovered"]
    rho = atp["correlations"]["|dL1| vs |AtP_dL1|"]["per_feature_mean"]["spearman"]
    assert f"Spearman between |attribution| and |ablation effect|, per feature: {rho:.3f}" in text
    edit = atp["baseline_edit_size"]["spearman_feat_absdL1_vs_act_sum"]
    assert f"edit size alone (summed activation) already gives Spearman {edit:.2f}" in text
    assert f"{int(round(atp['timing']['speedup_if_ranking_all_8192'], -2)):,}x" in text
    assert f"{atp['timing']['speedup_tested_features_only']:.1f}x" in text
