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


GITHUB = "https://github.com/jerrydjin/interpretableintelligence"
NAV = ["/explore/", "/live/", "/start/", "/blog/", GITHUB]
PAGES = {
    "index.html": None,
    "explore/index.html": "/explore/",
    "explore/vjepa2-vitl/L16/270/index.html": "/explore/",
    "start/index.html": "/start/",
    "blog/index.html": "/blog/",
    "blog/groot-1028/index.html": "/blog/",
    "blog/vjepa2ac-gripper/index.html": "/blog/",
    "live/index.html": "/live/",
}


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _nav(page: str) -> str:
    return re.search(r'<nav class="site-nav"[^>]*>(.*?)</nav>', page, re.S).group(1)


def _summary() -> dict:
    return json.loads((_repo() / "demo-data" / "groot" / "groot-1028-ablate-summary.json").read_text(encoding="utf-8"))


def _row(summary: dict, cond: str) -> dict:
    return next(row for row in summary["table"] if row["cond"] == cond)


def test_every_page_shares_one_header_and_one_stylesheet():
    for name, current in PAGES.items():
        page = (ROOT / name).read_text(encoding="utf-8")
        nav = _nav(page)
        assert re.findall(r'href="([^"]+)"', nav) == NAV, name
        marked = re.findall(r'href="([^"]+)" aria-current="page"', nav)
        assert marked == ([current] if current else []), name
        sheets = re.findall(r'rel="stylesheet" href="([^"]+)"', page)
        assert sheets[0].endswith("static/site.css"), name
        assert "demo.css" not in page and "static/style.css" not in page, name
    assert not (ROOT / "static" / "demo.css").exists()
    assert not (ROOT / "static" / "style.css").exists()


def test_cut_pages_forward_to_where_they_went():
    from interpretableintelligence.export_site import REDIRECTS

    for old in ("droid", "fake", "demos", "picker", "results", "vjepa", "groot"):
        page = (ROOT / old / "index.html").read_text(encoding="utf-8")
        assert f'http-equiv="refresh" content="0; url={REDIRECTS[old]}"' in page
        assert sorted(p.name for p in (ROOT / old).iterdir()) == ["index.html"], old


def test_home_leads_with_the_world_model():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    text = _text(page)
    assert 'href="/explore/" data-testid="cta"' in page
    assert 'href="/live/"' in page and 'href="/start/"' in page
    assert 'href="/blog/vjepa2ac-gripper/" data-testid="post-link"' in page
    assert "world model" in text
    assert 'src="/explore/media/droid_ep068.mp4"' in page
    assert 'href="/explore/vjepa2-vitl/L16/270/" data-testid="hero-feature"' in page
    for gone in ("/droid/", "/results/", "/demos/", "/fake/", "/picker/", "Synthetic"):
        assert gone not in page
    assert "\u2014" not in page
    assert "mailto:" not in page and not re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    assert "$" not in text


def test_start_page_commands_exist():
    from typer.main import get_command

    from interpretableintelligence.cli import app

    cli = set(get_command(app).commands)
    page = (ROOT / "start" / "index.html").read_text(encoding="utf-8")
    text = _text(page)
    assert "uv sync" in text and "uv run ii try" in text and "ii run hf" in text and "ii get-clips" in text
    named = set(re.findall(r'data-cmd="([^"]+)"', page))
    assert named <= cli
    commands = re.search(r'<ul class="commands" data-testid="commands">(.*?)</ul>', page, re.S).group(1)
    assert set(re.findall(r'data-cmd="([^"]+)"', commands)) == {"run", "view", "rollouts", "validate"}
    assert "\u2014" not in page


def test_blog_post_reads_every_number_from_the_result_files():
    summary = _summary()
    groot = _repo() / "demo-data" / "groot"
    built = ROOT / "blog" / "groot-1028" / "groot-1028-ablate-summary.json"
    assert built.read_bytes() == (groot / "groot-1028-ablate-summary.json").read_bytes()
    page = (ROOT / "blog" / "groot-1028" / "index.html").read_text(encoding="utf-8")
    text = _text(page)
    baseline, single = _row(summary, "baseline"), _row(summary, "A-cand-720")
    controls = [row for row in summary["table"] if "-ctrl" in row["cond"]]
    matched = [row for row in controls if row["cond"].startswith("A-ctrl")]
    fell = sum(row["verdict"] == "fall" for row in controls)
    assert (
        f"Switching feature 720 off dropped success from {baseline['successes']}/{baseline['n']} "
        f"to {single['successes']}/{single['n']}."
    ) in text
    assert f"{sum(r['verdict'] == 'fall' for r in matched)} of {len(matched)} matched random features" in text
    assert f"{fell} of {len(controls)} random controls fell" in text
    assert summary["real"] is False and "real=false" in text
    assert 'data-testid="post-verdict">neither candidate passed it<' in page
    assert f"{summary['episodes_logged']} rollouts in total." in text
    for row in summary["table"]:
        assert f"{row['cond']} {row['successes']}/{row['n']}" in text
    recon = json.loads((groot / "recon-only-summary.json").read_text(encoding="utf-8"))["recon_check"]
    assert f"identical to the baseline in {recon['n_identical_actions']} of {recon['n_compared_with_baseline']}" in text
    sae = json.loads((groot / "sae-heldout-eval.json").read_text(encoding="utf-8"))
    assert f"held-out explained variance {sae['heldout']['ev_raw']:.3f}" in text
    assert sae["loss_recovered"] is None and "loss_recovered: not measured" in text
    assert "ablate_v9.py" in page
    lower = text.lower()
    for claim in ("causes failure", "caused the failure", "failure cause"):
        assert claim not in lower
    assert "\u2014" not in page


def test_gripper_post_reads_every_number_from_the_result_files():
    from interpretableintelligence.site_gripper_post import FILES

    data_dir = _repo() / "demo-data" / "vjepa2ac"
    post = ROOT / "blog" / "vjepa2ac-gripper"
    for name in FILES:
        assert (post / name).read_bytes() == (data_dir / name).read_bytes(), name
    page = (post / "index.html").read_text(encoding="utf-8")
    text = _text(page)
    load = lambda name: json.loads((data_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    probe, trace, sae, circuit = (load(n) for n in ("gripper-probe.json", "gripper-trace.json", "gripper-sae-L9.json", "gripper-circuit.json"))
    pref = probe["stats"]["prefers_correct"]
    assert probe["verdict"] == "gripper_blind"
    assert f"{probe['n_events']} moments in {probe['n_episodes']} DROID episodes" in text
    assert f"right gripper command {100 * pref['value']:.0f}% of the time (95% interval {100 * pref['lo']:.0f}-{100 * pref['hi']:.0f}%)" in text
    ratio = trace["layers"]["grip_over_arm_frame"]
    assert f"grows to {max(ratio):.2f} times the arm's at block {ratio.index(max(ratio))}, then falls to {ratio[-1]:.2f}" in text
    tr = sae["arms"]["trained"]["consistency"]["value"]
    assert f"on {100 * tr:.0f}% of held-out events" in text
    steer = circuit["steer"]["1.0"]["prefers_correct"]
    assert f"preference to {100 * steer['value']:.0f}% ({100 * steer['lo']:.0f}-{100 * steer['hi']:.0f}%)" in text
    assert circuit["verdict"] == {"components": "none", "heads": "none", "steer": "no_fix"}
    assert 'data-testid="post-verdict">None of this is a finding.' in page
    assert f"AUROC {trace['readout']['encoder']['value']:.2f}" in text
    assert "\u2014" not in page


def test_blog_post_stop_rule_matches_the_runner():
    from interpretableintelligence.export_site import _sprt_spec

    spec = _sprt_spec()
    page = (ROOT / "blog" / "groot-1028" / "index.html").read_text(encoding="utf-8")
    assert f'data-stop-rule="{spec["stop_rule"]}"' in page
    assert f"Rise compares {spec['p0']} with {spec['p1_rise']}." in page


def test_export_site_regenerates_the_committed_site(tmp_path):
    """The committed demo-site is exactly what ii export-site writes, live playground aside."""
    from interpretableintelligence.export_site import export_site

    out = tmp_path / "site"
    export_site(out)
    generated = {p.relative_to(out) for p in out.rglob("*") if p.is_file()}
    committed = {
        p.relative_to(ROOT)
        for p in ROOT.rglob("*")
        if p.is_file() and p.relative_to(ROOT).parts[0] not in ("live", "vercel.json")
    }
    assert generated == committed
    for rel in generated:
        assert (out / rel).read_bytes() == (ROOT / rel).read_bytes(), rel


def _scoreboard() -> dict:
    return json.loads((_repo() / "demo-data" / "vjepa-slim" / "scoreboard.json").read_text(encoding="utf-8"))


def test_explore_lists_every_scored_feature_from_the_scoreboard():
    board = _scoreboard()
    page = (ROOT / "explore" / "index.html").read_text(encoding="utf-8")
    listed = re.search(r'<ul class="rows" data-testid="feature-list">(.*?)</ul>', page, re.S).group(1)
    ids = [int(i) for i in re.findall(r'href="/explore/vjepa2-vitl/L16/(\d+)/"', listed)]
    assert sorted(ids) == sorted(int(f["id"]) for f in board["features"])
    real = sum(f["real"] is True for f in board["features"])
    assert ("None has passed all four yet." in _text(page)) == (real == 0)
    names = re.findall(r'<span class="row-name">([^<]+)</span>', listed)
    assert len(names) == len(ids)
    assert "\u2014" not in page


def test_feature_pages_read_their_checks_from_the_scoreboard():
    from interpretableintelligence.site_explore import display_name

    for row in _scoreboard()["features"]:
        page = (ROOT / "explore" / "vjepa2-vitl" / "L16" / str(row["id"]) / "index.html").read_text(encoding="utf-8")
        text = _text(page)
        checks = re.search(r'<ul class="checks" data-testid="checks">(.*?)</ul>', page, re.S).group(1)
        states = re.findall(r'class="check check-(\w+)"', checks)
        expected = [
            "pass" if row["distinct_clips_top20"] >= 5 else "fail",
            "pass" if row["label_holds_heldout"] else "fail",
            "pass" if row["random_percentile"] >= 95 else "fail",
            "pending" if row["absent_in_random_init"] is None else ("pass" if row["absent_in_random_init"] else "fail"),
        ]
        assert states == expected, row["id"]
        assert f"{expected.count('pass')} of 4 checks" in text
        assert f"{row['distinct_clips_top20']} of its top 20 moments" in text
        assert f"of {row['n_random']} matched random features" in text
        assert f"#{row['id']}" in text
        assert display_name(row["label"])[0].split(",")[0] in text
        assert (ROOT / "explore" / "media" / f"feature_{int(row['id']):05d}.png").is_file()
        assert "\u2014" not in page


def test_explore_names_are_short_and_distinct_where_they_can_be():
    from interpretableintelligence.site_explore import display_name, load_run

    assert display_name("robot gripper / wrist (end effector and its cable)") == ("Robot gripper / wrist", "content")
    assert display_name("positional: band around row h~9, scattered along it") == ("Band around row 9", "position")
    assert display_name("positional: last tubelet (t=7), scattered tokens", 2) == ("Last frames, scattered tokens", "position")
    run = load_run(_repo() / "demo-data" / "vjepa-slim")
    position = [f["name"] for f in run["features"] if f["kind"] == "position"]
    assert len(position) == len(set(position))
