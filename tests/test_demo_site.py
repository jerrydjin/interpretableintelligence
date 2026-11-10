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
            target = _resolve(page, ref)
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
    lead = (
        f"{summary['episodes_logged']} rollouts, {len(controls)} random controls, "
        "and an offline lead rejected"
    )
    assert lead in page
    assert page.index(lead) < page.index("Tested: no causal effect found")
    assert 'data-testid="groot-status">Tested: no causal effect found<' in page
    assert "Not yet validated" not in page
    assert "Not yet validated" not in (ROOT / "index.html").read_text(encoding="utf-8")
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
