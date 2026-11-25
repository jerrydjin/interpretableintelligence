"""Explore: the V-JEPA 2 features on robot video, one page each.

Reads the vendored slim run in ``demo-data/vjepa-slim`` (scoreboard, picture grids,
one clip). Every number on these pages comes from that scoreboard; nothing is
computed here, and the scoreboard is never written.
"""

from __future__ import annotations

import json
import re
import shutil
from html import escape
from pathlib import Path

RUN_SLUG = "vjepa2-vitl/L16"
HERO_CLIP = "droid_ep068.mp4"
# The viz grids are 1120x588: two rows of four 266px tiles. Crop inside each, clear of the captions.
GRID_W, GRID_H, TILE = 1120, 588, 254
TILE_X = (17, 294, 572, 849)
TILE_Y = (41, 314)


def load_run(run: Path) -> dict:
    board = json.loads((run / "scoreboard.json").read_text(encoding="utf-8"))
    source = json.loads((run / "SOURCE.json").read_text(encoding="utf-8"))
    features = [_feature(row) for row in board["features"]]
    _tell_apart(features)
    features.sort(key=lambda f: (-f["passed"], f["kind"] != "content", f["id"]))
    return {"board": board, "source": source, "features": features, "dir": run}


def _tell_apart(features: list[dict]) -> None:
    """Two position features with the same short name keep one more part of their label."""
    for feature in features:
        twins = [f for f in features if f["name"] == feature["name"]]
        if len(twins) > 1 and feature["kind"] == "position":
            for twin in twins:
                twin["name"] = display_name(twin["label"], 2)[0]


def _feature(row: dict) -> dict:
    name, kind = display_name(row["label"])
    checks = [
        ("Fires broadly", _bool(row["distinct_clips_top20"] >= 5), f"{row['distinct_clips_top20']} of its top 20 moments are from different clips"),
        ("Name holds up", _bool(row["label_holds_heldout"]), "Fits clips held back while naming" if row["label_holds_heldout"] else "Does not fit clips held back while naming"),
        ("Beats random edits", _bool(row["random_percentile"] >= 95), f"{_ordinal(row['random_percentile'])} percentile of {row['n_random']} matched random features; needs 95th"),
        ("Was learnt", _bool(row["absent_in_random_init"]), _learnt_text(row["absent_in_random_init"])),
    ]
    return {
        "id": int(row["id"]),
        "name": name,
        "kind": kind,
        "label": row["label"],
        "checks": checks,
        "passed": sum(state == "pass" for _, state, _ in checks),
        "real": row["real"] is True,
        "effect": row["ablation_effect"],
        "control_median": row["control_effect_median"],
        "control_max": row["control_effect_max"],
    }


def display_name(label: str, parts_kept: int = 1) -> tuple[str, str]:
    """A short name from a scoreboard label, and whether it is about content or position."""
    text = label.strip()
    kind = "content"
    if text.lower().startswith("positional:"):
        kind, text = "position", text.split(":", 1)[1]
    elif text.lower().startswith("no consistent pattern"):
        kind = "unclear"
    parts = [part.strip() for part in re.sub(r"\s*\([^)]*\)", "", text).split(",")]
    text = ", ".join(parts[:parts_kept])
    text = re.sub(r"\s+found$", "", text)
    text = re.sub(r"\btubelet\b", "frames", re.sub(r"\bh~(\d+)", r"\1", text))
    return text[:1].upper() + text[1:], kind


def _bool(value) -> str:
    if value is None:
        return "pending"
    return "pass" if value else "fail"


def _ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _learnt_text(value) -> str:
    if value is None:
        return "Not settled: the untrained-model comparison is confounded at this layer"
    return "Missing from an untrained copy of the model" if value else "Also fires in an untrained copy of the model"


def copy_media(run: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / HERO_CLIP, dest / HERO_CLIP)
    for png in sorted((run / "features" / "viz").glob("feature_*.png")):
        shutil.copy2(png, dest / png.name)


def tile_style(feature_id: int, index: int) -> str:
    """CSS that shows one tile of a feature's picture grid."""
    x, y = TILE_X[index % 4], TILE_Y[index // 4]
    size = GRID_W / TILE * 100
    px = x / (GRID_W - TILE) * 100
    py = y / (GRID_H - TILE) * 100
    return (
        f"background-image:url(/explore/media/feature_{feature_id:05d}.png);"
        f"background-size:{size:.2f}% auto;background-position:{px:.2f}% {py:.2f}%"
    )


def feature_href(feature_id: int) -> str:
    return f"/explore/{RUN_SLUG}/{feature_id}/"


def evidence(feature: dict) -> str:
    """One dot per check: filled when it holds, hollow when it does not, dashed while unsettled."""
    dots = "".join(f'<span class="dot dot-{state}"></span>' for _, state, _ in feature["checks"])
    return f'<span class="evidence" role="img" aria-label="Evidence: {feature["passed"]} of 4 checks">{dots}</span>'


def lead_feature(run: dict) -> dict | None:
    """The strongest feature about what is in the scene, not where: the one the page opens with."""
    return next((f for f in run["features"] if f["kind"] == "content"), None)


def index_body(run: dict) -> str:
    lead = lead_feature(run)
    rest = [f for f in run["features"] if f is not lead]
    rows = "".join(
        f'<li><a href="{feature_href(f["id"])}"><span class="row-name">{escape(f["name"])}</span>'
        f'<span class="id">#{f["id"]}</span>{evidence(f)}</a></li>'
        for f in rest
    )
    return f"""
    <h1>Inside a robot world model</h1>
    <p class="lede">What V-JEPA 2 picks out as it watches robot arms at work, found with a sparse autoencoder.</p>
    {_lead(lead) if lead else ""}
    <ul class="rows" data-testid="feature-list">{rows}</ul>
    {checks_explainer()}
    {_credit(run)}
    """


def _lead(feature: dict) -> str:
    tiles = "".join(f'<span class="tile" style="{tile_style(feature["id"], i)}"></span>' for i in range(4))
    return f"""
    <a class="lead" href="{feature_href(feature["id"])}" data-testid="lead-feature">
      <span class="moments">{tiles}</span>
      <span class="lead-name"><span class="row-name">{escape(feature["name"])}</span> <span class="id">#{feature["id"]}</span>{evidence(feature)}</span>
    </a>"""


def checks_explainer() -> str:
    return """
    <h2>Evidence</h2>
    <p>Each dot is a check a feature has to clear before it counts as verified.</p>
    <ul class="plain">
      <li><b>Fires broadly.</b> Across many clips, not one scene.</li>
      <li><b>Name holds up.</b> Still fits clips held back while it was named.</li>
      <li><b>Beats random edits.</b> Removing it changes the prediction more than 95% of matched random features.</li>
      <li><b>Was learnt.</b> Missing from an untrained copy of the model.</li>
    </ul>"""


def _credit(run: dict) -> str:
    source = run["source"]
    return (
        f'<p class="fine">Layer {run["board"]["run"]["layer"]} of <a href="https://huggingface.co/{escape(run["board"]["run"]["model"])}">{escape(run["board"]["run"]["model"])}</a>. '
        f'Clips from <a href="{escape(source["url"])}">DROID</a> (CC BY 4.0).</p>'
    )


def feature_body(run: dict, feature: dict) -> str:
    checks = "".join(
        f'<li class="check check-{state}"><span class="mark-{state}" aria-hidden="true"></span>'
        f'<span><b>{escape(name)}</b><span>{escape(detail)}</span></span></li>'
        for name, state, detail in feature["checks"]
    )
    tiles = "".join(f'<span class="tile" style="{tile_style(feature["id"], i)}"></span>' for i in range(8))
    return f"""
    <p class="back"><a href="/explore/">Inside a robot world model</a></p>
    <h1>{escape(feature["name"])} <span class="id">#{feature["id"]}</span></h1>
    <div class="moments" role="img" aria-label="Where it fires most strongly, on eight robot clips">{tiles}</div>
    <h2>Evidence</h2>
    <ul class="checks" data-testid="checks">{checks}</ul>
    <p>Removing it moved the prediction by {_sci(feature["effect"])} (1 &minus; cosine). Matched random features: median {_sci(feature["control_median"])}, largest {_sci(feature["control_max"])}.</p>
    """


def _sci(value: float) -> str:
    mantissa, exp = f"{value:.1e}".split("e")
    return f"{mantissa}&times;10<sup>{int(exp)}</sup>"
