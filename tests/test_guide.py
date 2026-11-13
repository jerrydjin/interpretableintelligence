"""The viewer's plain-English gate guide (static/guide.js)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from interpretableintelligence import thresholds
from interpretableintelligence.view import STATIC_DIR

GUIDE = STATIC_DIR / "guide.js"


def test_guide_thresholds_match_the_scorer():
    text = GUIDE.read_text(encoding="utf-8")
    block = re.search(r"const THRESHOLDS = \{(.*?)\};", text, re.S)
    assert block, "THRESHOLDS block missing from guide.js"
    found = {name: float(value) for name, value in re.findall(r"(\w+): ([\d.]+)", block.group(1))}
    assert found, "no thresholds parsed"
    for name, value in found.items():
        assert float(getattr(thresholds, name)) == value, name


def _run_guide(script: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    program = f"const g = require({json.dumps(str(GUIDE))});\n{script}"
    out = subprocess.run([node, "-e", program], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


FAILING = {
    "id": 7,
    "label": "stripe",
    "badge": "status",
    "status": "failed: random_percentile",
    "scored": True,
    "distinct_clips_top20": 2,
    "label_holds_heldout": False,
    "ablation_effect": 2.3e-8,
    "random_percentile": 10.0,
    "n_random": 100,
    "absent_in_random_init": False,
}

REAL = {
    "id": 0,
    "label": "stripe",
    "badge": "real",
    "scored": True,
    "distinct_clips_top20": 20,
    "label_holds_heldout": True,
    "ablation_effect": 3e-4,
    "random_percentile": 100.0,
    "n_random": 100,
    "absent_in_random_init": True,
}


def test_every_failed_or_skipped_gate_suggests_a_next_experiment():
    unscored = {"id": 3, "label": "", "badge": "unverified", "scored": True}
    unusable = {"usable": False, "loss_recovered": 0.41}
    out = _run_guide(
        "console.log(JSON.stringify(["
        f"g.feature({json.dumps(FAILING)}, {{ sae: {json.dumps(unusable)}, steerHref: '#/clips/1?frame=2&feature=7' }}),"
        f"g.feature({json.dumps(unscored)}, {{}}),"
        "]));"
    )
    for read in out:
        assert read["real"] is False
        assert read["verdict"].startswith("Exploratory")
        for gate in read["gates"]:
            assert gate["passed"] is not True
            assert gate["next"] and gate["next"]["text"], gate["key"]
            assert gate["title"] and gate["result"] and gate["asks"]
    failing = {gate["key"]: gate for gate in out[0]["gates"]}
    assert set(failing) == {"usable", "distinct_clips_top20", "label_holds_heldout", "random_percentile", "absent_in_random_init"}
    assert "41%" in failing["usable"]["result"]
    assert "only 2 clips" in failing["distinct_clips_top20"]["result"]
    assert "10% of 100" in failing["random_percentile"]["result"]
    assert "--extra-effects group" in failing["random_percentile"]["next"]["text"]
    assert failing["random_percentile"]["next"]["href"] == "#/clips/1?frame=2&feature=7"
    skipped = {gate["key"]: gate for gate in out[1]["gates"]}
    assert all(gate["passed"] is None for gate in skipped.values())
    assert "--prefilter" in skipped["random_percentile"]["result"]


def test_real_feature_is_verified_and_has_no_next_steps():
    out = _run_guide(
        f"console.log(JSON.stringify(g.feature({json.dumps(REAL)}, {{ sae: {{ usable: true, loss_recovered: 0.97 }} }})));"
    )
    assert out["real"] is True
    assert out["verdict"].startswith("Verified")
    assert all(gate["passed"] is True and gate["next"] is None for gate in out["gates"])
    assert "97%" in out["gates"][0]["result"]


def test_guide_never_marks_a_passing_record_real_without_the_badge():
    # Every gate passes, but the scorer did not mark it real: the guide must not either.
    record = dict(REAL, badge="unverified")
    out = _run_guide(f"console.log(JSON.stringify(g.feature({json.dumps(record)}, {{}})));")
    assert out["real"] is False
    assert not out["verdict"].startswith("Verified")


def test_status_notes_and_index_summary():
    out = _run_guide(
        "console.log(JSON.stringify({"
        "pending: g.statusNote('ablation-passed, control pending'),"
        "estimate: g.statusNote('estimate (ablation incomplete)'),"
        "failed: g.statusNote('failed: random_percentile'),"
        "none: g.statusNote(null),"
        f"summary: g.summary([{json.dumps(FAILING)}, {json.dumps(REAL)}, {json.dumps(dict(FAILING, id=8, distinct_clips_top20=9))}]),"
        "}));"
    )
    assert "still running" in out["pending"]
    assert "estimate" in out["estimate"]
    assert out["failed"] is None and out["none"] is None
    lines = {line["key"]: line["count"] for line in out["summary"]["lines"]}
    assert lines == {"random_percentile": 2, "label_holds_heldout": 2, "absent_in_random_init": 2, "distinct_clips_top20": 1}
    assert out["summary"]["scored"] == 3
