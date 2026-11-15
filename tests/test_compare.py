"""ii compare: matching SAE features across runs on the same clips."""

from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.compare import compare_runs, match_pair, render_html


def test_match_pair_recovers_planted_partners():
    rng = np.random.default_rng(0)
    n_clips = 80
    shared = rng.gamma(1.0, 1.0, size=(n_clips, 3)) * (rng.random((n_clips, 3)) < 0.4)
    a = np.concatenate([shared, rng.gamma(1.0, 1.0, size=(n_clips, 20)) * (rng.random((n_clips, 20)) < 0.3)], axis=1)
    # b holds the same three profiles at other ids, with noise, among its own unrelated features.
    b = rng.gamma(1.0, 1.0, size=(n_clips, 30)) * (rng.random((n_clips, 30)) < 0.3)
    planted = {0: 17, 1: 4, 2: 25}
    for i, j in planted.items():
        b[:, j] = shared[:, i] + 0.05 * rng.random(n_clips)
    result = match_pair(a, b, seed=0, n_perm=50)
    found = {match["a"]: match["b"] for match in result["matches"]}
    assert found == planted
    assert all(match["r"] > result["threshold"] for match in result["matches"])
    assert 0.0 < result["threshold"] < 0.9


def test_match_pair_finds_nothing_in_noise():
    rng = np.random.default_rng(1)
    a = rng.gamma(1.0, 1.0, size=(60, 15)) * (rng.random((60, 15)) < 0.4)
    b = rng.gamma(1.0, 1.0, size=(60, 15)) * (rng.random((60, 15)) < 0.4)
    assert len(match_pair(a, b, seed=0, n_perm=100)["matches"]) <= 1


def test_a_run_matches_itself(demo_run, tmp_path):
    report = compare_runs([demo_run, demo_run], n_perm=20, log=lambda *_: None)
    assert report["n_shared_clips"] >= 20
    for group in report["shared"]:
        ids = {card["feature"] for card in group["features"]}
        assert len(ids) == 1
        assert group["min_r"] > 0.999
    verified = {card["label"] for group in report["shared"] for card in group["features"] if card["verified"]}
    assert verified == {"stripe", "pulse", "corner"}
    assert report["unmatched_verified"] == []
    public = {key: value for key, value in report.items() if not key.startswith("_")}
    page = render_html(public)
    assert "<title>What robot models share</title>" in page
    assert "verified" in page


def test_cli_writes_report(demo_run, tmp_path):
    out = tmp_path / "cmp"
    result = CliRunner().invoke(app, ["compare", str(demo_run), str(demo_run), "--out", str(out)])
    assert result.exit_code == 0, result.output
    document = json.loads((out / "compare.json").read_text())
    assert document["compare_version"] == 1
    assert not any(key.startswith("_") for key in document)
    assert (out / "index.html").is_file()
