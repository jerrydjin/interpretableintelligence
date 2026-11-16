"""`ii export`: a run as a static site with a precomputed playground."""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from test_steer import _chrome, _real_ids
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.export_run import STEER_PRESETS, export_run, steer_key
from interpretableintelligence.steer import Steerer
from interpretableintelligence.view import load_run

runner = CliRunner()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def site(demo, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("export") / "site"
    before = _digest(demo.run / "scoreboard.json")
    result = runner.invoke(app, ["export", str(demo.run), "--clips", str(demo.clips), "--out", str(out), "--steer-clips", "3", "--steer-features", "3"])
    assert result.exit_code == 0, result.output
    assert "playground on 3 clips" in result.output
    assert _digest(demo.run / "scoreboard.json") == before
    return out


def test_steer_key_prints_numbers_like_javascript():
    assert steer_key(0, 9, "scale", 3.0) == "0:9:scale:3"
    assert steer_key(1, 2, "scale", 0.5) == "1:2:scale:0.5"
    assert steer_key(0, 2, "add", -1.0) == "0:2:add:-1"


def test_every_preset_holds_the_viewers_default():
    app_js = (Path(__file__).resolve().parents[1] / "src" / "interpretableintelligence" / "static" / "app.js").read_text(encoding="utf-8")
    for kind, presets in STEER_PRESETS.items():
        default = float(app_js.split(f"  {kind}: {{", 1)[1].split("value: ", 1)[1].split(" ", 1)[0])
        assert default in presets, kind


def test_export_is_static_and_relative(site: Path):
    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'data-ii-root="data"' in html
    assert 'src="static/app.js"' in html
    assert "/static/" not in html and "/api/" not in html
    assert 'property="og:title"' in html
    css = (site / "static" / "style.css").read_text(encoding="utf-8")
    assert "/static/fonts" not in css
    assert (site / "static" / "fonts" / "geist-400.woff2").is_file()
    run = json.loads((site / "data" / "run.json").read_text(encoding="utf-8"))
    for clip in run["clips"]:
        assert (site / "data" / "clips" / f"{clip['clip_id']}.json").is_file()
    for feature in run["features"]:
        assert (site / "data" / "features" / f"{feature['id']}.json").is_file()


def test_precomputed_edits_match_the_live_playground(demo, site: Path):
    run = json.loads((site / "data" / "run.json").read_text(encoding="utf-8"))
    steer = run["steer"]
    assert steer["enabled"] is True and steer["static"] is True
    assert len(steer["clips"]) == 3
    clip_id = steer["clips"][0]
    pack = json.loads((site / "data" / "steer" / f"{clip_id}.json").read_text(encoding="utf-8"))
    key = next(key for key in pack if key.endswith(":clamp:3"))
    window, feature, _kind, _value = key.split(":")
    stored = pack[key]
    assert "activation" not in stored and stored["evidence"] is False
    live = Steerer(demo.run, demo.clips).steer({"clip_id": clip_id, "window": int(window), "features": [int(feature)], "kind": "clamp", "value": 3.0})
    assert stored["summary"] == pytest.approx(live["summary"], rel=1e-5, abs=1e-9)


def test_real_features_top_clips_are_precomputed(demo, site: Path):
    run = json.loads((site / "data" / "run.json").read_text(encoding="utf-8"))
    for feature_id in _real_ids(demo.run):
        detail = json.loads((site / "data" / "features" / f"{feature_id}.json").read_text(encoding="utf-8"))
        assert detail["top_clips"][0]["clip_id"] in run["steer"]["clips"]


def test_refuses_to_overwrite_a_folder_it_did_not_write(demo, tmp_path: Path):
    out = tmp_path / "mine"
    out.mkdir()
    (out / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError):
        export_run(demo.run, out, clips_dir=demo.clips, steer=False)
    assert (out / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_no_steer_leaves_the_playground_off(demo, tmp_path: Path):
    summary = export_run(demo.run, tmp_path / "plain", clips_dir=demo.clips, steer=False)
    run = json.loads((tmp_path / "plain" / "data" / "run.json").read_text(encoding="utf-8"))
    assert "steer" not in run and summary.steer_clips == []
    assert not (tmp_path / "plain" / "data" / "steer").exists()


def test_playwright_static_playground_under_a_subpath(demo, site: Path):
    sync_api = pytest.importorskip("playwright.sync_api")
    chrome = _chrome()
    if chrome is None:
        pytest.skip("Chrome is not installed; the static playground test cannot run")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site.parent))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/{site.name}/"
    feature = _real_ids(demo.run)[0]
    detail = json.loads((site / "data" / "features" / f"{feature}.json").read_text(encoding="utf-8"))
    top = detail["top_clips"][0]
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            failed: list[str] = []
            page.on("response", lambda response: failed.append(response.url) if response.status >= 400 else None)
            page.goto(f"{base}#/clips/{top['clip_id']}?frame={top['frame']}&feature={feature}", wait_until="domcontentloaded")
            page.wait_for_selector("[data-testid=steer-panel]")
            page.select_option("[data-testid=steer-kind]", "clamp")
            page.wait_for_selector("[data-testid=steer-mean]")
            assert page.get_attribute("[data-testid=heatmap]", "data-steer") == "true"
            assert float(page.get_attribute("[data-testid=steer-result]", "data-mean-shift")) > 1e-3
            page.eval_on_selector("[data-testid=steer-value]", "el => { el.value = '0'; el.dispatchEvent(new Event('input', { bubbles: true })); }")
            page.wait_for_function("document.querySelector('[data-testid=steer-readout]').textContent === '-1'")
            assert failed == []
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_heat_pack_holds_the_same_cells_as_the_live_heatmap(demo, site: Path):
    model = load_run(demo.run, demo.clips)
    run = json.loads((site / "data" / "run.json").read_text(encoding="utf-8"))
    clip_id = run["clips"][0]["clip_id"]
    pack = json.loads((site / "data" / "clips" / clip_id / "heat.json").read_text(encoding="utf-8"))
    height, width = pack["shape"]
    assert pack["cells"]
    expected = 0
    for frame in range(run["clips"][0]["n_frames"]):
        for feature in run["features"]:
            live = model.heatmap(clip_id, frame, feature["id"])
            cells = pack["cells"].get(f"{frame}:{feature['id']}")
            if not live or not any(value > 0 for row in live for value in row):
                assert cells is None
                continue
            expected += 1
            grid = [[0.0] * width for _ in range(height)]
            for h, w, value in zip(cells[0::3], cells[1::3], cells[2::3]):
                grid[h][w] = value
            assert grid == pytest.approx(np.array(live), rel=1e-3)
    assert expected == len(pack["cells"])


def test_max_clips_keeps_the_verified_examples_and_says_so(demo, tmp_path: Path):
    out = tmp_path / "small"
    summary = export_run(demo.run, out, clips_dir=demo.clips, steer=False, max_clips=4)
    run = json.loads((out / "data" / "run.json").read_text(encoding="utf-8"))
    kept = {clip["clip_id"] for clip in run["clips"]}
    assert summary.clips == len(kept) == 4
    assert run["export"]["clips_shown"] == 4 and run["export"]["clips_total"] > 4
    assert {path.name for path in (out / "data" / "clips").iterdir() if path.is_dir()} == kept
    full = load_run(demo.run, demo.clips)
    for feature_id in _real_ids(demo.run):
        detail = json.loads((out / "data" / "features" / f"{feature_id}.json").read_text(encoding="utf-8"))
        assert detail["top_clips"], "a verified feature lost every example"
        assert {top["clip_id"] for top in detail["top_clips"]} <= kept
        assert detail["top_clips"][0]["clip_id"] == full.feature(feature_id)["top_clips"][0]["clip_id"]
