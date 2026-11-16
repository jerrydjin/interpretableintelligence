"""`ii export`: a run as a static site with a precomputed playground."""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest
from test_steer import _chrome, _real_ids
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.export_run import STEER_PRESETS, export_run, steer_key
from interpretableintelligence.steer import Steerer

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


def test_frame_heatmaps_match_the_per_feature_heatmap(demo):
    """The export reads one clip's cells in one query; the grids are the ones ``heatmap`` serves."""
    from interpretableintelligence.view import load_run

    model = load_run(demo.run, demo.clips)
    ids = [int(feature["id"]) for feature in model.run_payload["features"]]
    checked = 0
    for clip in model.run_payload["clips"][:4]:
        clip_id = str(clip["clip_id"])
        maps = model.frame_heatmaps(clip_id, ids)
        assert len(maps) == int(model.clips[clip_id]["n_frames"])
        for frame, by_feature in enumerate(maps):
            expected = {}
            for feature_id in ids:
                grid = model.heatmap(clip_id, frame, feature_id)
                if grid and any(value > 0 for row in grid for value in row):
                    expected[feature_id] = grid
            assert list(by_feature) == list(expected)
            assert by_feature == expected
            checked += len(expected)
    assert checked > 0


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
