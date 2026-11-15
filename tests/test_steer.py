"""Playground steering: the edit rule, the Steerer on the fake acceptance run, and the viewer route."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from interpretableintelligence.sae import SAE
from interpretableintelligence.score import ablate_edit
from interpretableintelligence.steer import SteerError, Steerer, SteerRequest, steer_edit
from interpretableintelligence.view import bind_server, enable_steering, load_run


def _sae(d_model: int = 6, n_features: int = 4, seed: int = 0) -> SAE:
    torch.manual_seed(seed)
    sae = SAE(d_model, n_features)
    with torch.no_grad():
        sae.W_enc.normal_()
        sae.b_enc.fill_(0.1)
        sae.W_dec.copy_(torch.nn.functional.normalize(torch.randn(d_model, n_features), dim=0))
        sae.b_dec.zero_()
    return sae


def test_ablate_matches_the_scorers_single_feature_ablation():
    sae = _sae()
    acts = torch.randn(2, 3, 3, 6)
    with torch.no_grad():
        ours = steer_edit(sae, [1], "ablate", 0.0)(acts)
        scorer = ablate_edit(sae, 1)(acts)
    assert torch.allclose(ours, scorer, atol=1e-6)


@pytest.mark.parametrize(("kind", "value"), [("scale", 1.0), ("add", 0.0)])
def test_identity_edits_leave_activations_unchanged(kind: str, value: float):
    sae = _sae()
    acts = torch.randn(4, 6)
    with torch.no_grad():
        edited = steer_edit(sae, [0, 2], kind, value)(acts)
    assert torch.allclose(edited, acts, atol=1e-6)


def test_clamp_moves_along_the_decoder_column_and_keeps_the_rest():
    sae = _sae()
    acts = torch.randn(5, 6)
    with torch.no_grad():
        before = sae.encode(acts)[:, 3]
        edited = steer_edit(sae, [3], "clamp", 2.5)(acts)
    expected = acts + (2.5 - before).unsqueeze(1) * sae.W_dec[:, 3]
    assert torch.allclose(edited, expected, atol=1e-6)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"features": [0]},
        {"clip_id": "c", "features": []},
        {"clip_id": "c", "features": [99]},
        {"clip_id": "c", "features": [0], "kind": "explode"},
        {"clip_id": "c", "features": [0], "kind": "scale", "value": "big"},
        {"clip_id": "c", "features": [0], "kind": "scale", "value": 1e9},
        {"clip_id": "c", "features": [0], "window": -1},
        {"clip_id": "c", "features": list(range(9))},
    ],
)
def test_bad_requests_are_refused(payload):
    with pytest.raises(SteerError):
        SteerRequest.parse(payload, n_features=12)


def _real_ids(run) -> list[int]:
    scoreboard = json.loads((run / "scoreboard.json").read_text(encoding="utf-8"))
    return [feature["id"] for feature in scoreboard["features"] if feature["real"]]


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_steering_a_planted_feature_moves_the_prediction_and_writes_nothing(demo):
    before = _digest(demo.run / "scoreboard.json")
    steerer = Steerer(demo.run, demo.clips)
    clip_id = steerer.ctx.manifest["clips"][0]["clip_id"]
    feature = _real_ids(demo.run)[0]

    unchanged = steerer.steer({"clip_id": clip_id, "features": [feature], "kind": "scale", "value": 1.0})
    clamped = steerer.steer({"clip_id": clip_id, "features": [feature], "kind": "clamp", "value": 2.0})

    assert unchanged["summary"]["mean_shift"] < 1e-5
    assert clamped["summary"]["mean_shift"] > 1e-3
    assert clamped["summary"]["loss_change"] > 0
    assert clamped["evidence"] is False
    grid = clamped["grid"]
    assert len(clamped["shift"]) == grid["t"]
    assert len(clamped["shift"][0]) == grid["h"]
    assert len(clamped["shift"][0][0]) == grid["w"]
    assert any(value is not None and value > 0 for plane in clamped["shift"] for row in plane for value in row)
    assert _digest(demo.run / "scoreboard.json") == before


def test_unknown_clip_and_window_are_refused(demo):
    steerer = Steerer(demo.run, demo.clips)
    with pytest.raises(SteerError):
        steerer.steer({"clip_id": "no-such-clip", "features": [0]})
    clip_id = steerer.ctx.manifest["clips"][0]["clip_id"]
    with pytest.raises(SteerError):
        steerer.steer({"clip_id": clip_id, "window": 50, "features": [0]})


def _post(url: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _serve(model):
    httpd = bind_server(model, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    return httpd, f"http://{host}:{port}"


def test_viewer_route_is_off_by_default_and_answers_when_enabled(demo):
    model = load_run(demo.run, demo.clips)
    clip_id = model.run_payload["clips"][0]["clip_id"]
    feature = _real_ids(demo.run)[0]
    body = {"clip_id": clip_id, "features": [feature], "kind": "clamp", "value": 2.0}

    httpd, url = _serve(model)
    try:
        status, payload = _post(url + "/api/steer", body)
        assert status == 404
        assert "--steer" in payload["error"]
        assert "steer" not in model.run_payload

        enable_steering(model, demo.clips)
        with urllib.request.urlopen(url + "/api/run", timeout=30) as response:
            assert json.loads(response.read())["steer"]["enabled"] is True
        status, payload = _post(url + "/api/steer", body)
        assert status == 200
        assert payload["summary"]["mean_shift"] > 1e-3
        status, payload = _post(url + "/api/steer", {**body, "features": [999]})
        assert status == 400
        assert "unknown feature" in payload["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _chrome() -> str | None:
    """A Chrome or Chromium binary for the browser tests, or ``None`` (the test then skips).

    ``II_CHROME`` wins. Otherwise: an installed Chrome or Chromium (macOS or Linux), then a
    Chromium that ``playwright install chromium`` downloaded (``PLAYWRIGHT_BROWSERS_PATH``, else
    Playwright's default cache directory).
    """
    import os
    import sys
    from pathlib import Path

    forced = os.environ.get("II_CHROME")
    if forced:
        return forced if Path(forced).is_file() else None
    for candidate in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/local/bin/google-chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ):
        if Path(candidate).is_file():
            return candidate
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH"), "/opt/pw-browsers"]
    if sys.platform == "darwin":
        roots.append(str(Path.home() / "Library" / "Caches" / "ms-playwright"))
    else:
        roots.append(str(Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ms-playwright"))
    patterns = (
        "chromium-*/chrome-linux*/chrome",
        "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium",
        "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        "chromium/chrome-linux*/chrome",
        "chrome-linux*/chrome",
    )
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        for pattern in patterns:
            found = sorted(Path(root).glob(pattern))
            if found:
                return str(found[-1])
    return None


def test_playwright_playground_steers_and_draws_the_change(demo):
    sync_api = pytest.importorskip("playwright.sync_api")
    chrome = _chrome()
    if chrome is None:
        pytest.skip("Chrome is not installed; the playground browser test cannot run")
    model = load_run(demo.run, demo.clips)
    enable_steering(model, demo.clips)
    clip_id = model.run_payload["clips"][0]["clip_id"]
    feature = _real_ids(demo.run)[0]
    httpd, url = _serve(model)
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(f"{url}/#/clips/{clip_id}?frame=0&feature={feature}", wait_until="domcontentloaded")
            page.wait_for_selector("[data-testid=steer-panel]")
            page.select_option("[data-testid=steer-kind]", "clamp")
            page.click("[data-testid=steer-run]")
            page.wait_for_selector("[data-testid=steer-mean]")
            assert page.get_attribute("[data-testid=heatmap]", "data-steer") == "true"
            assert "prediction change" in page.text_content("[data-testid=heatmap-label]")
            assert float(page.get_attribute("[data-testid=steer-result]", "data-mean-shift")) > 1e-3
            page.uncheck("[data-testid=steer-show]")
            assert "prediction change" not in page.text_content("[data-testid=heatmap-label]")
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
