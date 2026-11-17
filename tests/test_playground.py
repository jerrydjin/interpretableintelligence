"""The hosted playground backend: API v1 (docs/PLAYGROUND-API.md), ii serve-live and ii deploy-live."""

from __future__ import annotations

import functools
import http.server
import json
import shutil
import socket
import threading
import time
import urllib.parse

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from test_live import LIVE, REPO, _Quiet, tiny_run  # noqa: F401  (tiny_run is a fixture)
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.playground.compute import PackModel
from interpretableintelligence.playground.deploy import DeployError, find_url, write_config
from interpretableintelligence.playground.http import create_app

OCTET = {"content-type": "application/octet-stream"}


@pytest.fixture(scope="module")
def served(tiny_run):  # noqa: F811
    model = PackModel(tiny_run["pack"], device=torch.device("cpu"), cache_images=4)
    return {**tiny_run, "backend": model, "client": TestClient(create_app({"tiny": model}))}


def _picture(seed: int, size: int = 32) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8).tobytes()


def _reference(served, rgb: bytes, feature: int | None = None, kind: str = "scale", value: float = 3.0):
    """The same read and edit straight from torch, with the pack's SAE weights.

    The pack's fp16 weights, not the run's fp32 ones: a top-k SAE can swap its k-th feature
    between the two on another CPU, which is a precision fact, not a backend bug
    (test_pack_ships_its_sae_for_the_backend checks the weights themselves)."""
    from interpretableintelligence.adapters import get_adapter
    from interpretableintelligence.adapters.hf import make_spec
    from interpretableintelligence.live import split_model
    from safetensors.torch import load_file

    from interpretableintelligence.sae import SAE
    from interpretableintelligence.steer import steer_edit

    pack = json.loads((served["pack"] / "pack.json").read_text(encoding="utf-8"))
    adapter = get_adapter(make_spec(str(served["model"])))(layer=1, device=torch.device("cpu"))
    adapter.load()
    info = pack["sae"]
    sae = SAE(pack["model"]["d_model"], info["n_features"], info["activation"], info.get("k"), info.get("threshold"))
    sae.load_state_dict({k: v.float() for k, v in load_file(str(served["pack"] / "sae.safetensors")).items()})
    sae.eval()
    front, back = split_model(adapter)
    mean = torch.tensor(pack["model"]["mean"]).view(3, 1, 1)
    std = torch.tensor(pack["model"]["std"]).view(3, 1, 1)
    x = torch.tensor(np.frombuffer(rgb, dtype=np.uint8).reshape(32, 32, 3)).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        hidden = front(((x - mean) / std)[None])
        acts = sae.encode(hidden[0, 1:])
        if feature is None:
            return acts, None
        base = back(hidden)[0, 1:]
        edited_patches = steer_edit(sae, [feature], kind, value)(hidden[:, 1:])
        edited = back(torch.cat([hidden[:, :1], edited_patches], dim=1))[0, 1:]
        shift = 1 - torch.nn.functional.cosine_similarity(base, edited, dim=-1)
    return acts, shift


def _dense(body: dict, n_tokens: int, n_features: int) -> np.ndarray:
    patches = body["patches"]
    out = np.zeros((n_tokens, n_features), dtype=np.float32)
    for t in range(n_tokens):
        a, b = patches["offsets"][t], patches["offsets"][t + 1]
        out[t, patches["ids"][a:b]] = patches["values"][a:b]
    return out


def test_health_and_packs(served):
    client = served["client"]
    assert client.get("/v1/health").json() == {"api": "ii-playground", "version": 1, "packs": ["tiny"]}
    info = client.get("/v1/packs/tiny").json()
    assert info["grid"] == {"h": 4, "w": 4} and info["image_size"] == 32 and info["device"] == "cpu"
    assert client.get("/v1/packs").json()["packs"] == [info]
    missing = client.get("/v1/packs/nope")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "pack_not_found"


def test_read_and_steer_match_torch(served):
    client = served["client"]
    rgb = _picture(1)
    body = client.post("/v1/packs/tiny/read", content=rgb, headers=OCTET).json()
    n = served["backend"].n_features
    got = _dense(body, 16, n)
    # Every row strongest first, and the patches' CSR offsets cover the lists.
    patches = body["patches"]
    assert len(patches["offsets"]) == 17 and patches["offsets"][-1] == len(patches["ids"]) == len(patches["values"])
    for t in range(16):
        row = patches["values"][patches["offsets"][t] : patches["offsets"][t + 1]]
        assert row == sorted(row, reverse=True) and all(v > 0 for v in row)
    acts, _ = _reference(served, rgb)
    assert np.allclose(got, acts.numpy(), atol=1e-4 * float(acts.max()) + 1e-5)
    totals = got.sum(axis=0)
    feature = int(totals.argmax())
    assert feature == int(acts.sum(dim=0).argmax())

    steer = client.post("/v1/packs/tiny/steer", json={"image_id": body["image_id"], "features": [feature], "kind": "scale", "value": 3})
    assert steer.status_code == 200, steer.text
    result = steer.json()
    _, shift = _reference(served, rgb, feature, "scale", 3.0)
    assert len(result["shift"]) == 16
    assert np.allclose(result["shift"], shift.numpy(), atol=1e-5 + 1e-3 * float(shift.max()))
    assert result["max"] == pytest.approx(max(result["shift"]), abs=1e-6) and result["max"] > 0

    # The same picture gets the same id: reads are deterministic.
    again = client.post("/v1/packs/tiny/read", content=rgb, headers=OCTET).json()
    assert again["image_id"] == body["image_id"] and again["patches"] == body["patches"]


def test_errors_are_one_shape(served):
    client = served["client"]
    image_id = client.post("/v1/packs/tiny/read", content=_picture(2), headers=OCTET).json()["image_id"]

    def error(response, status, code):
        assert response.status_code == status, response.text
        assert response.json()["error"]["code"] == code
        assert response.json()["error"]["message"]

    error(client.post("/v1/packs/tiny/read", content=b"\0" * 10, headers=OCTET), 400, "bad_request")
    error(client.post("/v1/packs/tiny/read", content=_picture(2), headers={"content-type": "image/png"}), 400, "bad_request")
    error(client.post("/v1/packs/nope/read", content=_picture(2), headers=OCTET), 404, "pack_not_found")
    steer = functools.partial(client.post, "/v1/packs/tiny/steer")
    error(steer(json={"image_id": "0" * 32, "features": [0], "kind": "ablate"}), 404, "unknown_image")
    error(steer(json={"image_id": image_id, "features": [0], "kind": "melt"}), 400, "bad_request")
    error(steer(json={"image_id": image_id, "features": [10**6], "kind": "ablate"}), 400, "bad_request")
    error(steer(json={"image_id": image_id, "features": [], "kind": "ablate"}), 400, "bad_request")
    error(steer(json={"image_id": image_id, "features": [1, 1], "kind": "ablate"}), 400, "bad_request")
    error(steer(json={"image_id": image_id, "features": [1], "kind": "scale", "value": 1e9}), 400, "bad_request")
    error(steer(json={"image_id": image_id, "features": ["1"], "kind": "ablate"}), 400, "bad_request")
    error(steer(json={"image_id": image_id, "features": [1], "kind": "ablate", "extra": 1}), 400, "bad_request")


def test_old_pictures_are_forgotten_and_read_again(served):
    client = served["client"]
    ids = [client.post("/v1/packs/tiny/read", content=_picture(10 + i), headers=OCTET).json()["image_id"] for i in range(5)]
    first = client.post("/v1/packs/tiny/steer", json={"image_id": ids[0], "features": [0], "kind": "ablate"})
    assert first.json()["error"]["code"] == "unknown_image"  # four kept, the oldest went
    last = client.post("/v1/packs/tiny/steer", json={"image_id": ids[-1], "features": [0], "kind": "ablate"})
    assert last.status_code == 200


def test_pack_ships_its_sae_for_the_backend(served):
    from safetensors.torch import load_file

    from interpretableintelligence.sae import load_sae

    pack = json.loads((served["pack"] / "pack.json").read_text(encoding="utf-8"))
    assert pack["files"]["sae_weights"] == "sae.safetensors"
    assert set(pack["digests"]) == set(pack["files"])
    stored = load_file(str(served["pack"] / "sae.safetensors"))
    sae, _ = load_sae(served["run"] / "sae", torch.device("cpu"))
    for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
        assert torch.allclose(stored[name].float(), getattr(sae, name).float(), atol=1e-3)


def test_the_published_pack_has_its_sae():
    pack_dir = LIVE / "packs" / "dinov2-small-droid"
    pack = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    assert (pack_dir / pack["files"]["sae_weights"]).is_file()
    assert json.loads((LIVE / "config.json").read_text(encoding="utf-8")).keys() == {"api"}


def test_deploy_live_writes_the_page_config(tmp_path):
    assert find_url("✓ Created web endpoint => https://someone--ii-playground-playground-web.modal.run\n") == (
        "https://someone--ii-playground-playground-web.modal.run"
    )
    assert find_url("no url here") is None
    with pytest.raises(DeployError):
        write_config(tmp_path, "javascript:alert(1)")
    runner = CliRunner()
    result = runner.invoke(app, ["deploy-live", "--live-dir", str(tmp_path), "--api", "https://x--y.modal.run/"])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "config.json").read_text()) == {"api": "https://x--y.modal.run"}
    result = runner.invoke(app, ["deploy-live", "--live-dir", str(tmp_path), "--api", "none"])
    assert result.exit_code == 0 and json.loads((tmp_path / "config.json").read_text()) == {"api": None}


def test_modal_image_carries_only_what_the_server_needs(tmp_path):
    pytest.importorskip("modal", reason="the Modal client is not installed (uv sync --extra remote)")
    from interpretableintelligence.remote import playground_app

    staged = playground_app._stage(LIVE / "packs")
    try:
        files = sorted(p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file())
    finally:
        shutil.rmtree(staged)
    assert files == ["dinov2-small-droid/pack.json", "dinov2-small-droid/sae.safetensors"]
    assert playground_app.app.name == "ii-playground"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_page_runs_on_the_backend(served, tmp_path):
    """The page with ?api= reads and edits through the backend (CORS, CSR, retry after eviction)."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import uvicorn
    from test_steer import _chrome

    chrome = _chrome()
    if chrome is None:
        pytest.skip("Chrome is not installed; the live playground test cannot run")

    site = tmp_path / "site"
    shutil.copytree(LIVE, site / "live", ignore=shutil.ignore_patterns("packs"))
    shutil.copytree(served["pack"], site / "live" / "packs" / "tiny")
    shutil.copytree(REPO / "demo-site" / "static" / "fonts", site / "static" / "fonts")
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(site)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    model = PackModel(served["pack"], device=torch.device("cpu"), cache_images=2)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app({"tiny": model}), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    rgb = _picture(3)
    expected = served["client"].post("/v1/packs/tiny/read", content=rgb, headers=OCTET).json()
    feature = int(_dense(expected, 16, model.n_features).sum(axis=0).argmax())
    direct = served["client"].post(
        "/v1/packs/tiny/steer", json={"image_id": expected["image_id"], "features": [feature], "kind": "scale", "value": 3}
    ).json()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/live/?pack=packs/tiny/&api=http://127.0.0.1:{port}"
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(url)
            page.wait_for_selector("#work:not([hidden])", timeout=60_000)
            page.wait_for_selector(".frow")
            assert "in the cloud" in page.text_content("#model-pill")
            got = page.evaluate(
                """async ([rgb, feature]) => {
                  const e = window.__ii.engine;
                  const state = await e.encodeRgb(Uint8Array.from(rgb));
                  const first = await e.edit(state, [feature], 'scale', 3.0);
                  // Push the picture out of the server's two-picture memory, then edit again.
                  for (const seed of [7, 8, 9]) await e.encodeRgb(new Uint8Array(32 * 32 * 3).fill(seed * 20));
                  const second = await e.edit(state, [feature], 'scale', 3.0);
                  // Edits made while one is in flight collapse into the latest.
                  const burst = await Promise.all([1, 2, 3].map((v) => e.edit(state, [feature], 'scale', v)));
                  return { backend: e.backend, imageId: state.imageId, totals: Array.from(state.totals),
                           first: Array.from(first.shift), second: Array.from(second.shift),
                           burst: burst.map((r) => r.mean) };
                }""",
                [list(rgb), feature],
            )
            page.click(".frow")
            page.wait_for_selector("[data-kind=ablate]")
            page.click("[data-kind=ablate]")
            page.wait_for_function("() => window.__ii.ui.result !== null")
            # The address bar holds the picture, feature and edit, and a fresh page opened on it
            # comes back to the same edit with the same result.
            sample = json.loads((served["pack"] / "pack.json").read_text(encoding="utf-8"))["samples"][1]["clip_id"]
            page.click(f".thumb[data-clip='{sample}']")
            page.click("[data-kind=scale]")
            page.wait_for_function("() => window.__ii.ui.kind === 'scale' && window.__ii.ui.result !== null")
            page.wait_for_function(f"() => location.hash.includes('p={sample}') && location.hash.includes('e=scale')")
            before = page.evaluate("window.__ii.ui.result.mean")
            again = browser.new_page()
            again.on("pageerror", lambda exc: errors.append(str(exc)))
            again.goto(page.url)
            again.wait_for_function("() => window.__ii && window.__ii.ui.result !== null", timeout=60_000)
            restored = again.evaluate("({f: window.__ii.ui.feature, k: window.__ii.ui.kind, v: window.__ii.ui.value.scale, m: window.__ii.ui.result.mean})")
            assert restored["k"] == "scale" and restored["v"] == 2 and restored["m"] == pytest.approx(before, abs=1e-9)
            # Your own picture travels inside the link.
            again.set_input_files("#file", str(served["pack"] / "samples" / f"{sample}.jpg"))
            again.wait_for_function(
                "() => { const ui = window.__ii.ui; return ui.picture && ui.picture.img && ui.result !== null && !ui.busy"
                " && !location.hash.includes('p=') && location.hash.includes('e=scale'); }"
            )
            link = again.evaluate("window.__ii.link({ withPicture: true })")
            params = urllib.parse.parse_qs(link)
            assert "img" in params and "p" not in params and params["e"] == ["scale"]
            third = browser.new_page()
            third.goto(f"{url}#{link}")
            third.wait_for_function("() => window.__ii && window.__ii.ui.result !== null", timeout=60_000)
            assert third.evaluate("window.__ii.ui.picture.img != null && window.__ii.ui.kind") == "scale"
            assert errors == []
            browser.close()
    finally:
        server.should_exit = True
        httpd.shutdown()
        httpd.server_close()
    assert got["backend"] == "remote" and got["imageId"] == expected["image_id"]
    assert np.allclose(got["totals"], _dense(expected, 16, model.n_features).sum(axis=0), atol=1e-4)
    assert np.allclose(got["first"], direct["shift"], atol=1e-6)
    assert np.allclose(got["second"], direct["shift"], atol=1e-6)
    assert got["burst"][1] == got["burst"][2]  # the second and third collapsed into one request
