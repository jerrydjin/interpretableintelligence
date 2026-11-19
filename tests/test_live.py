"""``ii export-live`` and the live in-browser playground (demo-site/live/)."""

from __future__ import annotations

import functools
import http.server
import json
import re
import shutil
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from interpretableintelligence.cli import app

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "demo-site" / "live"
ORT_PROBE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.min.mjs"


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A tiny ViT run stopped before scoring, plus its pack."""
    from transformers import AutoModel, ViTConfig

    from interpretableintelligence.adapters.vjepa2 import generate_tiny_clips

    root = tmp_path_factory.mktemp("live")
    torch.manual_seed(0)
    model_dir = root / "vit"
    AutoModel.from_config(
        ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    ).save_pretrained(model_dir)
    clips = root / "clips"
    generate_tiny_clips(clips, 20)
    run = root / "run"
    result = CliRunner().invoke(
        app, ["run", "hf", "--model", str(model_dir), "--clips", str(clips), "--out", str(run), "--no-score", "--no-serve"]
    )
    assert result.exit_code == 0, result.output
    pack = root / "pack"
    result = CliRunner().invoke(app, ["export-live", str(run), "--out", str(pack), "--clips", str(clips), "--samples", "3"])
    assert result.exit_code == 0, result.output
    return {"root": root, "model": model_dir, "clips": clips, "run": run, "pack": pack}


def test_pack_layout(tiny_run):
    pack_dir = tiny_run["pack"]
    pack = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    assert pack["format"] == "ii-live-pack" and pack["version"] == 1
    assert pack["evidence"] is False
    model = pack["model"]
    assert model["grid"] == {"h": 4, "w": 4} and model["prefix_tokens"] == 1 and model["d_model"] == 32
    assert model["layer"] == 1 and model["image_size"] == 32
    for name in pack["files"].values():
        assert (pack_dir / name).is_file(), name
    n = pack["sae"]["n_features"]
    assert (pack_dir / "decoder.bin").stat().st_size == n * model["d_model"] * 2
    features = json.loads((pack_dir / "features.json").read_text(encoding="utf-8"))
    assert [f["id"] for f in features] == list(range(n))
    # An unscored run has nothing verified, and the pack never invents a badge.
    assert not any(f.get("real") for f in features) and pack["n_real"] == 0
    assert all(f["max"] >= 0 and 0 <= f["freq"] <= 1 for f in features)
    assert len(pack["samples"]) == 3
    for sample in pack["samples"]:
        data = (pack_dir / sample["file"]).read_bytes()
        assert data[:2] == b"\xff\xd8"  # a JPEG


def test_split_model_reproduces_the_full_forward(tiny_run):
    from interpretableintelligence.adapters import get_adapter
    from interpretableintelligence.adapters.hf import make_spec
    from interpretableintelligence.live import split_model

    adapter = get_adapter(make_spec(str(tiny_run["model"])))(layer=1, device=torch.device("cpu"))
    adapter.load()
    front, back = split_model(adapter)
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        full = adapter.model(pixel_values=pixels, output_hidden_states=True)
        hidden = front(pixels)
        assert torch.allclose(hidden, full.hidden_states[1], atol=1e-5)
        assert torch.allclose(back(hidden), full.last_hidden_state, atol=1e-5)


def test_onnx_pack_matches_torch(tiny_run):
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed; the ONNX parity check needs it")
    from interpretableintelligence.adapters import get_adapter
    from interpretableintelligence.adapters.hf import make_spec
    from interpretableintelligence.sae import load_sae

    pack = tiny_run["pack"]
    adapter = get_adapter(make_spec(str(tiny_run["model"])))(layer=1, device=torch.device("cpu"))
    adapter.load()
    sae, _config = load_sae(tiny_run["run"] / "sae", torch.device("cpu"))
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        full = adapter.model(pixel_values=pixels, output_hidden_states=True)
        hidden = full.hidden_states[1]
        pre = torch.relu((hidden[0, 1:] - sae.b_dec) @ sae.W_enc.T + sae.b_enc)
    got_hidden = ort.InferenceSession(str(pack / "front.onnx")).run(None, {"pixel_values": pixels.numpy()})[0]
    got_final = ort.InferenceSession(str(pack / "back.onnx")).run(None, {"hidden": hidden.numpy()})[0]
    got_pre = ort.InferenceSession(str(pack / "sae.onnx")).run(None, {"x": hidden[0, 1:].numpy()})[0]

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))

    # Weights are stored as float16, so expect agreement to about 1e-3, not bit for bit.
    assert rel(hidden.numpy(), got_hidden) < 1e-2
    assert rel(full.last_hidden_state.numpy(), got_final) < 1e-2
    assert rel(pre.numpy(), got_pre) < 1e-2
    # No float64 anywhere: ONNX Runtime Web's WebGPU build has no float64 kernels (the SDPA
    # export's float64 Sqrt stopped the model loading on GPUs).
    onnx = pytest.importorskip("onnx")
    for name in ("front.onnx", "back.onnx", "sae.onnx"):
        graph = onnx.load(str(pack / name)).graph
        doubles = [node.name for node in graph.node if node.op_type == "Constant" and node.attribute[0].t.data_type == onnx.TensorProto.DOUBLE]
        doubles += [init.name for init in graph.initializer if init.data_type == onnx.TensorProto.DOUBLE]
        assert doubles == [], (name, doubles[:3])
    decoder = np.frombuffer((pack / "decoder.bin").read_bytes(), dtype="<f2").reshape(sae.n_features, -1)
    assert np.allclose(decoder, sae.W_dec.detach().T.numpy(), atol=1e-3)


def test_export_live_refuses_a_non_hf_run(tmp_path):
    from interpretableintelligence.adapters.fake import generate_clips
    from interpretableintelligence.pipeline import execute_pipeline

    clips = tmp_path / "clips"
    generate_clips(clips, n=20, seed=0)
    run = tmp_path / "run"
    execute_pipeline(adapter_name="fake", clips_dir=clips, out_dir=run, layer=16, seed=0, heldout_frac=0.2, score=False, serve=False, open_browser=False, log=lambda _m: None)
    result = CliRunner().invoke(app, ["export-live", str(run), "--out", str(tmp_path / "pack")])
    assert result.exit_code == 1
    assert "Hugging Face image models" in result.output


def test_badges_from_an_export_need_the_same_dictionary(tiny_run, tmp_path):
    """--meta-from copies labels and badges only when this SAE reproduces the export's top activations."""
    from safetensors.numpy import load_file

    from interpretableintelligence.live import LiveExportError, verify_against_export
    from interpretableintelligence.sae import load_sae

    run = tiny_run["run"]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    sae, _config = load_sae(run / "sae", torch.device("cpu"))
    shard = load_file(str(sorted((run / "acts").glob("shard-*.safetensors"))[0]))
    row = 0
    with torch.no_grad():
        acts = sae.encode(torch.from_numpy(shard["acts"][row : row + 1].astype(np.float32)))[0]
    feature = int(acts.argmax())
    clip_id = manifest["clips"][int(shard["clip_idx"][row])]["clip_id"]
    top = {key: int(shard[key][row]) for key in ("window", "t", "h", "w")}
    export = tmp_path / "export"
    (export / "data" / "features").mkdir(parents=True)
    features = [{"id": feature, "label": "a planted name", "badge": "real", "real": True}]
    (export / "data" / "run.json").write_text(json.dumps({"model": manifest["model"], "layer": manifest["layer"], "features": features}))

    def write_top(value: float) -> None:
        detail = {"id": feature, "top_clips": [{"clip_id": clip_id, **top, "value": value}]}
        (export / "data" / "features" / f"{feature}.json").write_text(json.dumps(detail))

    write_top(float(acts[feature]))
    assert verify_against_export(export, manifest, run, sae, features) == 1
    write_top(float(acts[feature]) + 0.5)
    with pytest.raises(LiveExportError, match="different dictionary"):
        verify_against_export(export, manifest, run, sae, features)


def test_demo_site_links_the_live_playground_and_keeps_it():
    home = (REPO / "demo-site" / "index.html").read_text(encoding="utf-8")
    assert 'href="/live/" data-testid="cta"' in home
    pack = json.loads((LIVE / "packs" / "dinov2-small-droid" / "pack.json").read_text(encoding="utf-8"))
    assert pack["model"]["name"] == "facebook/dinov2-small" and pack["model"]["layer"] == 6
    assert pack["meta_source"] == "export" and pack["n_real"] == 5
    features = json.loads((LIVE / "packs" / "dinov2-small-droid" / "features.json").read_text(encoding="utf-8"))
    assert sum(1 for f in features if f.get("real")) == pack["n_real"]
    headers = json.loads((REPO / "demo-site" / "vercel.json").read_text(encoding="utf-8"))["headers"]
    assert any(h["source"] == "/live/(.*)" for h in headers)
    from interpretableintelligence.export_site import LIVE_KEEP, SECTION_OF, site_header

    assert "live" in LIVE_KEEP and "vercel.json" in LIVE_KEEP
    assert SECTION_OF["live"] == "play"
    page = (LIVE / "index.html").read_text(encoding="utf-8")
    # The playground carries the same header as every generated page, on the shared stylesheet.
    nav = re.search(r'<nav class="site-nav".*?</nav>', site_header("live"), re.S).group(0)
    assert nav in page
    assert '<link rel="stylesheet" href="../static/site.css">' in page


def _cdn_reachable() -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(ORT_PROBE, method="HEAD"), timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def test_browser_engine_matches_torch(tiny_run, tmp_path):
    """The page's engine (ONNX Runtime Web) gives the same features and edit as torch, on the same pixels."""
    sync_api = pytest.importorskip("playwright.sync_api")
    from test_steer import _chrome

    from interpretableintelligence.adapters import get_adapter
    from interpretableintelligence.adapters.hf import make_spec
    from interpretableintelligence.live import split_model
    from interpretableintelligence.sae import load_sae
    from interpretableintelligence.steer import steer_edit

    chrome = _chrome()
    if chrome is None:
        pytest.skip("Chrome is not installed; the live playground test cannot run")
    if not _cdn_reachable():
        pytest.skip("cdn.jsdelivr.net is not reachable; the page loads ONNX Runtime Web from it")

    site = tmp_path / "site"
    shutil.copytree(LIVE, site / "live", ignore=shutil.ignore_patterns("packs"))
    shutil.copytree(tiny_run["pack"], site / "live" / "packs" / "tiny")
    shutil.copytree(REPO / "demo-site" / "static", site / "static")

    adapter = get_adapter(make_spec(str(tiny_run["model"])))(layer=1, device=torch.device("cpu"))
    adapter.load()
    sae, _config = load_sae(tiny_run["run"] / "sae", torch.device("cpu"))
    front, back = split_model(adapter)
    torch.manual_seed(1)
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        hidden = front(pixels)
        patches = hidden[0, 1:]
        feats = sae.encode(patches)
        totals = feats.sum(dim=0)
        feature = int(totals.argmax())
        base = back(hidden)[0, 1:]
        edited_patches = steer_edit(sae, [feature], "scale", 3.0)(patches[None])[0]
        edited = back(torch.cat([hidden[:, :1], edited_patches[None]], dim=1))[0, 1:]
        shift = 1 - torch.nn.functional.cosine_similarity(base, edited, dim=-1)

    handler = functools.partial(_Quiet, directory=str(site))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/live/?pack=packs/tiny/"
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(url)
            page.wait_for_selector("#work:not([hidden])", timeout=120_000)
            page.wait_for_selector(".frow")
            got = page.evaluate(
                """async ([px, feature]) => {
                  const e = window.__ii.engine;
                  const state = await e.encodePixels(Float32Array.from(px));
                  const result = await e.edit(state, [feature], 'scale', 3.0);
                  return { totals: Array.from(state.totals), shift: Array.from(result.shift) };
                }""",
                [pixels.flatten().tolist(), feature],
            )
            assert errors == []
            # Rows open: the overview lists features and a click opens one with its edit controls.
            page.click(".frow")
            page.wait_for_selector("[data-kind=ablate]")
            page.click("[data-kind=ablate]")
            page.wait_for_function("() => window.__ii.ui.result !== null")
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert np.allclose(np.array(got["totals"]), totals.numpy(), atol=2e-2 * float(totals.max()))
    assert int(np.argmax(got["totals"])) == feature
    assert np.allclose(np.array(got["shift"]), shift.numpy(), atol=2e-3 + 0.05 * float(shift.max()))


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass
