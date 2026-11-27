"""``ii export-live --readout words``: the words readout for CLIP models (live_words.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from interpretableintelligence.cli import app

pytest.importorskip("onnxscript", reason="the words readout's gradient graph needs onnxscript")

WORDS = ["cat", "dog", "robot arm", "apple", "cup"]


def _tiny_tokenizer(folder: Path) -> None:
    """A byte-level CLIP tokenizer with single characters only (no merges), so tests need no download."""
    from transformers import CLIPTokenizer

    chars = list("abcdefghijklmnopqrstuvwxyz.,")
    vocab = {}
    for c in chars:
        vocab[c] = len(vocab)
    for c in chars:
        vocab[c + "</w>"] = len(vocab)
    for special in ("<|startoftext|>", "<|endoftext|>"):
        vocab[special] = len(vocab)
    (folder / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (folder / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    CLIPTokenizer(str(folder / "vocab.json"), str(folder / "merges.txt")).save_pretrained(folder)


@pytest.fixture(scope="module")
def words_run(tmp_path_factory):
    from transformers import CLIPConfig, CLIPModel

    from interpretableintelligence.adapters.vjepa2 import generate_tiny_clips
    from interpretableintelligence.live import encode_jpeg

    root = tmp_path_factory.mktemp("words")
    torch.manual_seed(0)
    model_dir = root / "clip"
    model_dir.mkdir()
    _tiny_tokenizer(model_dir)
    config = CLIPConfig(
        text_config={"vocab_size": 58, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2, "max_position_embeddings": 64, "eos_token_id": 57, "bos_token_id": 56, "pad_token_id": 57},
        vision_config={"image_size": 32, "patch_size": 8, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3, "num_attention_heads": 2},
        projection_dim=16,
    )
    CLIPModel(config).eval().save_pretrained(model_dir)
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "CLIPImageProcessor", "crop_size": {"height": 32, "width": 32}, "size": {"shortest_edge": 32},
                    "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711]}),
        encoding="utf-8",
    )
    clips = root / "clips"
    generate_tiny_clips(clips, 20)
    run = root / "run"
    result = CliRunner().invoke(
        app, ["run", "hf", "--model", str(model_dir), "--clips", str(clips), "--out", str(run), "--layer", "2", "--no-score", "--no-serve"]
    )
    assert result.exit_code == 0, result.output
    pictures = root / "pictures"
    pictures.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        (pictures / f"p{i}.jpg").write_bytes(encode_jpeg(rng.integers(0, 255, size=(40, 48, 3), dtype=np.uint8)))
    (pictures / "credits.json").write_text(json.dumps({"p1.jpg": "Photo by someone, CC BY"}), encoding="utf-8")
    words = root / "words.txt"
    words.write_text("# test words\n" + "\n".join(WORDS + ["cat"]) + "\n", encoding="utf-8")
    pack = root / "pack"
    result = CliRunner().invoke(
        app,
        ["export-live", str(run), "--out", str(pack), "--readout", "words", "--words", str(words), "--sample-dir", str(pictures), "--samples", "3"],
    )
    assert result.exit_code == 0, result.output
    return {"root": root, "model": model_dir, "run": run, "pack": pack}


def test_words_pack_layout(words_run):
    pack_dir = words_run["pack"]
    pack = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    assert pack["version"] == 2 and pack["model"]["readout"] == "words" and pack["evidence"] is False
    assert "back" not in pack["files"]
    for name in pack["files"].values():
        assert (pack_dir / name).is_file(), name
    vocab = json.loads((pack_dir / "vocab.json").read_text(encoding="utf-8"))
    assert vocab == WORDS  # comments and the repeated word dropped, order kept
    words = pack["words"]
    assert words["n"] == len(WORDS) and words["dim"] == 16
    assert (pack_dir / "text.bin").stat().st_size == len(WORDS) * 16 * 2
    assert [s["clip_id"] for s in pack["samples"]] == ["p0", "p1", "p2"]
    assert pack["samples"][1]["credit"] == "Photo by someone, CC BY" and "credit" not in pack["samples"][0]
    features = json.loads((pack_dir / "features.json").read_text(encoding="utf-8"))
    with_sees = [f for f in features if f.get("sees")]
    assert with_sees, "some feature should push some word up"
    for f in with_sees:
        assert all(row["word"] in WORDS and row["rise"] > 0 and 0 <= row["top_on"] <= row["of"] == 3 for row in f["sees"])


def test_words_graph_matches_torch_gradient(words_run):
    ort = pytest.importorskip("onnxruntime")
    from transformers import CLIPModel

    from interpretableintelligence import live_words as lw

    pack = words_run["pack"]
    model = CLIPModel.from_pretrained(words_run["model"], attn_implementation="eager").eval()
    model.requires_grad_(False)
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        hidden = model.vision_model(pixel_values=pixels, output_hidden_states=True).hidden_states[2]
    u = torch.randn(1, 16)
    head = lw.WordsHead(model, 2)
    e, grad = head(hidden, u)
    got_hidden = ort.InferenceSession(str(pack / "front.onnx")).run(None, {"pixel_values": pixels.numpy()})[0]
    got_e, got_grad = ort.InferenceSession(str(pack / "words.onnx")).run(None, {"hidden": hidden.numpy(), "u": u.numpy()})

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))

    # Weight matrices are stored as int8 per channel (vectors as float16), so expect about 1e-2.
    assert rel(hidden.numpy(), got_hidden) < 3e-2
    assert rel(e.numpy(), got_e) < 3e-2
    assert rel(grad.numpy(), got_grad) < 5e-2
    # The text table is the model's own normalised text embeddings of the prompts.
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(words_run["model"])
    table = lw.text_table(model, tokenizer, WORDS)
    stored = np.frombuffer((pack / "text.bin").read_bytes(), dtype="<f2").reshape(len(WORDS), 16)
    assert np.allclose(stored, table, atol=2e-3)


def test_words_readout_refuses_a_plain_vision_model(tmp_path):
    from transformers import AutoModel, ViTConfig

    from interpretableintelligence.adapters.vjepa2 import generate_tiny_clips

    model_dir = tmp_path / "vit"
    AutoModel.from_config(ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)).save_pretrained(model_dir)
    clips = tmp_path / "clips"
    generate_tiny_clips(clips, 20)
    run = tmp_path / "run"
    assert CliRunner().invoke(app, ["run", "hf", "--model", str(model_dir), "--clips", str(clips), "--out", str(run), "--no-score", "--no-serve"]).exit_code == 0
    result = CliRunner().invoke(app, ["export-live", str(run), "--out", str(tmp_path / "pack"), "--readout", "words", "--clips", str(clips)])
    assert result.exit_code == 1
    assert "CLIP" in result.output


def test_load_words_default_list():
    from interpretableintelligence.live_words import load_words, prompt

    words = load_words()
    assert len(words) > 500 and len(words) == len(set(words))
    assert "robot arm" in words and "cup" in words
    assert prompt("apple") == "a photo of an apple." and prompt("cup") == "a photo of a cup."


def test_split_large_and_digests_cover_every_part(tmp_path):
    from interpretableintelligence.live import file_digests, split_large

    (tmp_path / "big.onnx").write_bytes(bytes(range(256)) * 10)  # 2,560 bytes
    (tmp_path / "small.bin").write_bytes(b"abc")
    files = split_large(tmp_path, {"big": "big.onnx", "small": "small.bin"}, part_bytes=1000)
    assert files == {"big": ["big.onnx.0", "big.onnx.1", "big.onnx.2"], "small": "small.bin"}
    assert not (tmp_path / "big.onnx").exists()
    joined = b"".join((tmp_path / p).read_bytes() for p in files["big"])
    assert joined == bytes(range(256)) * 10
    digests = file_digests(tmp_path, files)
    import hashlib

    assert digests["big"] == hashlib.sha256(joined).hexdigest()[:16]


def test_backend_skips_browser_only_packs(words_run, tmp_path):
    import shutil

    from interpretableintelligence.playground.compute import PackLoadError, load_packs

    packs = tmp_path / "packs"
    shutil.copytree(words_run["pack"], packs / "words")
    pack = json.loads((packs / "words" / "pack.json").read_text(encoding="utf-8"))
    assert pack["runs_in"] == ["browser"] and "sae_weights" not in pack["files"]
    with pytest.raises(PackLoadError, match="no packs"):
        load_packs(packs)


def test_assign_names_needs_a_word_that_stands_out_on_both_halves():
    from interpretableintelligence.live_names import assign_names

    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(20)]
    deltas = rng.normal(0, 1, size=(60, 4, 20)).astype(np.float16)
    deltas[0, :, 7] += 12  # feature 0 pushes w7 on every picture
    deltas[1, 0::2, 3] += 24  # feature 1 pushes w3 on half the pictures only
    deltas[2, :, 5] += 12  # always-on: never named
    skip = np.zeros(60, bool)
    skip[2] = True
    names = assign_names(deltas, skip, vocab)
    assert names[0] == "w7"
    assert names[1] is None and names[2] is None
    assert sum(n is not None for n in names) <= 3


def test_name_live_writes_names_through_the_pack(words_run, tmp_path):
    import shutil

    pytest.importorskip("onnxruntime")
    pack = tmp_path / "pack"
    shutil.copytree(words_run["pack"], pack)
    before = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    result = CliRunner().invoke(app, ["name-live", str(pack), "--pictures", "2", "--z-min", "0"])
    assert result.exit_code == 0, result.output
    meta = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    features = json.loads((pack / "features.json").read_text(encoding="utf-8"))
    named = [f for f in features if f.get("name")]
    assert meta["names"]["named"] == len(named) > 0
    assert all(f["name"] in WORDS for f in named)
    assert meta["digests"]["features"] != before["digests"]["features"]


def test_bake_live_saves_the_pages_first_answer(words_run, tmp_path):
    """ii bake-live runs the playground on every example picture and saves its answer; the page reads it back."""
    import os
    import shutil

    pytest.importorskip("playwright.sync_api")
    from test_live import LIVE, REPO, _cdn_reachable

    from interpretableintelligence.live_bake import bake_live, find_chrome

    if find_chrome() is None:
        pytest.skip("Chrome is not installed; baking runs the page in it")
    ort_dir = os.environ.get("II_ORT_DIR")  # a local onnxruntime-web dist/, for machines without the CDN
    if not ort_dir and not _cdn_reachable():
        pytest.skip("cdn.jsdelivr.net is not reachable; the page loads ONNX Runtime Web from it")

    site = tmp_path / "site"
    shutil.copytree(LIVE, site / "live", ignore=shutil.ignore_patterns("packs"))
    shutil.copytree(REPO / "demo-site" / "static", site / "static")
    pack = site / "live" / "packs" / "tiny"
    shutil.copytree(words_run["pack"], pack)
    summary = bake_live(pack, ort_dir=Path(ort_dir) if ort_dir else None, log=lambda *_: None)

    baked = json.loads((pack / "instant.json").read_text(encoding="utf-8"))
    meta = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    assert summary.samples == len(meta["samples"]) == len(baked["samples"]) == 3
    assert meta["files"]["instant"] == "instant.json"
    # Only the files that change the numbers are pinned, so new feature names keep it valid.
    assert baked["digests"] == {k: meta["digests"][k] for k in ("front", "sae", "decoder", "words", "text", "vocab")}
    for rec in baked["samples"].values():
        assert 0 <= rec["word"] < len(WORDS)
        assert abs(sum(rec["prob"].values()) - 1) < 0.05  # the 5 words, all kept
        assert set(rec["maps"]) == {str(i) for i, _ in rec["why"]}
        assert sorted(rec["off"]["ids"]) == sorted(i for i, _ in rec["why"])
        # What lights up inside, which the page opens on: switches by how much of the picture they
        # cover, strongest first, each with its map (here or among the why maps).
        scores = [score for _, score in rec["active"]["ids"]]
        assert scores and scores == sorted(scores, reverse=True) and min(scores) > 0
        assert {i for i, _ in rec["active"]["ids"]} <= {int(k) for k in rec["active"]["maps"]} | {int(k) for k in rec["maps"]}
        # The opening move: the switch for the pack's last word ("cup"), with a random switch beside it.
        flip = rec.get("flip")
        if flip:
            assert flip["word"] == len(WORDS) - 1 != rec["word"]
            assert flip["on"] == [flip["toward"][0][0]] and len(flip["control"]["ids"]) == 1
            assert str(flip["word"]) in flip["prob"] and str(flip["word"]) in flip["boost"] and str(flip["word"]) in flip["control"]["prob"]
    assert any("flip" in rec for rec in baked["samples"].values())

    # The page with that answer: it opens on the switch at once and goes live without an error.
    import functools
    import http.server
    import threading

    from playwright.sync_api import sync_playwright

    from interpretableintelligence.live_bake import ORT_URL, _Quiet

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(site)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    errors: list[str] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=find_chrome(), headless=True, args=["--no-sandbox", "--disable-gpu"])
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            if ort_dir:

                def serve_ort(route):
                    path = Path(ort_dir) / ORT_URL.match(route.request.url).group(1)
                    kind = "application/wasm" if path.name.endswith(".wasm") else "text/javascript"
                    route.fulfill(status=200, body=path.read_bytes(), headers={"content-type": kind, "access-control-allow-origin": "*"})

                context.route(ORT_URL, serve_ort)
            tab = context.new_page()
            tab.on("pageerror", lambda exc: errors.append(str(exc)))
            tab.goto(f"http://127.0.0.1:{httpd.server_address[1]}/live/index.html?cpu&pack=packs/tiny/")
            tab.wait_for_function("() => document.querySelector('[data-testid=see-input]') && window.__ii.ui.live && !window.__ii.ui.busy && !window.__ii.ui.state.baked", timeout=180_000)
            opened = tab.evaluate(
                "() => ({ view: window.__ii.ui.view, word: window.__ii.ui.word, rows: document.querySelectorAll('[data-testid=active] .frow').length })"
            )
            # Point at a spot: the list becomes the switches on there.
            box = tab.locator("#stage").bounding_box()
            tab.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            tab.wait_for_function("() => window.__ii.ui.spot != null")
            spot = tab.inner_text("[data-testid=active] h2")
            # Make it see the last word: the switch is found and turned on beside a random one.
            top = tab.evaluate("() => { const p = window.__ii.ui.state.seen.prob; return p.indexOf(Math.max(...p)); }")
            target = next(i for i in range(len(WORDS) - 1, -1, -1) if i != top)
            tab.fill("[data-testid=see-input]", WORDS[target])
            tab.dispatch_event("[data-testid=see-input]", "change")
            tab.wait_for_function("() => window.__ii.ui.boost && window.__ii.ui.boostControl && !window.__ii.ui.busy", timeout=120_000)
            made = tab.evaluate("() => ({ word: window.__ii.ui.word, on: [...window.__ii.ui.on], view: window.__ii.ui.view })")
            tab.click("[data-testid=back]")
            back = tab.evaluate("() => ({ view: window.__ii.ui.view, word: window.__ii.ui.word, on: window.__ii.ui.on.size })")
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert not errors, errors
    assert opened == {"view": "inside", "word": None, "rows": opened["rows"]} and opened["rows"] > 0
    assert spot == "Lit up at this spot"
    assert made["view"] == "word" and made["word"] == target and len(made["on"]) == 1
    assert back == {"view": "inside", "word": None, "on": 0}


def test_words_pack_reads_any_word(words_run):
    """export-live also ships the text half: it reproduces the pack's own word embeddings from token ids."""
    ort = pytest.importorskip("onnxruntime")
    from transformers import AutoTokenizer

    from interpretableintelligence.live_words import prompt

    pack_dir = words_run["pack"]
    pack = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    assert pack["files"]["text_encoder"] == "text_encoder.onnx" and pack["files"]["tokenizer"] == "tokenizer.json"
    spec = json.loads((pack_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(words_run["model"])
    assert spec["vocab"][spec["bos"]] == "<|startoftext|>" and spec["vocab"][spec["eos"]] == "<|endoftext|>"
    table = np.frombuffer((pack_dir / "text.bin").read_bytes(), dtype="<f2").astype(np.float32).reshape(len(WORDS), 16)
    session = ort.InferenceSession(str(pack_dir / "text_encoder.onnx"))
    for i, word in enumerate(WORDS):
        ids = tokenizer([prompt(word)])["input_ids"][0]
        got = session.run(None, {"input_ids": np.asarray([ids], dtype=np.int64)})[0][0]
        assert float(table[i] @ got) > 0.99


def test_page_tokenizer_matches_clip_tokenizer(tmp_path):
    """demo-site/live/text.js tokenises like Hugging Face's CLIPTokenizer (byte-level BPE with merges)."""
    import shutil
    import subprocess

    from transformers import CLIPTokenizer

    from interpretableintelligence.live_words import tokenizer_spec

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    keep = [*range(33, 127), *range(161, 173), *range(174, 256)]
    chars, extra = [], 0
    for b in range(256):
        chars.append(chr(b) if b in keep else chr(256 + extra))
        extra += b not in keep
    merges = ["t h", "th e</w>", "o f</w>", "p h", "ph o", "pho t", "phot o</w>", "c a", "ca t</w>", "r o", "ro b", "rob o", "robo t</w>", "e r</w>", "t er</w>"]
    vocab = chars + [c + "</w>" for c in chars] + ["".join(m.split()) for m in merges] + ["<|startoftext|>", "<|endoftext|>"]
    (tmp_path / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"))
    spec = tokenizer_spec(tokenizer)
    texts = ["a photo of a cat.", "A photo of a ROBOT arm.", "a photo of an otter's  tail!", "a photo of a crème brûlée 42.", "the  thermometer, then the cat"]
    want = [tokenizer(t)["input_ids"] for t in texts]
    page = Path(__file__).resolve().parents[1] / "demo-site" / "live" / "text.js"
    program = (
        f"import({json.dumps(page.as_uri())}).then(({{ Tokenizer }}) => {{"
        f" const t = new Tokenizer({json.dumps(spec)});"
        f" console.log(JSON.stringify({json.dumps(texts)}.map((x) => t.encode(x)))); }});"
    )
    got = json.loads(subprocess.run([node, "--input-type=module", "-e", program], check=True, capture_output=True, text=True).stdout)
    assert got == want


def test_gallery_live_keeps_each_features_strongest_pictures(words_run, tmp_path):
    """ii gallery-live reads a folder through the pack's own graphs and keeps, per feature, where it fires most."""
    import base64
    import shutil

    pytest.importorskip("onnxruntime")
    from interpretableintelligence.live import encode_jpeg

    pack = tmp_path / "pack"
    shutil.copytree(words_run["pack"], pack)
    photos = tmp_path / "photos"
    photos.mkdir()
    rng = np.random.default_rng(1)
    for i in range(6):
        (photos / f"img{i}.jpg").write_bytes(encode_jpeg(rng.integers(0, 255, size=(48, 40, 3), dtype=np.uint8)))
    (tmp_path / "credits.json").write_text(json.dumps({"img2": "Photo by someone, CC BY 2.0"}), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["gallery-live", str(pack), "--images", str(photos), "--credits", str(tmp_path / "credits.json"), "--per-feature", "9", "--min-frac", "0"]
    )
    assert result.exit_code == 0, result.output
    meta = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    assert meta["files"]["gallery"] == "gallery/index.json" and meta["digests"]["gallery"]
    index = json.loads((pack / "gallery" / "index.json").read_text(encoding="utf-8"))
    assert index["read"] == 6 + len(meta["samples"])
    seen, files, credited = 0, set(), False
    for s in range(index["shards"]):
        shard = json.loads((pack / "gallery" / f"top-{s:03d}.json").read_text(encoding="utf-8"))
        for fid, rows in shard.items():
            assert int(fid) % index["shards"] == s and 0 < len(rows) <= 9
            peaks = [row["peak"] for row in rows]
            assert min(peaks) > 0
            for row in rows:
                assert (pack / row["file"]).is_file()
                files.add(row["file"])
                credited |= row.get("credit") == "Photo by someone, CC BY 2.0"
                q = np.frombuffer(base64.b64decode(row["map"]), np.uint8)
                assert q.size == index["grid"] and q.max() == 255
            seen += 1
    assert seen > 0 and credited
    assert len([f for f in files if f.startswith("gallery/")]) == len(list((pack / "gallery").glob("*.jpg"))) <= index["pictures"]
    # firsts.json: each feature's first pictures in one file, as squares of patches inside the grid.
    firsts = json.loads((pack / "gallery" / index["firsts"]).read_text(encoding="utf-8"))
    assert len(firsts["top"]) == seen and set(firsts["pictures"]) <= files
    side = int(round(index["grid"] ** 0.5))
    for rows in firsts["top"].values():
        assert 0 < len(rows) <= 3
        for pic, x, y, n in rows:
            assert 0 <= pic < len(firsts["pictures"]) and 0 < n <= side and 0 <= x <= side - n and 0 <= y <= side - n
    # ii look-live: crops of each feature's gallery pictures vote for a word; named only by a half.
    from interpretableintelligence.live_looks import PARTS

    before = meta["digests"].get("features")
    result = CliRunner().invoke(app, ["look-live", str(pack), "--least", "1", "--checkpoint", str(tmp_path / "votes.jsonl")])
    assert result.exit_code == 0, result.output
    vocab = set(json.loads((pack / meta["files"]["vocab"]).read_text(encoding="utf-8"))) | set(PARTS)
    votes = {}
    for line in (tmp_path / "votes.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        votes[row["id"]] = row
    assert len(votes) == len(json.loads((pack / meta["files"]["features"]).read_text(encoding="utf-8")))
    for f in json.loads((pack / meta["files"]["features"]).read_text(encoding="utf-8")):
        row = votes[f["id"]]
        if "looks" in f:
            assert f["looks"] in vocab and 2 * row["votes"][f["looks"]] >= row["n"] >= 1
        else:
            assert not row["n"] or 2 * max(row["votes"].values()) < row["n"]
    assert json.loads((pack / "pack.json").read_text(encoding="utf-8"))["digests"]["features"] != before or not any(
        "looks" in f for f in json.loads((pack / meta["files"]["features"]).read_text(encoding="utf-8"))
    )
