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
