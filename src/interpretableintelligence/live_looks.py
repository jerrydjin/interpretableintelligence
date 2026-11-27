"""What each switch of a words pack fires on, in a word (``ii look-live``).

``ii name-live`` names a feature by what writing it in makes the model say, which can read wrong
beside a picture (a snout switch named "camera"). This names it by where it fires instead: for each
feature, the square of patches where it lights up on each of its gallery pictures (``ii
gallery-live``, the same squares as ``firsts.json``) is cut out, resized to the model's input and
read through the pack's own ``front`` and ``words`` graphs; every crop votes for the word it is most
likely to be called, among the pack's word list and :data:`PARTS` (parts and surfaces the list
lacks, read through the pack's text encoder when it has one). A feature gets that word as
``looks`` in ``features.json`` only when

* it has at least ``least`` gallery pictures, and
* at least half of them vote for the same word.

Otherwise it has no ``looks`` and the page falls back to its other name or its number. Always-on
features have no gallery pictures, so they are never named. Like ``name``, ``looks`` is a short
description for people, measured and not verified: it never sets a badge.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from interpretableintelligence.live_names import _join

# Parts and surfaces a crop is often of, which an object word list lacks.
PARTS = (
    "nose", "snout", "muzzle", "dog face", "cat face", "animal face", "human face", "ear", "eye",
    "eyes", "mouth", "teeth", "tongue", "lips", "paw", "fur", "whiskers", "beak", "feather", "wing",
    "tail", "horn", "mane", "hoof", "fin", "scales", "wheel", "tire", "headlight", "window", "door",
    "roof", "handle", "button", "knob", "screen", "keyboard", "text", "letters", "writing",
    "numbers", "logo", "sign", "label", "stripes", "checkered pattern", "dots", "grid", "pattern",
    "fabric", "wood", "metal", "glass", "plastic", "paper", "water", "sky", "cloud", "grass", "tree",
    "leaves", "brick", "tile", "floor", "wall", "ceiling", "shadow", "light", "lamp", "edge",
    "corner", "handrail", "fence", "pole", "rope", "string", "net", "mesh", "hole", "reflection",
    "shiny surface", "blurry background", "dark background", "white background", "texture",
    "fingers", "hand", "shoe", "hat", "glasses", "shirt", "jacket",
)  # fmt: skip


def _byte_chars() -> list[str]:
    keep = [*range(33, 127), *range(161, 173), *range(174, 256)]
    out, extra = [], 0
    for b in range(256):
        if b in keep:
            out.append(chr(b))
        else:
            out.append(chr(256 + extra))
            extra += 1
    return out


# Letters, digits and other runs, as text.js splits text (enough for the plain words used here).
_PAT = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+", re.IGNORECASE)


def encode(spec: dict, text: str) -> list[int]:
    """CLIP token ids for ``text`` from the pack's ``tokenizer.json`` (as demo-site/live/text.js)."""
    ids = {t: i for i, t in enumerate(spec["vocab"])}
    ranks = {m: r for r, m in enumerate(spec["merges"])}
    chars = _byte_chars()
    out = [spec["bos"]]
    for piece in _PAT.findall(" ".join(text.split()).lower()):
        word = [chars[b] for b in piece.encode("utf-8")]
        word[-1] += "</w>"
        while len(word) > 1:
            pairs = [(ranks.get(f"{a} {b}"), i) for i, (a, b) in enumerate(zip(word, word[1:], strict=False))]
            pairs = [p for p in pairs if p[0] is not None]
            if not pairs:
                break
            first, second = word[min(pairs)[1]], word[min(pairs)[1] + 1]
            nxt, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    nxt.append(first + second)
                    i += 2
                else:
                    nxt.append(word[i])
                    i += 1
            word = nxt
        out += [ids.get(p, spec["eos"]) for p in word]
    return [*out[:76], spec["eos"]]


def _prompt(template: str, word: str) -> str:
    return template.replace("{article}", "an" if word[:1].lower() in "aeiou" else "a").replace("{word}", word)


def word_table(pack_dir: Path, pack: dict, extra: tuple[str, ...] = PARTS) -> tuple[list[str], np.ndarray]:
    """The pack's words and ``extra`` (those not already in it, when the pack has a text encoder), unit rows."""
    import onnxruntime as ort

    files = pack["files"]
    vocab = json.loads((pack_dir / files["vocab"]).read_text(encoding="utf-8"))
    dim = int(pack["words"]["dim"])
    raw = (pack_dir / files["text"]).read_bytes()
    dtype = np.float16 if len(raw) == len(vocab) * dim * 2 else np.float32
    table = [np.frombuffer(raw, dtype=dtype).reshape(len(vocab), dim).astype(np.float32)]
    words = list(vocab)
    extra = tuple(w for w in extra if w not in set(vocab))
    if extra and files.get("text_encoder") and files.get("tokenizer"):
        spec = json.loads((pack_dir / files["tokenizer"]).read_text(encoding="utf-8"))
        session = ort.InferenceSession(_join(pack_dir, files["text_encoder"]))
        rows = []
        for w in extra:
            ids = np.array([encode(spec, _prompt(pack["words"]["prompt"], w))], dtype=np.int64)
            rows.append(session.run(None, {"input_ids": ids})[0][0])
        table.append(np.array(rows, dtype=np.float32))
        words += list(extra)
    t = np.concatenate(table)
    return words, t / np.linalg.norm(t, axis=1, keepdims=True)


def _crop(path: Path, x: int, y: int, side: int, grid: int, size: int) -> np.ndarray:
    """The square ``[x, y, side]`` (in patches) of a gallery picture, resized to ``size``, uint8."""
    import av

    with av.open(str(path)) as box:
        rgb = next(box.decode(video=0)).to_ndarray(format="rgb24")
    px = rgb.shape[0] / grid
    a, b = round(y * px), round((y + side) * px)
    c, d = round(x * px), round((x + side) * px)
    crop = np.ascontiguousarray(rgb[a:b, c:d])
    frame = av.VideoFrame.from_ndarray(crop, format="rgb24").reformat(width=size, height=size, interpolation="BICUBIC")
    return frame.to_ndarray(format="rgb24")


def look_pack(
    pack_dir: Path,
    *,
    least: int = 4,
    extra: tuple[str, ...] = PARTS,
    only: list[int] | None = None,
    checkpoint: Path | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> dict:
    """Write ``looks`` into the pack's ``features.json`` (see the module docstring).

    ``only`` limits the features read (the rest keep what they had); ``checkpoint`` is a JSON-lines
    file of votes already counted, read first and appended to, so a long run can stop and resume.
    """
    import onnxruntime as ort

    from interpretableintelligence.live_gallery import _square_of

    pack_dir = Path(pack_dir)
    pack_path = pack_dir / "pack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    files, model = pack["files"], pack["model"]
    if not files.get("gallery"):
        raise ValueError("this pack has no gallery: run ii gallery-live first")
    grid, size = int(model["grid"]["w"]), int(model["image_size"])
    mean, std = np.asarray(model["mean"], np.float32), np.asarray(model["std"], np.float32)
    words, table = word_table(pack_dir, pack, extra)
    front = ort.InferenceSession(_join(pack_dir, files["front"]))
    head = ort.InferenceSession(_join(pack_dir, files["words"]))
    scale = float(pack["words"].get("logit_scale", 100.0))
    index = json.loads((pack_dir / files["gallery"]).read_text(encoding="utf-8"))
    shards = int(index["shards"])

    done: dict[int, dict] = {}
    if checkpoint and Path(checkpoint).exists():
        for line in Path(checkpoint).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[int(row["id"])] = row
    sink = open(checkpoint, "a", encoding="utf-8") if checkpoint else None  # noqa: SIM115

    def votes_for(fid: int) -> dict:
        shard = json.loads((pack_dir / "gallery" / f"top-{fid % shards:03d}.json").read_text(encoding="utf-8"))
        rows = shard.get(str(fid), [])
        counts: dict[str, int] = {}
        for i, row in enumerate(rows):
            x, y, side = _square_of(base64.b64decode(row["map"]), grid, grid)
            rgb = _crop(pack_dir / row["file"], x, y, side, grid, size)
            pixels = ((rgb.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
            hidden = front.run(None, {"pixel_values": pixels})[0]
            e = head.run(["embed"], {"hidden": hidden, "u": np.zeros((1, table.shape[1]), np.float32)})[0][0]
            word = words[int(np.argmax(scale * table @ (e / np.linalg.norm(e))))]
            counts[word] = counts.get(word, 0) + 1
            # Stop once no word can reach half of the pictures.
            if 2 * (max(counts.values()) + len(rows) - i - 1) < len(rows):
                break
        return {"id": fid, "n": len(rows), "votes": counts}

    feat_path = pack_dir / files["features"]
    features = json.loads(feat_path.read_text(encoding="utf-8"))
    ids = only if only is not None else [f["id"] for f in features]
    for k, fid in enumerate(ids):
        if fid in done:
            continue
        done[fid] = votes_for(fid)
        if sink:
            sink.write(json.dumps(done[fid]) + "\n")
            sink.flush()
        if k % 100 == 0:
            log(f"{k}/{len(ids)} features read")
    if sink:
        sink.close()

    named = 0
    for f in features:
        row = done.get(f["id"])
        if row is None:
            continue
        f.pop("looks", None)
        if row["n"] >= least and row["votes"]:
            word, count = max(row["votes"].items(), key=lambda kv: kv[1])
            if 2 * count >= row["n"]:
                f["looks"] = word
                named += 1
    feat_path.write_text(json.dumps(features, separators=(",", ":")) + "\n", encoding="utf-8")
    pack.setdefault("digests", {})["features"] = hashlib.sha256(feat_path.read_bytes()).hexdigest()[:16]
    pack_path.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    return {"read": len([i for i in ids if i in done]), "named": named}
