"""Names for the features of a words pack (``ii name-live``).

A feature of the dictionary lives halfway through the model, so its direction cannot be read
against the text embeddings directly. Instead each feature is written onto every patch of a few
of the pack's sample pictures (``strength`` times its training peak, the playground's force on),
the rest of the model runs (the pack's own ``front`` and ``words`` graphs, through ONNX Runtime),
and the change in every word's log-probability is recorded.

Some words rise for almost any edit, so each feature's rise is compared with every other
feature's: for each picture and word, ``z = (rise - mean over features) / std over features``.
A feature gets the word with the highest mean ``z`` as its ``name`` only when

* that mean ``z`` is at least ``z_min``, and
* the word is among the ``top`` words of both halves of the pictures (even and odd), measured
  separately, so a name has to hold on pictures that did not pick it.

Otherwise the feature has no name and the page shows its number. The always-on features (see
:func:`background`) are never named. A name is a short description of what writing the feature
in does to the model's words: it is measured, not verified, and it never sets a badge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np


def background(features: list[dict]) -> np.ndarray:
    """Always-on features, as the page leaves them out: mean activation >= 3 or firing on half the patches."""
    return np.array([(f.get("mean") or 0) >= 3 or (f.get("freq") or 0) >= 0.5 for f in features])


def _join(base: Path, entry) -> bytes:
    return b"".join((base / part).read_bytes() for part in ([entry] if isinstance(entry, str) else entry))


def _picture(path: Path, size: int, mean, std, crop: bool) -> np.ndarray:
    """A sample picture as the page feeds it: the middle square (CLIP) resized to ``size``, normalised."""
    import torch

    from interpretableintelligence.live import read_image

    rgb = read_image(path)
    if crop:
        h, w = rgb.shape[:2]
        s = min(h, w)
        rgb = rgb[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)[0]
    x = (x.permute(1, 2, 0).numpy() - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def word_deltas(
    pack_dir: Path,
    n_pictures: int = 8,
    strength: float = 2.0,
    log: Callable[[str], None] = lambda _m: None,
    checkpoint: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """``[n_features, n_pictures, n_words]`` float16: change in log p(word) when each feature is written in.

    It takes about an hour on a laptop CPU for 6,144 features; with ``checkpoint`` (an ``.npz``
    file) it saves its progress every 250 features and resumes from there.
    """
    import onnxruntime as ort

    pack_dir = Path(pack_dir)
    pack = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    files, model = pack["files"], pack["model"]
    if model.get("readout") != "words":
        raise ValueError(f"{pack_dir} is not a words pack (ii export-live --readout words)")
    features = json.loads((pack_dir / files["features"]).read_text(encoding="utf-8"))
    d = int(model["d_model"])
    prefix = int(model.get("prefix_tokens", 1))
    text = np.frombuffer(_join(pack_dir, files["text"]), dtype="<f2").astype(np.float32).reshape(-1, pack["words"]["dim"])
    decoder = np.frombuffer(_join(pack_dir, files["decoder"]), dtype="<f2").astype(np.float32).reshape(-1, d)
    scale = float(pack["words"]["logit_scale"])
    peaks = np.array([f.get("max") or 0.0 for f in features], np.float32)
    options = ort.SessionOptions()
    # The pack stores int8 weights behind DequantizeLinear; let ONNX Runtime fold them to float once.
    options.add_session_config_entry("session.disable_quant_qdq", "1")
    front = ort.InferenceSession(_join(pack_dir, files["front"]), options)
    words = ort.InferenceSession(_join(pack_dir, files["words"]), options)
    zero_u = np.zeros((1, text.shape[1]), np.float32)

    def logp(hidden: np.ndarray) -> np.ndarray:
        e = words.run(["embed"], {"hidden": hidden, "u": zero_u})[0][0]
        logits = scale * (text @ (e / np.linalg.norm(e)))
        logits -= logits.max()
        return logits - np.log(np.exp(logits).sum())

    samples = pack.get("samples") or []
    chosen = samples[::2][:n_pictures] if len(samples) >= 2 * n_pictures else samples[:n_pictures]
    if len(chosen) < 2:
        raise ValueError("naming needs at least 2 sample pictures in the pack")
    crop = model.get("resize") == "crop"
    hiddens = [front.run(None, {"pixel_values": _picture(pack_dir / s["file"], int(model["image_size"]), model["mean"], model["std"], crop)})[0] for s in chosen]
    bases = [logp(h) for h in hiddens]
    out = np.zeros((len(features), len(chosen), text.shape[0]), np.float16)
    start = 0
    if checkpoint is not None and Path(checkpoint).is_file():
        saved = np.load(checkpoint)
        if saved["deltas"].shape == out.shape:
            out, start = saved["deltas"], int(saved["done"])
            log(f"resuming at feature {start}")
    for f in range(start, len(features)):
        if checkpoint is not None and f % 250 == 0 and f > start:
            np.savez(checkpoint, deltas=out, done=f)
        if f % 500 == 0:
            log(f"writing in each feature: {f}/{len(features)}")
        if peaks[f] <= 0:
            continue
        push = strength * peaks[f] * decoder[f]
        for j, (hidden, base) in enumerate(zip(hiddens, bases)):
            edited = hidden.copy()
            edited[0, prefix:] += push
            out[f, j] = logp(edited) - base
    return out, [s["file"] for s in chosen]


def assign_names(deltas: np.ndarray, skip: np.ndarray, vocab: list[str], z_min: float = 3.0, top: int = 3) -> list[str | None]:
    """One word per feature, or None, by the rule in the module docstring."""
    rise = deltas.astype(np.float32)
    live = ~skip & (np.abs(rise).sum((1, 2)) > 0)
    ref = rise[live]
    z = (rise - ref.mean(0)) / (ref.std(0) + 1e-6)  # [features, pictures, words]
    mean_z = z.mean(1)
    halves = (z[:, 0::2].mean(1), z[:, 1::2].mean(1))
    names: list[str | None] = []
    for f in range(len(rise)):
        w = int(mean_z[f].argmax())
        ok = live[f] and mean_z[f, w] >= z_min and all(w in np.argsort(-h[f])[:top] for h in halves)
        names.append(vocab[w] if ok else None)
    return names


def name_pack(
    pack_dir: Path, n_pictures: int = 8, z_min: float = 3.0, log: Callable[[str], None] = lambda _m: None, checkpoint: Path | None = None
) -> dict:
    """Write ``name`` into the pack's ``features.json`` (and refresh its digest). Returns counts."""
    pack_dir = Path(pack_dir)
    pack_path = pack_dir / "pack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    feat_path = pack_dir / pack["files"]["features"]
    features = json.loads(feat_path.read_text(encoding="utf-8"))
    vocab = json.loads((pack_dir / pack["files"]["vocab"]).read_text(encoding="utf-8"))
    deltas, pictures = word_deltas(pack_dir, n_pictures=n_pictures, log=log, checkpoint=checkpoint)
    names = assign_names(deltas, background(features), vocab, z_min=z_min)
    for feature, name in zip(features, names):
        feature.pop("name", None)
        if name:
            feature["name"] = name
    feat_path.write_text(json.dumps(features, separators=(",", ":")) + "\n", encoding="utf-8")
    pack["names"] = {"pictures": len(pictures), "strength": 2.0, "z_min": z_min, "named": sum(1 for n in names if n)}
    pack.setdefault("digests", {})["features"] = hashlib.sha256(feat_path.read_bytes()).hexdigest()[:16]
    pack_path.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    return {"named": pack["names"]["named"], "features": len(features), "pictures": pictures}
