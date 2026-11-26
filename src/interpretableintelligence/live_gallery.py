"""Where each switch lights up most, on a big set of photos (``ii gallery-live``).

A pack ships a handful of sample pictures, too few for most of its features to fire strongly on
any of them. This reads a larger folder of pictures through the pack's own ONNX graphs (``front``
then ``sae``, top-k per patch as the page does), keeps for every feature the pictures where it
fires most, and writes them into the pack:

* ``gallery/NNNNN.jpg``: only the pictures some feature keeps, as the model's input square
  (middle square, resized to ``image_size``), so a click reads the same pixels the map came from;
* ``gallery/index.json``: how many shards, pictures read and kept, the patch grid;
* ``gallery/top-SSS.json``: ``{"<feature>": [{"file", "credit", "sample"?, "peak", "map"}]}``
  for features ``id % shards == SSS``, most first (activation summed over the patches), so the page fetches one small file per
  switch. ``map`` is the feature over the patches, base64 of one byte per patch scaled to
  ``peak`` (the page's ``unq8``); ``sample`` names a pack sample picture.

A picture is kept for a feature only where the feature reaches ``min_frac`` of its training peak
(the page's "by itself" bar), so a faint flicker does not pass for what it stands for. Always-on
features (:func:`live_names.background`) get no pictures. Exploration only: nothing here is
scored or sets a badge.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from interpretableintelligence.live import encode_jpeg, read_image
from interpretableintelligence.live_names import _join, background

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _square(path: Path, size: int) -> np.ndarray:
    """The middle square of a picture, resized to ``size`` (as the page's crop resize), ``[size, size, 3]`` uint8.

    Through PyAV, as ``live.read_image`` reads pictures, so no imaging dependency is needed.
    """
    import av


    rgb = read_image(path)
    h, w = rgb.shape[:2]
    s = min(h, w)
    rgb = np.ascontiguousarray(rgb[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s])
    if s == size:
        return rgb
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(width=size, height=size, interpolation="AREA" if s > size else "BICUBIC")
    return frame.to_ndarray(format="rgb24")


def _q8(values: np.ndarray) -> tuple[str, float]:
    peak = float(values.max()) if values.size else 0.0
    q = np.round(np.clip(values, 0, None) / (peak or 1.0) * 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii"), float(f"{peak:.5g}")


def build_gallery(
    pack_dir: Path,
    images: Path,
    credits: dict[str, str] | None = None,
    per_feature: int = 6,
    min_frac: float = 0.25,
    shards: int = 128,
    quality: int = 4,
    with_samples: bool = True,
    limit: int | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> dict:
    """Write ``gallery/`` into a pack and point ``pack.json`` at it. Returns counts."""
    import onnxruntime as ort

    pack_dir, images = Path(pack_dir), Path(images)
    pack_path = pack_dir / "pack.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    files, model = pack["files"], pack["model"]
    features = json.loads((pack_dir / files["features"]).read_text(encoding="utf-8"))
    n_feat = len(features)
    size, d = int(model["image_size"]), int(model["d_model"])
    prefix = int(model.get("prefix_tokens", 1))
    sae = pack["sae"]
    k = int(sae["k"]) if sae.get("activation") == "topk" else 0
    threshold = float(sae.get("threshold") or 0.0) if sae.get("activation") == "jumprelu" else 0.0
    mean = np.asarray(model["mean"], np.float32)
    std = np.asarray(model["std"], np.float32)
    peaks = np.array([f.get("max") or 0.0 for f in features], np.float32)
    skip = background(features) | (peaks <= 0)

    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_quant_qdq", "1")
    front = ort.InferenceSession(_join(pack_dir, files["front"]), options)
    sae_s = ort.InferenceSession(_join(pack_dir, files["sae"]), options)

    pool: list[dict] = []
    if with_samples:
        for s in pack.get("samples") or []:
            pool.append({"path": pack_dir / s["file"], "file": s["file"], "credit": s.get("credit", ""), "sample": s["clip_id"]})
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in EXTS)
    if limit:
        paths = paths[:limit]
    credits = credits or {}
    for p in paths:
        pool.append({"path": p, "file": None, "credit": credits.get(p.stem) or credits.get(p.name) or ""})

    # For every feature, the best `per_feature` pictures so far: (total over patches, picture, map).
    # Ranked by the total, so a picture where the thing is big beats one where a corner of it fires.
    best_total = np.full((n_feat, per_feature), -1.0, np.float32)
    best_pic = np.full((n_feat, per_feature), -1, np.int32)
    best_map: dict[tuple[int, int], np.ndarray] = {}
    n_tok = None
    for i, entry in enumerate(pool):
        if i % 250 == 0:
            log(f"reading pictures: {i}/{len(pool)}")
        im = _square(entry["path"], size)
        x = ((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        hidden = front.run(None, {"pixel_values": x})[0][0, prefix:]
        pre = sae_s.run(None, {"x": np.ascontiguousarray(hidden.reshape(-1, d), np.float32)})[0]
        n_tok = pre.shape[0]
        if k:  # the k largest on each patch, as the page keeps them
            top = np.argpartition(-pre, k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(pre, top, axis=1)
            dense = np.zeros_like(pre)
            np.put_along_axis(dense, top, np.where(vals > 0, vals, 0), axis=1)
        else:
            dense = np.where(pre > threshold, pre, 0).astype(np.float32)
        fpeak = dense.max(axis=0)
        total = dense.sum(axis=0)
        hit = np.nonzero((fpeak >= min_frac * peaks) & ~skip & (fpeak > 0))[0]
        for f in hit:
            j = int(np.argmin(best_total[f]))
            if total[f] > best_total[f, j]:
                old = int(best_pic[f, j])
                if old >= 0:
                    best_map.pop((int(f), old), None)
                best_total[f, j], best_pic[f, j] = total[f], i
                best_map[(int(f), i)] = dense[:, f].astype(np.float32)

    # Which pictures are used, renumbered; samples keep their own file.
    used = sorted({int(p) for p in best_pic.ravel() if p >= 0})
    out_dir = pack_dir / "gallery"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*"):
        old.unlink()
    number: dict[int, int] = {}
    pictures = []
    for i in used:
        entry = pool[i]
        number[i] = len(pictures)
        if entry["file"] is None:
            name = f"gallery/{len(pictures):05d}.jpg"
            (pack_dir / name).write_bytes(encode_jpeg(_square(entry["path"], size), max_edge=size, quality=quality))
            rec = {"file": name}
        else:
            rec = {"file": entry["file"], "sample": entry["sample"]}
        if entry["credit"]:
            rec["credit"] = entry["credit"]
        pictures.append(rec)

    tops: list[dict] = [{} for _ in range(shards)]
    kept = 0
    for f in range(n_feat):
        order = [j for j in np.argsort(-best_total[f]) if best_pic[f, j] >= 0]
        if not order:
            continue
        kept += 1
        rows = []
        for j in order:
            pic = int(best_pic[f, j])
            b64, peak = _q8(best_map[(f, pic)])
            rows.append({**pictures[number[pic]], "peak": peak, "map": b64})
        tops[f % shards][str(f)] = rows
    for s, shard in enumerate(tops):
        (out_dir / f"top-{s:03d}.json").write_text(json.dumps(shard, separators=(",", ":")) + "\n", encoding="utf-8")
    index = {"shards": shards, "per_feature": per_feature, "min_frac": min_frac, "read": len(pool), "pictures": len(pictures), "grid": n_tok}
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")) + "\n", encoding="utf-8")
    digest = hashlib.sha256()
    for p in sorted(out_dir.glob("*.json")):
        digest.update(p.read_bytes())
    pack["files"]["gallery"] = "gallery/index.json"
    pack.setdefault("digests", {})["gallery"] = digest.hexdigest()[:16]
    pack_path.write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    return {"read": len(pool), "pictures": len(pictures), "features": kept, "of": int((~skip).sum())}
