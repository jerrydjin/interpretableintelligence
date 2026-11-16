"""Pack a run for the live in-browser playground (``ii export-live``).

The live playground runs the model and the SAE in the visitor's browser
(ONNX Runtime Web, WebGPU with a WebAssembly fallback). Nothing about a
visitor's image is precomputed: the page encodes whatever image it is given,
reads every feature on every patch, and reruns the rest of the network when a
feature is edited. This module writes the files that page needs:

``pack.json``
    What the page reads first: model, grid, preprocessing, SAE shape, file
    names, and the run it came from. ``format: ii-live-pack``, ``version: 1``.
``front.onnx``
    Pixels ``[1, 3, S, S]`` to the hidden state after block ``layer``
    (``[1, prefix + h*w, d]``), the layer the SAE was trained on.
``back.onnx``
    That hidden state to the model's final tokens (the hf adapter's readout,
    ``last_hidden_state``). The page edits the patch tokens between the two.
``sae.onnx``
    Patch tokens ``[n, d]`` to ReLU pre-activations ``[n, n_features]``. The
    page applies ``topk`` / ``jumprelu`` itself, exactly as ``sae.SAE.encode``.
``decoder.bin``
    ``W_dec`` transposed, ``[n_features, d]`` float16, so one feature's
    direction is one contiguous row.
``features.json``
    Per feature: label, badge and the scorer's checks (copied, never
    recomputed), plus activation statistics over the train split.
``samples/``
    A few frames from the run's clips (held-out clips first) to try first.

Weights are stored as float16 and cast to float32 inside the graph, so the
browser computes in float32 and the download is half the size.

The edit the page applies is the playground's (``steer.steer_edit``)::

    x_edited = x + sum_i (f'_i - f_i) * W_dec[:, i]

and its readout is the local playground's: per-patch cosine distance between
the clean and edited final tokens. It is exploration only. The pack copies the
scoreboard's badges; it never writes to the run folder, and nothing in the
browser can make a feature ``real``.
"""

from __future__ import annotations

import io
import json
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

FORMAT = "ii-live-pack"
VERSION = 1
DEFAULT_SAMPLES = 16
SAMPLE_PX = 448
SAMPLE_SHORT_EDGE = 288
# Feature checks copied from the scoreboard into the pack, when present.
CHECK_KEYS = (
    "scored",
    "distinct_clips_top20",
    "label_holds_heldout",
    "ablation_effect",
    "random_percentile",
    "n_random",
    "absent_in_random_init",
)


class LiveExportError(ValueError):
    """The run cannot be packed for the live playground."""


@dataclass
class LiveSummary:
    out_dir: Path
    n_features: int
    n_real: int
    samples: int
    bytes_total: int
    files: dict[str, int] = field(default_factory=dict)
    meta_source: str = "run"


class _Float32From16(torch.nn.Module):
    """Parametrisation: the weight is stored as float16 and used as float32."""

    def forward(self, stored: torch.Tensor) -> torch.Tensor:
        return stored.float()


def store_half(module: torch.nn.Module, min_elems: int = 1024) -> None:
    """Store every large float parameter of ``module`` as float16, cast back in the graph."""
    from torch.nn.utils import parametrize

    for sub in list(module.modules()):
        for name, param in list(sub.named_parameters(recurse=False)):
            if param.dtype == torch.float32 and param.numel() >= min_elems:
                setattr(sub, name, torch.nn.Parameter(param.detach().half(), requires_grad=False))
                parametrize.register_parametrization(sub, name, _Float32From16(), unsafe=True)


def _call_blocks(blocks, hidden: torch.Tensor) -> torch.Tensor:
    for block in blocks:
        out = block(hidden)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


class _Front(torch.nn.Module):
    def __init__(self, embeddings: torch.nn.Module, blocks) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return _call_blocks(self.blocks, self.embeddings(pixel_values))


class _Back(torch.nn.Module):
    def __init__(self, blocks, norm: torch.nn.Module | None) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        self.norm = norm if norm is not None else torch.nn.Identity()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(_call_blocks(self.blocks, hidden))


class _SaePre(torch.nn.Module):
    def __init__(self, sae) -> None:
        super().__init__()
        self.W_enc = torch.nn.Parameter(sae.W_enc.detach().float().cpu().clone())
        self.b_enc = torch.nn.Parameter(sae.b_enc.detach().float().cpu().clone())
        self.b_dec = torch.nn.Parameter(sae.b_dec.detach().float().cpu().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)


def split_model(adapter) -> tuple[torch.nn.Module, torch.nn.Module]:
    """The adapter's model cut at its hooked block: ``front`` (pixels to block ``layer``) and ``back``."""
    model = adapter.model
    if getattr(adapter, "is_video", False) or getattr(adapter, "_layout", "tokens") != "tokens":
        raise LiveExportError("the live playground takes image transformers (ViT, DINOv2, CLIP/SigLIP towers) for now")
    embeddings = getattr(model, "embeddings", None)
    if embeddings is None:
        raise LiveExportError(f"{type(model).__name__} has no .embeddings, so it cannot be split for the browser")
    norm = None
    for name in ("layernorm", "post_layernorm", "norm"):
        candidate = getattr(model, name, None)
        if isinstance(candidate, torch.nn.Module):
            norm = candidate
            break
    blocks = list(adapter._blocks)
    layer = int(adapter.layer)
    return _Front(embeddings, blocks[:layer]).eval(), _Back(blocks[layer:], norm).eval()


def _export_onnx(module: torch.nn.Module, example: torch.Tensor, path: Path, input_name: str, output_name: str, dynamic: bool = False) -> None:
    store_half(module)
    axes = {input_name: {0: "n"}, output_name: {0: "n"}} if dynamic else None
    # The tracer warns about Python branches on fixed shapes; the pack fixes the input size, so they are expected.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            (example,),
            str(path),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=18,
            dynamic_axes=axes,
            dynamo=False,
            # Folding would fold the float16-to-float32 casts back into float32 weights.
            do_constant_folding=False,
        )


def export_live(
    run_dir: Path,
    out_dir: Path,
    clips_dir: Path | None = None,
    meta_from: Path | None = None,
    n_samples: int = DEFAULT_SAMPLES,
    title: str | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> LiveSummary:
    """Write a live-playground pack for ``run_dir`` into ``out_dir``. See the module docstring."""
    from interpretableintelligence.adapters import get_adapter
    from interpretableintelligence.sae import load_sae
    from interpretableintelligence.artifacts import read_json

    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    manifest = read_json(run_dir / "manifest.json")
    model_info = manifest.get("model") or {}
    adapter_name = str(model_info.get("adapter") or "")
    if not adapter_name.startswith("hf:"):
        raise LiveExportError(f"the live playground runs Hugging Face image models (hf:<model>); this run uses {adapter_name or 'no adapter'}")
    if not (run_dir / "sae" / "weights.safetensors").is_file():
        raise LiveExportError(f"{run_dir} has no trained SAE (sae/weights.safetensors); run ii run first")
    layer = int(manifest["layer"])
    device = torch.device("cpu")

    log("loading the model and SAE")
    adapter = get_adapter(adapter_name)(layer=layer, device=device)
    adapter.load(random_init=False)
    sha = adapter.model_sha()
    if sha != model_info.get("sha"):
        raise LiveExportError(f"loaded weights sha {sha} does not match the run's model sha {model_info.get('sha')}")
    sae, config = load_sae(run_dir / "sae", device)
    sae.eval()

    features = _feature_meta(run_dir, manifest, meta_from, sae, log)
    stats = feature_stats(run_dir / "acts", sae, log)
    for feature, (peak, freq, mean) in zip(features, stats):
        feature["max"] = _round(peak)
        feature["freq"] = _round(freq, 6)
        feature["mean"] = _round(mean)

    out_dir.mkdir(parents=True, exist_ok=True)
    size = int(adapter._size)
    grid_h, grid_w = int(adapter._h), int(adapter._w)
    prefix = int(adapter._prefix)
    d_model = int(adapter.d_model)

    log("exporting the model to ONNX")
    front, back = split_model(adapter)
    pixels = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        hidden = front(pixels)
        reference = back(hidden)
        full = adapter.model(pixel_values=pixels, **adapter._extra)
    gap = float((reference - full.last_hidden_state).abs().max())
    if gap > 1e-4:
        raise LiveExportError(f"split model does not reproduce the full forward (max gap {gap:.2e})")
    if tuple(hidden.shape) != (1, prefix + grid_h * grid_w, d_model):
        raise LiveExportError(f"block {layer} output {tuple(hidden.shape)} is not {prefix} prefix + {grid_h}x{grid_w} patch tokens")
    _export_onnx(front, pixels, out_dir / "front.onnx", "pixel_values", "hidden")
    _export_onnx(back, hidden, out_dir / "back.onnx", "hidden", "final")
    _export_onnx(_SaePre(sae).eval(), torch.zeros(grid_h * grid_w, d_model), out_dir / "sae.onnx", "x", "pre", dynamic=True)
    decoder = sae.W_dec.detach().float().cpu().T.contiguous().numpy().astype("<f2")
    (out_dir / "decoder.bin").write_bytes(decoder.tobytes())

    log("writing sample frames")
    samples = _write_samples(manifest, run_dir, clips_dir, out_dir / "samples", n_samples)

    (out_dir / "features.json").write_text(json.dumps(features, separators=(",", ":")) + "\n", encoding="utf-8")
    scoreboard = _read_optional(run_dir / "scoreboard.json")
    sae_block = (scoreboard or {}).get("sae") if isinstance(scoreboard, dict) else None
    if sae_block is None and meta_from is not None:
        sae_block = (_read_optional(Path(meta_from) / "data" / "run.json") or {}).get("sae")
    n_real = sum(1 for feature in features if feature.get("real") is True)
    name = str(model_info.get("name") or adapter_name[3:])
    pack = {
        "format": FORMAT,
        "version": VERSION,
        "title": title or f"{name.split('/')[-1]} on {manifest['run_id']}",
        "run_id": manifest["run_id"],
        "model": {
            "name": name,
            "adapter": adapter_name,
            "sha": sha,
            "layer": layer,
            "n_layers": len(adapter._blocks),
            "image_size": size,
            "patch_px": int(adapter._patch),
            "grid": {"h": grid_h, "w": grid_w},
            "prefix_tokens": prefix,
            "d_model": d_model,
            "mean": [float(v) for v in adapter._mean],
            "std": [float(v) for v in adapter._std],
            "resize": "squash",
            "readout": "last_hidden_state",
        },
        "sae": {
            "n_features": int(config.n_features),
            "activation": config.activation,
            "k": config.k,
            "threshold": config.threshold,
            "usable": bool(isinstance(sae_block, dict) and sae_block.get("usable") is True),
            "ev_heldout": (sae_block or {}).get("ev_heldout") if isinstance(sae_block, dict) else None,
            "loss_recovered": (sae_block or {}).get("loss_recovered") if isinstance(sae_block, dict) else None,
        },
        "n_real": n_real,
        "files": {
            "front": "front.onnx",
            "back": "back.onnx",
            "sae": "sae.onnx",
            "decoder": "decoder.bin",
            "features": "features.json",
        },
        "samples": samples,
        "train_clips": sum(1 for clip in manifest.get("clips") or [] if clip.get("split") == "train"),
        "meta_source": "export" if meta_from is not None else "run",
        "evidence": False,
        "note": "Exploration only. Badges are copied from the run's scoreboard; nothing in the browser changes them.",
    }
    (out_dir / "pack.json").write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    files = {path.relative_to(out_dir).as_posix(): path.stat().st_size for path in sorted(out_dir.rglob("*")) if path.is_file()}
    return LiveSummary(
        out_dir=out_dir,
        n_features=int(config.n_features),
        n_real=n_real,
        samples=len(samples),
        bytes_total=sum(files.values()),
        files=files,
        meta_source=pack["meta_source"],
    )


def feature_stats(acts_dir: Path, sae, log: Callable[[str], None] = lambda _msg: None, chunk: int = 8192) -> list[tuple[float, float, float]]:
    """Per feature over the run's train tokens: peak activation, firing rate, mean when firing."""
    from safetensors.numpy import load_file

    shards = sorted(Path(acts_dir).glob("shard-*.safetensors"))
    if not shards:
        raise LiveExportError(f"{acts_dir} has no activation shards")
    split_codes = _train_mask_source(Path(acts_dir).parent)
    n = int(sae.n_features)
    peak = torch.zeros(n)
    count = torch.zeros(n, dtype=torch.float64)
    total = torch.zeros(n, dtype=torch.float64)
    rows_seen = 0
    log(f"activation statistics over {len(shards)} shards")
    with torch.no_grad():
        for shard in shards:
            data = load_file(str(shard))
            acts = data["acts"]
            keep = split_codes(data["clip_idx"]) if split_codes is not None else np.ones(len(acts), dtype=bool)
            acts = acts[keep]
            for start in range(0, len(acts), chunk):
                x = torch.from_numpy(np.ascontiguousarray(acts[start : start + chunk], dtype=np.float32))
                f = sae.encode(x)
                peak = torch.maximum(peak, f.max(dim=0).values)
                fired = f > 0
                count += fired.sum(dim=0).double()
                total += f.sum(dim=0).double()
                rows_seen += len(x)
    rows_seen = max(rows_seen, 1)
    out = []
    for i in range(n):
        c = float(count[i])
        out.append((float(peak[i]), c / rows_seen, float(total[i]) / c if c else 0.0))
    return out


def _train_mask_source(run_dir: Path):
    """``clip_idx -> bool`` for train clips, from the manifest's clip order and split labels."""
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    clips = manifest.get("clips") or []
    train = np.array([clip.get("split") == "train" for clip in clips], dtype=bool)
    if not train.any():
        return None
    return lambda clip_idx: train[np.asarray(clip_idx, dtype=np.int64)]


def _feature_meta(run_dir: Path, manifest: dict, meta_from: Path | None, sae, log) -> list[dict]:
    if meta_from is None:
        from interpretableintelligence.view import load_run

        source = load_run(run_dir).run_payload["features"]
    else:
        source = _export_features(Path(meta_from), manifest, run_dir, sae, log)
    by_id = {int(item["id"]): item for item in source if isinstance(item, dict) and "id" in item}
    out = []
    for index in range(int(sae.n_features)):
        item = by_id.get(index, {})
        entry: dict = {"id": index}
        label = item.get("label")
        if isinstance(label, str) and label:
            entry["label"] = label
        entry["badge"] = item.get("badge") or "unverified"
        if item.get("real") is True:
            entry["real"] = True
        checks = {key: item[key] for key in CHECK_KEYS if item.get(key) is not None}
        if checks:
            entry["checks"] = checks
        out.append(entry)
    return out


def _export_features(export_dir: Path, manifest: dict, run_dir: Path, sae, log) -> list[dict]:
    """Labels and badges from an ``ii export`` of the same run, after checking the dictionary matches.

    A badge belongs to one dictionary. The export's top activations are recomputed with this run's
    SAE on this run's activations; any mismatch refuses the copy.
    """
    run_json = _read_optional(export_dir / "data" / "run.json")
    if not isinstance(run_json, dict):
        raise LiveExportError(f"{export_dir} is not an ii export (no data/run.json)")
    model = run_json.get("model") or {}
    if model.get("sha") != (manifest.get("model") or {}).get("sha") or int(run_json.get("layer", -1)) != int(manifest["layer"]):
        raise LiveExportError("the export is of a different model or layer")
    features = [item for item in run_json.get("features") or [] if isinstance(item, dict)]
    checked = verify_against_export(export_dir, manifest, run_dir, sae, features)
    log(f"dictionary matches the export on {checked} top activations")
    return features


def verify_against_export(export_dir: Path, manifest: dict, run_dir: Path, sae, features: list[dict], max_features: int = 24, tol: float = 2e-3) -> int:
    from safetensors.numpy import load_file

    clip_index = {str(clip["clip_id"]): i for i, clip in enumerate(manifest.get("clips") or [])}
    wanted: list[tuple[int, int, int, int, int, int, float]] = []
    chosen = sorted(features, key=lambda item: (item.get("real") is not True, int(item["id"])))[:max_features]
    for item in chosen:
        detail = _read_optional(export_dir / "data" / "features" / f"{int(item['id'])}.json") or {}
        for top in (detail.get("top_clips") or [])[:3]:
            clip_id = str(top.get("clip_id"))
            if clip_id not in clip_index:
                raise LiveExportError(f"export clip {clip_id} is not in this run")
            wanted.append((int(item["id"]), clip_index[clip_id], int(top["window"]), int(top["t"]), int(top["h"]), int(top["w"]), float(top["value"])))
    if not wanted:
        raise LiveExportError("the export lists no top activations to check the dictionary against")
    found: dict[tuple[int, int, int, int, int], np.ndarray] = {}
    keys = {(c, win, t, h, w) for _f, c, win, t, h, w, _v in wanted}
    for shard in sorted((run_dir / "acts").glob("shard-*.safetensors")):
        data = load_file(str(shard))
        stacked = np.stack([data["clip_idx"], data["window"], data["t"], data["h"], data["w"]], axis=1).astype(np.int64)
        for row, key in enumerate(map(tuple, stacked)):
            if key in keys:
                found[key] = data["acts"][row].astype(np.float32)
    with torch.no_grad():
        for feature_id, c, win, t, h, w, value in wanted:
            row = found.get((c, win, t, h, w))
            if row is None:
                raise LiveExportError(f"no activation row for clip {c} t={t} h={h} w={w}")
            got = float(sae.encode(torch.from_numpy(row)[None])[0, feature_id])
            if abs(got - value) > tol * max(1.0, abs(value)):
                raise LiveExportError(
                    f"this SAE gives feature {feature_id} {got:.5f} where the export recorded {value:.5f}; "
                    "the export is of a different dictionary, so its badges cannot be copied"
                )
    return len(wanted)


def _write_samples(manifest: dict, run_dir: Path, clips_dir: Path | None, dest: Path, n_samples: int) -> list[dict]:
    """Middle frames of up to ``n_samples`` clips (held-out first) as JPEGs."""
    from interpretableintelligence.clips import decode_video
    from interpretableintelligence.view import _resolve_pixels

    if n_samples <= 0:
        return []
    pixels = _resolve_pixels(manifest, run_dir, clips_dir)
    clips = [clip for clip in manifest.get("clips") or [] if isinstance(clip, dict)]
    ordered = [c for c in clips if c.get("split") == "heldout"] + [c for c in clips if c.get("split") != "heldout"]
    # Spread the choice over the list so neighbouring episodes do not crowd it.
    step = max(len(ordered) // max(n_samples, 1), 1)
    picked = ordered[::step][:n_samples] if len(ordered) > n_samples else ordered
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for clip in picked:
        loaded = pixels.get(str(clip["clip_id"]))
        if loaded is None or loaded.status != "ok":
            continue
        if loaded.array is not None:
            frames = np.asarray(loaded.array)
        elif loaded.path is not None:
            frames, _fps = decode_video(loaded.path, short_edge=SAMPLE_SHORT_EDGE)
        else:
            continue
        frame = frames[len(frames) // 2]
        if frame.dtype != np.uint8:
            peak = float(frame.max()) or 1.0
            frame = np.clip(frame / (peak if peak > 1.5 else 1.0) * (255 if peak <= 1.5 else 1), 0, 255).astype(np.uint8)
        name = f"{clip['clip_id']}.jpg"
        (dest / name).write_bytes(encode_jpeg(frame))
        out.append({"file": f"samples/{name}", "clip_id": str(clip["clip_id"]), "split": clip.get("split"), "frame": len(frames) // 2})
    return out


def encode_jpeg(rgb: np.ndarray, max_edge: int = SAMPLE_PX, quality: int = 4) -> bytes:
    """``[H, W, 3]`` uint8 to JPEG bytes through PyAV's MJPEG encoder (no imaging dependency)."""
    import av

    height, width = rgb.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    out_w, out_h = max(2, int(round(width * scale / 2)) * 2), max(2, int(round(height * scale / 2)) * 2)
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[..., :3]), format="rgb24")
    if (out_w, out_h) != (width, height):
        frame = frame.reformat(width=out_w, height=out_h, interpolation="AREA")
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="image2pipe") as container:
        stream = container.add_stream("mjpeg")
        stream.width, stream.height = out_w, out_h
        stream.pix_fmt = "yuvj420p"
        stream.options = {"qmin": str(quality), "qmax": str(quality)}
        for packet in stream.encode(frame.reformat(format="yuvj420p")):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def _read_optional(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _round(value: float, digits: int = 4) -> float:
    value = float(value)
    return round(value, digits) if math.isfinite(value) else 0.0
