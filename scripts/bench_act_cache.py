"""Time reruns and playground edits with and without the activation cache.

Uses only APIs that predate the cache (``execute_pipeline`` and ``Steerer``),
and switches the cache with ``II_ACTS_CACHE``, so the same script measures the
branch and an older checkout (``PYTHONPATH=<checkout>/src``).

    uv run python scripts/bench_act_cache.py setup  --work W [--model vit-b|videomae-b|fake]
    uv run python scripts/bench_act_cache.py rerun  --work W --out W/runs/x
    uv run python scripts/bench_act_cache.py steer  --work W [--requests 5]

``setup`` writes random-weight clips, a randomly initialised Hugging Face
model of a real size (forward cost does not depend on the weights, and nothing
is downloaded), and one scored reference run. ``rerun`` runs the pipeline into
a new folder without scoring and prints the stage times. ``steer`` loads the
playground on the reference run and times the first edit on a window and a
few slider tweaks after it. Each prints one JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODELS = ("vit-b", "videomae-b", "fake")


def _colour_similarity(self, frames):
    """Stand-in for the CLIP labeller (no download): concept i is colour channel i % 3 on a 4x4 grid."""
    pixels = torch.from_numpy(np.ascontiguousarray(frames)).float().permute(0, 3, 1, 2)
    pooled = torch.nn.functional.adaptive_avg_pool2d(pixels, 4)
    return torch.stack([pooled[:, i % 3] for i in range(len(self.names))], dim=1)


def _no_clip_download() -> None:
    from interpretableintelligence.labelers import ClipConceptSource

    ClipConceptSource._dense_similarity = _colour_similarity


def _settings(work: Path) -> dict:
    return json.loads((work / "bench.json").read_text(encoding="utf-8"))


def _adapter_name(settings: dict) -> str:
    if settings["model"] == "fake":
        return "fake"
    from interpretableintelligence.adapters.hf import make_spec

    return make_spec(settings["model_dir"])


def _layer(settings: dict) -> int:
    return 16 if settings["model"] == "fake" else 6


def setup(args) -> dict:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    clips = work / "clips"
    settings = {"model": args.model, "clips": str(clips), "n_clips": args.clips}
    if args.model == "fake":
        from interpretableintelligence.adapters.fake import generate_clips

        generate_clips(clips, n=args.clips, seed=0)
    else:
        from transformers import AutoModel, VideoMAEConfig, ViTConfig

        config = ViTConfig() if args.model == "vit-b" else VideoMAEConfig()  # ViT-B/16 at 224 px; VideoMAE-B, 16 frames
        torch.manual_seed(0)
        model_dir = work / args.model
        AutoModel.from_config(config).save_pretrained(model_dir)
        settings["model_dir"] = str(model_dir.resolve())
        clips.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for index in range(args.clips):
            np.save(clips / f"clip_{index:03d}.npy", rng.random((16, 240, 320, 3), dtype=np.float32))
    (work / "bench.json").write_text(json.dumps(settings), encoding="utf-8")
    started = time.perf_counter()
    _pipeline(settings, work / "runs" / "reference", score=True)
    return {"phase": "setup", "seconds": time.perf_counter() - started, **settings}


def _pipeline(settings: dict, out: Path, score: bool) -> dict:
    from interpretableintelligence.pipeline import execute_pipeline

    _no_clip_download()
    return execute_pipeline(
        adapter_name=_adapter_name(settings),
        clips_dir=Path(settings["clips"]),
        out_dir=out,
        layer=_layer(settings),
        seed=0,
        heldout_frac=0.2,
        prefilter=None if settings["model"] == "fake" else 2,
        sae_features=None if settings["model"] == "fake" else 256,
        score=score,
        serve=False,
        log=lambda line: print(line, file=sys.stderr),
    )


def rerun(args) -> dict:
    settings = _settings(Path(args.work))
    started = time.perf_counter()
    result = _pipeline(settings, Path(args.out), score=args.score)
    total = time.perf_counter() - started
    stages = result["manifest"]["stages"]
    return {
        "phase": "rerun",
        "cache": os.environ.get("II_ACTS_CACHE", "<default>"),
        "total_s": round(total, 3),
        **{f"{name}_s": round(float(stage["wall_s"]), 3) for name, stage in stages.items() if name in ("capture", "score")},
    }


def steer(args) -> dict:
    from interpretableintelligence.steer import Steerer

    work = Path(args.work)
    settings = _settings(work)
    run = work / "runs" / "reference"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    clip_ids = [clip["clip_id"] for clip in manifest["clips"]]
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    started = time.perf_counter()
    steerer = Steerer(run, Path(settings["clips"]))
    load_s = time.perf_counter() - started
    first, tweaks = [], []
    for clip_id in clip_ids[: args.clips_steered]:
        request = {"clip_id": clip_id, "window": 0, "features": [0], "kind": "scale", "value": 2.0}
        started = time.perf_counter()
        steerer.steer(request)
        first.append(time.perf_counter() - started)
        for step in range(args.requests):
            started = time.perf_counter()
            steerer.steer({**request, "value": 0.5 + step})
            tweaks.append(time.perf_counter() - started)
    return {
        "phase": "steer",
        "cache": os.environ.get("II_ACTS_CACHE", "<default>"),
        "load_s": round(load_s, 3),
        "first_edit_ms": round(1000 * float(np.median(first)), 1),
        "tweak_ms": round(1000 * float(np.median(tweaks)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("setup", "rerun", "steer"))
    parser.add_argument("--work", required=True)
    parser.add_argument("--model", choices=MODELS, default="vit-b")
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--out")
    parser.add_argument("--score", action="store_true", help="rerun: also run the score stage")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--clips-steered", type=int, default=3)
    args = parser.parse_args()
    import interpretableintelligence

    print(f"using {interpretableintelligence.__file__}", file=sys.stderr)
    result = {"setup": setup, "rerun": rerun, "steer": steer}[args.phase](args)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
