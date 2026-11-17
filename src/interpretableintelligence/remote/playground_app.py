"""Modal deployment of the playground backend (``ii deploy-live``). Bills the deployer's Modal account.

One class, ``Playground``, serves API v1 (docs/PLAYGROUND-API.md) as a web endpoint:

- Scale to zero: no container runs while nobody uses the playground. A container stays warm for
  ``II_PLAYGROUND_IDLE`` seconds (default 300) after the last request, and at most
  ``II_PLAYGROUND_MAX_CONTAINERS`` (default 2) run at once, which caps the bill under load.
- Cold starts: the model weights are baked into the image at deploy time, and the loaded models
  are kept in a CPU memory snapshot, so a cold container restores them instead of loading them
  again, then moves them to the GPU.
- ``II_PLAYGROUND_GPU`` picks the GPU (default T4; ``none`` runs on CPU, which is enough for
  DINOv2-small and costs about a tenth as much).

Only each pack's ``pack.json`` and ``sae.safetensors`` go into the image; the browser fetches
labels and sample pictures from the static site as before.

Settings are read when the app is deployed (environment variables of ``ii deploy-live``).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from interpretableintelligence.remote.image import PYTHON_VERSION, RUNTIME_DEPENDENCIES

APP_NAME = "ii-playground"
PACKS_IN_IMAGE = "/packs"
HF_IN_IMAGE = "/root/hf"
REPO = Path(__file__).resolve().parents[3]

PACKS_DIR = Path(os.environ.get("II_PLAYGROUND_PACKS") or REPO / "demo-site" / "live" / "packs")
GPU = os.environ.get("II_PLAYGROUND_GPU", "T4")
GPU = None if GPU.lower() in ("", "none", "cpu") else GPU
IDLE_S = int(os.environ.get("II_PLAYGROUND_IDLE", "300"))
MAX_CONTAINERS = int(os.environ.get("II_PLAYGROUND_MAX_CONTAINERS", "2"))
ORIGINS = os.environ.get("II_PLAYGROUND_ORIGINS", "*")


def _stage(packs_dir: Path) -> Path:
    """A temporary copy of each pack's pack.json and sae.safetensors, the files the server needs."""
    import shutil
    import tempfile

    staged = Path(tempfile.mkdtemp(prefix="ii-playground-packs-"))
    for pack_json in sorted(Path(packs_dir).glob("*/pack.json")):
        dest = staged / pack_json.parent.name
        dest.mkdir()
        for name in ("pack.json", "sae.safetensors"):
            if (pack_json.parent / name).is_file():
                shutil.copy2(pack_json.parent / name, dest / name)
    return staged


# Runs in the image build, before this package is in the image, so it is a plain command.
_DOWNLOAD_WEIGHTS = (
    "python -c \"import glob, json, os; from huggingface_hub import snapshot_download; "
    "names = {json.load(open(p))['model']['name'] for p in glob.glob('%s/*/pack.json')}; "
    "[snapshot_download(n) for n in sorted(names) if not os.path.exists(n)]\"" % PACKS_IN_IMAGE
)


image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .uv_pip_install(*RUNTIME_DEPENDENCIES)
    .env({"HF_HOME": HF_IN_IMAGE, "PYTHONUNBUFFERED": "1", "II_PLAYGROUND_ORIGINS": ORIGINS})
    .add_local_dir(_stage(PACKS_DIR) if modal.is_local() else PACKS_DIR, PACKS_IN_IMAGE, copy=True)
    .run_commands(_DOWNLOAD_WEIGHTS)
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_python_source("interpretableintelligence")
)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu=GPU,
    cpu=2.0,
    memory=4096,
    scaledown_window=IDLE_S,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    enable_memory_snapshot=True,
    timeout=600,
)
@modal.concurrent(max_inputs=16)
class Playground:
    @modal.enter(snap=True)
    def load(self) -> None:
        import torch

        from interpretableintelligence.playground.compute import load_packs

        self.packs = load_packs(Path(PACKS_IN_IMAGE), device=torch.device("cpu"))

    @modal.enter(snap=False)
    def to_gpu(self) -> None:
        import torch

        if torch.cuda.is_available():
            for model in self.packs.values():
                model.to(torch.device("cuda"))

    @modal.asgi_app()
    def web(self):
        from interpretableintelligence.playground.http import create_app

        origins = [o.strip() for o in os.environ.get("II_PLAYGROUND_ORIGINS", "*").split(",") if o.strip()]
        return create_app(lambda: self.packs, allow_origins=origins)
