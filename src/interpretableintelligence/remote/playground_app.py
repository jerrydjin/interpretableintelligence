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

Bring your own model (``II_PLAYGROUND_BUILDS=1``, on by default in ``ii deploy-live``): the page can
name a Hugging Face vision model, and ``build`` (a separate GPU function) trains a dictionary for
it on robot clips baked into its image, then writes a pack to the ``ii-playground`` Volume, where
the web containers find it. Cost guards: one build at a time (``max_containers=1``), the
function's timeout (``II_BUILD_TIMEOUT``) as a hard stop, a daily cap and a parameter cap checked
before anything is downloaded (``playground.builds.Limits``). Built packs are kept per model,
commit and layer, so the same model is never built twice. Model weights downloaded by builds stay
on the Volume and the web containers read them from there.

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
BUILDS = os.environ.get("II_PLAYGROUND_BUILDS", "0") == "1"
BUILD_GPU = os.environ.get("II_BUILD_GPU", "L4")
BUILD_TIMEOUT = int(os.environ.get("II_BUILD_TIMEOUT", "1200"))
BUILD_CLIPS = int(os.environ.get("II_BUILD_CLIPS", "40"))
# Passed through to the containers, where playground.builds.Limits.from_env reads them.
BUILD_SETTINGS = {key: os.environ[key] for key in ("II_BUILD_PER_DAY", "II_BUILD_TIMEOUT", "II_BUILD_MAX_PARAMS", "II_BUILD_CLIPS") if os.environ.get(key)}
# The containers import this module again, so BUILDS must read the same there as at deploy time:
# otherwise the build function and the Volume exist locally but not remotely, and Modal refuses
# to start the containers ("Function has 1 dependencies but container got 2 object ids").
BUILD_SETTINGS["II_PLAYGROUND_BUILDS"] = "1" if BUILDS else "0"
DATA = "/data"
BUILDS_ROOT = f"{DATA}/builds"
BUILD_HF = f"{DATA}/hf"
CLIPS_IN_IMAGE = "/clips"


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
    .env({"HF_HOME": HF_IN_IMAGE, "PYTHONUNBUFFERED": "1", "II_PLAYGROUND_ORIGINS": ORIGINS, **BUILD_SETTINGS})
    .add_local_dir(_stage(PACKS_DIR) if modal.is_local() else PACKS_DIR, PACKS_IN_IMAGE, copy=True)
    .run_commands(_DOWNLOAD_WEIGHTS)
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_python_source("interpretableintelligence")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(f"{APP_NAME}-builds", create_if_missing=True) if BUILDS else None
_web_volumes = {DATA: volume} if volume is not None else {}

# The build function's image: the same dependencies, the package, and DROID robot clips (MIT
# licence) fetched once at image build, so a build downloads only the model it was asked for.
build_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .uv_pip_install(*RUNTIME_DEPENDENCIES)
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/root/clips-hf"})
    .add_local_python_source("interpretableintelligence", copy=True)
    # The source lands in /root, which Modal puts on the path only in running containers, not
    # in image build steps, so the build command adds it itself.
    .run_commands(
        "PYTHONPATH=/root python -c \"from interpretableintelligence.sample_clips import fetch_droid_clips; "
        f"fetch_droid_clips('{CLIPS_IN_IMAGE}', n={BUILD_CLIPS})\""
    )
    .env({"HF_HOME": BUILD_HF, **BUILD_SETTINGS})
)

if BUILDS:

    @app.function(
        image=build_image,
        gpu=BUILD_GPU,
        cpu=4.0,
        memory=16 * 1024,
        volumes={DATA: volume},
        timeout=BUILD_TIMEOUT,
        min_containers=0,
        max_containers=1,
        retries=0,
    )
    def build(plan: dict) -> None:
        from interpretableintelligence.playground.builds import JobStore, ModelPlan, execute_job

        store = JobStore(Path(BUILDS_ROOT), sync=volume.reload, persist=volume.commit)
        execute_job(ModelPlan(**plan), store, Path(CLIPS_IN_IMAGE))
        volume.commit()


def _link_weights(pack_dir: Path) -> None:
    """Make a built pack's model weights (on the Volume) visible to this container's offline Hub cache."""
    import json

    name = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))["model"]["name"]
    folder = "models--" + name.replace("/", "--")
    source = Path(BUILD_HF) / "hub" / folder
    target = Path(HF_IN_IMAGE) / "hub" / folder
    if source.is_dir() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)


def _throttled(fn, every_s: float = 2.0):
    import time

    last = [0.0]

    def call() -> None:
        if time.monotonic() - last[0] < every_s:
            return
        last[0] = time.monotonic()
        try:
            fn()
        except Exception:  # noqa: BLE001  (a reload can fail while a file is open; the next one catches up)
            pass

    return call


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
    volumes=_web_volumes,
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            for model in self.packs.values():
                model.to(device)
        self.table = self.packs
        self.builds = None
        if BUILDS:
            from interpretableintelligence.playground.builds import Builds, JobStore, Limits
            from interpretableintelligence.playground.registry import PackRegistry

            sync = _throttled(volume.reload)
            store = JobStore(Path(BUILDS_ROOT), sync=sync, persist=volume.commit)
            self.builds = Builds(store, lambda plan: build.spawn(plan.as_dict()), Limits.from_env())
            self.table = PackRegistry(self.packs, built_dir=store.root / "packs", device=device, before_load=_link_weights, sync=sync)

    @modal.asgi_app()
    def web(self):
        from interpretableintelligence.playground.http import create_app

        origins = [o.strip() for o in os.environ.get("II_PLAYGROUND_ORIGINS", "*").split(",") if o.strip()]
        return create_app(lambda: self.table, allow_origins=origins, builds=lambda: self.builds)
