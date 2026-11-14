"""Modal definitions for ``ii run --remote``. Imported only when a remote run starts.

One Volume, ``interpretableintelligence``, holds everything the container keeps between runs:

- ``clips/<key>/``: uploaded clip directories, keyed by content, so a second run on the same
  clips uploads nothing,
- ``runs/<key>/<run_id>/``: run folders. A rerun into the same local ``--out`` resumes here and
  skips the stages that are already valid, exactly like a local rerun,
- ``hf-cache/``: Hugging Face weights, so a model is downloaded (and paid for) once,
- ``acts-cache/``: the shared activation cache (``act_cache``, ``ii cache``), so a remote rerun
  with a new seed, split or SAE size skips the GPU forward passes. Its usual size cap applies.

The image installs the same dependencies as ``pyproject.toml`` (CUDA PyTorch from PyPI) and
mounts this package's source, so local edits reach the container without a rebuild. GR00T gets
its own app and image because LeRobot is a large install the other adapters do not need.
"""

from __future__ import annotations

from pathlib import Path

import modal

from interpretableintelligence.remote.image import GROOT_DEPENDENCIES, PYTHON_VERSION, RUNTIME_DEPENDENCIES
from interpretableintelligence.remote.pricing import CPU_CORES, DEFAULT_GPU, MEMORY_GIB

APP_NAME = "interpretableintelligence"
VOLUME_NAME = "interpretableintelligence"
VOLUME_ROOT = "/vol"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _image(extra: tuple[str, ...] = ()) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .uv_pip_install(*RUNTIME_DEPENDENCIES, *extra)
        .env({"HF_HOME": f"{VOLUME_ROOT}/hf-cache", "II_ACTS_CACHE": f"{VOLUME_ROOT}/acts-cache", "PYTHONUNBUFFERED": "1"})
        .add_local_python_source("interpretableintelligence")
    )


_FUNCTION = {
    "gpu": DEFAULT_GPU,
    "cpu": CPU_CORES,
    "memory": MEMORY_GIB * 1024,
    "volumes": {VOLUME_ROOT: volume},
    # Replaced per run by the --max-cost timeout. Generator functions are never retried, so a
    # failed run reports its error once instead of billing twice.
    "timeout": 3600,
}

app = modal.App(APP_NAME)
groot_app = modal.App(f"{APP_NAME}-groot")


def _events(request: dict):
    from interpretableintelligence.remote.worker import run_events  # noqa: PLC0415

    yield from run_events(request, Path(VOLUME_ROOT), commit=volume.commit)


@app.function(image=_image(), **_FUNCTION)
def run_pipeline(request: dict):
    yield from _events(request)


@groot_app.function(image=_image(GROOT_DEPENDENCIES), **_FUNCTION)
def run_pipeline_groot(request: dict):
    yield from _events(request)
