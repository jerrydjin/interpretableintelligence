"""Run the real-GR00T steering scorecard (scripts/groot_steer_scorecard.sh) on a Modal GPU, billed to your own account.

    modal volume put ii-groot-steer groot-1028-v9-n100.tar.xz /capture/                           # once
    modal run scripts/modal_groot_steer.py::prefetch                                              # once, CPU only
    modal run scripts/modal_groot_steer.py::smoke                                                 # ~10 GPU minutes
    modal run --detach scripts/modal_groot_steer.py::scorecard                                    # the full run
    modal volume get ii-groot-steer results ./results                                            # fetch results

Everything persistent lives on the Volume ``ii-groot-steer``: the capture, Hugging Face weights, the LIBERO-plus
assets, and the work folder (clips, run, steering). The work folder is committed every few minutes, so a run that
hits its timeout or is stopped resumes where it left off when started again. The GPU function has a hard
``--hours`` timeout (default 3), which caps the bill: about 0.6 USD per T4 hour at Modal's list price.
Guide: docs/groot-steer-scorecard.md.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
VOL = "/vol"
PLUS_DIR = "/opt/LIBERO-plus"
CAPTURE = f"{VOL}/capture/groot-1028-v9-n100.tar.xz"
WORK = f"{VOL}/work"
RESULTS = f"{VOL}/results"
ASSETS_UNZIP = f"{VOL}/libero-assets"
CHECKPOINT = "nvidia/gr00t17-lerobot-libero_10-640"
VLM_PROCESSOR = "Qwen/Qwen3-VL-2B-Instruct"
GPU = os.environ.get("II_MODAL_GPU", "T4")
HOURS = float(os.environ.get("II_MODAL_HOURS", "3"))

volume = modal.Volume.from_name("ii-groot-steer", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "xz-utils", "unzip", "libegl1", "libgl1", "libgles2", "libosmesa6", "libglib2.0-0",
        "libexpat1", "libfontconfig1-dev", "libmagickwand-dev",
    )
    # lerobot's libero extra brings mujoco and robosuite through hf-libero; LIBERO-plus then replaces hf-libero's package.
    .pip_install("lerobot[groot,libero]==0.7.0")
    .run_commands(
        "pip uninstall -y -q hf-libero",
        f"git clone -q --depth 1 https://github.com/sylvestf/LIBERO-plus.git {PLUS_DIR}",
        f"if [ -f {PLUS_DIR}/extra_requirements.txt ]; then pip install -q -r {PLUS_DIR}/extra_requirements.txt; fi",
        f"pip install -q -e {PLUS_DIR} --config-settings editable_mode=compat",
        "pip install -q 'gym==0.25.2'",  # LIBERO's envs import the old gym API
        # LIBERO asks for its paths on stdin the first time it is imported without a config; write it up front.
        f"mkdir -p /root/.libero && printf 'benchmark_root: {PLUS_DIR}/libero/libero\\nbddl_files: {PLUS_DIR}/libero/libero/bddl_files\\n"
        f"init_states: {PLUS_DIR}/libero/libero/init_files\\ndatasets: {PLUS_DIR}/libero/datasets\\nassets: {PLUS_DIR}/libero/libero/assets\\n'"
        " > /root/.libero/config.yaml",
        "python -c 'import mujoco, robosuite, libero.libero; print(\"libero ok\", libero.libero.__file__)'",
    )
    .add_local_dir(
        str(REPO), "/repo", copy=True,
        ignore=[".git", ".venv", "target", "node_modules", "runs", "clips", ".libero-plus", "**/__pycache__", "demo-site"],
    )
    .run_commands("pip install -q -e /repo")
    .env({"HF_HOME": f"{VOL}/hf-cache", "PYTHONUNBUFFERED": "1", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
          "NVIDIA_DRIVER_CAPABILITIES": "all"})
)
app = modal.App("ii-groot-steer", image=image)


@app.function(volumes={VOL: volume}, timeout=3 * 3600, cpu=4, memory=8192)
def prefetch() -> None:
    """CPU only: GR00T weights, the VLM processor and the LIBERO-plus assets onto the Volume."""
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(CHECKPOINT)
    snapshot_download(VLM_PROCESSOR, allow_patterns=["*.json", "*.txt", "*.jinja", "*.model"])
    volume.commit()
    marker = Path(ASSETS_UNZIP) / ".done"
    if not marker.is_file():
        import zipfile

        zip_path = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset", cache_dir="/tmp/hf")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ASSETS_UNZIP)
        marker.write_text("ok\n")
        volume.commit()
    print("prefetch done")


def _prepare() -> dict:
    from interpretableintelligence.adapters.groot.lib.place_libero_assets import find_assets_root

    root = find_assets_root(Path(ASSETS_UNZIP))
    dest = Path(PLUS_DIR) / "libero" / "libero" / "assets"
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        import shutil

        shutil.rmtree(dest)
    dest.symlink_to(root, target_is_directory=True)
    env = dict(os.environ, PLUS_DIR=PLUS_DIR)
    probe = "import mujoco; ctx = mujoco.GLContext(64, 64); ctx.make_current(); print('render ok')"
    if subprocess.run(["python", "-c", probe], env=env).returncode != 0:
        print("EGL rendering failed; falling back to OSMesa (slower)")
        env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    return env


def _run(cmd: list[str], env: dict) -> None:
    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(300):
            volume.commit()

    thread = threading.Thread(target=committer, daemon=True)
    thread.start()
    try:
        subprocess.run(cmd, env=env, cwd="/repo", check=True)
    finally:
        stop.set()
        volume.commit()


@app.function(volumes={VOL: volume}, timeout=600)
def check() -> None:
    """CPU only: the env stack imports and the assets resolve."""
    env = _prepare()
    code = "import lerobot.sim.integrations.libero, libero.libero; from libero.libero import benchmark; print(len(benchmark.get_benchmark_dict()['libero_10']().tasks), 'libero_10 tasks')"
    subprocess.run(["python", "-c", code], env=env, check=True)


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=1800, cpu=4, memory=24576)
def smoke() -> None:
    """One short episode: checks rendering, the policy load and the input parity check."""
    env = _prepare()
    _run(["ii", "record-libero", "--out", f"{VOL}/smoke/clips", "--episodes", "1", "--max-steps", "48", "--plus-dir", PLUS_DIR], env)


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), cpu=4, memory=24576)
def scorecard(features: str = "6262,7353,7775", episodes: int = 24) -> None:
    env = _prepare()
    env.update(FEATURES=features, EPISODES=str(episodes), RESULTS=RESULTS)
    _run(["bash", "scripts/groot_steer_scorecard.sh", CAPTURE, WORK], env)
