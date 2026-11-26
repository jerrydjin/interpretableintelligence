"""Run ``ii freeze-check`` on a Modal T4, billed to your own Modal account.

    modal run scripts/modal_freeze_check.py::load                                    # CPU: load the model, 1 clip
    modal run scripts/modal_freeze_check.py::smoke                                   # T4, 15-min cap: 5 clips of DINO-WM
    modal run --detach scripts/modal_freeze_check.py::run --model jepa-wms:dino_wm_droid   # T4: all clips, 1-hour cap
    modal run scripts/modal_freeze_check.py::fetch                                   # print results, save them locally

Reads the DROID clips (with arm poses) and V-JEPA 2-AC checkpoint that ``scripts/modal_vjepa2ac.py::prep``
put on the Volume ``ii-vjepa2ac``. Meta's jepa-wms code is cloned at ``JEPAWMS_COMMIT``; its checkpoints
download to the Volume on first use. Each GPU function has a hard timeout (``II_MODAL_HOURS``, default 1),
which caps the bill at about 0.6 USD per T4 hour at Modal's list price.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
VOL = "/vol"
CKPT = f"{VOL}/vjepa2-ac-vitg.pt"
CLIPS = f"{VOL}/clips/droid-h2"
RESULTS = f"{VOL}/results/freeze-check"
VJEPA2 = "/opt/vjepa2"
VJEPA2_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
JEPAWMS = "/opt/jepa-wms"
JEPAWMS_COMMIT = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
GPU = os.environ.get("II_MODAL_GPU", "T4")
HOURS = float(os.environ.get("II_MODAL_HOURS", "1"))

volume = modal.Volume.from_name("ii-vjepa2ac", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "pillow", "einops", "timm", "tensordict", "omegaconf", "pyyaml",
                 "ruamel.yaml", "matplotlib", "tqdm", "huggingface_hub", "lpips", "scipy", "decord", "h5py", "pandas",
                 "opencv-python-headless")
    .run_commands(
        f"git clone -q https://github.com/facebookresearch/vjepa2.git {VJEPA2}",
        f"cd {VJEPA2} && git checkout -q {VJEPA2_COMMIT}",
        f"git clone -q https://github.com/facebookresearch/jepa-wms.git {JEPAWMS}",
        f"cd {JEPAWMS} && git checkout -q {JEPAWMS_COMMIT}",
    )
    .add_local_dir(
        str(REPO),
        "/repo",
        copy=True,
        ignore=[".git", ".venv", "runs", "clips", "node_modules", "demo-site", "target", "**/__pycache__"],
    )
    .run_commands("pip install -q /repo")
    .env({"HF_HOME": f"{VOL}/hf", "TORCH_HOME": f"{VOL}/torch", "JEPAWM_HOME": "/opt", "JEPAWM_LOGS": "/tmp/jepawm-logs",
          "JEPAWM_CKPT": "/tmp/jepawm-ckpt", "JEPAWM_DSET": "/tmp/jepawm-dset", "JEPAWM_OSSCKPT": "/tmp/jepawm-oss"})
)
app = modal.App("ii-freeze-check", image=image)


def _dest(model: str, tag: str = "") -> str:
    return f"{RESULTS}/{model.replace(':', '-')}{tag}"


def _check(model: str, *extra: str, tag: str = "") -> dict:
    import shutil
    import time

    dest = _dest(model, tag)
    shutil.rmtree(dest, ignore_errors=True)
    code = VJEPA2 if model == "vjepa2-ac" else JEPAWMS
    cmd = ["ii", "freeze-check", "--clips", CLIPS, "--out", dest, "--model", model, "--code-dir", code, *extra]
    if model == "vjepa2-ac":
        cmd += ["--checkpoint", CKPT]
    if model == "jepa-wms:vjepa2_ac_droid":
        # The retrained V-JEPA 2-AC keeps Meta's frozen V-JEPA 2 ViT-g encoder, which its config reads from
        # $JEPAWM_OSSCKPT; the public V-JEPA 2-AC checkpoint on the Volume holds that encoder under "encoder".
        link = Path(os.environ["JEPAWM_OSSCKPT"], "vjepa2_opensource", "vjepa2_vit_giant.pth")
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(CKPT)
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout[-20000:], flush=True)
    if proc.returncode:
        raise RuntimeError(f"freeze-check failed ({proc.returncode})")
    load_lines = [ln[-400:] for ln in proc.stdout.splitlines() if "loaded pretrained" in ln or "IncompatibleKeys" in ln]
    volume.commit()
    out = json.loads(Path(dest, "freeze_check.json").read_text())
    out.pop("steps", None)
    out.pop("control_steps", None)
    out["wall_s"] = time.time() - t0
    out["load_log"] = load_lines
    return out


@app.function(volumes={VOL: volume}, timeout=1200, cpu=4, memory=16384)
def load_job(model: str = "jepa-wms:dino_wm_droid") -> dict:
    """CPU only: download and load the model, then one clip through the check (does the plumbing work?)."""
    return _check(model, "--max-clips", "1", "--bootstrap", "20", tag="-load")


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=900, memory=32768)
def smoke_job(model: str = "jepa-wms:dino_wm_droid") -> dict:
    return _check(model, "--max-clips", "5", "--bootstrap", "200", tag="-smoke")


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def full_job(model: str = "jepa-wms:dino_wm_droid", max_clips: int = 100) -> dict:
    return _check(model, "--max-clips", str(max_clips))


@app.function(volumes={VOL: volume}, timeout=300)
def read_job() -> dict:
    out = {}
    root = Path(RESULTS)
    for d in sorted(root.glob("*")) if root.exists() else []:
        f = d / "freeze_check.json"
        if f.exists():
            payload = json.loads(f.read_text())
            payload.pop("steps", None)
            payload.pop("control_steps", None)
            out[d.name] = payload
    return out


@app.function(volumes={VOL: volume}, timeout=300)
def file_job(name: str, rel: str) -> bytes:
    """One file alone: large function results go through Modal's blob store, which some networks block."""
    return Path(RESULTS, name, rel).read_bytes()


def _report(results: dict) -> None:
    dest = Path(os.environ.get("II_FREEZE_LOCAL", "freeze-check-results"))
    dest.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        (dest / f"{name}.json").write_text(json.dumps(r, indent=2) + "\n")
        for rel in ("index.html", "heatmaps/moving_frozen.png"):
            try:
                (dest / f"{name}-{rel.replace('/', '-')}").write_bytes(file_job.remote(name, rel))
            except Exception as exc:  # noqa: BLE001 - a missing picture is not worth the run
                print(name, rel, "not fetched:", exc)
        m, c = r["model"], r.get("control")
        print(f"{name}: verdict {r['verdict']}  {m['n_steps']} steps / {m['n_episodes']} episodes  wall_s {r.get('wall_s')}")
        for k in ("frozen_wins", "frozen_wins_zero_action", "change_predicted", "change_from_action", "moving_noticed"):
            ctl = f"   untrained {c[k]['value']:.3f} [{c[k]['lo']:.3f},{c[k]['hi']:.3f}]" if c else ""
            print(f"  {k:26s} {m[k]['value']:.3f} [{m[k]['lo']:.3f},{m[k]['hi']:.3f}]{ctl}")


@app.local_entrypoint()
def load(model: str = "jepa-wms:dino_wm_droid") -> None:
    r = load_job.remote(model)
    _report({Path(_dest(model, "-load")).name: r})


@app.local_entrypoint()
def smoke(model: str = "jepa-wms:dino_wm_droid") -> None:
    r = smoke_job.remote(model)
    _report({Path(_dest(model, "-smoke")).name: r})


@app.local_entrypoint()
def run(model: str = "jepa-wms:dino_wm_droid", max_clips: int = 100) -> None:
    full_job.spawn(model, max_clips)
    print(f"spawned freeze-check on {model}; fetch with ::fetch")


@app.local_entrypoint()
def fetch() -> None:
    _report(read_job.remote())
