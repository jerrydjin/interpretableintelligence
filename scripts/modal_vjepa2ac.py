"""Run ``ii action-effects`` (V-JEPA 2-AC, hypothesis H2) on a Modal T4, billed to your own Modal account.

    modal run scripts/modal_vjepa2ac.py::prep                     # CPU: DROID clips with arm poses + checkpoint
    modal run scripts/modal_vjepa2ac.py::smoke                    # T4, ~10 min: fp16 check + 3 clips
    modal run --detach scripts/modal_vjepa2ac.py::run             # T4: all clips, trained and random predictor
    modal run scripts/modal_vjepa2ac.py::fetch                    # print results, save regions.png locally

Everything persistent lives on the Volume ``ii-vjepa2ac``: Meta's checkpoint (``vjepa2-ac-vitg.pt``), the clips
and the results. Each GPU function has a hard timeout (``II_MODAL_HOURS``, default 1), which caps the bill at
about 0.6 USD per T4 hour at Modal's list price. Meta's model code is cloned at ``VJEPA2_COMMIT``.
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
RESULTS = f"{VOL}/results"
CODE = "/opt/vjepa2"
VJEPA2_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
GPU = os.environ.get("II_MODAL_GPU", "T4")
HOURS = float(os.environ.get("II_MODAL_HOURS", "1"))
N_CLIPS = int(os.environ.get("II_H2_CLIPS", "100"))
SECONDS = float(os.environ.get("II_H2_SECONDS", "12"))

volume = modal.Volume.from_name("ii-vjepa2ac", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "pillow", "einops", "timm")
    .run_commands(
        f"git clone -q https://github.com/facebookresearch/vjepa2.git {CODE}",
        f"cd {CODE} && git checkout -q {VJEPA2_COMMIT}",
    )
    .add_local_dir(
        str(REPO),
        "/repo",
        copy=True,
        ignore=[".git", ".venv", "runs", "clips", "node_modules", "demo-site", "target", "**/__pycache__"],
    )
    .run_commands("pip install -q /repo")
    .env({"HF_HOME": f"{VOL}/hf", "TORCH_HOME": f"{VOL}/torch"})
)
app = modal.App("ii-vjepa2ac", image=image)


def _sh(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(volumes={VOL: volume}, timeout=1800, cpu=2, memory=8192)
def prep() -> dict:
    """DROID clips with per-frame arm poses, and Meta's checkpoint, onto the Volume (CPU only)."""
    import urllib.request

    if not os.path.exists(CKPT):
        urllib.request.urlretrieve(CHECKPOINT_URL, CKPT + ".part")
        os.rename(CKPT + ".part", CKPT)
    _sh(["ii", "get-clips", "--out", CLIPS, "--n", str(N_CLIPS), "--seconds", str(SECONDS), "--with-states"])
    volume.commit()
    index = json.loads(Path(CLIPS, "clips.json").read_text())
    return {"checkpoint_bytes": os.path.getsize(CKPT), "clips": len(index["clips"])}


def _fp16_check() -> dict:
    """Same frames and actions through the fp32 and fp16 model: fp16 must not change the answer."""
    import numpy as np
    import torch

    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, poses_to_diff
    from interpretableintelligence.worldmodel.action_effects import load_trajectories

    tr = load_trajectories(Path(CLIPS), max_clips=1)[0]
    frames, states = tr.frames[:4], tr.states[:4]
    acts = torch.from_numpy(poses_to_diff(states))
    out = {}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        m = load_vjepa2ac("cuda", code_dir=CODE, checkpoint=CKPT, dtype=dtype)
        z = m.encode_frames(frames)
        p = m.predict_next(z[:-1], torch.from_numpy(states[:-1]), acts)
        out[name] = (z, p)
        out["info"] = m.info
        del m
        torch.cuda.empty_cache()
    (z32, p32), (z16, p16) = out["fp32"], out["fp16"]
    cos = torch.nn.functional.cosine_similarity
    copy = (z32[:-1] - z32[1:]).abs().mean().item()
    pred = (p32 - z32[1:]).abs().mean().item()
    return {
        "encoder_cos_fp16_vs_fp32": float(cos(z16.flatten(1), z32.flatten(1)).min()),
        "predictor_cos_fp16_vs_fp32": float(cos(p16.flatten(1), p32.flatten(1)).min()),
        "finite": bool(torch.isfinite(z16).all() and torch.isfinite(p16).all()),
        "copy_l1": copy,
        "pred_l1": pred,
        "pred_beats_copy": pred < copy,
        "load_info": out["info"],
        "n_frames": int(len(frames)),
        "shape": list(z32.shape),
        "gripper_range": [float(np.min(states[:, 6])), float(np.max(states[:, 6]))],
    }


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=1200, memory=32768)
def smoke_job(fp16_check: bool = True) -> dict:
    """fp16 vs fp32 on one clip, then ``ii action-effects`` on a few clips (``II_H2_SMOKE_CLIPS``, default 6)."""
    check = _fp16_check() if fp16_check else {}
    print(json.dumps(check, indent=1), flush=True)
    out = f"{RESULTS}/smoke"
    _sh(["ii", "action-effects", "--clips", CLIPS, "--out", out, "--code-dir", CODE, "--checkpoint", CKPT,
         "--max-clips", os.environ.get("II_H2_SMOKE_CLIPS", "6"), "--bootstrap", "200"])
    volume.commit()
    result = json.loads(Path(out, "action_effects.json").read_text())
    png = Path(out, "regions.png")
    return {"fp16_check": check, "action_effects": result, "regions_png": png.read_bytes() if png.exists() else None}


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def full() -> dict:
    """All clips, trained predictor, then the random-weights predictor control."""
    out = {}
    for name, extra in (("trained", []), ("random-predictor", ["--random-predictor"])):
        dest = f"{RESULTS}/{name}"
        if not Path(dest, "action_effects.json").exists():
            _sh(["ii", "action-effects", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT, *extra])
            volume.commit()
        out[name] = json.loads(Path(dest, "action_effects.json").read_text())
    return out


@app.function(volumes={VOL: volume}, timeout=300)
def read_results() -> dict:
    out = {}
    for d in sorted(Path(RESULTS).glob("*")):
        f = d / "action_effects.json"
        if f.exists():
            png = d / "regions.png"
            out[d.name] = {"json": json.loads(f.read_text()), "png": png.read_bytes() if png.exists() else None}
    return out


def _save_local(name: str, payload: dict | None, png: bytes | None) -> None:
    dest = Path(os.environ.get("II_H2_LOCAL", "h2-results"))
    dest.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (dest / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    if png:
        (dest / f"{name}-regions.png").write_bytes(png)
    print(f"saved {dest}/{name}*", flush=True)


@app.local_entrypoint()
def smoke(fp16_check: bool = False) -> None:
    out = smoke_job.remote(fp16_check)
    print(json.dumps(out["fp16_check"], indent=1))
    m = out["action_effects"]["moving"]
    print("smoke verdict", m["verdict"], m["n_transitions"], "moving transitions;", out["action_effects"]["n_transitions"], "total")
    _save_local("smoke", out["action_effects"] | {"fp16_check": out["fp16_check"]}, out["regions_png"])


@app.local_entrypoint()
def run() -> None:
    full.spawn()
    print("spawned the full run; fetch results with ::fetch when it is done")


@app.local_entrypoint()
def fetch() -> None:
    for name, item in read_results.remote().items():
        _save_local(name, item["json"], item["png"])
        m = item["json"]["moving"]
        print(name, m["verdict"], m["n_transitions"], "transitions", m["n_episodes"], "episodes")


# ----------------------------------------------------------------------------- H5: ii plan-probe

PP_STRIDE = os.environ.get("II_H5_STRIDE", "16")
PP_SAMPLES = os.environ.get("II_H5_SAMPLES", "100")
PP_STEPS = os.environ.get("II_H5_STEPS", "6")


def _plan_probe(dest: str, *extra: str) -> dict:
    if not Path(dest, "plan_probe.json").exists():
        base = ["--stride", PP_STRIDE] if "--stride" not in extra else []
        _sh(["ii", "plan-probe", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT,
             *base, "--samples", PP_SAMPLES, "--steps", PP_STEPS, *extra])
        volume.commit()
    return json.loads(Path(dest, "plan_probe.json").read_text())


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=1200, memory=32768)
def h5_smoke_job() -> dict:
    import time

    t0 = time.time()
    import shutil

    shutil.rmtree(f"{RESULTS}/h5-smoke", ignore_errors=True)
    out = _plan_probe(f"{RESULTS}/h5-smoke", "--max-clips", "4", "--stride", "8", "--bootstrap", "200")
    out["wall_s"] = time.time() - t0
    return out


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def h5_full_job() -> dict:
    return {
        "trained": _plan_probe(f"{RESULTS}/h5-trained"),
        "random-predictor": _plan_probe(f"{RESULTS}/h5-random-predictor", "--random-predictor"),
    }


@app.local_entrypoint()
def h5_smoke() -> None:
    out = h5_smoke_job.remote()
    _save_local("h5-smoke", out, None)
    print("h5 smoke", out["verdict"], out["n_transitions"], "transitions in", round(out["wall_s"]), "s")


@app.local_entrypoint()
def h5_run() -> None:
    h5_full_job.spawn()
    print("spawned H5; fetch with ::h5_fetch")


@app.function(volumes={VOL: volume}, timeout=300)
def h5_read() -> dict:
    out = {}
    for name in ("h5-trained", "h5-random-predictor"):
        f = Path(RESULTS, name, "plan_probe.json")
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


@app.local_entrypoint()
def h5_fetch() -> None:
    for name, payload in h5_read.remote().items():
        _save_local(name, payload, None)
        print(name, payload["verdict"], payload["n_transitions"], "transitions")


# ----------------------------------------------------------------------------- H1: ii gripper-probe


def _gripper_probe(dest: str, *extra: str) -> dict:
    if not Path(dest, "gripper_probe.json").exists():
        _sh(["ii", "gripper-probe", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT, *extra])
        volume.commit()
    out = json.loads(Path(dest, "gripper_probe.json").read_text())
    out.pop("events", None)
    return out


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def h1_job() -> dict:
    return {
        "h1-trained": _gripper_probe(f"{RESULTS}/h1-trained"),
        "h1-random-predictor": _gripper_probe(f"{RESULTS}/h1-random-predictor", "--random-predictor"),
    }


@app.local_entrypoint()
def h1() -> None:
    for name, payload in h1_job.remote().items():
        _save_local(name, payload, None)
        print(name, payload["verdict"], payload["n_events"], "events", payload["n_episodes"], "episodes")
        for k, v in payload["stats"].items():
            print(f"  {k:20s} {v['value']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
        print("  by direction", payload["by_direction"])


# ----------------------------------------------------------------------------- inside the model: ii gripper-trace


def _gripper_trace(dest: str, *extra: str) -> dict:
    if not Path(dest, "gripper_trace.json").exists():
        _sh(["ii", "gripper-trace", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT, *extra])
        volume.commit()
    out = json.loads(Path(dest, "gripper_trace.json").read_text())
    out.pop("events", None)
    return out


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def trace_job(smoke: bool = True) -> dict:
    if smoke:
        return {"trace-smoke": _gripper_trace(f"{RESULTS}/trace-smoke", "--max-clips", "8", "--bootstrap", "200")}
    return {
        "trace-trained": _gripper_trace(f"{RESULTS}/trace-trained"),
        "trace-random-predictor": _gripper_trace(f"{RESULTS}/trace-random-predictor", "--random-predictor"),
    }


@app.local_entrypoint()
def trace(full: bool = False) -> None:
    for name, payload in trace_job.remote(not full).items():
        _save_local(name, payload, None)
        print(name, payload["verdict"], payload["n_events"], "events", payload["n_episodes"], "episodes")
        for k, v in payload["readout"].items():
            print("  readout", k, v)
        for k, v in payload["stats"].items():
            print(f"  {k:20s} {v['value']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
        for k, v in payload["layers"].items():
            print(f"  {k:20s}", " ".join(f"{x:.3f}" for x in v))
