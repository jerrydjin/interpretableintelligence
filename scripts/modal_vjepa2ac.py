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


# ----------------------------------------------------------------------------- inside the model: ii gripper-sae


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=49152)
def sae_job(layer: int, smoke: bool = True) -> dict:
    name = f"sae-smoke-L{layer}" if smoke else f"sae-L{layer}"
    dest = f"{RESULTS}/{name}"
    extra = ["--max-clips", "12", "--steps", "300", "--bootstrap", "200"] if smoke else []
    if not Path(dest, "gripper_sae.json").exists():
        _sh(["ii", "gripper-sae", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT, "--layer", str(layer), *extra])
        volume.commit()
    out = json.loads(Path(dest, "gripper_sae.json").read_text())
    out.pop("events", None)
    return {name: out}


@app.local_entrypoint()
def sae(layer: int, full: bool = False) -> None:
    for name, payload in sae_job.remote(layer, not full).items():
        _save_local(name, payload, None)
        print(name, payload["verdict"], "layer", payload["layer"], payload["n_test_events"], "held-out events",
              "FVU", round(payload["sae"]["final_fvu"], 4))
        for arm, v in payload["arms"].items():
            c, r = v["consistency"], v["recovered"]
            print(f"  {arm:24s} latents {v['latents']} consistency {c['value']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] "
                  f"recovered {r['value']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")


# ----------------------------------------------------------------------------- inside the model: ii gripper-circuit


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def circuit_job(smoke: bool = True) -> dict:
    name = "circuit-smoke" if smoke else "circuit"
    dest = f"{RESULTS}/{name}"
    extra = ["--max-clips", "10", "--random-sets", "4", "--bootstrap", "200"] if smoke else []
    if smoke or not Path(dest, "gripper_circuit.json").exists():
        _sh(["ii", "gripper-circuit", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT, *extra])
        volume.commit()
    out = json.loads(Path(dest, "gripper_circuit.json").read_text())
    out.pop("events", None)
    return {name: out}


@app.local_entrypoint()
def circuit(full: bool = False) -> None:
    for name, payload in circuit_job.remote(not full).items():
        _save_local(name, payload, None)
        print(name, payload["verdict"], payload["n_tested"], "held-out events", payload["n_episodes"], "episodes")
        print("  ratio curve", " ".join(f"{x:.3f}" for x in payload["ratio_curve"]))
        for f, p in payload["picks"].items():
            print("  fold", f, p)
        for kind, k in payload["knockout"].items():
            r, pc = k["recovered"], k["prefers_correct"]
            print(f"  knockout {kind}: recovered {r['value']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] random mean {k.get('random_mean', float('nan')):+.3f} "
                  f"max {k.get('random_max', float('nan')):+.3f}; prefers_correct {pc['value']:.3f} [{pc['lo']:.3f}, {pc['hi']:.3f}]")
        print("  baseline", payload["baseline"])
        for a, s in payload["steer"].items():
            print(f"  steer {a}: {s}")
        print("  steer random", payload["steer_random"])
        print("  attention", payload["attention"])
        print("  baseline check", payload["baseline_check"])
        print("  wrong-command drift per block", " ".join(f"{x:.3f}" for x in payload["curve"]))
        for kind, k in payload["knockout"].items():
            print(f"  knocked {kind} drift", " ".join(f"{x:.3f}" for x in k["curve"]))
            print(f"  random {kind} drift ", " ".join(f"{x:.3f}" for x in k.get("random_curve", [])))


# ----------------------------------------------------------------------------- screen: ii impossible-events


def _impossible(dest: str, *extra: str, fresh: bool = False) -> dict:
    import shutil

    if fresh:
        shutil.rmtree(dest, ignore_errors=True)
    if not Path(dest, "impossible_events.json").exists():
        _sh(["ii", "impossible-events", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT, *extra])
        volume.commit()
    out = json.loads(Path(dest, "impossible_events.json").read_text())
    out.pop("transitions", None)
    return out


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=1200, memory=32768)
def ie_smoke_job() -> dict:
    import time

    t0 = time.time()
    out = _impossible(f"{RESULTS}/ie-smoke", "--max-clips", "5", "--bootstrap", "200", fresh=True)
    out["wall_s"] = time.time() - t0
    return {"ie-smoke": out}


@app.function(volumes={VOL: volume}, timeout=300)
def ie_png(name: str, fname: str) -> bytes:
    """One heatmap file, alone: large function results go through Modal's blob store, which this network blocks."""
    return Path(RESULTS, name, "heatmaps", fname).read_bytes()


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def ie_full_job() -> dict:
    return {
        "ie-trained": _impossible(f"{RESULTS}/ie-trained"),
        "ie-random-predictor": _impossible(f"{RESULTS}/ie-random-predictor", "--random-predictor"),
    }


@app.function(volumes={VOL: volume}, timeout=300)
def ie_read() -> dict:
    out = {}
    for name in ("ie-smoke", "ie-trained", "ie-random-predictor"):
        f = Path(RESULTS, name, "impossible_events.json")
        if f.exists():
            payload = json.loads(f.read_text())
            payload.pop("transitions", None)
            payload["heatmap_files"] = sorted(p.name for p in Path(RESULTS, name, "heatmaps").glob("*.png"))
            out[name] = payload
    return out


def _ie_report(results: dict) -> None:
    dest = Path(os.environ.get("II_H2_LOCAL", "h2-results"))
    for name, payload in results.items():
        for fname in payload.pop("heatmap_files", []):
            try:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / f"{name}-{fname}").write_bytes(ie_png.remote(name, fname))
            except Exception as exc:  # noqa: BLE001 - a heatmap that cannot come back is not worth the run
                print("heatmap", fname, "not fetched:", exc)
        _save_local(name, payload, None)
        print(name, payload["n_transitions"], "transitions", payload["n_episodes"], "episodes", "wall_s", payload.get("wall_s"))
        for e, v in payload["edits"].items():
            d = v["detected"]
            print(f"  {e:13s} {v['verdict']:17s} " + "  ".join(f"{k} {d[k]['value']:.2f} [{d[k]['lo']:.2f},{d[k]['hi']:.2f}]" for k in d)
                  + f"  ratio {v['surprise_ratio']['model']:.3f} n={v['n_transitions']}")


@app.local_entrypoint()
def ie_smoke() -> None:
    ie_smoke_job.remote()
    _ie_report({k: v for k, v in ie_read.remote().items() if k == "ie-smoke"})


@app.local_entrypoint()
def ie_run() -> None:
    ie_full_job.spawn()
    print("spawned the impossible-events screen; fetch with ::ie_fetch")


@app.local_entrypoint()
def ie_fetch() -> None:
    _ie_report(ie_read.remote())


# ----------------------------------------------------------------------------- impossible-events with more context

CTX_START = "6"  # leaves room for 7 context frames (Meta trained on 8-frame clips: 7 in, 7 out), so every context scores the same transitions


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=1200, memory=32768)
def ctx_smoke_job() -> dict:
    return {f"ctx-smoke-{k}": _impossible(f"{RESULTS}/ctx-smoke-{k}", "--max-clips", "3", "--bootstrap", "200",
                                          "--context", k, "--start", CTX_START, fresh=True) for k in ("1", "7")}


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def ctx_full_job(contexts: str = "1,4,7", clips: int = 50) -> dict:
    return {f"ctx-{k}": _impossible(f"{RESULTS}/ctx-{k}", "--context", k, "--start", CTX_START, "--max-clips", str(clips))
            for k in contexts.split(",")}


@app.function(volumes={VOL: volume}, timeout=300)
def ctx_read() -> dict:
    out = {}
    for f in sorted(Path(RESULTS).glob("ctx-*/impossible_events.json")):
        payload = json.loads(f.read_text())
        payload.pop("transitions", None)
        out[f.parent.name] = payload
    return out


@app.local_entrypoint()
def ctx_smoke() -> None:
    import time

    t0 = time.time()
    ctx_smoke_job.remote()
    print("wall incl. startup", round(time.time() - t0), "s")
    _ie_report({k: v for k, v in ctx_read.remote().items() if k in ("ctx-smoke-1", "ctx-smoke-7")})


@app.local_entrypoint()
def ctx_run(contexts: str = "1,4,7", clips: int = 50) -> None:
    ctx_full_job.spawn(contexts, clips)
    print("spawned; fetch with ::ctx_fetch")


@app.local_entrypoint()
def ctx_fetch() -> None:
    _ie_report(ctx_read.remote())


# ----------------------------------------------------------------------------- does the freeze bias hurt planning: ii plan-stall


def _plan_stall(dest: str, *extra: str, fresh: bool = False) -> dict:
    import shutil

    if fresh:
        shutil.rmtree(dest, ignore_errors=True)
    if not Path(dest, "plan_stall.json").exists():
        _sh(["ii", "plan-stall", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT, *extra])
        volume.commit()
    out = json.loads(Path(dest, "plan_stall.json").read_text())
    out.pop("transitions", None)
    return out


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=1200, memory=32768)
def ps_smoke_job() -> dict:
    import time

    t0 = time.time()
    out = _plan_stall(f"{RESULTS}/ps-smoke", "--max-clips", "4", "--bootstrap", "200", fresh=True)
    out["wall_s"] = time.time() - t0
    return {"ps-smoke": out}


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def ps_full_job(clips: int = 100) -> dict:
    return {
        "ps-trained": _plan_stall(f"{RESULTS}/ps-trained", "--max-clips", str(clips)),
        "ps-random-predictor": _plan_stall(f"{RESULTS}/ps-random-predictor", "--random-predictor", "--max-clips", str(clips)),
    }


@app.function(volumes={VOL: volume}, timeout=300)
def ps_read() -> dict:
    out = {}
    for f in sorted(Path(RESULTS).glob("ps-*/plan_stall.json")):
        payload = json.loads(f.read_text())
        payload.pop("transitions", None)
        out[f.parent.name] = payload
    return out


def _ps_report(results: dict) -> None:
    for name, payload in results.items():
        _save_local(name, payload, None)
        print(name, "wall_s", payload.get("wall_s"))
        for h, v in payload["horizons"].items():
            print(f" goal +{h}: {v['n_transitions']} transitions {v['n_episodes']} episodes  hurt {v['hurt']}  fix {v['fix']}")
            for s, st in v["scores"].items():
                print(f"  {s:7s} " + "  ".join(f"{k} {x['value']:.3f} [{x['lo']:.3f},{x['hi']:.3f}]" for k, x in st.items()))
            for k, x in {**v["paired"], **v["zero_minus_real_split"]}.items():
                print(f"  {k:28s} {x['value']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]")


@app.local_entrypoint()
def ps_smoke() -> None:
    import time

    t0 = time.time()
    ps_smoke_job.remote()
    print("wall incl. startup", round(time.time() - t0), "s")
    _ps_report({k: v for k, v in ps_read.remote().items() if k == "ps-smoke"})


@app.local_entrypoint()
def ps_run(clips: int = 100) -> None:
    ps_full_job.spawn(clips)
    print("spawned plan-stall; fetch with ::ps_fetch")


@app.local_entrypoint()
def ps_fetch() -> None:
    _ps_report({k: v for k, v in ps_read.remote().items() if k != "ps-smoke"})


# ----------------------------------------------------------------------------- does it hold the motion back: ii action-amp


def _action_amp(dest: str, *extra: str, fresh: bool = False) -> dict:
    import shutil

    if fresh:
        shutil.rmtree(dest, ignore_errors=True)
    if not Path(dest, "action_amp.json").exists():
        _sh(["ii", "action-amp", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT, *extra])
        volume.commit()
    out = json.loads(Path(dest, "action_amp.json").read_text())
    out.pop("transitions", None)
    return out


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=1200, memory=32768)
def aa_smoke_job() -> dict:
    import time

    t0 = time.time()
    out = _action_amp(f"{RESULTS}/aa-smoke", "--max-clips", "4", "--bootstrap", "200", fresh=True)
    out["wall_s"] = time.time() - t0
    return {"aa-smoke": out}


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def aa_full_job(clips: int = 60) -> dict:
    return {
        "aa-trained": _action_amp(f"{RESULTS}/aa-trained", "--max-clips", str(clips)),
        "aa-random-predictor": _action_amp(f"{RESULTS}/aa-random-predictor", "--random-predictor", "--max-clips", str(clips)),
    }


@app.function(volumes={VOL: volume}, timeout=300)
def aa_read() -> dict:
    out = {}
    for f in sorted(Path(RESULTS).glob("aa-*/action_amp.json")):
        payload = json.loads(f.read_text())
        payload.pop("transitions", None)
        out[f.parent.name] = payload
    return out


def _aa_report(results: dict) -> None:
    for name, payload in results.items():
        _save_local(name, payload, None)
        print(name, "wall_s", payload.get("wall_s"), payload["n_transitions"], "transitions", payload["n_episodes"], "episodes")
        print(" verdicts", payload["verdicts"])
        for arm in ("action", "shuffled", "change"):
            v = payload["prediction"][arm]
            g = v["gain"]
            print(f"  {arm:8s} gain {g['value']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] alpha {v['alpha_per_fold']}"
                  f" err {[round(x, 4) for x in v['err_by_alpha']]} closer {[round(x, 2) for x in v['closer_real_by_alpha']]}")
        for k, x in payload["prediction"]["paired"].items():
            print(f"  {k:22s} {x['value']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]")
        for h, v in payload["planning"].items():
            for arm in ("action", "change"):
                a = v[arm]
                print(f"  plan +{h} {arm:7s} gain {a['gain']['value']:+.3f} [{a['gain']['lo']:+.3f},{a['gain']['hi']:+.3f}]"
                      f" alpha {a['alpha_per_fold']} cos {[round(x, 3) for x in a['cos_by_alpha']]}"
                      f" stall {[round(x, 2) for x in a['stall_by_alpha']]}")


@app.local_entrypoint()
def aa_smoke() -> None:
    import time

    t0 = time.time()
    aa_smoke_job.remote()
    print("wall incl. startup", round(time.time() - t0), "s")
    _aa_report({k: v for k, v in aa_read.remote().items() if k == "aa-smoke"})


@app.local_entrypoint()
def aa_run(clips: int = 60) -> None:
    aa_full_job.spawn(clips)
    print("spawned action-amp; fetch with ::aa_fetch")


@app.local_entrypoint()
def aa_fetch() -> None:
    _aa_report({k: v for k, v in aa_read.remote().items() if k != "aa-smoke"})


# ----------------------------------------------------------------------------- do a few neurons carry the arm action: ii action-neurons


def _action_neurons(dest: str, *extra: str, fresh: bool = False) -> dict:
    import shutil

    if fresh:
        shutil.rmtree(dest, ignore_errors=True)
    if not Path(dest, "action_neurons.json").exists():
        _sh(["ii", "action-neurons", "--clips", CLIPS, "--out", dest, "--code-dir", CODE, "--checkpoint", CKPT, *extra])
        volume.commit()
    out = json.loads(Path(dest, "action_neurons.json").read_text())
    out.pop("transitions", None)
    return out


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=1200, memory=32768)
def an_smoke_job() -> dict:
    import time

    t0 = time.time()
    out = _action_neurons(f"{RESULTS}/an-smoke", "--max-clips", "4", "--bootstrap", "200", fresh=True)
    out["wall_s"] = time.time() - t0
    return {"an-smoke": out}


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), memory=32768)
def an_full_job(clips: int = 40) -> dict:
    return {"an-trained": _action_neurons(f"{RESULTS}/an-trained", "--max-clips", str(clips))}


@app.function(volumes={VOL: volume}, timeout=300)
def an_read() -> dict:
    out = {}
    for f in sorted(Path(RESULTS).glob("an-*/action_neurons.json")):
        payload = json.loads(f.read_text())
        payload.pop("transitions", None)
        out[f.parent.name] = payload
    return out


def _an_report(results: dict) -> None:
    for name, payload in results.items():
        _save_local(name, payload, None)
        print(name, "wall_s", payload.get("wall_s"), payload["n_transitions"], "transitions", payload["n_episodes"], "episodes",
              payload["n_neurons"], "neurons; verdict", payload["verdict"], "k_carry", payload["k_carry"])
        for k, v in payload["per_k"].items():
            t = v["top_remain"]
            print(f"  top {k:>5s} remain {t['value']:.3f} [{t['lo']:.3f},{t['hi']:.3f}] random mean {v['random_remain_mean']:.3f}"
                  f" min {min(v['random_remain_per_set']):.3f}  err rise top {v['top_err_rise']['value']:+.4f}"
                  f" random {v['random_err_rise_mean']:+.4f}")
        print("  top-1000 blocks per test fold", payload["top_blocks_by_test_fold"])
        print("  top-10 per test fold", payload["top_neurons_by_test_fold"])


@app.local_entrypoint()
def an_smoke() -> None:
    import time

    t0 = time.time()
    an_smoke_job.remote()
    print("wall incl. startup", round(time.time() - t0), "s")
    _an_report({k: v for k, v in an_read.remote().items() if k == "an-smoke"})


@app.local_entrypoint()
def an_run(clips: int = 40) -> None:
    an_full_job.spawn(clips)
    print("spawned action-neurons; fetch with ::an_fetch")


@app.local_entrypoint()
def an_fetch() -> None:
    _an_report({k: v for k, v in an_read.remote().items() if k != "an-smoke"})
