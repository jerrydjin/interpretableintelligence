"""Run the real-GR00T steering scorecard (scripts/groot_steer_scorecard.sh) on a Modal GPU, billed to your own account.

    modal volume put ii-groot-steer groot-1028-v9-n100.tar.xz /capture/                           # once
    modal run scripts/modal_groot_steer.py::prefetch                                              # once, CPU only
    modal run scripts/modal_groot_steer.py::smoke                                                 # ~10 GPU minutes
    modal run --detach scripts/modal_groot_steer.py::scorecard                                    # the full run
    modal run scripts/modal_groot_steer.py::fetch                                                 # print results

Matched-feature controls (other live SAE features instead of random directions), reusing the recorded clips and run:

    modal run scripts/modal_groot_steer.py::steer_smoke                                           # 1 held-out episode
    II_MODAL_HOURS=1 modal run --detach scripts/modal_groot_steer.py::scorecard --controls-from features
    modal run scripts/modal_groot_steer.py::fetch --controls-from features

Task 1028 test of feature 6262 on the rescaled (channel-normalised) SAE, which lives on the retrain sweep's Volume:

    modal run scripts/modal_groot_steer.py::map_feature                                           # checks 0 and 1, CPU only

The rescaled SAE's own picks, scored offline against matched real-feature controls:

    modal run scripts/modal_groot_steer.py::pick_rescaled                                         # picker top 3, CPU only
    modal run scripts/modal_groot_steer.py::rescaled_scorecard --features A,B,C --smoke           # capture + 1 episode
    II_MODAL_HOURS=1.5 modal run --detach scripts/modal_groot_steer.py::rescaled_scorecard --features A,B,C

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
RETRAIN_VOL = "/retrain"
retrain_volume = modal.Volume.from_name("ii-groot-sae-retrain", create_if_missing=False)
RESCALED_SAE = f"{RETRAIN_VOL}/runs/auxk-8192-chnorm/sae"
CAPTURE_DIR = f"{WORK}/groot-1028-v9-n100"
MAP_RESULTS = f"{VOL}/results-map"
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


def _results_dir(controls_from: str) -> str:
    return RESULTS if controls_from == "random" else f"{RESULTS}-{controls_from}"


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), cpu=4, memory=24576)
def scorecard(features: str = "6262,7353,7775", episodes: int = 24, controls_from: str = "random") -> None:
    env = _prepare()
    env.update(FEATURES=features, EPISODES=str(episodes), RESULTS=_results_dir(controls_from), CONTROLS_FROM=controls_from)
    _run(["bash", "scripts/groot_steer_scorecard.sh", CAPTURE, WORK], env)


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=1200, cpu=4, memory=24576)
def steer_smoke(features: str = "6262,7353,7775", controls_from: str = "features") -> None:
    """The scorecard step alone on one held-out episode of the existing run (needs a finished ``scorecard`` first)."""
    env = _prepare()
    run = f"{WORK}/run"
    if not Path(run, "manifest.json").is_file():
        raise SystemExit(f"no run at {run}; run scorecard first")
    _run(["ii", "steer-scorecard", run, "--clips", f"{WORK}/clips-run", "--features", features, "--controls", "19",
          "--controls-from", controls_from, "--max-episodes", "1", "--bootstrap", "50"], env)


def _rescaled_sae() -> str:
    """The channel-norm SAE was written before PR 59 under its old file name; give it the standard one."""
    sae = Path(RESCALED_SAE)
    old, new = sae / "weights.channel-norm.safetensors", sae / "weights.safetensors"
    if not new.is_file():
        if not old.is_file():
            raise SystemExit(f"no rescaled SAE weights in {sae}")
        old.rename(new)
        retrain_volume.commit()
    return str(sae)


@app.function(volumes={VOL: volume, RETRAIN_VOL: retrain_volume}, timeout=1800, cpu=4, memory=16384)
def map_feature_remote(feature: int = 6262, name: str = "6262") -> dict:
    """Checks 0 and 1 of the 6262 design: ii map-feature from the capture SAE into the rescaled SAE, with the picker's
    failure-direction check. CPU only. Returns mapping.json."""
    import json
    import shutil

    to_sae = _rescaled_sae()
    out = Path(MAP_RESULTS) / name
    if out.exists():
        shutil.rmtree(out)
    _run(["ii", "map-feature", CAPTURE_DIR, "--from-sae", f"{CAPTURE_DIR}/sae", "--to-sae", to_sae,
          "--feature", str(int(feature)), "--out", str(out)], dict(os.environ))
    return json.loads((out / "mapping.json").read_text())


@app.local_entrypoint()
def map_feature(feature: int = 6262, out: str = "mapping-6262.json") -> None:
    import json

    report = map_feature_remote.remote(feature, str(feature))
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["decision"], indent=1))
    print(json.dumps(report.get("direction"), indent=1))
    print(f"wrote {out}")


RESCALED_RUN = f"{WORK}/run-rescaled"
RESCALED_RESULTS = f"{VOL}/results-rescaled"
PICK_SEED = 40000  # ii rollouts --pick's default base seed, which the picker runs with


@app.function(volumes={VOL: volume, RETRAIN_VOL: retrain_volume}, timeout=3600, cpu=4, memory=16384)
def pick_rescaled_remote(top: int = 3) -> dict:
    """The GR00T picker (ii rollouts --pick's arguments) on the capture through the rescaled SAE. CPU only."""
    import json
    import shutil

    from interpretableintelligence.rollout_groot_pick import ranked_features, run_picker

    out = Path(RESCALED_RESULTS) / "pick"
    if out.exists():
        shutil.rmtree(out)
    run = run_picker(Path(CAPTURE_DIR), Path(_rescaled_sae()), out, seed=PICK_SEED, top=20)
    ranked = ranked_features(out, run["result"])
    contrast = {}
    for line in (out / "contrast.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if int(row["feature_id"]) in ranked[:top]:
                contrast[str(row["feature_id"])] = row
    report = {"picks": ranked[:top], "candidate_ids": [int(x) for x in run["result"].get("candidate_ids") or []],
              "argv": run["argv"], "contrast": contrast}
    (out / "picks.json").write_text(json.dumps(report, indent=1) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def pick_rescaled(top: int = 3, out: str = "rescaled-picks.json") -> None:
    import json

    report = pick_rescaled_remote.remote(top)
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("picks", "candidate_ids")}))
    print(f"wrote {out}")


@app.function(volumes={VOL: volume, RETRAIN_VOL: retrain_volume}, gpu=GPU, timeout=int(HOURS * 3600), cpu=4, memory=24576)
def rescaled_scorecard_remote(features: str, smoke: bool = False) -> dict:
    """Capture the recorded clips with the rescaled SAE frozen in (once), then the matched-feature scorecard.
    ``smoke``: one held-out episode, 50 bootstrap resamples, results not copied."""
    import json
    import shutil

    env = _prepare()
    run, clips = RESCALED_RUN, f"{WORK}/clips-run"
    if not Path(run, "features", "stats.json").is_file():
        if Path(run).exists():
            shutil.rmtree(run)
        _run(["ii", "run", "groot", "--clips", clips, "--out", run, "--sae-from", _rescaled_sae(), "--heldout-frac", "0.75",
              "--no-score", "--no-serve"], env)
    cmd = ["ii", "steer-scorecard", run, "--clips", clips, "--features", features, "--controls", "19", "--controls-from", "features"]
    if smoke:
        cmd += ["--max-episodes", "1", "--bootstrap", "50"]
    _run(cmd, env)
    card = json.loads(Path(run, "steering", "scorecard.json").read_text())
    stats = json.loads(Path(run, "features", "stats.json").read_text())
    if not smoke:
        results = Path(RESCALED_RESULTS)
        results.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(run, "steering", "scorecard.json"), results / "scorecard.json")
        shutil.copy(Path(run, "features", "stats.json"), results / "features-stats.json")
        volume.commit()
    wanted = {int(x) for x in features.split(",") if x}
    return {"scorecard.json": card, "stats": {**{k: v for k, v in stats.items() if k != "features"},
            "picked": [f for f in stats.get("features", []) if int(f.get("id", -1)) in wanted]}}


@app.local_entrypoint()
def rescaled_scorecard(features: str, smoke: bool = False, out: str = "rescaled-scorecard.json") -> None:
    import json

    result = rescaled_scorecard_remote.remote(features, smoke)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    for row in result["scorecard.json"]["rows"]:
        excess = (row.get("excess") or {}).get("mean")
        print(row["feature_id"], row["kind"], "windows", row["n_windows"], "excess", None if excess is None else round(excess, 4),
              "pct", row["control_percentile"], "beats", row["beats_controls"])
    print(f"wrote {out}")


@app.function(volumes={VOL: volume}, timeout=300)
def _results(controls_from: str = "random") -> dict:
    out = {}
    for name in ("scorecard.json", "record.json", "features-stats.json"):
        path = Path(_results_dir(controls_from)) / name
        if path.is_file():
            data = __import__("json").loads(path.read_text())
            if name == "features-stats.json":
                picked = {6262, 7353, 7775}
                data = {**{k: v for k, v in data.items() if k != "features"},
                        "picked": [{k: f[k] for k in ("id", "freq", "max_act")} for f in data["features"] if f["id"] in picked]}
            out[name] = data
    return out


@app.local_entrypoint()
def fetch(out: str = "groot-steer-results.json", controls_from: str = "random") -> None:
    """Copy the small result files off the Volume through the API (no direct storage download)."""
    import json

    results = _results.remote(controls_from)
    Path(out).write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {out}: {sorted(results)}")


ROLLOUT_RESULTS = f"{VOL}/results-rollouts"
RESCALED_CAPTURE = f"{WORK}/capture-rescaled"


def _rescaled_capture() -> str:
    """The v9 capture's manifest and split with the rescaled SAE as its sae/: what `ii rollouts --capture` reads."""
    import shutil

    view = Path(RESCALED_CAPTURE)
    view.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "split.json"):
        if not (view / name).is_file():
            shutil.copy(Path(CAPTURE_DIR, name), view / name)
    sae = view / "sae"
    if not sae.exists():
        sae.symlink_to(_rescaled_sae(), target_is_directory=True)
    return str(view)


@app.function(volumes={VOL: volume, RETRAIN_VOL: retrain_volume}, gpu=GPU, timeout=int(HOURS * 3600), cpu=8, memory=32768)
def rollouts_remote(name: str, features: str, controls: str = "", episodes: int = 3, direction: str = "fall",
                    seed: int = 70000, hours: float = 0.3) -> dict:
    """`ii rollouts --policy groot-1028` for hand-named rescaled-SAE features, on LIBERO-plus in this container.
    Each `name` gets its own run folder (a copy of run-rescaled's manifest and split), so plans never collide."""
    import json
    import shutil

    env = _prepare()
    env["GROOT_ABLATE_HOURS"] = str(hours + 0.25)  # the v9 runner stops dispatching once under 0.25 h are left
    run = Path(WORK, f"rollouts-{name}")
    run.mkdir(parents=True, exist_ok=True)
    for item in ("manifest.json", "split.json"):
        if not (run / item).is_file():
            shutil.copy(Path(RESCALED_RUN, item), run / item)
    cmd = ["ii", "rollouts", str(run), "--policy", "groot-1028", "--capture", _rescaled_capture(), "--features", features,
           "--episodes", str(episodes), "--direction", direction, "--seed", str(seed)]
    if controls:
        cmd += ["--control-features", controls]
    _run(cmd, env)
    out = {"summary.json": json.loads((run / "rollouts" / "summary.json").read_text()),
           "ablate_plan.json": json.loads((run / "rollouts" / "ablate_plan.json").read_text())}
    results = Path(ROLLOUT_RESULTS, name)
    results.mkdir(parents=True, exist_ok=True)
    for key, data in out.items():
        (results / key).write_text(json.dumps(data, indent=1) + "\n")
    volume.commit()
    return out


@app.local_entrypoint()
def rollouts(name: str, features: str, controls: str = "", episodes: int = 3, direction: str = "fall", seed: int = 70000,
             hours: float = 0.3, out: str = "") -> None:
    import json

    result = rollouts_remote.remote(name, features, controls, episodes, direction, seed, hours)
    Path(out or f"rollouts-{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    summary = result["summary.json"]
    for row in summary.get("features", []):
        print(row.get("feature_id"), row.get("role"), "delta", row.get("delta"), "p", row.get("p_value"), "causal", row.get("causal"))
    print(json.dumps({k: summary.get(k) for k in ("baseline", "recon_only", "direction")}, indent=1)[:2000])


@app.function(volumes={VOL: volume, RETRAIN_VOL: retrain_volume}, gpu=GPU, timeout=int(HOURS * 3600), cpu=4, memory=24576)
def action_screen_remote(min_rate: float = 0.10, top: int = 16, controls: int = 3, episodes: int = 6, features: str = "") -> dict:
    """Cheap remove/boost screen of the rescaled SAE's densest features on recorded clips (exploration, never a finding):
    which edit moves GR00T's planned actions most. Features firing on >= min_rate of held-out tokens, by rate x max act."""
    import json
    import shutil

    env = _prepare()
    run = RESCALED_RUN
    stats = json.loads(Path(run, "features", "stats.json").read_text())
    n = float(stats["n_heldout"])
    dense = [f for f in stats["features"] if f["freq"] / n >= min_rate]
    dense.sort(key=lambda f: -f["freq"] * f["max_act"])
    ids = [int(x) for x in features.split(",") if x] or [int(f["id"]) for f in dense[:top]]
    _run(["ii", "steer-scorecard", run, "--clips", f"{WORK}/clips-run", "--features", ",".join(map(str, ids)), "--controls",
          str(controls), "--controls-from", "features", "--max-episodes", str(episodes), "--bootstrap", "200"], env)
    card = json.loads(Path(run, "steering", "scorecard.json").read_text())
    tag = "screen" if not features else "screen-" + features.replace(",", "-")
    results = Path(ROLLOUT_RESULTS, tag)
    results.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(run, "steering", "scorecard.json"), results / "scorecard.json")
    volume.commit()
    meta = [{"id": f["id"], "rate": f["freq"] / n, "max_act": f["max_act"]} for f in dense]
    return {"scorecard.json": card, "dense": meta, "n_dense": len(dense), "screened": ids}


@app.local_entrypoint()
def action_screen(min_rate: float = 0.10, top: int = 16, controls: int = 3, episodes: int = 6, features: str = "",
                  out: str = "action-screen.json") -> None:
    import json

    result = action_screen_remote.remote(min_rate, top, controls, episodes, features)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print("dense features:", result["n_dense"])
    for row in result["scorecard.json"]["rows"]:
        excess = (row.get("excess") or {}).get("mean")
        print(row["feature_id"], row["kind"], "change", row.get("action_change"), "excess", excess, "ratio", row.get("ratio"),
              "pct", row["control_percentile"])
