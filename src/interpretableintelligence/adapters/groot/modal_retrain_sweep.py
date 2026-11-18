"""Run the GR00T SAE retrain sweep (the Colab notebook's seven configs) on a Modal T4 instead of Colab.

    pip install modal            # or: uv sync --extra remote
    modal run src/interpretableintelligence/adapters/groot/modal_retrain_sweep.py --out runs/groot-sae-retrain

Bills the invoking user's own Modal account (credentials from ``modal token set`` or MODAL_TOKEN_ID /
MODAL_TOKEN_SECRET). One T4 container runs the configs one after another, with a hard timeout, so the cost is
bounded up front: about $0.75 an hour at Modal's 2026 list price for a T4 plus the 2 cores and 8 GiB reserved
here. The sweep takes about an hour on a T4 (the two 8192-latent configs take about 15 minutes each).

The container downloads the capture tarball (``data/groot-1028.md``: release ``data-groot-1028-v1``), checks its
sha256, and keeps it and every finished config on the Volume ``ii-groot-sae-retrain``, so a rerun after a
timeout or failure skips what is already done. The local entrypoint writes ``sweep-summary.json`` (same shape
as the notebook's) plus each run's ``retrain.json`` under ``--out`` and prints the comparison table; the new
``sae/`` folders stay on the Volume (``modal volume get ii-groot-sae-retrain runs/<name>/sae <dest>``).

Offline only, like ``ii retrain-sae``: nothing here changes ``split.json``, a scoreboard, or a rollout result.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "ii-groot-sae-retrain"
VOLUME_ROOT = "/vol"
GPU = "T4"
CPU_CORES = 2.0
MEMORY_MIB = 8 * 1024
TIMEOUT_S = 120 * 60

CAPTURE_NAME = "groot-1028-v9-n100"
CAPTURE_URL = "https://github.com/jerrydjin/interpretableintelligence/releases/download/data-groot-1028-v1/groot-1028-v9-n100.tar.xz"
CAPTURE_SHA256 = "0f8954e52ae737aec8a9cd734d58efc3e43ad3725d623fae2e5a6eac858cf00e"

# Same sweep and hyperparameters as the Colab notebook (colab/build_retrain_notebook.py).
SWEEP = [
    ("none-8192", 8192, "none"),
    ("auxk-8192", 8192, "auxk"),
    ("auxk-4096", 4096, "auxk"),
    ("auxk-2048", 2048, "auxk"),
    ("auxk-1024", 1024, "auxk"),
    ("none-2048", 2048, "none"),
    ("resample-2048", 2048, "resample"),
    # auxk-8192 with per-channel input standardisation (norm="channel", a [D] norm_scale). The Nov 17 2026 run on
    # the Volume predates vector support and named its weights sae/weights.channel-norm.safetensors.
    ("auxk-8192-chnorm", 8192, "auxk", "channel"),
]
STEPS, BATCH, LR, K = 6000, 4096, 4e-4, 32

volume = modal.Volume.from_name(APP_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("xz-utils")
    .uv_pip_install("torch>=2.2", "numpy>=1.26,<3", "safetensors>=0.4")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("interpretableintelligence")
)
app = modal.App(APP_NAME)


def _fetch_capture(vol: Path) -> Path:
    """Download and verify the tarball once (kept on the Volume), then unpack it to local disk."""
    import hashlib
    import subprocess
    import urllib.request

    tar = vol / f"{CAPTURE_NAME}.tar.xz"
    if not tar.is_file():
        part = tar.with_suffix(".part")
        with urllib.request.urlopen(CAPTURE_URL) as src, open(part, "wb") as dst:
            while block := src.read(1 << 22):
                dst.write(block)
        part.rename(tar)
        volume.commit()
    digest = hashlib.sha256()
    with open(tar, "rb") as fh:
        while block := fh.read(1 << 22):
            digest.update(block)
    if digest.hexdigest() != CAPTURE_SHA256:
        tar.unlink()
        volume.commit()
        raise RuntimeError("capture tarball sha256 does not match data/groot-1028.md")
    capture = Path("/tmp") / CAPTURE_NAME
    if not (capture / "acts").is_dir():
        subprocess.run(["tar", "-xJf", str(tar), "-C", "/tmp"], check=True)
    return capture


@app.function(image=image, gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MIB, timeout=TIMEOUT_S, volumes={VOLUME_ROOT: volume})
def sweep(only: list[str] | None = None, steps: int = 0) -> dict:
    """Run every config not already on the Volume; return {name: {"retrain": report}}.

    The new SAEs stay on the Volume (runs/<name>/sae/): returning ~100 MB of weights per config would go through
    Modal's blob store, which restricted networks often block. Fetch one with
    ``modal volume get ii-groot-sae-retrain runs/<name>/sae <dest>``.
    """
    import shutil

    import torch

    from interpretableintelligence.adapters.groot.retrain_sae import RetrainConfig, format_report, retrain

    vol = Path(VOLUME_ROOT)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    capture = None
    results = {}
    for name, n_latents, revive, *rest in SWEEP:
        norm = rest[0] if rest else "keep"
        if only and name not in only:
            continue
        # steps > 0 is a smoke test: a short run kept apart from the real one, so it is never mistaken for it.
        run_name = f"{name}-smoke{steps}" if steps else name
        done = vol / "runs" / run_name
        if not (done / "retrain.json").is_file():
            capture = capture or _fetch_capture(vol)
            work = Path("/tmp/sae-retrain") / name
            shutil.rmtree(work, ignore_errors=True)
            config = RetrainConfig(n_latents=n_latents, k=K, steps=steps or STEPS, batch_tokens=BATCH, lr=LR, revive=revive, norm=norm)
            print(f"== {name}")
            report = retrain(capture, work, config, torch.device("cuda"), log=print)
            print(format_report(report))
            shutil.rmtree(done, ignore_errors=True)
            (done / "sae").mkdir(parents=True)
            shutil.copytree(work / "sae", done / "sae", dirs_exist_ok=True)
            shutil.copy(work / "retrain.json", done / "retrain.json")
            volume.commit()
        else:
            print(f"done already: {run_name}")
        results[name] = {"retrain": json.loads((done / "retrain.json").read_text())}
    return results


def summarise(results: dict) -> tuple[dict, str]:
    """The notebook's comparison: old SAE first, then each config, plus the input-variance check."""
    rows = []
    for name, *_rest in SWEEP:
        if name not in results:
            continue
        rep = results[name]["retrain"]
        if not rows and rep["old"]:
            rows.append(("old (capture)", rep["old"]))
        rows.append((name, rep["new"]))
    lines = [f"{'SAE':<18}{'latents':>8}{'alive tr':>9}{'alive ho':>9}{'dense ho':>9}{'L0 ho':>7}{'EV ho':>8}{'EV/dim ho':>10}"]
    for name, r in rows:
        tr, ho = r["train"], r["heldout"]
        lines.append(
            f"{name:<18}{tr['alive_latents'] + tr['dead_latents']:>8}{tr['alive_latents']:>9}{ho['alive_latents']:>9}"
            f"{ho['dense_latents']:>9}{ho['l0_mean']:>7.2f}{ho['ev_raw']:>8.4f}{ho['ev_per_dim_mean']:>10.4f}"
        )
    ho = rows[0][1]["heldout"]
    lines.append(
        f"\nInput check: the {len(ho['top_dims'])} largest-variance channels hold {ho['variance_share_top_dims']:.1%} "
        f"of held-out variance (channels {ho['top_dims']})."
    )
    return {name: r for name, r in rows}, "\n".join(lines)


@app.local_entrypoint()
def main(out: str = "runs/groot-sae-retrain", only: str = "", steps: int = 0, timeout_min: int = 0) -> None:
    """``--steps N`` runs a short smoke test kept apart on the Volume; ``--timeout-min M`` lowers the hard cap."""
    names = [n for n in only.split(",") if n] or None
    timeout_s = min(timeout_min * 60, TIMEOUT_S) if timeout_min > 0 else TIMEOUT_S
    print(f"Modal {GPU}, timeout {timeout_s // 60} min: at most about ${0.75 * timeout_s / 3600:.2f} at list price.")
    fn = sweep.with_options(timeout=timeout_s) if timeout_s != TIMEOUT_S else sweep
    results = fn.remote(names, steps)
    dest = Path(out)
    for name, res in results.items():
        (dest / name).mkdir(parents=True, exist_ok=True)
        (dest / name / "retrain.json").write_text(json.dumps(res["retrain"], indent=1) + "\n")
    summary, table = summarise(results)
    (dest / "sweep-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(table)
    print(f"wrote {dest} (new SAEs stay on the Volume {APP_NAME}: modal volume get {APP_NAME} runs/<name>/sae <dest>)")
