"""Write the Colab notebook that retrains the GR00T capture SAE (``ii retrain-sae``) and compares settings.

Notebooks are generated, not committed (``*.ipynb`` is gitignored)::

    uv run python src/interpretableintelligence/adapters/groot/colab/build_retrain_notebook.py --out groot_retrain_sae.ipynb

Then open the file in Colab (File > Upload notebook), pick a GPU runtime, and Run all. Steps and how to read the
result: ``docs/groot-retrain-sae.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "https://github.com/jerrydjin/interpretableintelligence"
RELEASE_TAG = "data-groot-1028-v1"
ASSET = "groot-1028-v9-n100.tar.xz"
ASSET_SHA256 = "0f8954e52ae737aec8a9cd734d58efc3e43ad3725d623fae2e5a6eac858cf00e"
# (name, n_latents, revive). The first row is the old SAE's recipe (plain BatchTopK, 8192 latents) retrained here,
# so the sweep can tell "this trainer" apart from "this dictionary size" and "this revival method".
SWEEP = (
    ("none-8192", 8192, "none"),
    ("auxk-8192", 8192, "auxk"),
    ("auxk-4096", 4096, "auxk"),
    ("auxk-2048", 2048, "auxk"),
    ("auxk-1024", 1024, "auxk"),
    ("none-2048", 2048, "none"),
    ("resample-2048", 2048, "resample"),
)


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def cells(branch: str) -> list[dict]:
    sweep = ",\n".join(f"    ({name!r}, {n}, {revive!r})" for name, n, revive in SWEEP)
    return [
        _md(
            """
# Retrain the GR00T task-1028 SAE

The capture SAE (`groot-1028-v9-n100/sae`, BatchTopK, 8192 latents, k = 32) has about 7,700 latents that never fire.
This notebook retrains it on the same capture (train split only) with dead-latent revival and smaller dictionaries,
and compares every result with the old SAE on the same train and held-out tokens.

Runtime: **GPU** (T4 or L4). About 30 minutes for the whole sweep on a T4. Nothing here changes `split.json`, a
scoreboard, or a rollout result. Steps and how to read the table: `docs/groot-retrain-sae.md`.
"""
        ),
        _code(
            f"""
# 1. Code: the ii repo at the retrain branch. Colab's own torch is used; no other heavy installs.
BRANCH = {branch!r}
!git clone -q --branch $BRANCH {REPO}.git /content/interpretableintelligence || (cd /content/interpretableintelligence && git pull -q)
!pip install -q --no-deps -e /content/interpretableintelligence
!pip install -q typer safetensors
!ii retrain-sae --help | head -5
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (switch the runtime to GPU)")
"""
        ),
        _code(
            f"""
# 2. Capture: unpack groot-1028-v9-n100 from Drive if it is there, else download the draft release asset
#    (needs a GitHub token with access to the repo in Colab secrets as GH_TOKEN).
import hashlib, json, os, pathlib, subprocess, urllib.request
from google.colab import drive
drive.mount("/content/drive")
CAPTURE_TAR = "/content/drive/MyDrive/groot/{ASSET}"   # edit if your copy lives elsewhere
CAPTURE = "/content/groot-1028-v9-n100"
RESULTS = "/content/drive/MyDrive/groot/sae-retrain"     # copied here at the end
if not pathlib.Path(CAPTURE, "acts").is_dir():
    tar = pathlib.Path(CAPTURE_TAR)
    if not tar.is_file():
        from google.colab import userdata
        token = userdata.get("GH_TOKEN")
        api = "https://api.github.com/repos/jerrydjin/interpretableintelligence"
        head = {{"Authorization": f"Bearer {{token}}", "Accept": "application/vnd.github+json"}}
        releases = json.load(urllib.request.urlopen(urllib.request.Request(api + "/releases?per_page=100", headers=head)))
        release = next(r for r in releases if r["tag_name"] == {RELEASE_TAG!r})
        asset = next(a for a in release["assets"] if a["name"] == {ASSET!r})
        tar = pathlib.Path("/content", {ASSET!r})
        req = urllib.request.Request(asset["url"], headers={{**head, "Accept": "application/octet-stream"}})
        with urllib.request.urlopen(req) as src, open(tar, "wb") as dst:
            while block := src.read(1 << 22):
                dst.write(block)
    digest = hashlib.sha256()
    with open(tar, "rb") as fh:
        while block := fh.read(1 << 22):
            digest.update(block)
    assert digest.hexdigest() == {ASSET_SHA256!r}, "capture tarball sha256 does not match data/groot-1028.md"
    subprocess.run(["tar", "-xJf", str(tar), "-C", "/content"], check=True)
print(sorted(p.name for p in pathlib.Path(CAPTURE).iterdir()))
"""
        ),
        _code(
            f"""
# 3. Sweep. Each run writes /content/sae-retrain/<name>/ (new sae/, retrain.json, links to the capture).
SWEEP = [
{sweep},
]
STEPS, BATCH, LR, K = 6000, 4096, 4e-4, 32
for name, n_latents, revive in SWEEP:
    out = f"/content/sae-retrain/{{name}}"
    if pathlib.Path(out, "retrain.json").is_file():
        print("done already:", name); continue
    !ii retrain-sae $CAPTURE --out $out --n-latents $n_latents --revive $revive --k $K --steps $STEPS --batch-tokens $BATCH --lr $LR
"""
        ),
        _code(
            """
# 4. Compare. "old" is the capture's SAE on the same tokens (should show ~456 alive of 8192 on train).
import shutil
rows = []
for name, _n, _r in SWEEP:
    rep = json.load(open(f"/content/sae-retrain/{name}/retrain.json"))
    if not rows and rep["old"]:
        rows.append(("old (capture)", rep["old"]))
    rows.append((name, rep["new"]))
print(f"{'SAE':<16}{'latents':>8}{'alive tr':>9}{'alive ho':>9}{'dense ho':>9}{'L0 ho':>7}{'EV ho':>8}{'EV/dim ho':>10}")
for name, r in rows:
    tr, ho = r["train"], r["heldout"]
    print(f"{name:<16}{tr['alive_latents'] + tr['dead_latents']:>8}{tr['alive_latents']:>9}{ho['alive_latents']:>9}"
          f"{ho['dense_latents']:>9}{ho['l0_mean']:>7.2f}{ho['ev_raw']:>8.4f}{ho['ev_per_dim_mean']:>10.4f}")
ho = rows[0][1]["heldout"]
print(f"\\nInput check: the {len(ho['top_dims'])} largest-variance channels hold {ho['variance_share_top_dims']:.1%} of held-out variance "
      f"(channels {ho['top_dims']}).")
summary = {name: r for name, r in rows}
os.makedirs(RESULTS, exist_ok=True)
json.dump(summary, open(f"{RESULTS}/sweep-summary.json", "w"), indent=1)
for name, _n, _r in SWEEP:
    dst = pathlib.Path(RESULTS, name)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(f"/content/sae-retrain/{name}/retrain.json", dst / "retrain.json")
    shutil.copytree(f"/content/sae-retrain/{name}/sae", dst / "sae", dirs_exist_ok=True)
print("copied to", RESULTS, "- send sweep-summary.json back to the thread")
"""
        ),
    ]


def build(branch: str) -> dict:
    return {
        "cells": cells(branch),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("groot_retrain_sae.ipynb"))
    parser.add_argument("--branch", default="main", help="Repo branch the notebook clones.")
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(args.branch), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
