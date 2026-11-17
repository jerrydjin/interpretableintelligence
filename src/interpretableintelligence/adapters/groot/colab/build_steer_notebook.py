"""Write the Colab notebook that runs the steering scorecard on the real GR00T model (task 1028).

Notebooks are generated, not committed (``*.ipynb`` is gitignored)::

    uv run python src/interpretableintelligence/adapters/groot/colab/build_steer_notebook.py --out groot_steer_scorecard.ipynb

Then open the file in Colab (File > Upload notebook), pick a GPU runtime (T4 or L4), and Run all. Every heavy step
is one shell script, ``scripts/groot_steer_scorecard.sh``, so the same run works on any GPU box with a shell.
Steps and how to read the result: ``docs/groot-steer-scorecard.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "https://github.com/jerrydjin/interpretableintelligence"
RELEASE_TAG = "data-groot-1028-v1"
ASSET = "groot-1028-v9-n100.tar.xz"
ASSET_SHA256 = "0f8954e52ae737aec8a9cd734d58efc3e43ad3725d623fae2e5a6eac858cf00e"  # data/groot-1028.md
# Picked by `ii rollouts --pick` on the capture SAE (PR 50). None passed the offline gates; this measures them.
FEATURES = (6262, 7353, 7775)
EPISODES = 24
HELDOUT_FRAC = 0.75


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def cells(branch: str) -> list[dict]:
    features = ",".join(str(f) for f in FEATURES)
    return [
        _md(
            f"""
# Steering scorecard on the real GR00T model (LIBERO-plus task 1028)

Records {EPISODES} closed-loop GR00T episodes on task 1028, captures DiT block 5 on them with the task-1028 capture SAE
frozen in (so feature ids mean what they meant in PR 50), then removes and boosts features **{features}** on the held-out
episodes against 19 matched random-direction controls and reports how much, and how reliably, the planned action
chunk moves.

Runtime: **GPU** (T4 or L4). About 1.5 to 2 hours on a T4, most of it recording. Every step resumes, so after a
disconnect just Run all again. Offline measurement only: it never touches a scoreboard, and it is not a rollout
finding. Guide: `docs/groot-steer-scorecard.md`.
"""
        ),
        _code(
            f"""
# 1. Code and one-time setup: this repo, lerobot[groot]==0.7.0, LIBERO-plus and its assets (about 15 minutes).
#    Everything heavy runs as shell commands, so no runtime restart is needed after the installs.
BRANCH = {branch!r}
!git clone -q --branch $BRANCH {REPO}.git /content/interpretableintelligence || (cd /content/interpretableintelligence && git pull -q)
%cd /content/interpretableintelligence
!PLUS_DIR=/content/LIBERO-plus bash scripts/groot_gpu_setup.sh
"""
        ),
        _code(
            f"""
# 2. Capture (for its SAE): from Drive if it is there, else the draft release asset
#    (needs a GitHub token with access to the repo in Colab secrets as GH_TOKEN). The run script checks the sha256.
import json, pathlib, urllib.request
from google.colab import drive
drive.mount("/content/drive")
CAPTURE_TAR = "/content/drive/MyDrive/groot/{ASSET}"   # edit if your copy lives elsewhere
RESULTS = "/content/drive/MyDrive/groot/steer-scorecard"   # small result files are copied here at the end
if not pathlib.Path(CAPTURE_TAR).is_file():
    from google.colab import userdata
    token = userdata.get("GH_TOKEN")
    api = "https://api.github.com/repos/jerrydjin/interpretableintelligence"
    head = {{"Authorization": f"Bearer {{token}}", "Accept": "application/vnd.github+json"}}
    releases = json.load(urllib.request.urlopen(urllib.request.Request(api + "/releases?per_page=100", headers=head)))
    release = next(r for r in releases if r["tag_name"] == {RELEASE_TAG!r})
    asset = next(a for a in release["assets"] if a["name"] == {ASSET!r})
    CAPTURE_TAR = "/content/{ASSET}"
    if not pathlib.Path(CAPTURE_TAR).is_file():
        req = urllib.request.Request(asset["url"], headers={{**head, "Accept": "application/octet-stream"}})
        with urllib.request.urlopen(req) as src, open(CAPTURE_TAR, "wb") as dst:
            while block := src.read(1 << 22):
                dst.write(block)
print("capture:", CAPTURE_TAR)
"""
        ),
        _code(
            f"""
# 3. Record, capture, score. Edit the knobs here; the script's header lists them all.
FEATURES, EPISODES, HELDOUT_FRAC = {features!r}, {EPISODES}, {HELDOUT_FRAC}
!cd /content/interpretableintelligence && MUJOCO_GL=egl PLUS_DIR=/content/LIBERO-plus RESULTS=$RESULTS FEATURES=$FEATURES EPISODES=$EPISODES HELDOUT_FRAC=$HELDOUT_FRAC bash scripts/groot_steer_scorecard.sh $CAPTURE_TAR /content/groot-1028-steer
"""
        ),
        _md(
            """
## Reading the result

The table printed by step 3 has one line per feature and edit (remove, boost). **change vs controls** is the mean
relative L2 change of the planned 40-step action chunk for the feature edit and for the matched random controls;
**beats every control on X%** is how often the feature moved the chunk more than all 19 controls on the same window
(chance is 5%); **beats controls** needs p ≤ 0.05 and an excess whose 95% episode-bootstrap interval is above 0.

Check the two lines under it first. Input parity must pass (it stops the run otherwise), and the SAE's held-out EV
on the new clips should be close to the capture's 0.998. If it is far lower, the adapter's hook or inputs do not match
the capture and the scorecard does not mean anything yet.

Send `scorecard.json` and `record.json` from `MyDrive/groot/steer-scorecard/` back to the thread.
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
    parser.add_argument("--out", type=Path, default=Path("groot_steer_scorecard.ipynb"))
    parser.add_argument("--branch", default="main", help="Repo branch the notebook clones.")
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(args.branch), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
