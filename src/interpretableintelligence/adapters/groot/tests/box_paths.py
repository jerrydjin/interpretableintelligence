"""Where the GR00T box scripts look for the checkpoint, the Colab checkout and run folders.

The box scripts in this directory (``check_*.py``, ``mock_*.py``, ``v9_standin.py``) need a GPU
box's trees. None of them exist on a fresh clone, and each script skips itself when its inputs are
missing. Nothing here hardcodes a user's home: the checkpoint is found through the Hugging Face
cache (``HF_HOME`` / ``HF_HUB_CACHE`` are honoured), and the two box trees can be moved with
``II_GROOT_COLAB_DIR`` and ``II_GROOT_RUNS_DIR`` (defaults are the box's ``/workspace`` layout).
"""

from __future__ import annotations

import os
from pathlib import Path

CHECKPOINT = "nvidia/gr00t17-lerobot-libero_10-640"
COLAB = Path(os.environ.get("II_GROOT_COLAB_DIR") or "/workspace/groot-colab")
RUNS = Path(os.environ.get("II_GROOT_RUNS_DIR") or "/workspace/runs")


def groot_snapshot(repo_id: str = CHECKPOINT) -> str | None:
    """Local snapshot directory of ``repo_id`` if it is already in the HF cache, else ``None``. Never downloads."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id, local_files_only=True)
    except Exception:  # noqa: BLE001 (not cached, offline, or huggingface_hub missing: all mean "absent")
        return None
