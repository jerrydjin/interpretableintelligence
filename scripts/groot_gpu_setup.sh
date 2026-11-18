#!/usr/bin/env bash
# One-time setup of a GPU box (Colab T4/L4, or any Ubuntu machine with an NVIDIA GPU) for GR00T on LIBERO-plus.
#
#   bash scripts/groot_gpu_setup.sh            # from the repo root
#
# Installs, into the current Python (Colab's own, or an activated venv with a CUDA torch already in it):
#   * this repo (pip install -e ., so `ii` is on PATH; the CUDA torch already installed satisfies its torch pin)
#   * lerobot[groot]==0.7.0 (GR00T N1.7 and the LIBERO env wrapper)
#   * LIBERO-plus (github.com/sylvestf/LIBERO-plus) and its assets (huggingface.co/datasets/Sylvest/LIBERO-plus)
# and writes the LIBERO config. Set PLUS_DIR to put the LIBERO-plus checkout elsewhere (default ./.libero-plus).
# Rerunning skips what is already there.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUS_DIR="${PLUS_DIR:-$REPO_ROOT/.libero-plus}"
PY="${PYTHON:-python3}"
SUDO=""
if [ "$(id -u)" != "0" ] && command -v sudo >/dev/null; then SUDO="sudo"; fi

echo "== system packages (EGL rendering for MuJoCo, LIBERO-plus extras)"
if command -v apt-get >/dev/null; then
  $SUDO apt-get -qq update
  $SUDO apt-get -qq install -y libegl1 libgl1 libosmesa6 libexpat1 libfontconfig1-dev libmagickwand-dev >/dev/null
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "== python packages"
"$PY" -m pip install -q -e "$REPO_ROOT"
# lerobot's libero extra brings mujoco and robosuite through hf-libero; LIBERO-plus replaces hf-libero's package below.
"$PY" -m pip install -q "lerobot[groot,libero]==0.7.0"
"$PY" -m pip uninstall -y -q hf-libero || true

echo "== LIBERO-plus at $PLUS_DIR"
if [ ! -d "$PLUS_DIR/.git" ]; then
  git clone -q --depth 1 https://github.com/sylvestf/LIBERO-plus.git "$PLUS_DIR"
fi
if [ -f "$PLUS_DIR/extra_requirements.txt" ]; then "$PY" -m pip install -q -r "$PLUS_DIR/extra_requirements.txt"; fi
"$PY" -m pip install -q -e "$PLUS_DIR" --config-settings editable_mode=compat
"$PY" -m pip install -q "gym==0.25.2"  # LIBERO's envs import the old gym API

ASSETS="$PLUS_DIR/libero/libero/assets"
if [ ! -f "$ASSETS/wall.xml" ]; then
  echo "== LIBERO-plus assets (several GB, once)"
  "$PY" - "$PLUS_DIR" <<'PYEOF'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from interpretableintelligence.adapters.groot.lib.place_libero_assets import extract_zip_and_place

plus = Path(sys.argv[1])
zip_path = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset")
# The zip is not rooted at assets/; find the real root and link it where LIBERO looks.
print("assets at", extract_zip_and_place(Path(zip_path), plus / ".assets-unzip", plus / "libero" / "libero" / "assets"))
PYEOF
fi

echo "== LIBERO config"
"$PY" - "$PLUS_DIR" <<'PYEOF'
import sys
from interpretableintelligence.adapters.groot.record_libero import _load_v8
print(_load_v8().write_libero_config(sys.argv[1], log=lambda _m: None))
PYEOF

"$PY" -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo "setup done. export MUJOCO_GL=egl PLUS_DIR=$PLUS_DIR before running scripts/groot_steer_scorecard.sh"
