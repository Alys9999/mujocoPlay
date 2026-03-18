#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-linux-gpu}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ffmpeg \
  git \
  libegl1 \
  libgl1 \
  libglew2.2 \
  libglfw3 \
  libosmesa6 \
  libxcursor1 \
  libxinerama1 \
  libxi6 \
  python3-venv

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
python -m pip install -r "$ROOT_DIR/requirements-linux.txt" --no-deps

cat <<EOF
Native Linux GPU environment is ready.

Activate:
  source "$VENV_DIR/bin/activate"

Recommended headless render setting:
  export MUJOCO_GL=egl

Verification:
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
EOF
