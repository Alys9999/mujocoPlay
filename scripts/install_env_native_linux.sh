#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-native}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
CUDA_WHL_TAG="${CUDA_WHL_TAG:-cu128}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sudo apt-get update
sudo apt-get install -y \
  "$PYTHON_BIN" \
  "${PYTHON_BIN}-venv" \
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
  libxi6

"$PYTHON_BIN" -m venv "$ROOT_DIR/$VENV_DIR"
source "$ROOT_DIR/$VENV_DIR/bin/activate"

PIP_CONFIG_FILE=/dev/null python -m pip install --upgrade pip
PIP_CONFIG_FILE=/dev/null python -m pip install \
  "torch==${TORCH_VERSION}+${CUDA_WHL_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+${CUDA_WHL_TAG}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHL_TAG}"
PIP_CONFIG_FILE=/dev/null python -m pip install \
  --index-url "https://pypi.org/simple" \
  -r "$ROOT_DIR/requirements.txt"

python "$ROOT_DIR/verify_env.py"
