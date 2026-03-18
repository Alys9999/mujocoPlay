#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mujocoplay}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
CUDA_WHL_TAG="${CUDA_WHL_TAG:-cu128}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install \
  "torch==${TORCH_VERSION}+${CUDA_WHL_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+${CUDA_WHL_TAG}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHL_TAG}"
python -m pip install -r "$ROOT_DIR/requirements.txt"

python "$ROOT_DIR/verify_env.py"
