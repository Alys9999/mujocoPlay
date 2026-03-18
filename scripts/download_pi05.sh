#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-lerobot/pi05_base}"
TARGET_DIR="${TARGET_DIR:-$ROOT_DIR/external/models/pi05_base}"

mkdir -p "$(dirname "$TARGET_DIR")"
python -m huggingface_hub download \
  "$MODEL_ID" \
  --local-dir "$TARGET_DIR" \
  --local-dir-use-symlinks False
