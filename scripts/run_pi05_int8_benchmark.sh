#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_KWARGS="$(tr -d '\n' < "$ROOT_DIR/configs/pi05_int8_linux.json")"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

python -m phase1.policy_benchmark \
  --task pick_place \
  --pipeline all \
  --policies pi05-int8 \
  --policy-kwargs "$POLICY_KWARGS" \
  --output "$ROOT_DIR/benchmark_results/pi05_int8_benchmark.md" \
  "$@"
