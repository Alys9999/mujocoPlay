#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mujocoplay}"
REPO_ID="${REPO_ID:-google/paligemma-3b-pt-224}"
OUT_DIR="${OUT_DIR:-/root/models/paligemma-3b-pt-224}"
HF_TOKEN="${HF_TOKEN:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_paligemma_tokenizer.sh

Optional environment overrides:
  ENV_NAME   Conda environment that contains the hf CLI. Default: mujocoplay
  REPO_ID    Hugging Face repo id to download from. Default: google/paligemma-3b-pt-224
  OUT_DIR    Local directory to write tokenizer files to. Default: /root/models/paligemma-3b-pt-224
  HF_TOKEN   Optional Hugging Face token passed to hf download

What it downloads:
  - config.json
  - tokenizer.json
  - tokenizer.model
  - tokenizer_config.json
  - special_tokens_map.json
  - added_tokens.json
  - spiece.model

Notes:
  - This repo is gated. Your Hugging Face account must already have access.
  - The downloaded directory can be passed to
    PI05SequentialPolicy(..., tokenizer_name_or_path="/path/to/dir")
EOF
}

log() {
  printf '[download_paligemma_tokenizer] %s\n' "$*"
}

die() {
  printf '[download_paligemma_tokenizer] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  die "conda was not found in PATH. Activate your conda setup first."
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  die "Conda environment '$ENV_NAME' was not found."
fi

mkdir -p "$OUT_DIR"

hf_cmd=(
  hf download
  "$REPO_ID"
  --local-dir "$OUT_DIR"
  --include "config.json"
  --include "tokenizer.json"
  --include "tokenizer.model"
  --include "tokenizer_config.json"
  --include "special_tokens_map.json"
  --include "added_tokens.json"
  --include "spiece.model"
)

if [[ -n "$HF_TOKEN" ]]; then
  hf_cmd+=(--token "$HF_TOKEN")
fi

log "Downloading tokenizer files from ${REPO_ID} into ${OUT_DIR}"
if ! conda run -n "$ENV_NAME" "${hf_cmd[@]}"; then
  die "Download failed. Make sure your Hugging Face account is logged in and approved for ${REPO_ID}."
fi

required_one_of=0
for candidate in tokenizer.json tokenizer.model spiece.model; do
  if [[ -f "$OUT_DIR/$candidate" ]]; then
    required_one_of=1
    break
  fi
done

if [[ ! -f "$OUT_DIR/tokenizer_config.json" || "$required_one_of" != "1" ]]; then
  die "Download finished but the tokenizer directory looks incomplete: ${OUT_DIR}"
fi

log "Tokenizer files are ready at ${OUT_DIR}"
printf '%s\n' "$OUT_DIR"
