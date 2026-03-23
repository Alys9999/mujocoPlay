#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mujocoplay}"
REPO_ID="${REPO_ID:-pcuenq/gemma-tokenizer}"
OUT_DIR="${OUT_DIR:-/root/models/gemma-tokenizer-public}"
HF_TOKEN="${HF_TOKEN:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_public_gemma_tokenizer.sh

Optional environment overrides:
  ENV_NAME   Conda environment that contains the hf CLI. Default: mujocoplay
  REPO_ID    Hugging Face repo id to download from. Default: pcuenq/gemma-tokenizer
  OUT_DIR    Local directory to write tokenizer files to. Default: /root/models/gemma-tokenizer-public
  HF_TOKEN   Optional Hugging Face token passed to hf download

What it downloads:
  - config.json
  - tokenizer.json
  - tokenizer.model
  - tokenizer_config.json
  - special_tokens_map.json

Notes:
  - This repo is public and does not require gated PaliGemma access.
  - The downloaded directory can be passed to
    PI05SequentialPolicy(..., tokenizer_name_or_path="/path/to/dir")
EOF
}

log() {
  printf '[download_public_gemma_tokenizer] %s\n' "$*"
}

die() {
  printf '[download_public_gemma_tokenizer] ERROR: %s\n' "$*" >&2
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
)

if [[ -n "$HF_TOKEN" ]]; then
  hf_cmd+=(--token "$HF_TOKEN")
fi

log "Downloading tokenizer files from ${REPO_ID} into ${OUT_DIR}"
if ! conda run -n "$ENV_NAME" "${hf_cmd[@]}"; then
  die "Download failed for ${REPO_ID}."
fi

if [[ ! -f "$OUT_DIR/tokenizer_config.json" ]]; then
  die "Download finished but tokenizer_config.json is missing: ${OUT_DIR}"
fi

if [[ ! -f "$OUT_DIR/tokenizer.json" && ! -f "$OUT_DIR/tokenizer.model" ]]; then
  die "Download finished but no tokenizer serialization file was found in: ${OUT_DIR}"
fi

log "Tokenizer files are ready at ${OUT_DIR}"
printf '%s\n' "$OUT_DIR"
