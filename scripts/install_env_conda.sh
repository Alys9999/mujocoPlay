#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mujocoplay}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
CUDA_WHL_TAG="${CUDA_WHL_TAG:-cu128}"
SKIP_APT="${SKIP_APT:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
  printf '[install_env_conda] %s\n' "$*"
}

apt_run() {
  if [[ "$SKIP_APT" == "1" ]]; then
    return 0
  fi

  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get "$@"
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install Linux runtime packages. Re-run with sudo or set SKIP_APT=1 if they are already present." >&2
    exit 1
  fi

  sudo apt-get "$@"
}

first_available_package() {
  local package_name
  for package_name in "$@"; do
    if apt-cache show "$package_name" >/dev/null 2>&1; then
      printf '%s\n' "$package_name"
      return 0
    fi
  done
  return 1
}

ubuntu_sources_mismatched() {
  local target_codename current_codename
  [[ -f /etc/os-release ]] || return 1
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" ]] || return 1

  target_codename="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
  [[ -n "$target_codename" ]] || return 1
  [[ -f /etc/apt/sources.list ]] || return 1

  current_codename="$(
    awk '
      $1 == "deb" && $2 ~ /ubuntu/ {
        split($3, parts, "-")
        print parts[1]
        exit
      }
    ' /etc/apt/sources.list 2>/dev/null
  )"
  [[ -n "$current_codename" ]] || return 1
  [[ "$current_codename" != "$target_codename" ]]
}

repair_ubuntu_sources_if_needed() {
  if [[ "$SKIP_APT" == "1" ]]; then
    return 0
  fi

  if [[ "$(uname -s)" != "Linux" ]] || ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi

  if ! ubuntu_sources_mismatched; then
    return 0
  fi

  log "Detected Ubuntu apt sources that do not match /etc/os-release; applying scripts/repair_ubuntu_apt_sources.sh --apply."
  if [[ "$(id -u)" -eq 0 ]]; then
    bash "$ROOT_DIR/scripts/repair_ubuntu_apt_sources.sh" --apply
    return 0
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "Ubuntu apt sources appear mismatched, but sudo is unavailable to repair them. Run bash scripts/repair_ubuntu_apt_sources.sh --apply or set SKIP_APT=1 if the host is already fixed." >&2
    exit 1
  fi

  sudo bash "$ROOT_DIR/scripts/repair_ubuntu_apt_sources.sh" --apply
}

ensure_linux_runtime_packages() {
  local glew_runtime_package
  if [[ "$SKIP_APT" == "1" ]]; then
    return 0
  fi

  if [[ "$(uname -s)" != "Linux" ]] || ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi

  repair_ubuntu_sources_if_needed

  glew_runtime_package="$(first_available_package libglew2.2 libglew2.1 libglew2.0 || true)"
  if [[ -z "${glew_runtime_package:-}" ]]; then
    echo "Unable to find a supported libglew runtime package via apt-cache." >&2
    exit 1
  fi

  log "Installing Linux runtime libraries required for MuJoCo rendering."
  apt_run update
  apt_run install -y \
    ffmpeg \
    libegl1 \
    libgl1 \
    "$glew_runtime_package" \
    libglfw3 \
    libosmesa6 \
    libxcursor1 \
    libxinerama1 \
    libxi6
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

ensure_linux_runtime_packages

eval "$(conda shell.bash hook)"

conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
conda activate "$ENV_NAME"

PIP_CONFIG_FILE=/dev/null python -m pip install --upgrade pip
PIP_CONFIG_FILE=/dev/null python -m pip install \
  "torch==${TORCH_VERSION}+${CUDA_WHL_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+${CUDA_WHL_TAG}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHL_TAG}"
PIP_CONFIG_FILE=/dev/null python -m pip install \
  --index-url "https://pypi.org/simple" \
  -r "$ROOT_DIR/requirements.txt"

python "$ROOT_DIR/verify_env.py"
