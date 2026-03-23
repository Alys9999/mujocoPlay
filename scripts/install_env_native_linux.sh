#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-native}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCH_WHL_TAG="${TORCH_WHL_TAG:-${CUDA_WHL_TAG:-auto}}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-auto}"
SKIP_APT="${SKIP_APT:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
  printf '[install_env_native_linux] %s\n' "$*"
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
    echo "sudo is required to install system packages. Re-run with sudo or set SKIP_APT=1 if dependencies are already present." >&2
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

python_requirement_satisfied() {
  local interpreter="$1"
  "$interpreter" - <<'PY' >/dev/null 2>&1
import sys

raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
}

require_python() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    return
  fi

  if [[ "$PYTHON_BIN" != "python3" ]] && command -v python3 >/dev/null 2>&1 && python_requirement_satisfied python3; then
    log "Requested ${PYTHON_BIN} was not found; falling back to python3 from PATH because it already satisfies Python >= 3.12."
    PYTHON_BIN="python3"
    return
  fi

  if [[ "$SKIP_APT" == "1" ]]; then
    echo "Python 3.12+ is required. Interpreter '${PYTHON_BIN}' was not found and SKIP_APT=1 prevented installing it." >&2
    echo "Use scripts/install_env_conda.sh or install Python 3.12 separately and re-run with PYTHON_BIN=/path/to/python3.12." >&2
    exit 1
  fi

  if [[ "$PYTHON_BIN" =~ ^python3(\.[0-9]+)?$ ]]; then
    if ! apt-cache show "$PYTHON_BIN" >/dev/null 2>&1; then
      echo "Python 3.12+ is required, but apt does not provide package '${PYTHON_BIN}' in the current repositories." >&2
      echo "On Ubuntu 22.04 this usually means you should use scripts/install_env_conda.sh or preinstall Python 3.12 and set PYTHON_BIN." >&2
      exit 1
    fi
    apt_run update
    apt_run install -y "$PYTHON_BIN"
    return
  fi

  echo "Python 3.12+ interpreter '${PYTHON_BIN}' was not found. Set PYTHON_BIN to an installed Python 3.12 interpreter." >&2
  exit 1
}

check_python_version() {
  "$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit(f"Python >= 3.12 is required, found {sys.version.split()[0]}")
PY
}

ensure_venv_support() {
  local probe_dir
  probe_dir="$(mktemp -d)"
  if "$PYTHON_BIN" -m venv "$probe_dir/test" >/dev/null 2>&1; then
    rm -rf "$probe_dir"
    return
  fi
  rm -rf "$probe_dir"

  if [[ "$SKIP_APT" == "1" ]]; then
    echo "python -m venv is unavailable for ${PYTHON_BIN} and SKIP_APT=1 prevented installing venv support." >&2
    exit 1
  fi

  if [[ "$PYTHON_BIN" =~ ^python3(\.[0-9]+)?$ ]]; then
    if ! apt-cache show "${PYTHON_BIN}-venv" >/dev/null 2>&1; then
      echo "The venv package '${PYTHON_BIN}-venv' is not available in the current apt repositories." >&2
      echo "Use scripts/install_env_conda.sh or install a Python 3.12 build that includes venv support." >&2
      exit 1
    fi
    apt_run update
    apt_run install -y "${PYTHON_BIN}-venv"
    return
  fi

  echo "python -m venv failed for ${PYTHON_BIN}. Install the matching venv package and re-run." >&2
  exit 1
}

detect_torch_whl_tag() {
  if [[ "$TORCH_WHL_TAG" != "auto" ]]; then
    printf '%s\n' "$TORCH_WHL_TAG"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "cu128"
    return
  fi

  if [[ -e /dev/nvidiactl || -e /dev/nvidia0 ]]; then
    printf '%s\n' "cu128"
    return
  fi

  printf '%s\n' "cpu"
}

detect_mujoco_gl_backend() {
  if [[ -n "${MUJOCO_GL:-}" ]]; then
    printf '%s\n' "$MUJOCO_GL"
    return
  fi

  if [[ "$MUJOCO_GL_BACKEND" != "auto" ]]; then
    printf '%s\n' "$MUJOCO_GL_BACKEND"
    return
  fi

  if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    printf '%s\n' "egl"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "egl"
    return
  fi

  if [[ -e /dev/nvidiactl || -e /dev/nvidia0 || -e /dev/dri/renderD128 ]]; then
    printf '%s\n' "egl"
    return
  fi

  printf '%s\n' "osmesa"
}

require_python
check_python_version

if [[ "$SKIP_APT" != "1" ]]; then
  GLEW_RUNTIME_PACKAGE="$(first_available_package libglew2.2 libglew2.1 libglew2.0 || true)"
  if [[ -z "${GLEW_RUNTIME_PACKAGE:-}" ]]; then
    echo "Unable to find a supported libglew runtime package via apt-cache." >&2
    exit 1
  fi

  apt_run update
  apt_run install -y \
    build-essential \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    "$GLEW_RUNTIME_PACKAGE" \
    libglfw3 \
    libosmesa6 \
    libxcursor1 \
    libxinerama1 \
    libxi6
fi

ensure_venv_support

"$PYTHON_BIN" -m venv "$ROOT_DIR/$VENV_DIR"
source "$ROOT_DIR/$VENV_DIR/bin/activate"

TORCH_WHL_TAG="$(detect_torch_whl_tag)"
export MUJOCO_GL="$(detect_mujoco_gl_backend)"
if [[ "$MUJOCO_GL" == "egl" || "$MUJOCO_GL" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM="$MUJOCO_GL"
fi

log "Using python=$(python --version 2>&1)"
log "Using torch wheel tag=${TORCH_WHL_TAG}"
log "Using MUJOCO_GL=${MUJOCO_GL}"

PIP_CONFIG_FILE=/dev/null python -m pip install --upgrade pip
PIP_CONFIG_FILE=/dev/null python -m pip install \
  "torch==${TORCH_VERSION}+${TORCH_WHL_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+${TORCH_WHL_TAG}" \
  --index-url "https://download.pytorch.org/whl/${TORCH_WHL_TAG}"
PIP_CONFIG_FILE=/dev/null python -m pip install \
  --index-url "https://pypi.org/simple" \
  -r "$ROOT_DIR/requirements.txt"

python "$ROOT_DIR/verify_env.py"
