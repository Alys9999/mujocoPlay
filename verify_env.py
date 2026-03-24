from __future__ import annotations

import ctypes.util
import importlib
import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from phase1.mujoco_runtime import configure_mujoco_gl, get_mujoco_gl_diagnostic, probe_mujoco_renderer

configure_mujoco_gl()


REQUIRED_MODULES = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("mujoco", "mujoco"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors", "safetensors"),
    ("PIL", "pillow"),
    ("lerobot", "lerobot"),
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _distribution_version(distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _import_module(module_name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _compact_detail(detail: str, max_lines: int = 6, max_chars: int = 800) -> str:
    lines = [line.rstrip() for line in detail.splitlines() if line.strip()]
    if not lines:
        return detail[:max_chars]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    compact = " | ".join(lines)
    return compact[:max_chars]


def _check_python() -> CheckResult:
    version = platform.python_version()
    ok = sys.version_info >= (3, 12)
    detail = f"python={version} executable={sys.executable}"
    if not ok:
        detail += " required>=3.12"
    return CheckResult("python", ok, detail)


def _check_torch_runtime() -> list[CheckResult]:
    module, error = _import_module("torch")
    if module is None:
        return [CheckResult("torch_runtime", False, error or "torch import failed")]

    torch = module
    results = [
        CheckResult("torch", True, f"torch={getattr(torch, '__version__', 'unknown')}"),
        CheckResult("torch_cuda_build", True, f"torch.version.cuda={getattr(torch.version, 'cuda', None)}"),
        CheckResult("torch_cuda_available", bool(torch.cuda.is_available()), f"cuda_available={torch.cuda.is_available()}"),
    ]

    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            results.append(CheckResult("torch_cuda_device", True, f"device_count={device_count} first_device={device_name}"))
        except Exception as exc:
            results.append(CheckResult("torch_cuda_device", False, f"{type(exc).__name__}: {exc}"))
    else:
        results.append(CheckResult("torch_cuda_device", False, "no CUDA device available"))

    try:
        cudnn_version = torch.backends.cudnn.version()
        cudnn_ok = bool(cudnn_version)
        results.append(CheckResult("torch_cudnn", cudnn_ok, f"cudnn_version={cudnn_version}"))
    except Exception as exc:
        results.append(CheckResult("torch_cudnn", False, f"{type(exc).__name__}: {exc}"))

    return results


def _read_first_ubuntu_archive_codename(path: Path) -> str | None:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return None

    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3 or parts[0] != "deb" or "ubuntu" not in parts[1]:
            continue
        return parts[2].split("-", 1)[0]
    return None


def _check_linux_host_runtime() -> list[CheckResult]:
    if platform.system().lower() != "linux":
        return []

    results: list[CheckResult] = []
    os_release = platform.freedesktop_os_release() if hasattr(platform, "freedesktop_os_release") else {}
    if os_release.get("ID") == "ubuntu":
        target_codename = os_release.get("UBUNTU_CODENAME") or os_release.get("VERSION_CODENAME")
        source_codename = _read_first_ubuntu_archive_codename(Path("/etc/apt/sources.list"))
        if target_codename and source_codename:
            ok = source_codename == target_codename
            detail = f"os_release={target_codename} apt_sources={source_codename}"
            if not ok:
                detail += " run `bash scripts/repair_ubuntu_apt_sources.sh --apply`"
            results.append(CheckResult("apt_sources_codename", ok, detail))

    lib_gl = ctypes.util.find_library("GL")
    lib_egl = ctypes.util.find_library("EGL")
    lib_osmesa = ctypes.util.find_library("OSMesa")
    results.append(CheckResult("gl_runtime_opengl", bool(lib_gl), f"libGL={lib_gl or 'missing'}"))
    results.append(
        CheckResult(
            "gl_runtime_backend",
            bool(lib_egl or lib_osmesa),
            f"libEGL={lib_egl or 'missing'} libOSMesa={lib_osmesa or 'missing'}",
        )
    )
    return results


def _check_mujoco_runtime() -> list[CheckResult]:
    module, error = _import_module("mujoco")
    if module is None:
        return [CheckResult("mujoco_runtime", False, error or "mujoco import failed")]

    mujoco = module
    version = getattr(mujoco, "__version__", "unknown")
    gl_backend = os.environ.get("MUJOCO_GL", "<unset>")
    results = [
        CheckResult("mujoco", True, f"mujoco={version}"),
        CheckResult("mujoco_gl_env", True, f"MUJOCO_GL={gl_backend}"),
    ]
    diagnostic = get_mujoco_gl_diagnostic()
    if diagnostic:
        results.append(CheckResult("mujoco_gl_selection", True, _compact_detail(diagnostic)))

    renderer_ok, renderer_detail = probe_mujoco_renderer(gl_backend if gl_backend != "<unset>" else None)
    results.append(CheckResult("mujoco_renderer", renderer_ok, _compact_detail(renderer_detail)))
    return results


def _check_required_modules() -> list[CheckResult]:
    results: list[CheckResult] = []
    for module_name, distribution_name in REQUIRED_MODULES:
        module, error = _import_module(module_name)
        if module is None:
            results.append(CheckResult(module_name, False, error or "import failed"))
            continue
        version = getattr(module, "__version__", None) or _distribution_version(distribution_name)
        results.append(CheckResult(module_name, True, f"version={version}"))
    return results


def _print_results(results: list[CheckResult]) -> int:
    failures = 0
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        if not result.ok:
            failures += 1
    return failures


def main() -> int:
    print("Repository environment verification")
    print(f"platform={platform.platform()}")
    print(f"cwd={os.getcwd()}")
    print()

    results: list[CheckResult] = []
    results.append(_check_python())
    results.extend(_check_torch_runtime())
    results.extend(_check_linux_host_runtime())
    results.extend(_check_mujoco_runtime())
    results.extend(_check_required_modules())

    failures = _print_results(results)
    print()
    print(f"summary: total_checks={len(results)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
