from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Any


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


def _check_python() -> CheckResult:
    version = platform.python_version()
    ok = sys.version_info >= (3, 10)
    detail = f"python={version} executable={sys.executable}"
    if not ok:
        detail += " required>=3.10"
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


def _check_mujoco_runtime() -> list[CheckResult]:
    module, error = _import_module("mujoco")
    if module is None:
        return [CheckResult("mujoco_runtime", False, error or "mujoco import failed")]

    mujoco = module
    version = getattr(mujoco, "__version__", "unknown")
    gl_backend = os.environ.get("MUJOCO_GL", "<unset>")
    return [
        CheckResult("mujoco", True, f"mujoco={version}"),
        CheckResult("mujoco_gl_env", True, f"MUJOCO_GL={gl_backend}"),
    ]


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
    results.extend(_check_mujoco_runtime())
    results.extend(_check_required_modules())

    failures = _print_results(results)
    print()
    print(f"summary: total_checks={len(results)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
