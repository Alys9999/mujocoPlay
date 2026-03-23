from __future__ import annotations

import os
import platform
from pathlib import Path


def _has_headless_gpu_or_dri_device() -> bool:
    return any(
        Path(device_path).exists()
        for device_path in (
            "/dev/nvidiactl",
            "/dev/nvidia0",
            "/dev/dri/renderD128",
        )
    )


def configure_mujoco_gl() -> str | None:
    """Choose a headless-friendly MuJoCo GL backend when one is not preset."""
    existing = os.environ.get("MUJOCO_GL")
    if existing:
        if existing in {"egl", "osmesa"} and "PYOPENGL_PLATFORM" not in os.environ:
            os.environ["PYOPENGL_PLATFORM"] = existing
        return existing

    system = platform.system().lower()
    if system == "linux":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        backend = "egl" if has_display or _has_headless_gpu_or_dri_device() else "osmesa"
        os.environ["MUJOCO_GL"] = backend
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    return os.environ.get("MUJOCO_GL")
