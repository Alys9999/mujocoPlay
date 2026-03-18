from __future__ import annotations

import os
import platform


def configure_mujoco_gl() -> str | None:
    """Choose a headless-friendly MuJoCo GL backend when one is not preset."""
    existing = os.environ.get("MUJOCO_GL")
    if existing:
        if existing in {"egl", "osmesa"} and "PYOPENGL_PLATFORM" not in os.environ:
            os.environ["PYOPENGL_PLATFORM"] = existing
        return existing

    system = platform.system().lower()
    if system == "linux":
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return os.environ.get("MUJOCO_GL")
