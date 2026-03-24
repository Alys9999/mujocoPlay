from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


_PROBE_XML = """
<mujoco>
  <visual>
    <global offwidth="16" offheight="16"/>
  </visual>
  <worldbody/>
</mujoco>
""".strip()
_PROBE_SCRIPT = (
    "import mujoco; "
    f"model = mujoco.MjModel.from_xml_string({_PROBE_XML!r}); "
    "renderer = mujoco.Renderer(model, width=16, height=16); "
    "renderer.close()"
)
_PROBE_TIMEOUT_SEC = 15
_LAST_GL_DIAGNOSTIC: str | None = None
_PROBE_ERROR_PREFIXES = (
    "ImportError:",
    "RuntimeError:",
    "ValueError:",
    "AttributeError:",
    "ModuleNotFoundError:",
)


def _has_headless_gpu_or_dri_device() -> bool:
    return any(
        Path(device_path).exists()
        for device_path in (
            "/dev/nvidiactl",
            "/dev/nvidia0",
            "/dev/dri/renderD128",
        )
    )


def _set_gl_environment(backend: str) -> None:
    os.environ["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        os.environ["PYOPENGL_PLATFORM"] = backend
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)


def _backend_candidates() -> tuple[str, ...]:
    system = platform.system().lower()
    if system != "linux":
        return ()

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if has_display:
        return ("egl", "glfw", "osmesa")

    if _has_headless_gpu_or_dri_device():
        return ("egl", "osmesa")

    return ("osmesa", "egl")


def _summarize_probe_failure(output: str, returncode: int) -> str:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Exception ignored in:"):
            continue
        if stripped.startswith(_PROBE_ERROR_PREFIXES):
            return stripped
    return output or f"renderer probe exited with code {returncode}"


def probe_mujoco_renderer(backend: str | None = None) -> tuple[bool, str]:
    """Return whether MuJoCo can create a renderer for the selected backend."""
    selected_backend = backend or os.environ.get("MUJOCO_GL")
    if not selected_backend:
        return False, "MUJOCO_GL is unset."

    env = os.environ.copy()
    env["MUJOCO_GL"] = selected_backend
    if selected_backend in {"egl", "osmesa"}:
        env["PYOPENGL_PLATFORM"] = selected_backend
    else:
        env.pop("PYOPENGL_PLATFORM", None)

    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=_PROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return False, f"{selected_backend} probe timed out after {_PROBE_TIMEOUT_SEC}s."
    except OSError as exc:
        return False, f"{selected_backend} probe could not start: {type(exc).__name__}: {exc}"

    if result.returncode == 0:
        return True, f"{selected_backend} renderer probe succeeded."

    combined_output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip()).strip()
    return False, _summarize_probe_failure(combined_output, result.returncode)


def get_mujoco_gl_diagnostic() -> str | None:
    """Return the latest auto-selection diagnostic, if any."""
    return _LAST_GL_DIAGNOSTIC


def configure_mujoco_gl() -> str | None:
    """Choose a MuJoCo GL backend that can create a real renderer."""
    global _LAST_GL_DIAGNOSTIC

    existing = os.environ.get("MUJOCO_GL")
    if existing:
        _set_gl_environment(existing)
        _LAST_GL_DIAGNOSTIC = f"Using explicit MUJOCO_GL={existing}."
        return existing

    candidates = _backend_candidates()
    if not candidates:
        _LAST_GL_DIAGNOSTIC = f"No automatic MUJOCO_GL candidates for platform={platform.system()}."
        return os.environ.get("MUJOCO_GL")

    diagnostics: list[str] = []
    for backend in candidates:
        ok, detail = probe_mujoco_renderer(backend)
        diagnostics.append(detail)
        if ok:
            _set_gl_environment(backend)
            _LAST_GL_DIAGNOSTIC = "; ".join(diagnostics)
            return backend

    fallback = candidates[0]
    _set_gl_environment(fallback)
    _LAST_GL_DIAGNOSTIC = "; ".join(diagnostics)
    return fallback
