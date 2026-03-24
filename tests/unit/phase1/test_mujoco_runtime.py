import os
import subprocess

import pytest

import phase1.mujoco_runtime as mujoco_runtime


@pytest.fixture(autouse=True)
def _clear_gl_environment(monkeypatch):
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "MUJOCO_GL", "PYOPENGL_PLATFORM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(mujoco_runtime, "_LAST_GL_DIAGNOSTIC", None)


@pytest.mark.fast
def test_configure_mujoco_gl_respects_explicit_backend(monkeypatch):
    probe_calls: list[str | None] = []

    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setattr(
        mujoco_runtime,
        "probe_mujoco_renderer",
        lambda backend=None: (probe_calls.append(backend), (True, "ok"))[1],
    )

    backend = mujoco_runtime.configure_mujoco_gl()

    assert backend == "osmesa"
    assert probe_calls == []
    assert os.environ["PYOPENGL_PLATFORM"] == "osmesa"
    assert mujoco_runtime.get_mujoco_gl_diagnostic() == "Using explicit MUJOCO_GL=osmesa."


@pytest.mark.fast
def test_configure_mujoco_gl_falls_back_to_osmesa_when_egl_probe_fails(monkeypatch):
    probe_attempts: list[str | None] = []

    monkeypatch.setattr(mujoco_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mujoco_runtime, "_has_headless_gpu_or_dri_device", lambda: True)

    def _fake_probe(backend: str | None = None) -> tuple[bool, str]:
        probe_attempts.append(backend)
        if backend == "egl":
            return False, "egl renderer probe failed"
        if backend == "osmesa":
            return True, "osmesa renderer probe succeeded"
        raise AssertionError(f"unexpected backend {backend!r}")

    monkeypatch.setattr(mujoco_runtime, "probe_mujoco_renderer", _fake_probe)

    backend = mujoco_runtime.configure_mujoco_gl()

    assert backend == "osmesa"
    assert probe_attempts == ["egl", "osmesa"]
    assert os.environ["MUJOCO_GL"] == "osmesa"
    assert os.environ["PYOPENGL_PLATFORM"] == "osmesa"
    assert "egl renderer probe failed" in (mujoco_runtime.get_mujoco_gl_diagnostic() or "")


@pytest.mark.fast
def test_probe_mujoco_renderer_sets_backend_specific_environment(monkeypatch):
    captured_envs: list[dict[str, str]] = []

    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        command = args[0]
        captured_envs.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setattr(mujoco_runtime.subprocess, "run", _fake_run)

    ok, detail = mujoco_runtime.probe_mujoco_renderer("glfw")

    assert ok is True
    assert detail == "glfw renderer probe succeeded."
    assert captured_envs[0]["MUJOCO_GL"] == "glfw"
    assert "PYOPENGL_PLATFORM" not in captured_envs[0]


@pytest.mark.fast
def test_probe_mujoco_renderer_surfaces_primary_error(monkeypatch):
    def _fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        stderr = "\n".join(
            (
                "Traceback (most recent call last):",
                "ImportError: Cannot initialize a EGL device display.",
                "Exception ignored in: <function Renderer.__del__ at 0x0>",
                "AttributeError: 'Renderer' object has no attribute '_mjr_context'",
            )
        )
        return subprocess.CompletedProcess(["python", "-c", "..."], 1, "", stderr)

    monkeypatch.setattr(mujoco_runtime.subprocess, "run", _fake_run)

    ok, detail = mujoco_runtime.probe_mujoco_renderer("egl")

    assert ok is False
    assert detail == "ImportError: Cannot initialize a EGL device display."
