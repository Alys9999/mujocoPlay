from pathlib import Path

import numpy as np
import pytest

import phase1.policy_benchmark as policy_benchmark


class _StubPolicy:
    name = "stub"
    requires_image = True
    benchmark_interface = "default"

    def reset(self) -> None:
        return None

    def act(self, observation: dict[str, object]) -> np.ndarray:
        assert "overview_rgb" in observation
        return np.zeros(4, dtype=float)


class _StubSession:
    def __init__(self) -> None:
        self.capture_calls = 0
        self._step_index = 0

    def reset(self, **kwargs) -> dict[str, object]:
        assert kwargs["include_image"] is True
        self._step_index = 0
        return {"overview_rgb": np.full((2, 2, 3), 1, dtype=np.uint8)}

    def step(self, action: np.ndarray, include_image: bool) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        del action
        assert include_image is True
        self._step_index += 1
        observation = {"overview_rgb": np.full((2, 2, 3), self._step_index + 1, dtype=np.uint8)}
        terminated = self._step_index >= 2
        info = {"step_count": self._step_index, "episode_duration_sec": float(self._step_index)}
        return observation, 0.0, terminated, False, info

    def capture_frame(self) -> np.ndarray:
        self.capture_calls += 1
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def summarize_episode(self) -> dict[str, object]:
        return {"success": True, "step_count": self._step_index}


@pytest.mark.fast
def test_rollout_policy_reuses_observation_image_for_video(monkeypatch, tmp_path):
    recorded_frames: list[np.ndarray] = []
    session = _StubSession()

    class _StubWriter:
        def __init__(self, output_path: Path, fps: int, buffer_size: int) -> None:
            assert output_path == tmp_path / "episode.mp4"
            assert fps == 20
            assert buffer_size == 3

        def submit(self, frame: np.ndarray) -> None:
            recorded_frames.append(np.array(frame, copy=True))

        def close(self) -> None:
            return None

    monkeypatch.setattr(policy_benchmark, "AsyncVideoWriter", _StubWriter)

    summary = policy_benchmark.rollout_policy(
        policy=_StubPolicy(),
        session=session,
        seed=0,
        hidden_context={},
        target_xy=np.zeros(2, dtype=float),
        video_path=tmp_path / "episode.mp4",
        video_fps=20,
        video_buffer_size=3,
    )

    assert [int(frame.sum()) for frame in recorded_frames] == [12, 24, 36]
    assert session.capture_calls == 0
    assert summary["video_path"] == str(tmp_path / "episode.mp4")
