import time

import numpy as np
import pytest

from phase1.video_io import AsyncVideoWriter, resolve_video_frame


@pytest.mark.fast
def test_resolve_video_frame_reuses_observation_image():
    frame = np.full((2, 3, 3), 7, dtype=np.uint8)
    capture_calls = 0

    def _capture() -> np.ndarray:
        nonlocal capture_calls
        capture_calls += 1
        return np.zeros((2, 3, 3), dtype=np.uint8)

    resolved = resolve_video_frame({"overview_rgb": frame}, _capture)

    assert capture_calls == 0
    assert resolved is not frame
    assert np.array_equal(resolved, frame)


@pytest.mark.fast
def test_resolve_video_frame_falls_back_to_capture():
    capture_calls = 0
    expected = np.full((2, 2, 3), 5, dtype=np.uint8)

    def _capture() -> np.ndarray:
        nonlocal capture_calls
        capture_calls += 1
        return expected

    resolved = resolve_video_frame({}, _capture)

    assert capture_calls == 1
    assert resolved is expected


@pytest.mark.io
def test_async_video_writer_flushes_frames_in_order(monkeypatch, tmp_path):
    appended: list[np.ndarray] = []
    closed = False

    class FakeWriter:
        def append_data(self, frame: np.ndarray) -> None:
            appended.append(np.array(frame, copy=True))

        def close(self) -> None:
            nonlocal closed
            closed = True

    def _get_writer(path, fps):  # noqa: ANN001
        assert path == tmp_path / "episode.mp4"
        assert fps == 20
        return FakeWriter()

    monkeypatch.setattr("phase1.video_io.imageio.get_writer", _get_writer)

    writer = AsyncVideoWriter(output_path=tmp_path / "episode.mp4", fps=20, buffer_size=2)
    writer.submit(np.zeros((2, 2, 3), dtype=np.uint8))
    writer.submit(np.ones((2, 2, 3), dtype=np.uint8))
    writer.close()

    assert closed
    assert [int(frame.sum()) for frame in appended] == [0, 12]


@pytest.mark.io
def test_async_video_writer_propagates_background_failures(monkeypatch, tmp_path):
    class FailingWriter:
        def append_data(self, frame: np.ndarray) -> None:
            del frame
            raise RuntimeError("encode failed")

        def close(self) -> None:
            return None

    monkeypatch.setattr("phase1.video_io.imageio.get_writer", lambda path, fps: FailingWriter())

    writer = AsyncVideoWriter(output_path=tmp_path / "episode.mp4", fps=20, buffer_size=1)
    writer.submit(np.zeros((2, 2, 3), dtype=np.uint8))
    deadline = time.monotonic() + 2.0
    while writer._worker_error is None and time.monotonic() < deadline:
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="Async video writing failed"):
        writer.close()
