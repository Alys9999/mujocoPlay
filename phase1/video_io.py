from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np

_QUEUE_POLL_TIMEOUT_SEC = 0.1
_QUEUE_SENTINEL = object()


class AsyncVideoWriter:
    """Encode and write video frames on a background thread.

    Args:
        output_path: Video path to write.
        fps: Output frames per second.
        buffer_size: Maximum number of frames buffered ahead of the writer.
    """

    def __init__(self, output_path: Path, fps: int, buffer_size: int = 64) -> None:
        self.output_path = Path(output_path)
        self.fps = max(int(fps), 1)
        self.buffer_size = int(buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("AsyncVideoWriter buffer_size must be >= 1.")

        self._queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=self.buffer_size)
        self._closed = False
        self._worker_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run_writer,
            name=f"video-writer-{self.output_path.stem}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Queue one RGB frame for background encoding."""
        if self._closed:
            raise RuntimeError("Cannot submit a frame after closing the video writer.")
        self._raise_if_failed()

        array = np.asarray(frame)
        if array.ndim != 3:
            raise ValueError(f"Expected an HWC RGB frame, got shape={array.shape!r}.")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)

        self._put_with_backpressure(array)
        self._raise_if_failed()

    def close(self) -> None:
        """Flush the queue and stop the background writer."""
        if not self._closed:
            self._closed = True
            self._put_with_backpressure(_QUEUE_SENTINEL)
        self._thread.join()
        self._raise_if_failed()

    def _put_with_backpressure(self, item: np.ndarray | object) -> None:
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=_QUEUE_POLL_TIMEOUT_SEC)
                return
            except queue.Full:
                if not self._thread.is_alive():
                    self._raise_if_failed()

    def _run_writer(self) -> None:
        writer: Any | None = None
        try:
            while True:
                item = self._queue.get()
                if item is _QUEUE_SENTINEL:
                    break
                if writer is None:
                    self.output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(self.output_path, fps=self.fps)
                writer.append_data(item)
        except BaseException as exc:  # pragma: no cover - exercised via public error path.
            self._worker_error = exc
        finally:
            if writer is not None:
                try:
                    writer.close()
                except BaseException as exc:  # pragma: no cover - exercised via public error path.
                    if self._worker_error is None:
                        self._worker_error = exc

    def _raise_if_failed(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(f"Async video writing failed for {self.output_path}.") from self._worker_error


def compose_video_views(*frames: np.ndarray) -> np.ndarray:
    """Compose one or more HWC RGB views into a single side-by-side frame."""
    valid_frames = [np.asarray(frame) for frame in frames if frame is not None]
    if not valid_frames:
        raise ValueError("compose_video_views requires at least one frame.")
    if len(valid_frames) == 1:
        return np.array(valid_frames[0], copy=True)

    target_height = max(frame.shape[0] for frame in valid_frames)
    padded_frames: list[np.ndarray] = []
    for frame in valid_frames:
        if frame.ndim != 3:
            raise ValueError(f"Expected HWC RGB frame, got shape={frame.shape!r}.")
        if frame.shape[0] == target_height:
            padded_frames.append(np.array(frame, copy=True))
            continue
        pad_total = target_height - frame.shape[0]
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded_frames.append(
            np.pad(
                frame,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        )
    return np.concatenate(padded_frames, axis=1)


def resolve_video_frame(
    observation: dict[str, Any],
    capture_frame: Callable[[], np.ndarray],
) -> np.ndarray:
    """Reuse already-rendered PI05 input views when available."""
    base_frame = observation.get("base_rgb")
    if base_frame is None:
        base_frame = observation.get("overview_rgb")

    arm_frame = observation.get("arm_rgb")
    if arm_frame is None:
        arm_frame = observation.get("left_wrist_rgb")
    if arm_frame is None:
        arm_frame = observation.get("right_wrist_rgb")
    if base_frame is not None:
        if arm_frame is not None and not np.array_equal(base_frame, arm_frame):
            return compose_video_views(base_frame, arm_frame)
        return np.array(base_frame, copy=True)
    return capture_frame()
