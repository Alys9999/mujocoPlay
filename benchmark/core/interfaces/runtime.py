from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .control_schema import ControlSchemaAdapter
from .observation import ObservationBundle, StepResult
from benchmark.schemas.models.action_packet import ActionPacket


class SceneRuntime(Protocol):
    """Runtime wrapper around one concrete simulator/environment backend."""

    name: str
    control_adapter: ControlSchemaAdapter
    camera_names: tuple[str, ...]

    def reset(
        self,
        *,
        seed: int | None = None,
        hidden_context: dict[str, dict[str, Any]] | None = None,
        target_xy: np.ndarray | None = None,
    ) -> ObservationBundle:
        """Reset the simulator for one episode."""

    def step(self, action_packet: ActionPacket) -> StepResult:
        """Apply one canonical action packet and advance the runtime."""

    def summarize_episode(self) -> dict[str, Any]:
        """Return benchmark episode metrics after termination or truncation."""

    def get_privileged_context(self) -> dict[str, Any]:
        """Return hidden context for trace sinks only."""

    def close(self) -> None:
        """Release any renderer or backend resources."""
