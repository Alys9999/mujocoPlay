from __future__ import annotations

from typing import Any, Protocol

from .observation import ObservationBundle
from benchmark.schemas.models.action_packet import ActionPacket


class PolicyAdapter(Protocol):
    """Policy boundary shared by random, pi0.5, and smolVLA adapters."""

    name: str
    requires_image: bool
    required_cameras: tuple[str, ...]
    supported_schema_ids: tuple[str, ...]

    def reset(self) -> None:
        """Reset policy-side recurrent state for a fresh episode."""

    def act(self, observation: ObservationBundle) -> ActionPacket:
        """Produce one canonical action packet from the public observation."""

    def observe_transition(
        self,
        observation: ObservationBundle,
        action: ActionPacket,
        reward: float,
        next_observation: ObservationBundle,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Optionally observe one completed transition."""
