from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from benchmark.schemas.models.action_packet import ActionPacket


class CapabilityMismatchError(ValueError):
    """Raised when a policy schema cannot be consumed by the selected robot profile."""


class UnsupportedHandModeError(ValueError):
    """Raised when the hand sub-command cannot be executed by the robot profile."""


@dataclass(frozen=True, slots=True)
class RuntimeAction:
    """Robot-runtime action after schema translation."""

    kind: str
    values: np.ndarray


class ControlSchemaAdapter(Protocol):
    """Translate canonical action packets into runtime-native actions."""

    supported_schema_ids: tuple[str, ...]

    def validate(self, packet: ActionPacket) -> ActionPacket:
        """Validate one action packet against adapter capabilities."""

    def to_runtime_action(self, packet: ActionPacket) -> RuntimeAction:
        """Translate one action packet into a runtime action."""
