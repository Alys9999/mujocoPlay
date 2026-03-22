from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ObservationBundle:
    """Stable public observation passed to policy adapters."""

    joint_positions: np.ndarray
    ee_position: np.ndarray
    ee_quaternion: np.ndarray
    gripper_aperture: float
    wrist_force: np.ndarray
    wrist_torque: np.ndarray
    target_position: np.ndarray
    mocap_target: np.ndarray
    previous_action_vector: np.ndarray
    time_sec: float
    step_count: int
    instruction: str
    object_family: str
    task_variant: str
    images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def image(self, camera_name: str) -> np.ndarray:
        """Return one named camera image or raise a stable error."""
        try:
            return self.images[camera_name]
        except KeyError as exc:
            raise KeyError(f"Observation does not provide camera '{camera_name}'.") from exc


@dataclass(slots=True)
class StepResult:
    """One runtime step result returned to the scheduler."""

    observation: ObservationBundle
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
