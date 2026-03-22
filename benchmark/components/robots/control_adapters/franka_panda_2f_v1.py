from __future__ import annotations

import math

import numpy as np

from benchmark.core.interfaces.control_schema import (
    CapabilityMismatchError,
    RuntimeAction,
    UnsupportedHandModeError,
)
from benchmark.schemas.models.action_packet import ActionPacket


def _quat_to_rpy(quat: list[float]) -> np.ndarray:
    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.asarray([roll, pitch, yaw], dtype=float)


class FrankaPanda2FControlAdapter:
    """Canonical control adapter for the phase1 Franka Panda two-finger runtime."""

    supported_schema_ids = ("cartesian_gripper_v1", "cartesian_hand_vector_v2")
    supported_hand_modes = ("scalar_close", "finger_vector")

    def validate(self, packet: ActionPacket) -> ActionPacket:
        if packet.schema_id not in self.supported_schema_ids:
            raise CapabilityMismatchError(f"Unknown schema_id '{packet.schema_id}'.")
        if packet.hand.mode not in self.supported_hand_modes:
            raise UnsupportedHandModeError(f"Unsupported hand mode '{packet.hand.mode}'.")
        return packet

    def to_runtime_action(self, packet: ActionPacket) -> RuntimeAction:
        packet = self.validate(packet)
        close_value: float
        if packet.hand.mode == "scalar_close":
            assert packet.hand.value is not None
            close_value = float(packet.hand.value)
        else:
            assert packet.hand.values is not None
            close_value = float(np.mean(np.asarray(packet.hand.values, dtype=float)))

        if packet.arm.mode == "delta_pose":
            rpy = packet.arm.rpy or [0.0, 0.0, 0.0]
            values = np.asarray([*packet.arm.xyz[:3], close_value], dtype=float)
            return RuntimeAction(kind="delta", values=values)

        if packet.arm.mode == "absolute_pose":
            if packet.arm.rpy is not None:
                rpy = np.asarray(packet.arm.rpy, dtype=float)
            elif packet.arm.quat is not None:
                rpy = _quat_to_rpy(packet.arm.quat)
            else:
                raise CapabilityMismatchError("absolute_pose requires `rpy` or `quat`.")
            values = np.asarray([*packet.arm.xyz[:3], *rpy.tolist(), 1.0 - close_value], dtype=float)
            return RuntimeAction(kind="absolute", values=values)

        raise CapabilityMismatchError(f"Unsupported arm mode '{packet.arm.mode}'.")
