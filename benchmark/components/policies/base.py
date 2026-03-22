from __future__ import annotations

from typing import Any

import numpy as np

from phase1.franka_env import FrankaHiddenPhysicsPickPlaceEnv

from benchmark.core.interfaces.observation import ObservationBundle
from benchmark.schemas.models.action_packet import ActionPacket


def observation_to_state_vector(
    observation: ObservationBundle,
    *,
    max_episode_steps: int = 1400,
    state_dim: int = 32,
    max_open_aperture: float = 0.04,
    force_scale: float = 20.0,
    torque_scale: float = 2.0,
) -> np.ndarray:
    """Convert the stable public observation into the 32D LeRobot state vector."""
    cfg = FrankaHiddenPhysicsPickPlaceEnv.DEFAULT_CONFIG
    family_names = ("block", "cylinder", "small_box")
    task_names = ("pick_place",)
    family_index = family_names.index(observation.object_family) / max(len(family_names) - 1, 1)
    task_index = task_names.index(observation.task_variant) / max(len(task_names) - 1, 1)

    arm_qpos = np.clip(np.asarray(observation.joint_positions, dtype=np.float32) / np.pi, -1.0, 1.0)
    gripper_open = np.asarray(
        [np.clip(2.0 * float(observation.gripper_aperture) / max(max_open_aperture, 1e-6) - 1.0, -1.0, 1.0)],
        dtype=np.float32,
    )
    wrist_force = np.clip(np.asarray(observation.wrist_force, dtype=np.float32) / force_scale, -1.0, 1.0)
    wrist_torque = np.clip(np.asarray(observation.wrist_torque, dtype=np.float32) / torque_scale, -1.0, 1.0)
    workspace_low = np.asarray([cfg.workspace_x_min, cfg.workspace_y_min, cfg.workspace_z_min], dtype=np.float32)
    workspace_high = np.asarray([cfg.workspace_x_max, cfg.workspace_y_max, cfg.workspace_z_max], dtype=np.float32)

    def _scale_to_unit_interval(value: np.ndarray) -> np.ndarray:
        scaled = 2.0 * (value - workspace_low) / np.maximum(workspace_high - workspace_low, 1e-6) - 1.0
        return np.clip(scaled, -1.0, 1.0)

    target_pos = _scale_to_unit_interval(np.asarray(observation.target_position, dtype=np.float32))
    mocap_target = _scale_to_unit_interval(np.asarray(observation.mocap_target, dtype=np.float32))
    previous_action = np.asarray(observation.previous_action_vector, dtype=np.float32).reshape(4)
    previous_action = np.concatenate(
        [
            np.clip(previous_action[:3], -1.0, 1.0),
            np.asarray([2.0 * previous_action[3] - 1.0], dtype=np.float32),
        ]
    )
    time_frac = np.clip(
        float(observation.time_sec) / max(float(max_episode_steps) * cfg.timestep * cfg.control_substeps, 1e-6),
        0.0,
        1.0,
    )
    step_frac = np.clip(float(observation.step_count) / max(float(max_episode_steps), 1.0), 0.0, 1.0)

    parts = [
        arm_qpos,
        gripper_open,
        wrist_force,
        wrist_torque,
        target_pos,
        mocap_target,
        previous_action,
        np.asarray([2.0 * time_frac - 1.0], dtype=np.float32),
        np.asarray([2.0 * step_frac - 1.0], dtype=np.float32),
        np.asarray([2.0 * family_index - 1.0], dtype=np.float32),
        np.asarray([2.0 * task_index - 1.0], dtype=np.float32),
    ]
    state = np.concatenate(parts, dtype=np.float32)
    if state.shape[0] > state_dim:
        return state[:state_dim]
    if state.shape[0] < state_dim:
        padded = np.zeros(state_dim, dtype=np.float32)
        padded[: state.shape[0]] = state
        return padded
    return state


def delta_action_to_packet(action: np.ndarray, *, source: str) -> ActionPacket:
    """Wrap one legacy 4D delta action into the canonical ActionPacket."""
    action = np.asarray(action, dtype=np.float32).reshape(4)
    return ActionPacket(
        schema_id="cartesian_gripper_v1",
        arm={"mode": "delta_pose", "xyz": action[:3].tolist(), "rpy": [0.0, 0.0, 0.0]},
        hand={"mode": "scalar_close", "value": float(np.clip(action[3], 0.0, 1.0))},
        metadata={"source": source},
    )


class RandomPolicyAdapter:
    """Random smoke-test policy that already speaks ActionPacket."""

    name = "random"
    requires_image = False
    required_cameras: tuple[str, ...] = ()
    supported_schema_ids = ("cartesian_gripper_v1",)

    def __init__(self, seed: int = 0, action_scale: float = 1.0, close_bias: float = 0.5) -> None:
        self._seed = int(seed)
        self._action_scale = float(action_scale)
        self._close_bias = float(close_bias)
        self._rng = np.random.default_rng(self._seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def act(self, observation: ObservationBundle) -> ActionPacket:
        del observation
        action = self._rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
        action[:3] *= self._action_scale
        action[3] = float(np.clip(self._close_bias + 0.35 * self._rng.standard_normal(), 0.0, 1.0))
        return delta_action_to_packet(action, source="random")

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
        del observation, action, reward, next_observation, terminated, truncated, info
