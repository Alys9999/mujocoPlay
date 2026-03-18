from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .mujoco_runtime import configure_mujoco_gl

configure_mujoco_gl()

import mujoco
import numpy as np

from .benchmark_spec import get_object_family_spec, get_task_variant_spec
from .config import Phase1Config


class FrankaHiddenPhysicsPickPlaceEnv:
    """Phase 1 environment using the official MuJoCo Menagerie Panda model.
    """

    ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
    ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
    HOME_ARM_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float)
    ASSET_FILES = {
        "block": "phase1_franka_scene.xml",
        "cylinder": "phase1_franka_scene_cylinder.xml",
        "small_box": "phase1_franka_scene_small_box.xml",
    }
    DEFAULT_CONFIG = replace(
        Phase1Config(),
        control_substeps=28,
        max_episode_steps=1400,
        home_x=0.32,
        home_y=0.0,
        home_z=0.48,
        workspace_x_min=0.22,
        workspace_x_max=0.72,
        workspace_y_min=-0.35,
        workspace_y_max=0.35,
        workspace_z_min=0.015,
        workspace_z_max=0.62,
        default_object_x=0.52,
        default_object_y=-0.10,
        default_object_z=0.015,
        target_x=0.62,
        target_y=0.12,
        target_random_x_min=0.46,
        target_random_x_max=0.68,
        target_random_y_min=-0.18,
        target_random_y_max=0.18,
        target_random_min_distance=0.12,
        action_delta_limit=0.01,
        hold_steps=40,
    )
    OBJECT_TARGET_OFFSETS = {
        "block": (-0.025, 0.0, 0.025),
        "cylinder": (-0.022, 0.0, 0.028),
        "small_box": (-0.022, 0.0, 0.024),
    }
    MIN_CLOSE_FRACTION = {
        "block": 0.94,
        "cylinder": 0.98,
        "small_box": 0.96,
    }

    @staticmethod
    def _euler_xyz_to_matrix(angles: np.ndarray) -> np.ndarray:
        """Convert XYZ Euler angles into a rotation matrix.

        Args:
            angles: XYZ Euler angles in radians.
        """
        roll, pitch, yaw = np.asarray(angles, dtype=float).reshape(3)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return rz @ ry @ rx

    def __init__(
        self,
        config: Phase1Config | None = None,
        object_family: str = "block",
        task_variant: str = "standard",
    ) -> None:
        """Initialize FrankaHiddenPhysicsPickPlaceEnv.
        
        Args:
            config: Environment configuration to use; defaults are applied when None.
            object_family: Object family to simulate.
            task_variant: Task variant to simulate.
        """
        self.config = config or self.DEFAULT_CONFIG
        self.object_family = object_family
        self.task_variant = task_variant
        self.family_spec = get_object_family_spec(object_family)
        self.task_spec = get_task_variant_spec(task_variant)
        asset_path = Path(__file__).resolve().parent / "assets" / self.ASSET_FILES[object_family]
        self.model = mujoco.MjModel.from_xml_path(str(asset_path))
        self.model.opt.timestep = self.config.timestep
        self.data = mujoco.MjData(self.model)

        self._rng = np.random.default_rng()
        self.step_count = 0
        self._episode_started = False
        self._step_callback = None

        self.object_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.object_joint_qposadr = self.model.jnt_qposadr[self.object_joint_id]
        self.object_joint_dofadr = self.model.jnt_dofadr[self.object_joint_id]

        self.mocap_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
        self.hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_finger_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.mocap_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "mocap_site")
        self.object_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "object_site")
        self.wrist_force_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "wrist_force_site")
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.wrist_force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_force")
        self.wrist_torque_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_torque")
        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
        self._hide_auxiliary_sites_for_render()

        self.arm_joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.ARM_JOINTS],
            dtype=int,
        )
        self.arm_joint_qposadr = np.array([self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids], dtype=int)
        self.arm_joint_dofadr = np.array([self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids], dtype=int)
        self.arm_act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.ARM_ACTUATORS],
            dtype=int,
        )
        self.arm_lower = self.model.jnt_range[self.arm_joint_ids, 0].copy()
        self.arm_upper = self.model.jnt_range[self.arm_joint_ids, 1].copy()

        self.finger_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
            ],
            dtype=int,
        )
        self.finger_qposadr = np.array([self.model.jnt_qposadr[jid] for jid in self.finger_joint_ids], dtype=int)
        self.gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.finger_open_limit = float(self.model.jnt_range[self.finger_joint_ids[0], 1])
        self.base_object_mass = float(self.model.body_mass[self.object_bid])
        self.base_object_inertia = np.array(self.model.body_inertia[self.object_bid], copy=True)
        self.base_object_friction = np.array(self.model.geom_friction[self.object_geom_id], copy=True)
        self.base_object_rgba = np.array(self.model.geom_rgba[self.object_geom_id], copy=True)
        self.mass_values = np.array(self.config.object_masses, dtype=float)
        self.friction_values = np.array(self.config.object_frictions, dtype=float)
        self.stiffness_values = np.array(self.config.object_stiffnesses, dtype=float)

        self.variant: dict[str, Any] | None = None
        self._max_object_z = self.family_spec.rest_height
        self._ee_target_mat = np.eye(3, dtype=float)
        self._home_target = np.array([self.config.home_x, self.config.home_y, self.config.home_z], dtype=float)
        self._object_target_offset = np.array(
            self.OBJECT_TARGET_OFFSETS.get(object_family, (-0.022, 0.0, 0.025)),
            dtype=float,
        )
        self._last_close_fraction = 0.0
        self._peak_close_fraction = 0.0
        self._break_close_fraction = 1.0
        self._object_broken = False
        self._target_xy = np.array([self.config.target_x, self.config.target_y], dtype=float)

    def _hide_auxiliary_sites_for_render(self) -> None:
        """Hide non-task site markers from rendered images while keeping site kinematics."""
        hidden_site_ids = (
            self.mocap_site_id,
            self.object_site_id,
            self.wrist_force_site_id,
            self.ee_site_id,
        )
        for site_id in hidden_site_ids:
            if int(site_id) < 0:
                continue
            self.model.site_rgba[site_id, 3] = 0.0

    def seed(self, seed: int | None = None) -> None:
        """Set the random generator seed.
        
        Args:
            seed: Random seed for reproducible sampling or resets.
        """
        self._rng = np.random.default_rng(seed)

    def set_step_callback(self, callback) -> None:
        """Register a callback to run after each environment step.
        
        Args:
            callback: Value for the callback used by this routine.
        """
        self._step_callback = callback

    def reset(
        self,
        seed: int | None = None,
        variant: dict[str, float] | None = None,
        target_xy: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the object to a fresh episode state.
        
        Args:
            seed: Random seed for reproducible sampling or resets.
            variant: Hidden-physics variant dictionary for the episode.
            target_xy: Optional XY target override for the episode.
        """
        if seed is not None:
            self.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self._episode_started = True
        self._max_object_z = self.family_spec.rest_height
        self._last_close_fraction = 0.0
        self._peak_close_fraction = 0.0
        self._object_broken = False

        self.data.qpos[self.arm_joint_qposadr] = self.HOME_ARM_QPOS
        self.data.ctrl[self.arm_act_ids] = self.HOME_ARM_QPOS
        self.data.qpos[self.finger_qposadr] = self.finger_open_limit
        self.data.ctrl[self.gripper_act_id] = 255.0

        self.variant = self._sample_variant() if variant is None else variant
        self._apply_variant(self.variant)
        self._set_broken_state(False)
        self._set_target_xy(target_xy)
        self._set_object_pose(
            np.array(
                [
                    self.config.default_object_x,
                    self.config.default_object_y,
                    self.family_spec.rest_height,
                ],
                dtype=float,
            )
        )
        self._update_target_zone()

        mujoco.mj_forward(self.model, self.data)
        ee_home = np.array(self.data.site_xpos[self.ee_site_id], copy=True)
        self._home_target = ee_home.copy()
        self._ee_target_mat = np.array(self.data.site_xmat[self.ee_site_id].reshape(3, 3), copy=True)
        self._set_mocap_target(ee_home)

        for _ in range(self.config.settle_steps):
            self._apply_arm_tracking()
            mujoco.mj_step(self.model, self.data)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance the environment by one control step.
        
        Args:
            action: Normalized action vector to apply for this control step.
        """
        if not self._episode_started:
            raise RuntimeError("Call reset() before step().")
        clipped = np.asarray(action, dtype=float).copy()
        delta = np.clip(clipped[:3], -1.0, 1.0) * self.config.action_delta_limit
        target_pos = self.data.mocap_pos[0] + delta
        closure_fraction = float(np.clip(clipped[3], 0.0, 1.0))
        return self._step_to_target(target_pos=target_pos, closure_fraction=closure_fraction)

    def step_cartesian(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance the environment using an absolute Cartesian target action.

        Args:
            action: Absolute end-effector target action in `x, y, z, roll, pitch, yaw, gripper_open`.
        """
        if not self._episode_started:
            raise RuntimeError("Call reset() before step_cartesian().")
        clipped = np.asarray(action, dtype=float).copy().reshape(7)
        target_pos = clipped[:3]
        target_pos[0] = np.clip(target_pos[0], self.config.workspace_x_min, self.config.workspace_x_max)
        target_pos[1] = np.clip(target_pos[1], self.config.workspace_y_min, self.config.workspace_y_max)
        target_pos[2] = np.clip(target_pos[2], self.config.workspace_z_min, self.config.workspace_z_max)
        self._ee_target_mat = self._euler_xyz_to_matrix(clipped[3:6])
        gripper_open = float(np.clip(clipped[6], 0.0, 1.0))
        closure_fraction = 1.0 - gripper_open
        return self._step_to_target(target_pos=target_pos, closure_fraction=closure_fraction)

    def _step_to_target(
        self,
        target_pos: np.ndarray,
        closure_fraction: float,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Run one control step toward a target pose with a gripper command.

        Args:
            target_pos: Absolute mocap target position for this control step.
            closure_fraction: Normalized close command.
        """
        self._set_mocap_target(np.asarray(target_pos, dtype=float))
        self._last_close_fraction = float(np.clip(closure_fraction, 0.0, 1.0))
        self._peak_close_fraction = max(self._peak_close_fraction, self._last_close_fraction)
        for _ in range(self.config.control_substeps):
            self._apply_arm_tracking()
            self._set_aperture(self._last_close_fraction)
            mujoco.mj_step(self.model, self.data)
            self._max_object_z = max(self._max_object_z, float(self.data.xpos[self.object_bid][2]))
            self._update_breakage()

        self.step_count += 1
        obs = self._get_obs()
        info = self._get_info()
        success = bool(info["success"])
        terminated = success
        truncated = self.step_count >= self.config.max_episode_steps
        reward = self._compute_reward(info)
        if self._step_callback is not None:
            self._step_callback(self, obs, info)
        return obs, reward, terminated, truncated, info

    def move_to(
        self,
        target_pos: np.ndarray,
        aperture_fraction: float,
        max_steps: int = 160,
        step_fraction: float = 1.0,
    ) -> dict[str, Any]:
        """Move the end effector toward a target position.
        
        Args:
            target_pos: Target position to move the end effector toward.
            aperture_fraction: Normalized gripper command to hold or apply while moving.
            max_steps: Number of steps to use for max.
            step_fraction: Normalized fraction used for step.
        """
        target = np.asarray(target_pos, dtype=float)
        scaled_fraction = float(np.clip(step_fraction, 0.1, 1.0))
        for _ in range(max_steps):
            mocap_delta = target - self.data.mocap_pos[0]
            ee_delta = target - np.array(self.data.site_xpos[self.ee_site_id], copy=False)
            if (
                np.linalg.norm(mocap_delta) < self.config.move_tolerance * 1.5
                and np.linalg.norm(ee_delta) < self.config.move_tolerance * 2.0
            ):
                break
            action = np.zeros(4, dtype=float)
            action[:3] = scaled_fraction * np.clip(
                mocap_delta / max(self.config.action_delta_limit, 1e-6),
                -1.0,
                1.0,
            )
            action[3] = aperture_fraction
            _, _, terminated, truncated, info = self.step(action)
            if terminated or truncated:
                return info
        return self._get_info()

    def hold(self, aperture_fraction: float, steps: int | None = None) -> dict[str, Any]:
        """Hold the current pose for a number of control steps.
        
        Args:
            aperture_fraction: Normalized gripper command to hold or apply while moving.
            steps: Value for the steps used by this routine.
        """
        info = self._get_info()
        for _ in range(steps or self.config.hold_steps):
            _, _, terminated, truncated, info = self.step(
                np.array([0.0, 0.0, 0.0, aperture_fraction], dtype=float)
            )
            if terminated or truncated:
                break
        return info

    def execute_pick_and_place(self, close_fraction: float = 0.8) -> dict[str, Any]:
        """Execute a legacy scripted pick-and-place helper.
        
        Args:
            close_fraction: Normalized gripper closure command for the legacy helper.
        """
        object_pos = self.object_position
        close_fraction = max(close_fraction, self.MIN_CLOSE_FRACTION.get(self.object_family, 0.94))
        lift_fraction = 0.12 if self.object_family == "block" else 0.08
        transport_fraction = 0.06 if self.object_family == "block" else 0.04
        place_fraction = 0.05 if self.object_family == "block" else 0.035
        self.move_to(self._object_target(object_pos, self.family_spec.approach_height), 0.0)
        self.move_to(self._object_target(object_pos, self.family_spec.grasp_height), 0.0)
        self.hold(close_fraction, steps=max(self.task_spec.grasp_hold_steps, 40))
        self.move_to(
            self._absolute_target(object_pos[0], object_pos[1], self.task_spec.lift_height),
            close_fraction,
            max_steps=360,
            step_fraction=lift_fraction,
        )
        self.hold(close_fraction, steps=10)
        x_waypoints = np.linspace(object_pos[0], self._target_xy[0], 4, dtype=float)[1:]
        for x_target in x_waypoints:
            self.move_to(
                self._absolute_target(float(x_target), object_pos[1], self.task_spec.transport_height + 0.02),
                close_fraction,
                max_steps=220,
                step_fraction=transport_fraction,
            )
            self.hold(close_fraction, steps=8)
        y_waypoints = np.linspace(object_pos[1], self._target_xy[1], 4, dtype=float)[1:]
        for y_target in y_waypoints:
            self.move_to(
                self._absolute_target(self._target_xy[0], float(y_target), self.task_spec.transport_height + 0.02),
                close_fraction,
                max_steps=220,
                step_fraction=transport_fraction,
            )
            self.hold(close_fraction, steps=8)
        self.move_to(
            self._absolute_target(self._target_xy[0], self._target_xy[1], self.task_spec.place_height + 0.01),
            close_fraction,
            max_steps=260,
            step_fraction=place_fraction,
        )
        self.hold(close_fraction, steps=max(self.task_spec.settle_hold_steps, 10))
        self.hold(0.0, steps=self.task_spec.release_hold_steps)
        self.move_to(self._home_target + np.array([0.0, 0.0, 0.03]), 0.0)
        return self._get_info()

    @property
    def object_position(self) -> np.ndarray:
        """Run object position.
        """
        return np.array(self.data.site_xpos[self.object_site_id], copy=True)

    def _sample_variant(self) -> dict[str, Any]:
        """Run sample variant.
        """
        mass_idx = int(self._rng.integers(len(self.mass_values)))
        friction_idx = int(self._rng.integers(len(self.friction_values)))
        stiffness_idx = int(self._rng.integers(len(self.stiffness_values)))
        return {
            "mass": float(self.mass_values[mass_idx]),
            "mass_idx": mass_idx,
            "friction": float(self.friction_values[friction_idx]),
            "friction_idx": friction_idx,
            "stiffness": float(self.stiffness_values[stiffness_idx]),
            "stiffness_idx": stiffness_idx,
        }

    def _apply_variant(self, variant: dict[str, Any]) -> None:
        """Run apply variant.
        
        Args:
            variant: Hidden-physics variant dictionary for the episode.
        """
        mass = float(variant["mass"])
        friction = float(variant["friction"])
        scale = mass / self.base_object_mass
        self.model.body_mass[self.object_bid] = mass
        self.model.body_inertia[self.object_bid] = self.base_object_inertia * scale
        self.model.geom_friction[self.object_geom_id] = np.array(
            [friction, self.base_object_friction[1], self.base_object_friction[2]],
            dtype=float,
        )
        self._break_close_fraction = self._safe_close_fraction_from_variant(variant)
        mujoco.mj_forward(self.model, self.data)

    def _set_mocap_target(self, pos: np.ndarray) -> None:
        """Set mocap target.
        
        Args:
            pos: Value for the pos used by this routine.
        """
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _set_aperture(self, closure_fraction: float) -> None:
        """Set aperture.
        
        Args:
            closure_fraction: Normalized fraction used for closure.
        """
        open_fraction = float(np.clip(1.0 - closure_fraction, 0.0, 1.0))
        self.data.ctrl[self.gripper_act_id] = 255.0 * open_fraction

    def _set_object_pose(self, pos: np.ndarray) -> None:
        """Set object pose.
        
        Args:
            pos: Value for the pos used by this routine.
        """
        adr = self.object_joint_qposadr
        dof_adr = self.object_joint_dofadr
        clamped = np.array(pos, copy=True)
        clamped[2] = max(clamped[2], self.family_spec.rest_height)
        self.data.qpos[adr : adr + 3] = clamped
        self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.qvel[dof_adr : dof_adr + 6] = 0.0

    def _update_target_zone(self) -> None:
        """Run update target zone.
        """
        self.model.geom_pos[self.target_geom_id] = np.array(
            [self._target_xy[0], self._target_xy[1], 0.001],
            dtype=float,
        )
        self.model.geom_size[self.target_geom_id] = np.array(
            [self.task_spec.target_radius, 0.001, 0.0],
            dtype=float,
        )
        mujoco.mj_forward(self.model, self.data)

    def _set_target_xy(self, target_xy: np.ndarray | None) -> None:
        """Set the current episode target location.

        Args:
            target_xy: Optional XY target override for the current episode.
        """
        if target_xy is None:
            self._target_xy = np.array([self.config.target_x, self.config.target_y], dtype=float)
            return
        target = np.asarray(target_xy, dtype=float).reshape(2)
        target[0] = np.clip(target[0], self.config.workspace_x_min, self.config.workspace_x_max)
        target[1] = np.clip(target[1], self.config.workspace_y_min, self.config.workspace_y_max)
        self._target_xy = target

    def _safe_close_fraction_from_variant(self, variant: dict[str, Any]) -> float:
        """Run safe close fraction from variant.
        
        Args:
            variant: Hidden-physics variant dictionary for the episode.
        """
        stiffness = float(variant["stiffness"])
        stiffness_norm = np.clip(
            (stiffness - float(self.stiffness_values.min())) / max(float(np.ptp(self.stiffness_values)), 1e-6),
            0.0,
            1.0,
        )
        family_bias = {"block": 0.00, "cylinder": 0.02, "small_box": -0.02}.get(self.object_family, 0.0)
        return float(np.clip(0.86 + 0.10 * stiffness_norm + family_bias, 0.82, 0.99))

    def _set_broken_state(self, broken: bool) -> None:
        """Set broken state.
        
        Args:
            broken: Value for the broken used by this routine.
        """
        self._object_broken = broken
        if broken:
            self.model.geom_rgba[self.object_geom_id] = np.array([0.18, 0.18, 0.18, 1.0], dtype=float)
            self.model.geom_friction[self.object_geom_id] = np.array(
                [0.05, self.base_object_friction[1], self.base_object_friction[2]],
                dtype=float,
            )
        else:
            self.model.geom_rgba[self.object_geom_id] = self.base_object_rgba
            friction = self.base_object_friction if self.variant is None else np.array(
                [float(self.variant["friction"]), self.base_object_friction[1], self.base_object_friction[2]],
                dtype=float,
            )
            self.model.geom_friction[self.object_geom_id] = friction
        mujoco.mj_forward(self.model, self.data)

    def _update_breakage(self) -> None:
        """Run update breakage.
        """
        if self._object_broken:
            return
        left_contact, right_contact = self._finger_contact_state()
        if not (left_contact and right_contact):
            return
        if self._last_close_fraction <= self._break_close_fraction:
            return
        self._set_broken_state(True)

    def _object_target(self, object_pos: np.ndarray, relative_height: float) -> np.ndarray:
        """Run object target.
        
        Args:
            object_pos: Value for the object pos used by this routine.
            relative_height: Height used for relative.
        """
        target = np.array(object_pos, copy=True)
        target[:2] = object_pos[:2] + self._object_target_offset[:2]
        target[2] = object_pos[2] + relative_height - self._object_target_offset[2]
        return target

    def _absolute_target(self, x: float, y: float, z: float) -> np.ndarray:
        """Run absolute target.
        
        Args:
            x: Value for the x used by this routine.
            y: Value for the y used by this routine.
            z: Value for the z used by this routine.
        """
        target = np.array([x, y, z], dtype=float)
        target[:2] += self._object_target_offset[:2]
        target[2] -= self._object_target_offset[2]
        return target

    def _apply_arm_tracking(self) -> None:
        """Run apply arm tracking.
        """
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        j = np.vstack([jacp[:, self.arm_joint_dofadr], jacr[:, self.arm_joint_dofadr]])

        ee_pos = np.array(self.data.site_xpos[self.ee_site_id], copy=True)
        ee_mat = np.array(self.data.site_xmat[self.ee_site_id].reshape(3, 3), copy=True)
        pos_err = self.data.mocap_pos[0] - ee_pos
        rot_err = 0.5 * (
            np.cross(ee_mat[:, 0], self._ee_target_mat[:, 0])
            + np.cross(ee_mat[:, 1], self._ee_target_mat[:, 1])
            + np.cross(ee_mat[:, 2], self._ee_target_mat[:, 2])
        )

        error = np.concatenate([6.0 * pos_err, 0.05 * rot_err])
        damping = 1e-3 * np.eye(6, dtype=float)
        dq = j.T @ np.linalg.solve(j @ j.T + damping, error)
        current_qpos = np.array(self.data.qpos[self.arm_joint_qposadr], copy=True)
        target_qpos = current_qpos + 0.18 * dq
        target_qpos = np.clip(target_qpos, self.arm_lower, self.arm_upper)
        self.data.ctrl[self.arm_act_ids] = target_qpos

    def _finger_contact_state(self) -> tuple[bool, bool]:
        """Run finger contact state.
        """
        left_contact = False
        right_contact = False
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            pair = {body1, body2}
            if pair == {self.left_finger_bid, self.object_bid}:
                left_contact = True
            if pair == {self.right_finger_bid, self.object_bid}:
                right_contact = True
        return left_contact, right_contact

    def _compute_reward(self, info: dict[str, Any]) -> float:
        """Run compute reward.
        
        Args:
            info: Value for the info used by this routine.
        """
        reward = 0.0
        if info["success"]:
            reward += 5.0
        if info["object_lifted"]:
            reward += 0.5
        if info["object_in_gripper"]:
            reward += 0.5
        if info.get("object_broken", False):
            reward -= 3.0
        reward -= 0.002 * self.step_count
        return reward

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Return obs.
        """
        force_adr = self.model.sensor_adr[self.wrist_force_sensor_id]
        torque_adr = self.model.sensor_adr[self.wrist_torque_sensor_id]
        object_adr = self.object_joint_qposadr
        object_dof_adr = self.object_joint_dofadr
        finger_open = float(np.mean(self.data.qpos[self.finger_qposadr]))
        ee_quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(ee_quat, self.data.site_xmat[self.ee_site_id].reshape(-1))
        return {
            "gripper_pos": np.array(self.data.site_xpos[self.ee_site_id], copy=True),
            "ee_quat": ee_quat,
            "gripper_aperture": np.array([finger_open], dtype=float),
            "arm_qpos": np.array(self.data.qpos[self.arm_joint_qposadr], copy=True),
            "object_pos": np.array(self.data.qpos[object_adr : object_adr + 3], copy=True),
            "object_quat": np.array(self.data.qpos[object_adr + 3 : object_adr + 7], copy=True),
            "object_linvel": np.array(self.data.qvel[object_dof_adr : object_dof_adr + 3], copy=True),
            "object_angvel": np.array(self.data.qvel[object_dof_adr + 3 : object_dof_adr + 6], copy=True),
            "wrist_force": np.array(self.data.sensordata[force_adr : force_adr + 3], copy=True),
            "wrist_torque": np.array(self.data.sensordata[torque_adr : torque_adr + 3], copy=True),
            "target_pos": np.array(self.model.geom_pos[self.target_geom_id], copy=True),
            "mocap_target": np.array(self.data.mocap_pos[0], copy=True),
        }

    def _get_info(self) -> dict[str, Any]:
        """Return info.
        """
        object_pos = np.array(self.data.xpos[self.object_bid], copy=True)
        gripper_pos = np.array(self.data.site_xpos[self.ee_site_id], copy=True)
        gripper_object_distance = float(np.linalg.norm(object_pos - gripper_pos))
        left_contact, right_contact = self._finger_contact_state()
        dist_to_target = float(
            np.linalg.norm(object_pos[:2] - self._target_xy)
        )
        lifted = bool(object_pos[2] > self.family_spec.rest_height + 0.03)
        ever_lifted = bool(self._max_object_z > self.family_spec.rest_height + 0.03)
        in_gripper = bool(
            (left_contact and right_contact)
            or (
                np.linalg.norm(object_pos - gripper_pos) < 0.07
                and object_pos[2] > self.family_spec.rest_height + 0.015
            )
        )
        target_reached = bool(dist_to_target < self.task_spec.target_radius)
        placed = bool(target_reached and object_pos[2] < self.family_spec.rest_height + 0.03)
        slipped = bool(ever_lifted and not placed and not in_gripper)
        dropped = bool(slipped and object_pos[2] < self.family_spec.rest_height + 0.020)
        task_mode = self.task_spec.task_mode
        success = bool(placed and ever_lifted and not self._object_broken)
        return {
            "step_count": self.step_count,
            "variant": dict(self.variant or {}),
            "object_family": self.object_family,
            "task_variant": self.task_variant,
            "task_mode": task_mode,
            "object_pos": object_pos,
            "gripper_pos": gripper_pos,
            "gripper_object_distance": gripper_object_distance,
            "distance_to_target": dist_to_target,
            "placement_error": dist_to_target,
            "target_reached": target_reached,
            "object_lifted": lifted,
            "ever_lifted": ever_lifted,
            "object_in_gripper": in_gripper,
            "object_broken": bool(self._object_broken),
            "break_threshold": float(self._break_close_fraction),
            "commanded_close_fraction": float(self._peak_close_fraction),
            "object_dropped": dropped,
            "slip_detected": slipped,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "success": success,
            "episode_duration_sec": self.step_count * self.config.control_substeps * self.config.timestep,
            "config": asdict(self.config),
        }
