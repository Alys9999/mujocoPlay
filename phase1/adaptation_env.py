from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .config import Phase1Config
from .franka_env import FrankaHiddenPhysicsPickPlaceEnv
from .splits import sample_hidden_context


class FrankaLatentAdaptationEnv(FrankaHiddenPhysicsPickPlaceEnv):
    """Benchmark environment with hidden robot-body and object-environment physics.

    Args:
        config: Environment configuration to use; defaults are applied when None.
        object_family: Object family to simulate.
        task_variant: Task variant to simulate.
    """

    REACH_BODY_NAMES = (
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "hand",
        "left_finger",
        "right_finger",
    )

    MASS_BODY_NAMES = (
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "hand",
        "left_finger",
        "right_finger",
    )

    def __init__(
        self,
        config: Phase1Config | None = None,
        object_family: str = "block",
        task_variant: str = "standard",
    ) -> None:
        """Initialize FrankaLatentAdaptationEnv.

        Args:
            config: Environment configuration to use; defaults are applied when None.
            object_family: Object family to simulate.
            task_variant: Task variant to simulate.
        """
        super().__init__(config=config, object_family=object_family, task_variant=task_variant)
        self._hidden_body_context: dict[str, Any] | None = None
        self._hidden_env_context: dict[str, Any] | None = None

        self._reach_body_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.REACH_BODY_NAMES],
            dtype=int,
        )
        self._mass_body_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.MASS_BODY_NAMES],
            dtype=int,
        )
        self._base_arm_body_pos = np.array(self.model.body_pos[self._reach_body_ids], copy=True)
        self._base_arm_body_mass = np.array(self.model.body_mass[self._mass_body_ids], copy=True)
        self._base_arm_body_inertia = np.array(self.model.body_inertia[self._mass_body_ids], copy=True)
        self._payload_body_ids = np.array([self.hand_bid, self.left_finger_bid, self.right_finger_bid], dtype=int)
        self._base_payload_body_mass = np.array(self.model.body_mass[self._payload_body_ids], copy=True)
        self._base_payload_body_inertia = np.array(self.model.body_inertia[self._payload_body_ids], copy=True)
        self._base_arm_damping = np.array(self.model.dof_damping[self.arm_joint_dofadr], copy=True)
        self._base_arm_actuator_gain = np.array(self.model.actuator_gainprm[self.arm_act_ids], copy=True)
        self._base_arm_actuator_bias = np.array(self.model.actuator_biasprm[self.arm_act_ids], copy=True)
        self._base_arm_force_range = np.array(self.model.actuator_forcerange[self.arm_act_ids], copy=True)
        self._base_object_target_offset = np.array(self._object_target_offset, copy=True)

        finger_geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in {self.left_finger_bid, self.right_finger_bid}
        ]
        self._finger_geom_ids = np.array(finger_geom_ids, dtype=int)
        self._left_finger_geom_mask = np.array(
            [int(self.model.geom_bodyid[geom_id]) == self.left_finger_bid for geom_id in self._finger_geom_ids],
            dtype=bool,
        )
        self._right_finger_geom_mask = np.array(
            [int(self.model.geom_bodyid[geom_id]) == self.right_finger_bid for geom_id in self._finger_geom_ids],
            dtype=bool,
        )
        self._base_finger_geom_friction = np.array(self.model.geom_friction[self._finger_geom_ids], copy=True)

    def reset(
        self,
        seed: int | None = None,
        hidden_context: dict[str, dict[str, Any]] | None = None,
        target_xy: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment with a freshly sampled hidden body and env context.

        Args:
            seed: Random seed for reproducible sampling or resets.
            hidden_context: Optional hidden context override with `body` and `env` sections.
            target_xy: Optional XY target override for the episode.
        """
        if seed is not None:
            self.seed(seed)

        context = hidden_context or self._sample_hidden_context()
        self._hidden_body_context = dict(context.get("body", {}))
        self._hidden_env_context = dict(context.get("env", {}))

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self._episode_started = True
        self._max_object_z = self.family_spec.rest_height
        self._last_close_fraction = 0.0
        self._peak_close_fraction = 0.0
        self._object_broken = False

        self._apply_hidden_body_context(self._hidden_body_context)
        self._apply_hidden_env_context(self._hidden_env_context)
        self._set_broken_state(False)
        self._set_target_xy(target_xy)

        self.data.qpos[self.arm_joint_qposadr] = self.HOME_ARM_QPOS
        self.data.ctrl[self.arm_act_ids] = self.HOME_ARM_QPOS
        self.data.qpos[self.finger_qposadr] = self.finger_open_limit
        self.data.ctrl[self.gripper_act_id] = 255.0
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

    def debug_hidden_context(self) -> dict[str, dict[str, Any]]:
        """Return the internal hidden context for offline analysis only."""
        return {
            "body": dict(self._hidden_body_context or {}),
            "env": dict(self._hidden_env_context or {}),
        }

    def _sample_hidden_context(self) -> dict[str, dict[str, Any]]:
        """Sample a hidden context for the robot body and external environment."""
        return sample_hidden_context(split_name="full", rng=self._rng)

    def _sample_body_context(self) -> dict[str, Any]:
        """Sample hidden robot-body properties."""
        return sample_hidden_context(split_name="full", rng=self._rng)["body"]

    def _sample_env_context(self) -> dict[str, Any]:
        """Sample hidden object-side physical properties."""
        return sample_hidden_context(split_name="full", rng=self._rng)["env"]

    def _apply_hidden_body_context(self, context: dict[str, Any]) -> None:
        """Apply hidden robot-body properties to the MuJoCo model.

        Args:
            context: Hidden body context sampled for the current episode.
        """
        reach_scale = float(context.get("reach_scale", 1.0))
        arm_mass_scale = float(context.get("arm_mass_scale", 1.0))
        payload_scale = float(context.get("payload_scale", 1.0))
        joint_damping_scale = float(context.get("joint_damping_scale", 1.0))
        actuator_gain_scale = float(context.get("actuator_gain_scale", 1.0))
        fingertip_friction_scale = float(context.get("fingertip_friction_scale", 1.0))
        damage_joint_index = int(context.get("damage_joint_index", -1))
        damage_gain_scale = float(context.get("damage_gain_scale", 1.0))
        damage_damping_scale = float(context.get("damage_damping_scale", 1.0))
        local_finger_wear_side = str(context.get("local_finger_wear_side", "none"))
        local_finger_friction_scale = float(context.get("local_finger_friction_scale", 1.0))

        self.model.body_pos[self._reach_body_ids] = self._base_arm_body_pos * reach_scale
        self.model.body_mass[self._mass_body_ids] = self._base_arm_body_mass * arm_mass_scale
        self.model.body_inertia[self._mass_body_ids] = self._base_arm_body_inertia * arm_mass_scale
        self.model.body_mass[self._payload_body_ids] = self._base_payload_body_mass * arm_mass_scale * payload_scale
        self.model.body_inertia[self._payload_body_ids] = (
            self._base_payload_body_inertia * arm_mass_scale * payload_scale
        )

        arm_damping = self._base_arm_damping * joint_damping_scale
        arm_gain = self._base_arm_actuator_gain * actuator_gain_scale
        arm_bias = self._base_arm_actuator_bias * actuator_gain_scale
        arm_force = self._base_arm_force_range * actuator_gain_scale
        if 0 <= damage_joint_index < len(self.arm_joint_dofadr):
            arm_damping[damage_joint_index] *= damage_damping_scale
            arm_gain[damage_joint_index] *= damage_gain_scale
            arm_bias[damage_joint_index] *= damage_gain_scale
            arm_force[damage_joint_index] *= damage_gain_scale
        self.model.dof_damping[self.arm_joint_dofadr] = arm_damping
        self.model.actuator_gainprm[self.arm_act_ids] = arm_gain
        self.model.actuator_biasprm[self.arm_act_ids] = arm_bias
        self.model.actuator_forcerange[self.arm_act_ids] = arm_force

        finger_friction = np.array(self._base_finger_geom_friction, copy=True)
        finger_friction[:, 0] *= fingertip_friction_scale
        finger_friction[:, 1] *= fingertip_friction_scale
        if local_finger_wear_side == "left":
            finger_friction[self._left_finger_geom_mask, 0] *= local_finger_friction_scale
            finger_friction[self._left_finger_geom_mask, 1] *= local_finger_friction_scale
        elif local_finger_wear_side == "right":
            finger_friction[self._right_finger_geom_mask, 0] *= local_finger_friction_scale
            finger_friction[self._right_finger_geom_mask, 1] *= local_finger_friction_scale
        self.model.geom_friction[self._finger_geom_ids] = finger_friction
        self._object_target_offset = self._base_object_target_offset * reach_scale

        mujoco.mj_forward(self.model, self.data)

    def _apply_hidden_env_context(self, context: dict[str, Any]) -> None:
        """Apply hidden object-side physics to the MuJoCo model.

        Args:
            context: Hidden environment context sampled for the current episode.
        """
        mass = float(context["mass"])
        friction = float(context["friction"])
        scale = mass / self.base_object_mass
        self.model.body_mass[self.object_bid] = mass
        self.model.body_inertia[self.object_bid] = self.base_object_inertia * scale
        self.model.geom_friction[self.object_geom_id] = np.array(
            [friction, self.base_object_friction[1], self.base_object_friction[2]],
            dtype=float,
        )
        self._break_close_fraction = self._safe_close_fraction_from_variant(context)
        mujoco.mj_forward(self.model, self.data)

    def _set_broken_state(self, broken: bool) -> None:
        """Set the broken-object state without exposing hidden env parameters.

        Args:
            broken: Whether the object should be marked as broken.
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
            friction = float((self._hidden_env_context or {}).get("friction", self.base_object_friction[0]))
            self.model.geom_friction[self.object_geom_id] = np.array(
                [friction, self.base_object_friction[1], self.base_object_friction[2]],
                dtype=float,
            )
        mujoco.mj_forward(self.model, self.data)

    def _get_info(self) -> dict[str, Any]:
        """Return semantic episode info without exposing hidden context."""
        info = super()._get_info()
        info.pop("variant", None)
        info.pop("config", None)
        info["adaptation_success_horizon_steps"] = self.config.adaptation_success_horizon_steps
        return info
