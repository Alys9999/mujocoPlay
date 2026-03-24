from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from phase1.policy_benchmark import AdaptiveBenchmarkSession
from phase1.task_language import build_instruction

from benchmark.core.interfaces.observation import ObservationBundle, StepResult
from benchmark.schemas.models.action_packet import ActionPacket
from .control_adapters.franka_panda_2f_v1 import FrankaPanda2FControlAdapter


class Phase1FrankaPanda2FSceneRuntime:
    """SceneRuntime wrapper around the existing phase1 AdaptiveBenchmarkSession."""

    name = "franka_panda_2f_v1"

    def __init__(
        self,
        *,
        object_family: str,
        task_variant: str,
        camera_names: Iterable[str] = ("overview",),
    ) -> None:
        self.object_family = str(object_family)
        self.task_variant = str(task_variant)
        self.camera_names = tuple(dict.fromkeys(camera_names))
        self.control_adapter = FrankaPanda2FControlAdapter()
        self._session = AdaptiveBenchmarkSession(object_family=self.object_family, task_variant=self.task_variant)
        self._instruction = build_instruction(self.object_family, self.task_variant)
        self._latest_action_vector = np.zeros(4, dtype=float)

    def close(self) -> None:
        self._session.close()

    def _requires_rendered_images(self) -> bool:
        return bool(set(self.camera_names) & {"overview", "wrist_left", "wrist_right"})

    def reset(
        self,
        *,
        seed: int | None = None,
        hidden_context: dict[str, dict[str, Any]] | None = None,
        target_xy: np.ndarray | None = None,
    ) -> ObservationBundle:
        legacy_observation = self._session.reset(
            seed=seed,
            hidden_context=hidden_context,
            target_xy=target_xy,
            include_image=self._requires_rendered_images(),
        )
        return self._build_observation(legacy_observation)

    def step(self, action_packet: ActionPacket) -> StepResult:
        runtime_action = self.control_adapter.to_runtime_action(action_packet)
        include_image = self._requires_rendered_images()
        if runtime_action.kind == "delta":
            self._latest_action_vector = np.asarray(runtime_action.values, dtype=float).reshape(4)
            legacy_observation, reward, terminated, truncated, info = self._session.step(
                self._latest_action_vector,
                include_image=include_image,
            )
            return StepResult(
                observation=self._build_observation(legacy_observation),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )

        if runtime_action.kind != "absolute":
            raise ValueError(f"Unknown runtime action kind '{runtime_action.kind}'.")

        absolute_action = np.asarray(runtime_action.values, dtype=float).reshape(7)
        synthetic_delta = np.zeros(4, dtype=float)
        mocap_delta = absolute_action[:3] - self._session.env.data.mocap_pos[0]
        limit = max(float(self._session.env.config.action_delta_limit), 1e-6)
        synthetic_delta[:3] = np.clip(mocap_delta / limit, -1.0, 1.0)
        synthetic_delta[3] = 1.0 - absolute_action[6]
        self._latest_action_vector = synthetic_delta
        self._session._previous_action = np.array(synthetic_delta, copy=True)
        self._session._action_history.append(np.array(synthetic_delta, copy=True))
        obs, reward, terminated, truncated, info = self._session.env.step_cartesian(absolute_action)
        self._session._latest_obs = obs
        self._session._latest_info = info
        legacy_observation = self._session.get_public_observation(include_image=include_image)
        return StepResult(
            observation=self._build_observation(legacy_observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def summarize_episode(self) -> dict[str, Any]:
        return self._session.summarize_episode()

    def get_privileged_context(self) -> dict[str, Any]:
        if hasattr(self._session.env, "debug_hidden_context"):
            return self._session.env.debug_hidden_context()
        return {}

    def _build_observation(self, legacy_observation: dict[str, Any]) -> ObservationBundle:
        images: dict[str, np.ndarray] = {}
        if "overview_rgb" in legacy_observation:
            images["overview"] = np.asarray(legacy_observation["overview_rgb"], copy=True)
        if "left_wrist_rgb" in legacy_observation:
            images["wrist_left"] = np.asarray(legacy_observation["left_wrist_rgb"], copy=True)
        if "right_wrist_rgb" in legacy_observation:
            images["wrist_right"] = np.asarray(legacy_observation["right_wrist_rgb"], copy=True)
        metadata = {
            "available_cameras": tuple(sorted(images)),
            "instruction": self._instruction,
        }
        return ObservationBundle(
            joint_positions=np.asarray(legacy_observation["arm_qpos"], dtype=np.float32),
            ee_position=np.asarray(legacy_observation["gripper_pos"], dtype=np.float32),
            ee_quaternion=np.asarray(legacy_observation["ee_quat"], dtype=np.float32),
            gripper_aperture=float(np.asarray(legacy_observation["gripper_aperture"], dtype=np.float32).reshape(-1)[0]),
            wrist_force=np.asarray(legacy_observation["wrist_force"], dtype=np.float32),
            wrist_torque=np.asarray(legacy_observation["wrist_torque"], dtype=np.float32),
            target_position=np.asarray(legacy_observation["target_pos"], dtype=np.float32),
            mocap_target=np.asarray(legacy_observation["mocap_target"], dtype=np.float32),
            previous_action_vector=np.asarray(legacy_observation["previous_action"], dtype=np.float32),
            time_sec=float(legacy_observation["time_sec"]),
            step_count=int(legacy_observation["step_count"]),
            instruction=str(legacy_observation["instruction"]),
            object_family=str(legacy_observation["object_family"]),
            task_variant=str(legacy_observation["task_variant"]),
            images=images,
            metadata=metadata,
        )


def build_phase1_franka_runtime(
    *,
    object_family: str,
    task_variant: str,
    camera_names: Iterable[str] = ("overview",),
) -> Phase1FrankaPanda2FSceneRuntime:
    """Factory used by the default registry."""
    return Phase1FrankaPanda2FSceneRuntime(
        object_family=object_family,
        task_variant=task_variant,
        camera_names=camera_names,
    )
