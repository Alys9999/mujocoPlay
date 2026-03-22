from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

from benchmark.schemas.models.action_packet import ActionPacket
from .lerobot_common import AbstractLeRobotPolicyAdapter


class LeRobotSmolVLAAdapter(AbstractLeRobotPolicyAdapter):
    """LeRobot smolVLA/XVLA adapter that shares the same ActionPacket boundary."""

    policy_family = "smolvla"

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        max_episode_steps: int = 1400,
        duplicate_overview_to_all_cameras: bool = False,
        gripper_index: int = 9,
        num_inference_steps: int | None = None,
        image_key_mapping: dict[str, str] | None = None,
    ) -> None:
        self._gripper_index = int(gripper_index)
        super().__init__(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_episode_steps=max_episode_steps,
            duplicate_overview_to_all_cameras=duplicate_overview_to_all_cameras,
            num_inference_steps=num_inference_steps,
            state_dim=32,
            image_key_mapping=image_key_mapping,
        )

    def _load_policy_bundle(self) -> tuple[Any, Callable[[dict[str, Any]], dict[str, Any]], Callable[[Any], Any]]:
        config = XVLAConfig.from_pretrained(self._model_path)
        config.device = self._device_name
        config.dtype = self._dtype
        if self._num_inference_steps is not None:
            config.num_denoising_steps = self._num_inference_steps
        if not self._image_key_mapping:
            camera_order = ("overview", "wrist_left", "wrist_right")
            self._image_key_mapping = {
                image_key: camera_order[min(index, len(camera_order) - 1)]
                for index, image_key in enumerate(config.image_features)
            }
        policy = XVLAPolicy.from_pretrained(self._model_path, config=config).eval().to(self._device_name)
        if self._dtype == "bfloat16":
            policy.to(dtype=torch.bfloat16)
        else:
            policy.to(dtype=torch.float32)
        action_dim = config.max_action_dim
        if config.action_feature is not None and len(config.action_feature.shape) > 0:
            action_dim = max(action_dim, int(config.action_feature.shape[-1]))
        feature_stats = self._build_identity_quantile_stats(config.max_state_dim, action_dim)
        preprocess, postprocess = make_xvla_pre_post_processors(config, dataset_stats=feature_stats)
        return policy, preprocess, postprocess

    def _raw_action_to_packet(self, raw_action: np.ndarray) -> ActionPacket:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.shape[0] < max(4, self._gripper_index + 1):
            raise ValueError(f"Expected at least {self._gripper_index + 1} action dimensions from smolVLA, got {action.shape}.")
        delta_xyz = np.tanh(action[:3]).astype(np.float32)
        close_value = float(np.clip(action[self._gripper_index], 0.0, 1.0))
        return ActionPacket(
            schema_id="cartesian_gripper_v1",
            arm={"mode": "delta_pose", "xyz": delta_xyz.tolist(), "rpy": [0.0, 0.0, 0.0]},
            hand={"mode": "scalar_close", "value": close_value},
            metadata={"source": "smolvla", "policy_family": "smolvla"},
        )
