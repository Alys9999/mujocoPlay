from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from benchmark.schemas.models.action_packet import ActionPacket
from .lerobot_common import AbstractLeRobotPolicyAdapter

LOGGER = logging.getLogger(__name__)

SMOLVLA_IMAGE_KEYS = {
    "observation.image": "overview",
    "observation.image2": "wrist_left",
    "observation.image3": "wrist_right",
}


class LeRobotSmolVLAAdapter(AbstractLeRobotPolicyAdapter):
    """LeRobot SmolVLA adapter with a heuristic bridge into the benchmark action packet."""

    policy_family = "smolvla"

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        max_episode_steps: int = 1400,
        duplicate_overview_to_all_cameras: bool = False,
        gripper_index: int = 5,
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
            image_key_mapping=image_key_mapping or SMOLVLA_IMAGE_KEYS,
        )

    def _load_policy_bundle(self) -> tuple[Any, Callable[[dict[str, Any]], dict[str, Any]], Callable[[Any], Any]]:
        config = PreTrainedConfig.from_pretrained(self._model_path)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"Expected a SmolVLA checkpoint, got {type(config).__name__}.")
        config.device = self._device_name
        policy = SmolVLAPolicy.from_pretrained(self._model_path, config=config).eval().to(self._device_name)
        if self._dtype == "bfloat16":
            policy.to(dtype=torch.bfloat16)
        else:
            policy.to(dtype=torch.float32)

        preprocess, _ = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=self._model_path,
            preprocessor_overrides={"device_processor": {"device": self._device_name}},
        )

        LOGGER.warning(
            "SmolVLA base is an SO100-style joint-space checkpoint; this adapter heuristically maps "
            "normalized outputs into `cartesian_gripper_v1` for exploratory benchmark runs."
        )
        return policy, preprocess, lambda action: action

    def _raw_action_to_packet(self, raw_action: np.ndarray) -> ActionPacket:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.shape[0] < max(4, self._gripper_index + 1):
            raise ValueError(f"Expected at least {self._gripper_index + 1} action dimensions from smolVLA, got {action.shape}.")
        delta_xyz = np.tanh(action[:3]).astype(np.float32)
        close_value = 0.5 * (np.tanh(action[self._gripper_index]) + 1.0)
        return ActionPacket(
            schema_id="cartesian_gripper_v1",
            arm={"mode": "delta_pose", "xyz": delta_xyz.tolist(), "rpy": [0.0, 0.0, 0.0]},
            hand={"mode": "scalar_close", "value": float(np.clip(close_value, 0.0, 1.0))},
            metadata={"source": "smolvla", "policy_family": "smolvla"},
        )
