from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from benchmark.schemas.models.action_packet import ActionPacket
from .lerobot_common import AbstractLeRobotPolicyAdapter

LOGGER = logging.getLogger(__name__)

PI05_IMAGE_KEYS = {
    "observation.images.base_0_rgb": "overview",
    "observation.images.left_wrist_0_rgb": "wrist_left",
    "observation.images.right_wrist_0_rgb": "wrist_right",
}


class LeRobotPI05PolicyAdapter(AbstractLeRobotPolicyAdapter):
    """LeRobot pi0.5 policy adapter that emits canonical action packets."""

    policy_family = "pi05"

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        quantization: str = "none",
        dtype: str = "float32",
        max_episode_steps: int = 1400,
        duplicate_overview_to_all_cameras: bool = False,
        gripper_index: int = 6,
        num_inference_steps: int | None = None,
        image_key_mapping: dict[str, str] | None = None,
    ) -> None:
        self._quantization = str(quantization).strip().lower()
        self._gripper_index = int(gripper_index)
        super().__init__(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_episode_steps=max_episode_steps,
            duplicate_overview_to_all_cameras=duplicate_overview_to_all_cameras,
            num_inference_steps=num_inference_steps,
            state_dim=32,
            image_key_mapping=image_key_mapping or PI05_IMAGE_KEYS,
        )

    def _load_policy_bundle(self) -> tuple[Any, Callable[[dict[str, Any]], dict[str, Any]], Callable[[Any], Any]]:
        config = PI05Config.from_pretrained(self._model_path)
        config.device = "cpu" if self._quantization != "none" else self._device_name
        config.dtype = self._dtype
        if self._num_inference_steps is not None:
            config.num_inference_steps = self._num_inference_steps

        policy = PI05Policy.from_pretrained(self._model_path, config=config).eval()
        if self._quantization == "int8_dynamic":
            if self._device_name != "cpu":
                raise ValueError("`int8_dynamic` quantization is CPU-only.")
            policy = torch.ao.quantization.quantize_dynamic(policy, {torch.nn.Linear}, dtype=torch.qint8)
        elif self._quantization != "none":
            raise ValueError(f"Unsupported quantization mode: {self._quantization}")

        policy.to(self._device_name)
        if self._dtype == "bfloat16":
            policy.to(dtype=torch.bfloat16)
        else:
            policy.to(dtype=torch.float32)
        feature_stats = self._build_identity_quantile_stats(config.max_state_dim, config.max_action_dim)
        preprocess, postprocess = make_pre_post_processors(config, dataset_stats=feature_stats)
        return policy, preprocess, postprocess

    def _raw_action_to_packet(self, raw_action: np.ndarray) -> ActionPacket:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.shape[0] < max(7, self._gripper_index + 1):
            raise ValueError(f"Expected at least 7 action dimensions from pi05, got {action.shape}.")
        delta_xyz = np.tanh(action[:3]).astype(np.float32)
        close_value = 0.5 * (np.tanh(action[self._gripper_index]) + 1.0)
        return ActionPacket(
            schema_id="cartesian_gripper_v1",
            arm={"mode": "delta_pose", "xyz": delta_xyz.tolist(), "rpy": [0.0, 0.0, 0.0]},
            hand={"mode": "scalar_close", "value": float(np.clip(close_value, 0.0, 1.0))},
            metadata={"source": "pi05", "policy_family": "pi05"},
        )
