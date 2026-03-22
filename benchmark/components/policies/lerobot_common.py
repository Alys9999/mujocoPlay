from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from benchmark.core.interfaces.observation import ObservationBundle
from benchmark.schemas.models.action_packet import ActionPacket
from .base import observation_to_state_vector

LOGGER = logging.getLogger(__name__)


class AbstractLeRobotPolicyAdapter(ABC):
    """Shared LeRobot policy adapter boundary used by pi0.5 and smolVLA."""

    requires_image = True
    supported_schema_ids = ("cartesian_gripper_v1",)

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        max_episode_steps: int = 1400,
        duplicate_overview_to_all_cameras: bool = False,
        num_inference_steps: int | None = None,
        state_dim: int = 32,
        image_key_mapping: dict[str, str] | None = None,
    ) -> None:
        self._model_path = str(model_path)
        self._device_name = str(device).strip().lower()
        self._dtype = str(dtype).strip().lower()
        self._max_episode_steps = int(max_episode_steps)
        self._duplicate_overview_to_all_cameras = bool(duplicate_overview_to_all_cameras)
        self._num_inference_steps = None if num_inference_steps is None else int(num_inference_steps)
        self._state_dim = int(state_dim)
        self._image_key_mapping = dict(image_key_mapping or {})
        self._policy = None
        self._preprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._postprocess: Callable[[Any], Any] | None = None
        self.required_cameras = tuple(sorted(set(self._image_key_mapping.values())))
        self.name = self._build_name()

    def _build_name(self) -> str:
        parts = [self.policy_family]
        if self._device_name:
            parts.append(self._device_name)
        if self._model_path:
            parts.append(Path(self._model_path).name.replace(".", "_"))
        return "_".join(parts)

    @property
    @abstractmethod
    def policy_family(self) -> str:
        """Short adapter family name used in traces and results."""

    def reset(self) -> None:
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

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

    def act(self, observation: ObservationBundle) -> ActionPacket:
        self._ensure_loaded()
        assert self._policy is not None
        assert self._preprocess is not None
        assert self._postprocess is not None

        batch = self._build_batch(observation)
        processed = self._preprocess(batch)
        started_at = time.perf_counter()
        with torch.inference_mode():
            pred_action = self._policy.select_action(processed)
            pred_action = self._postprocess(pred_action)
        LOGGER.info("%s inference finished in %.2fs", self.policy_family, time.perf_counter() - started_at)
        raw_action = np.asarray(pred_action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        return self._raw_action_to_packet(raw_action)

    def _ensure_loaded(self) -> None:
        if self._policy is not None and self._preprocess is not None and self._postprocess is not None:
            return
        self._policy, self._preprocess, self._postprocess = self._load_policy_bundle()
        self.required_cameras = tuple(sorted(set(self._image_key_mapping.values())))
        self.name = self._build_name()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def _build_batch(self, observation: ObservationBundle) -> dict[str, Any]:
        state = torch.from_numpy(
            observation_to_state_vector(
                observation,
                max_episode_steps=self._max_episode_steps,
                state_dim=self._state_dim,
            )
        )
        batch: dict[str, Any] = {
            "observation.state": state,
            "task": observation.instruction,
        }
        for image_key, camera_name in self._image_key_mapping.items():
            image = observation.images.get(camera_name)
            if image is None and self._duplicate_overview_to_all_cameras:
                image = observation.images.get("overview")
            if image is None:
                raise ValueError(
                    f"{self.policy_family} requires camera '{camera_name}' for batch key '{image_key}'."
                )
            batch[image_key] = torch.from_numpy(np.asarray(image, dtype=np.uint8)).permute(2, 0, 1).contiguous()
        return batch

    def _build_identity_quantile_stats(self, state_dim: int, action_dim: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "observation.state": {
                "q01": -torch.ones(state_dim, dtype=torch.float32),
                "q99": torch.ones(state_dim, dtype=torch.float32),
            },
            "action": {
                "q01": -torch.ones(action_dim, dtype=torch.float32),
                "q99": torch.ones(action_dim, dtype=torch.float32),
            },
        }

    @abstractmethod
    def _load_policy_bundle(self) -> tuple[Any, Callable[[dict[str, Any]], dict[str, Any]], Callable[[Any], Any]]:
        """Load the backend policy and pre/post processors."""

    @abstractmethod
    def _raw_action_to_packet(self, raw_action: np.ndarray) -> ActionPacket:
        """Convert the backend-native action tensor into the canonical packet."""
