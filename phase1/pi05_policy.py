from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


PI05_IMAGE_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)


def _load_pi05_config(model_dir: Path) -> PI05Config:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"pi05 config not found: {config_path}")

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    # LeRobot 0.5 ships PI05Config behind a registry-based loader, but the
    # downloaded checkpoint also stores a top-level `type` field that direct
    # PI05Config parsing rejects. We strip that legacy field before decoding.
    config_data.pop("type", None)

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as handle:
        json.dump(config_data, handle)
        temp_config_path = handle.name

    with draccus.config_type("json"):
        return draccus.parse(PI05Config, temp_config_path, args=[])


def _scale_to_unit_interval(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    scaled = 2.0 * (value - low) / np.maximum(high - low, 1e-6) - 1.0
    return np.clip(scaled, -1.0, 1.0)


def benchmark_observation_to_pi05_state(
    observation: dict[str, Any],
    max_episode_steps: int = 1400,
    state_dim: int = 32,
    max_open_aperture: float = 0.04,
    force_scale: float = 20.0,
    torque_scale: float = 2.0,
) -> np.ndarray:
    from .franka_env import FrankaHiddenPhysicsPickPlaceEnv

    cfg = FrankaHiddenPhysicsPickPlaceEnv.DEFAULT_CONFIG
    family_names = ("block", "cylinder", "small_box")
    task_names = ("pick_place",)
    family_index = family_names.index(str(observation["object_family"])) / max(len(family_names) - 1, 1)
    task_index = task_names.index(str(observation["task_variant"])) / max(len(task_names) - 1, 1)

    arm_qpos = np.clip(np.asarray(observation["arm_qpos"], dtype=np.float32) / np.pi, -1.0, 1.0)
    gripper_open = float(np.asarray(observation["gripper_aperture"], dtype=np.float32).reshape(-1)[0]) / max(
        max_open_aperture, 1e-6
    )
    gripper_open = np.asarray([np.clip(2.0 * gripper_open - 1.0, -1.0, 1.0)], dtype=np.float32)
    wrist_force = np.clip(np.asarray(observation["wrist_force"], dtype=np.float32) / force_scale, -1.0, 1.0)
    wrist_torque = np.clip(np.asarray(observation["wrist_torque"], dtype=np.float32) / torque_scale, -1.0, 1.0)
    workspace_low = np.asarray([cfg.workspace_x_min, cfg.workspace_y_min, cfg.workspace_z_min], dtype=np.float32)
    workspace_high = np.asarray([cfg.workspace_x_max, cfg.workspace_y_max, cfg.workspace_z_max], dtype=np.float32)
    target_pos = _scale_to_unit_interval(np.asarray(observation["target_pos"], dtype=np.float32), workspace_low, workspace_high)
    mocap_target = _scale_to_unit_interval(
        np.asarray(observation["mocap_target"], dtype=np.float32),
        workspace_low,
        workspace_high,
    )
    previous_action = np.asarray(observation["previous_action"], dtype=np.float32).reshape(4)
    previous_action = np.concatenate(
        [
            np.clip(previous_action[:3], -1.0, 1.0),
            np.asarray([2.0 * previous_action[3] - 1.0], dtype=np.float32),
        ]
    )
    time_frac = np.clip(
        float(observation["time_sec"]) / max(float(max_episode_steps) * cfg.timestep * cfg.control_substeps, 1e-6),
        0.0,
        1.0,
    )
    step_frac = np.clip(float(observation["step_count"]) / max(float(max_episode_steps), 1.0), 0.0, 1.0)

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


class PI05SequentialPolicy:
    name = "pi05"
    requires_image = True
    benchmark_interface = "default"

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        max_episode_steps: int = 1400,
        duplicate_overview_to_all_cameras: bool | None = None,
        gripper_index: int = 6,
        quantization: str | None = None,
        dtype: str | None = None,
        num_inference_steps: int | None = None,
    ) -> None:
        self._model_path = None if model_path is None else str(model_path)
        self._device_name = None if device is None else str(device).strip().lower()
        self._max_episode_steps = int(max_episode_steps)
        self._duplicate_overview_to_all_cameras = duplicate_overview_to_all_cameras
        self._gripper_index = int(gripper_index)
        self._quantization = None if quantization is None else str(quantization).strip().lower()
        self._dtype = None if dtype is None else str(dtype).strip().lower()
        self._num_inference_steps = None if num_inference_steps is None else int(num_inference_steps)
        self._policy: PI05Policy | None = None
        self._preprocess = None
        self._postprocess = None
        name_parts = ["pi05"]
        if self._device_name:
            name_parts.append(self._device_name)
        if self._quantization:
            name_parts.append(self._quantization)
        if self._model_path:
            name_parts.append(Path(self._model_path).name.replace(".", "_"))
        self.name = "_".join(name_parts)

    def reset(self) -> None:
        if self._policy is not None:
            self._policy.reset()

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if "overview_rgb" not in observation:
            raise ValueError("PI05SequentialPolicy requires `overview_rgb` in the observation.")
        self._ensure_loaded()
        assert self._policy is not None
        assert self._preprocess is not None
        assert self._postprocess is not None

        batch = self._build_batch(observation)
        processed = self._preprocess(batch)
        with torch.inference_mode():
            pred_action = self._policy.select_action(processed)
            pred_action = self._postprocess(pred_action)
        raw_action = np.asarray(pred_action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        return self._to_benchmark_action(raw_action)

    def _ensure_loaded(self) -> None:
        if self._policy is not None and self._preprocess is not None and self._postprocess is not None:
            return

        self._validate_runtime()
        assert self._model_path is not None
        assert self._device_name is not None
        assert self._quantization is not None
        assert self._dtype is not None
        assert self._duplicate_overview_to_all_cameras is not None

        model_dir = Path(self._model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"pi05 model path does not exist: {model_dir}")

        config = _load_pi05_config(model_dir)
        config.device = "cpu" if self._quantization != "none" else self._device_name
        config.dtype = self._dtype
        if self._num_inference_steps is not None:
            config.num_inference_steps = self._num_inference_steps

        self._policy = PI05Policy.from_pretrained(str(model_dir), config=config).eval()
        if self._quantization == "int8_dynamic":
            self._policy = self._apply_dynamic_int8_quantization(self._policy)
        elif self._quantization != "none":
            raise ValueError(f"Unsupported quantization mode: {self._quantization}")

        feature_stats = self._build_identity_quantile_stats(config.max_state_dim, config.max_action_dim)
        self._preprocess, self._postprocess = make_pre_post_processors(config, dataset_stats=feature_stats)
        self._policy.reset()

    def _validate_runtime(self) -> None:
        if self._model_path is None:
            raise ValueError("PI05SequentialPolicy requires an explicit `model_path`.")
        if self._device_name is None:
            raise ValueError("PI05SequentialPolicy requires an explicit `device` such as `cpu` or `cuda`.")
        if self._quantization is None:
            raise ValueError("PI05SequentialPolicy requires an explicit `quantization` such as `none` or `int8_dynamic`.")
        if self._dtype is None:
            raise ValueError("PI05SequentialPolicy requires an explicit `dtype` such as `float32` or `bfloat16`.")
        if self._duplicate_overview_to_all_cameras is None:
            raise ValueError(
                "PI05SequentialPolicy requires an explicit `duplicate_overview_to_all_cameras` setting."
            )
        if self._quantization == "int8_dynamic" and self._device_name != "cpu":
            raise ValueError("`int8_dynamic` quantization is CPU-only. Use `device=cpu` or `quantization=none`.")
        if self._device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested for PI05, but `torch.cuda.is_available()` is false.")
        if self._device_name == "mps":
            if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
                raise ValueError("MPS was requested for PI05, but it is not available in this torch build/runtime.")
        if self._dtype == "bfloat16" and self._device_name not in {"cuda", "cpu"}:
            raise ValueError("`bfloat16` dtype is only supported here for `cpu` and `cuda`.")

    def _apply_dynamic_int8_quantization(self, policy: PI05Policy) -> PI05Policy:
        if self._device_name != "cpu":
            raise ValueError("`int8_dynamic` quantization is CPU-only. Use `device=cpu`.")
        quantized_model = torch.ao.quantization.quantize_dynamic(policy, {torch.nn.Linear}, dtype=torch.qint8)
        return quantized_model

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

    def _build_batch(self, observation: dict[str, Any]) -> dict[str, Any]:
        image = torch.from_numpy(np.asarray(observation["overview_rgb"], dtype=np.uint8)).permute(2, 0, 1).contiguous()
        state = torch.from_numpy(
            benchmark_observation_to_pi05_state(
                observation,
                max_episode_steps=self._max_episode_steps,
            )
        )
        batch: dict[str, Any] = {
            "observation.state": state,
            "task": str(observation["instruction"]),
            PI05_IMAGE_KEYS[0]: image,
        }
        if self._duplicate_overview_to_all_cameras:
            batch[PI05_IMAGE_KEYS[1]] = image
            batch[PI05_IMAGE_KEYS[2]] = image
        return batch

    def _to_benchmark_action(self, raw_action: np.ndarray) -> np.ndarray:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.shape[0] < max(7, self._gripper_index + 1):
            raise ValueError(f"Expected at least 7 action dimensions from pi05, got {action.shape}.")
        benchmark_action = np.zeros(4, dtype=np.float32)
        benchmark_action[:3] = np.tanh(action[:3])
        benchmark_action[3] = 0.5 * (np.tanh(action[self._gripper_index]) + 1.0)
        return benchmark_action
