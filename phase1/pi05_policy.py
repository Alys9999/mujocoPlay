from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)


PI05_IMAGE_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
PI05_DEFAULT_TOKENIZER_NAME = "google/paligemma-3b-pt-224"
PI05_PUBLIC_FALLBACK_TOKENIZER_REPO = "pcuenq/gemma-tokenizer"
PI05_PUBLIC_FALLBACK_TOKENIZER_DIR = Path.home() / "models" / "gemma-tokenizer-public"
PI05_TOKENIZER_REQUIRED_FILES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
)
PI05_TOKENIZER_SERIALIZATION_FILES = (
    "tokenizer.json",
    "tokenizer.model",
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
        action_chunk_size: int | None = 10,
        duplicate_overview_to_all_cameras: bool | None = None,
        gripper_index: int = 6,
        quantization: str | None = None,
        dtype: str | None = None,
        num_inference_steps: int | None = None,
        tokenizer_name_or_path: str | None = None,
    ) -> None:
        self._model_path = None if model_path is None else str(model_path)
        self._device_name = None if device is None else str(device).strip().lower()
        self._max_episode_steps = int(max_episode_steps)
        self._action_chunk_size = None if action_chunk_size is None else int(action_chunk_size)
        self._duplicate_overview_to_all_cameras = duplicate_overview_to_all_cameras
        self._gripper_index = int(gripper_index)
        self._quantization = None if quantization is None else str(quantization).strip().lower()
        self._dtype = None if dtype is None else str(dtype).strip().lower()
        self._effective_dtype = self._dtype
        self._num_inference_steps = None if num_inference_steps is None else int(num_inference_steps)
        self._tokenizer_name_or_path = (
            None if tokenizer_name_or_path is None else str(tokenizer_name_or_path).strip()
        )
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
        self._ensure_loaded()
        assert self._policy is not None
        assert self._preprocess is not None
        assert self._postprocess is not None

        cached_actions = self._cached_action_count()
        if cached_actions == 0 and "overview_rgb" not in observation:
            raise ValueError("PI05SequentialPolicy requires `overview_rgb` when a fresh action chunk is needed.")
        infer_started_at = time.perf_counter()
        LOGGER.info(
            "PI05 action request device=%s dtype=%s cached_actions=%s",
            self._device_name,
            self._dtype,
            cached_actions,
        )
        if cached_actions > 0:
            with torch.inference_mode():
                pred_action = self._policy.select_action({})
                pred_action = self._postprocess(pred_action)
            LOGGER.info(
                "PI05 reused cached action in %.2fs remaining_cached_actions=%s",
                time.perf_counter() - infer_started_at,
                self._cached_action_count(),
            )
        else:
            preprocess_started_at = time.perf_counter()
            batch = self._build_batch(observation)
            processed = self._preprocess(batch)
            preprocess_duration = time.perf_counter() - preprocess_started_at
            compute_started_at = time.perf_counter()
            if self._device_name == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            with torch.inference_mode():
                pred_action = self._policy.select_action(processed)
                pred_action = self._postprocess(pred_action)
            if self._device_name == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            LOGGER.info(
                "PI05 generated action chunk preprocess=%.2fs infer=%.2fs total=%.2fs remaining_cached_actions=%s",
                preprocess_duration,
                time.perf_counter() - compute_started_at,
                time.perf_counter() - infer_started_at,
                self._cached_action_count(),
            )
        raw_action = np.asarray(pred_action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        return self._to_benchmark_action(raw_action)

    def needs_image_next_step(self) -> bool:
        return self._cached_action_count() == 0

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

        tokenizer_started_at = time.perf_counter()
        tokenizer = self._resolve_processor_tokenizer()
        LOGGER.info(
            "PI05 tokenizer ready in %.2fs source=%s",
            time.perf_counter() - tokenizer_started_at,
            getattr(tokenizer, "name_or_path", "<object>"),
        )

        load_started_at = time.perf_counter()
        LOGGER.info("PI05 config load start model_dir=%s", model_dir)
        config = _load_pi05_config(model_dir)
        LOGGER.info("PI05 config load finished in %.2fs", time.perf_counter() - load_started_at)
        config.device = "cpu" if self._quantization != "none" else self._device_name
        self._effective_dtype = self._resolve_runtime_dtype()
        config.dtype = self._effective_dtype
        if self._action_chunk_size is not None:
            config.chunk_size = self._action_chunk_size
            config.n_action_steps = self._action_chunk_size
        if self._num_inference_steps is not None:
            config.num_inference_steps = self._num_inference_steps

        policy_load_started_at = time.perf_counter()
        LOGGER.info(
            "PI05 policy load start target_device=%s dtype=%s inference_steps=%s",
            config.device,
            config.dtype,
            config.num_inference_steps,
        )
        self._policy = PI05Policy.from_pretrained(str(model_dir), config=config).eval()
        LOGGER.info("PI05 policy load finished in %.2fs", time.perf_counter() - policy_load_started_at)
        if self._quantization == "int8_dynamic":
            self._policy = self._apply_dynamic_int8_quantization(self._policy)
        elif self._quantization != "none":
            raise ValueError(f"Unsupported quantization mode: {self._quantization}")

        runtime_started_at = time.perf_counter()
        LOGGER.info("PI05 runtime device/dtype application start")
        self._apply_runtime_device_and_dtype()
        LOGGER.info("PI05 runtime device/dtype application finished in %.2fs", time.perf_counter() - runtime_started_at)
        policy_config = self._policy.config

        preprocess_started_at = time.perf_counter()
        LOGGER.info("PI05 processor construction start")
        feature_stats = self._build_identity_quantile_stats(policy_config.max_state_dim, policy_config.max_action_dim)
        preprocessor_overrides: dict[str, dict[str, Any]] = {
            "normalizer_processor": {"stats": feature_stats},
            "device_processor": {"device": self._device_name},
            # Preload the tokenizer object so we do not depend on gated or partial
            # Hugging Face downloads during processor instantiation.
            "tokenizer_processor": {"tokenizer": tokenizer},
        }

        postprocessor_overrides: dict[str, dict[str, Any]] = {
            "unnormalizer_processor": {"stats": feature_stats},
            "device_processor": {"device": "cpu"},
        }

        # Prefer the checkpoint's saved processor configs so tokenizer/device wiring can
        # be overridden locally without patching LeRobot itself.
        self._preprocess, self._postprocess = make_pre_post_processors(
            policy_config,
            pretrained_path=str(model_dir),
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
            dataset_stats=feature_stats,
        )
        LOGGER.info("PI05 processor construction finished in %.2fs", time.perf_counter() - preprocess_started_at)
        self._policy.reset()
        first_param = next(self._policy.parameters())
        LOGGER.info(
            "PI05 ready param_device=%s param_dtype=%s cuda_mem_alloc=%s cuda_mem_reserved=%s",
            first_param.device,
            first_param.dtype,
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        )

    def _apply_runtime_device_and_dtype(self) -> None:
        assert self._policy is not None
        assert self._device_name is not None
        assert self._effective_dtype is not None

        self._policy.to(self._device_name)

        if hasattr(self._policy, "model") and hasattr(self._policy.model, "to_bfloat16_for_selected_params"):
            self._policy.model.to_bfloat16_for_selected_params(self._effective_dtype)
        elif self._effective_dtype == "bfloat16":
            self._policy.to(dtype=torch.bfloat16)
        elif self._effective_dtype == "float32":
            self._policy.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported dtype for runtime application: {self._effective_dtype}")

        self._policy.config.device = self._device_name
        self._policy.config.dtype = self._effective_dtype
        self._policy.eval()

    def _resolve_runtime_dtype(self) -> str:
        assert self._dtype is not None
        assert self._device_name is not None

        if self._device_name == "cuda" and self._dtype == "bfloat16":
            LOGGER.warning(
                "PI05 requested dtype=bfloat16 on CUDA, but this runtime currently hits mixed Float/BFloat16 "
                "errors during denoising. Falling back to float32 on GPU."
            )
            return "float32"
        return self._dtype

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
        if self._action_chunk_size is not None and self._action_chunk_size <= 0:
            raise ValueError("`action_chunk_size` must be a positive integer when provided.")

    def _resolve_processor_tokenizer(self) -> Any:
        requested = self._tokenizer_name_or_path
        attempts: list[str] = []

        if requested:
            requested_path = Path(requested).expanduser()
            if requested_path.exists():
                if self._tokenizer_dir_has_assets(requested_path):
                    try:
                        return AutoTokenizer.from_pretrained(str(requested_path), local_files_only=True)
                    except Exception as exc:  # pragma: no cover - defensive path for local env drift.
                        attempts.append(f"{requested_path}: {exc}")
                else:
                    attempts.append(
                        f"{requested_path}: missing tokenizer files "
                        f"({', '.join((*PI05_TOKENIZER_REQUIRED_FILES, *PI05_TOKENIZER_SERIALIZATION_FILES))})"
                    )
                    LOGGER.warning(
                        "PI05 tokenizer override path %s exists but is incomplete; falling back to a public Gemma tokenizer.",
                        requested_path,
                    )
            else:
                try:
                    return AutoTokenizer.from_pretrained(requested)
                except Exception as exc:
                    attempts.append(f"{requested}: {exc}")

        fallback_dir = PI05_PUBLIC_FALLBACK_TOKENIZER_DIR
        if self._tokenizer_dir_has_assets(fallback_dir):
            try:
                return AutoTokenizer.from_pretrained(str(fallback_dir), local_files_only=True)
            except Exception as exc:  # pragma: no cover - defensive path for local env drift.
                attempts.append(f"{fallback_dir}: {exc}")

        try:
            snapshot_download(
                repo_id=PI05_PUBLIC_FALLBACK_TOKENIZER_REPO,
                local_dir=str(fallback_dir),
            )
            return AutoTokenizer.from_pretrained(str(fallback_dir), local_files_only=True)
        except Exception as exc:
            attempts.append(f"{PI05_PUBLIC_FALLBACK_TOKENIZER_REPO}: {exc}")

        message_lines = [
            "PI05 could not load a tokenizer for the processor pipeline.",
            f"Requested tokenizer: {requested or PI05_DEFAULT_TOKENIZER_NAME}",
            "Tried the following sources:",
            *[f"- {attempt}" for attempt in attempts],
            (
                "If you have accepted the gated PaliGemma license, download "
                f"`{PI05_DEFAULT_TOKENIZER_NAME}` into a real local directory and pass it as "
                "`tokenizer_name_or_path`."
            ),
        ]
        raise ValueError("\n".join(message_lines))

    def _tokenizer_dir_has_assets(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        has_required_files = all((path / filename).exists() for filename in PI05_TOKENIZER_REQUIRED_FILES)
        has_serialized_tokenizer = any((path / filename).exists() for filename in PI05_TOKENIZER_SERIALIZATION_FILES)
        return has_required_files and has_serialized_tokenizer

    def _apply_dynamic_int8_quantization(self, policy: PI05Policy) -> PI05Policy:
        if self._device_name != "cpu":
            raise ValueError("`int8_dynamic` quantization is CPU-only. Use `device=cpu`.")
        quantized_model = torch.ao.quantization.quantize_dynamic(policy, {torch.nn.Linear}, dtype=torch.qint8)
        return quantized_model

    def _cached_action_count(self) -> int:
        if self._policy is None:
            return 0
        action_queue = getattr(self._policy, "_action_queue", None)
        if action_queue is None:
            return 0
        return len(action_queue)

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
        base_image = torch.from_numpy(
            np.asarray(observation.get("base_rgb", observation["overview_rgb"]), dtype=np.uint8)
        ).permute(2, 0, 1).contiguous()
        state = torch.from_numpy(
            benchmark_observation_to_pi05_state(
                observation,
                max_episode_steps=self._max_episode_steps,
            )
        )
        batch: dict[str, Any] = {
            "observation.state": state,
            "task": str(observation["instruction"]),
            PI05_IMAGE_KEYS[0]: base_image,
        }
        if self._duplicate_overview_to_all_cameras:
            batch[PI05_IMAGE_KEYS[1]] = torch.from_numpy(
                np.asarray(observation.get("left_wrist_rgb", observation["overview_rgb"]), dtype=np.uint8)
            ).permute(2, 0, 1).contiguous()
            batch[PI05_IMAGE_KEYS[2]] = torch.from_numpy(
                np.asarray(observation.get("right_wrist_rgb", observation["overview_rgb"]), dtype=np.uint8)
            ).permute(2, 0, 1).contiguous()
        else:
            if "left_wrist_rgb" in observation:
                batch[PI05_IMAGE_KEYS[1]] = torch.from_numpy(
                    np.asarray(observation["left_wrist_rgb"], dtype=np.uint8)
                ).permute(2, 0, 1).contiguous()
            if "right_wrist_rgb" in observation:
                batch[PI05_IMAGE_KEYS[2]] = torch.from_numpy(
                    np.asarray(observation["right_wrist_rgb"], dtype=np.uint8)
                ).permute(2, 0, 1).contiguous()
        return batch

    def _to_benchmark_action(self, raw_action: np.ndarray) -> np.ndarray:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.shape[0] < max(7, self._gripper_index + 1):
            raise ValueError(f"Expected at least 7 action dimensions from pi05, got {action.shape}.")
        benchmark_action = np.zeros(4, dtype=np.float32)
        benchmark_action[:3] = np.tanh(action[:3])
        benchmark_action[3] = 0.5 * (np.tanh(action[self._gripper_index]) + 1.0)
        return benchmark_action
