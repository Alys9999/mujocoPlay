from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Phase1Config
from .splits import sample_hidden_context


@dataclass(frozen=True)
class BenchmarkPipelineSpec:
    """Describe which sources of variation are active in one benchmark pipeline.

    Args:
        name: Stable pipeline identifier.
        randomize_body: Whether to randomize robot-body physics.
        randomize_env: Whether to randomize object/environment physics.
        description: Human-readable description of the pipeline.
    """

    name: str
    randomize_body: bool
    randomize_env: bool
    description: str


PIPELINE_SPECS: dict[str, BenchmarkPipelineSpec] = {
    "normal_pick_place": BenchmarkPipelineSpec(
        name="normal_pick_place",
        randomize_body=False,
        randomize_env=False,
        description="Nominal pick-and-place with fixed target, nominal body physics, and nominal object physics.",
    ),
    "body_random_pick_place": BenchmarkPipelineSpec(
        name="body_random_pick_place",
        randomize_body=True,
        randomize_env=False,
        description="Pick-and-place with randomized robot-body physics and nominal object physics.",
    ),
    "object_random_pick_place": BenchmarkPipelineSpec(
        name="object_random_pick_place",
        randomize_body=False,
        randomize_env=True,
        description="Pick-and-place with nominal robot-body physics and randomized object physics.",
    ),
    "both_random_pick_place": BenchmarkPipelineSpec(
        name="both_random_pick_place",
        randomize_body=True,
        randomize_env=True,
        description="Pick-and-place with randomized robot-body and object physics.",
    ),
}

DEFAULT_PIPELINE_NAME = "both_random_pick_place"
DEFAULT_PIPELINE_CONFIG_PATH = Path(__file__).resolve().parent / "benchmark_pipeline.json"


def default_body_context() -> dict[str, Any]:
    """Return the nominal hidden body context used by non-randomized pipelines."""
    return {
        "reach_scale": 1.0,
        "arm_mass_scale": 1.0,
        "payload_scale": 1.0,
        "joint_damping_scale": 1.0,
        "actuator_gain_scale": 1.0,
        "fingertip_friction_scale": 1.0,
        "damage_joint_index": -1,
        "damage_gain_scale": 1.0,
        "damage_damping_scale": 1.0,
        "local_finger_wear_side": "none",
        "local_finger_friction_scale": 1.0,
    }


def default_env_context(config: Phase1Config) -> dict[str, Any]:
    """Return the nominal hidden object context used by non-randomized pipelines.

    Args:
        config: Environment configuration that defines nominal physics values.
    """
    return {
        "mass": float(np.mean(config.object_masses)),
        "friction": float(np.mean(config.object_frictions)),
        "stiffness": float(np.mean(config.object_stiffnesses)),
    }


def resolve_pipeline_spec(name: str | None) -> BenchmarkPipelineSpec:
    """Resolve a pipeline name into a pipeline specification.

    Args:
        name: Requested pipeline name, or None to use the default preset.
    """
    pipeline_name = DEFAULT_PIPELINE_NAME if name is None else str(name)
    try:
        return PIPELINE_SPECS[pipeline_name]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline: {pipeline_name}") from exc


def load_pipeline_name(config_path: Path | None = None) -> str:
    """Load the active pipeline name from the shared benchmark config file.

    Args:
        config_path: Optional path to the benchmark pipeline JSON file.
    """
    path = DEFAULT_PIPELINE_CONFIG_PATH if config_path is None else Path(config_path)
    if not path.exists():
        return DEFAULT_PIPELINE_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("pipeline", DEFAULT_PIPELINE_NAME))


def sample_target_xy(config: Phase1Config, rng: np.random.Generator) -> np.ndarray:
    """Sample one randomized target location within the configured target bounds.

    Args:
        config: Environment configuration that defines target bounds.
        rng: Random-number generator used for sampling.
    """
    object_xy = np.array([config.default_object_x, config.default_object_y], dtype=float)
    for _ in range(64):
        candidate = np.array(
            [
                rng.uniform(config.target_random_x_min, config.target_random_x_max),
                rng.uniform(config.target_random_y_min, config.target_random_y_max),
            ],
            dtype=float,
        )
        if np.linalg.norm(candidate - object_xy) >= float(config.target_random_min_distance):
            return candidate
    return np.array([config.target_x, config.target_y], dtype=float)


def build_episode_setup(
    pipeline_name: str,
    split_name: str,
    config: Phase1Config,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    """Construct hidden context and target position for one benchmark episode.

    Args:
        pipeline_name: Name of the benchmark pipeline to instantiate.
        split_name: Hidden-context split used when randomization is enabled.
        config: Environment configuration used for nominal values and target bounds.
        rng: Random-number generator used for sampling.
    """
    pipeline = resolve_pipeline_spec(pipeline_name)
    sampled_context = sample_hidden_context(split_name=split_name, rng=rng)
    body = sampled_context["body"] if pipeline.randomize_body else default_body_context()
    env = sampled_context["env"] if pipeline.randomize_env else default_env_context(config)
    target_xy = np.array([config.target_x, config.target_y], dtype=float)
    return {"body": body, "env": env}, target_xy
