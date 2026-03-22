from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid")


class BenchmarkSection(StrictModel):
    """Top-level benchmark execution settings."""

    name: str
    seed: int = 0
    episodes: int = 1
    output_dir: Path = Path("benchmark_results/default")
    family_split: Literal["all", "seen", "heldout"] = "all"
    hidden_split: Literal["seen", "unseen", "full"] = "unseen"


class RuntimeSection(StrictModel):
    """Runtime backend settings."""

    backend: Literal["mujoco"] = "mujoco"
    max_steps: int = 0


class TopologyRandomizationConfig(StrictModel):
    """Optional topology-randomization configuration."""

    enabled: bool = False
    candidates: list[str] = Field(default_factory=list)


class RandomizationConfig(StrictModel):
    """Shared randomization section for robot and object sets."""

    enabled: bool = False
    topology: TopologyRandomizationConfig = Field(default_factory=TopologyRandomizationConfig)
    parameters: dict[str, Any] = Field(default_factory=dict)


class RobotConfig(StrictModel):
    """Robot profile selection settings."""

    profile_pool: list[str]
    selection_strategy: Literal["first", "weighted_sample"] = "first"
    randomization: RandomizationConfig = Field(default_factory=RandomizationConfig)


class TaskConfig(StrictModel):
    """Task-definition selection settings."""

    name: str


class ObjectsConfig(StrictModel):
    """Object-set selection settings."""

    set: str
    randomization: RandomizationConfig = Field(default_factory=RandomizationConfig)


class CameraRigConfig(StrictModel):
    """One camera rig definition."""

    name: str
    mount_type: Literal["world_fixed", "body_attached"]
    preset: str | None = None
    mount_frame: str | None = None
    pose_mode: Literal["preset", "relative_pose"] = "preset"
    xyz: list[float] | None = None
    rpy: list[float] | None = None
    quat: list[float] | None = None
    width: int = 320
    height: int = 240

    @model_validator(mode="after")
    def validate_mount(self) -> "CameraRigConfig":
        if self.mount_type == "body_attached" and not self.mount_frame:
            raise ValueError("`mount_frame` is required for `body_attached` camera rigs.")
        if self.pose_mode == "relative_pose" and self.xyz is None:
            raise ValueError("`xyz` is required when `pose_mode` is `relative_pose`.")
        if self.pose_mode == "relative_pose" and self.rpy is None and self.quat is None:
            raise ValueError("Either `rpy` or `quat` is required when `pose_mode` is `relative_pose`.")
        return self


class CamerasConfig(StrictModel):
    """Camera-rig collection."""

    rigs: list[CameraRigConfig] = Field(default_factory=list)


class PolicyConfig(StrictModel):
    """Policy adapter selection settings."""

    adapter: str = "random"
    kwargs: dict[str, Any] = Field(default_factory=dict)


class TraceSinkConfig(StrictModel):
    """One trace sink definition."""

    type: str
    include: list[str] = Field(default_factory=list)
    cameras: list[str] = Field(default_factory=list)
    include_privileged_context: bool = False
    output_path: Path | None = None


class TracingConfig(StrictModel):
    """Tracing sink collection."""

    sinks: list[TraceSinkConfig] = Field(default_factory=list)


class BenchmarkConfig(StrictModel):
    """Full typed benchmark configuration."""

    benchmark: BenchmarkSection
    runtime: RuntimeSection = Field(default_factory=RuntimeSection)
    robot: RobotConfig
    task: TaskConfig
    objects: ObjectsConfig
    cameras: CamerasConfig = Field(default_factory=CamerasConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
