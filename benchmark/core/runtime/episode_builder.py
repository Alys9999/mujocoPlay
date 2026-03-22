from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from phase1.franka_env import FrankaHiddenPhysicsPickPlaceEnv
from phase1.pipeline_config import build_episode_setup

from benchmark.core.config.models import BenchmarkConfig
from benchmark.core.registry.registry import BenchmarkRegistry


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One object-family/task pair to evaluate."""

    object_family: str
    task_variant: str


@dataclass(frozen=True, slots=True)
class EpisodeSetup:
    """Hidden context and target for one sampled episode."""

    seed: int
    hidden_context: dict[str, dict[str, Any]]
    target_xy: np.ndarray


class EpisodeBuilder:
    """Resolve config-driven benchmark cases and per-episode samples."""

    def __init__(self, config: BenchmarkConfig, registry: BenchmarkRegistry) -> None:
        self.config = config
        self.registry = registry
        self._task_definition = registry.create("task_definitions", config.task.name)
        self._object_set = registry.create("object_sets", config.objects.set)
        self._phase1_config = FrankaHiddenPhysicsPickPlaceEnv.DEFAULT_CONFIG

    @property
    def task_definition(self):
        return self._task_definition

    def resolve_eval_cases(self) -> tuple[EvaluationCase, ...]:
        families = self._object_set.resolve_families(self.config.benchmark.family_split)
        variants = self._task_definition.resolve_variants()
        return tuple(EvaluationCase(object_family=family, task_variant=variant) for family in families for variant in variants)

    def camera_names(self) -> tuple[str, ...]:
        return tuple(rig.name for rig in self.config.cameras.rigs)

    def build_runtime(self, case: EvaluationCase):
        robot_profile = self.config.robot.profile_pool[0]
        return self.registry.create(
            "robot_profiles",
            robot_profile,
            object_family=case.object_family,
            task_variant=case.task_variant,
            camera_names=self.camera_names(),
        )

    def build_policy(self):
        return self.registry.create("policy_adapters", self.config.policy.adapter, **self.config.policy.kwargs)

    def sample_episode(self, *, episode_index: int, rng: np.random.Generator) -> EpisodeSetup:
        hidden_context, target_xy = build_episode_setup(
            pipeline_name=self.config.benchmark.name,
            split_name=self.config.benchmark.hidden_split,
            config=self._phase1_config,
            rng=rng,
        )
        return EpisodeSetup(
            seed=int(self.config.benchmark.seed + episode_index),
            hidden_context=hidden_context,
            target_xy=np.asarray(target_xy, dtype=float),
        )
