from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ComponentRegistration(Generic[T]):
    """One registry entry."""

    name: str
    factory: Callable[..., T]
    description: str = ""


class ComponentRegistry(Generic[T]):
    """Simple named component registry."""

    def __init__(self) -> None:
        self._entries: dict[str, ComponentRegistration[T]] = {}

    def register(self, name: str, factory: Callable[..., T], description: str = "") -> None:
        if name in self._entries:
            raise KeyError(f"Registry entry '{name}' is already defined.")
        self._entries[name] = ComponentRegistration(name=name, factory=factory, description=description)

    def get(self, name: str) -> Callable[..., T]:
        try:
            return self._entries[name].factory
        except KeyError as exc:
            raise KeyError(f"Unknown registry entry '{name}'.") from exc

    def describe(self, name: str) -> ComponentRegistration[T]:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"Unknown registry entry '{name}'.") from exc

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        return self.get(name)(*args, **kwargs)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


class BenchmarkRegistry:
    """Grouped registries for all benchmark component categories."""

    CATEGORY_NAMES = (
        "presets",
        "robot_profiles",
        "task_definitions",
        "object_sets",
        "policy_adapters",
        "trace_sinks",
    )

    def __init__(self) -> None:
        self._categories: dict[str, ComponentRegistry[Any]] = {
            name: ComponentRegistry() for name in self.CATEGORY_NAMES
        }

    def category(self, name: str) -> ComponentRegistry[Any]:
        try:
            return self._categories[name]
        except KeyError as exc:
            raise KeyError(f"Unknown registry category '{name}'.") from exc

    def register(self, category: str, name: str, factory: Callable[..., Any], description: str = "") -> None:
        self.category(category).register(name, factory, description)

    def create(self, category: str, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.category(category).create(name, *args, **kwargs)

    def keys(self, category: str) -> tuple[str, ...]:
        return self.category(category).keys()


def create_default_registry() -> BenchmarkRegistry:
    """Build the default registry used by the benchmark scheduler."""
    from benchmark.components.objects.hidden_physics_blocks_v1 import HiddenPhysicsBlocksV1
    from benchmark.components.policies.base import RandomPolicyAdapter
    from benchmark.components.policies.lerobot_pi05 import LeRobotPI05PolicyAdapter
    from benchmark.components.policies.lerobot_smolvla import LeRobotSmolVLAAdapter
    from benchmark.components.robots.franka_panda_2f_v1 import build_phase1_franka_runtime
    from benchmark.components.tracing.sinks import JSONLTraceSink
    from benchmark.components.tasks.pick_place import PickPlaceTaskDefinition
    from benchmark.presets.benchmarks import BENCHMARK_PRESET_NAMES, get_benchmark_preset_payload

    registry = BenchmarkRegistry()
    for preset_name in BENCHMARK_PRESET_NAMES:
        registry.register(
            "presets",
            preset_name,
            lambda name=preset_name: get_benchmark_preset_payload(name),
            description=f"Built-in benchmark preset '{preset_name}'.",
        )
    registry.register(
        "robot_profiles",
        "franka_panda_2f_v1",
        build_phase1_franka_runtime,
        description="Phase 1 Franka Panda two-finger runtime wrapper.",
    )
    registry.register(
        "task_definitions",
        "pick_place",
        PickPlaceTaskDefinition,
        description="Pick-and-place task definition backed by phase1 task language.",
    )
    registry.register(
        "object_sets",
        "hidden_physics_blocks_v1",
        HiddenPhysicsBlocksV1,
        description="Phase 1 hidden-physics block object families.",
    )
    registry.register(
        "policy_adapters",
        "random",
        RandomPolicyAdapter,
        description="Random smoke-test policy adapter.",
    )
    registry.register(
        "policy_adapters",
        "lerobot.pi05",
        LeRobotPI05PolicyAdapter,
        description="LeRobot pi0.5 policy adapter.",
    )
    registry.register(
        "policy_adapters",
        "lerobot.smolvla",
        LeRobotSmolVLAAdapter,
        description="LeRobot SmolVLA policy adapter.",
    )
    registry.register(
        "trace_sinks",
        "jsonl",
        JSONLTraceSink,
        description="JSONL trace sink.",
    )
    return registry
