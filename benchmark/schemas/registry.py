from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmark.schemas.models import ActionPacket, BenchmarkResult, TraceEvent


@dataclass(frozen=True, slots=True)
class SchemaDefinition:
    """Schema registration metadata."""

    name: str
    model: type
    jsonschema_path: Path
    markdown_path: Path
    example_factory: Callable[[], dict[str, Any]]


_REPO_ROOT = Path(__file__).resolve().parents[2]


_SCHEMA_DEFINITIONS: tuple[SchemaDefinition, ...] = (
    SchemaDefinition(
        name="action-packet",
        model=ActionPacket,
        jsonschema_path=_REPO_ROOT / "contracts" / "jsonschema" / "action-packet.schema.json",
        markdown_path=_REPO_ROOT / "docs" / "schemas" / "action-packet.md",
        example_factory=lambda: ActionPacket(
            schema_id="cartesian_gripper_v1",
            arm={"mode": "delta_pose", "xyz": [0.0, 0.0, 0.01], "rpy": [0.0, 0.0, 0.0]},
            hand={"mode": "scalar_close", "value": 0.8},
            metadata={"source": "policy_adapter"},
        ).model_dump(mode="json"),
    ),
    SchemaDefinition(
        name="trace-event",
        model=TraceEvent,
        jsonschema_path=_REPO_ROOT / "contracts" / "jsonschema" / "trace-event.schema.json",
        markdown_path=_REPO_ROOT / "docs" / "schemas" / "trace-event.md",
        example_factory=lambda: TraceEvent(
            event_type="step",
            run_id="run_001",
            session_id="session_001",
            episode_id="ep_003",
            step_idx=27,
            sim_time=1.08,
            action_packet=ActionPacket(
                schema_id="cartesian_gripper_v1",
                arm={"mode": "delta_pose", "xyz": [0.0, 0.0, 0.01], "rpy": [0.0, 0.0, 0.0]},
                hand={"mode": "scalar_close", "value": 0.8},
                metadata={"source": "policy_adapter"},
            ),
            public_obs_summary={"has_overview": True},
            info_summary={"success": False, "slip": True},
            privileged_context={"body": {"reach_scale": 1.02}, "env": {"mass": 0.12}},
        ).model_dump(mode="json"),
    ),
    SchemaDefinition(
        name="benchmark-result",
        model=BenchmarkResult,
        jsonschema_path=_REPO_ROOT / "contracts" / "jsonschema" / "benchmark-result.schema.json",
        markdown_path=_REPO_ROOT / "docs" / "schemas" / "benchmark-result.md",
        example_factory=lambda: BenchmarkResult(
            run_id="run_001",
            benchmark_name="both_random_pick_place",
            policy_name="pi05_cpu",
            aggregate_metrics={"success_rate": 0.5},
            episode_rows=[{"success": 1.0, "object_family": "block"}],
            artifacts={"trace_jsonl": "benchmark_results/run_001/trace.jsonl"},
        ).model_dump(mode="json"),
    ),
)


def iter_schema_definitions() -> tuple[SchemaDefinition, ...]:
    """Return all registered schema definitions."""
    return _SCHEMA_DEFINITIONS


def get_schema_definition(name: str) -> SchemaDefinition:
    """Resolve one registered schema definition by name."""
    for definition in _SCHEMA_DEFINITIONS:
        if definition.name == name:
            return definition
    raise KeyError(f"Unknown schema definition '{name}'.")
