from __future__ import annotations

from typing import Any

from pydantic import Field

from .action_packet import SchemaModel


class BenchmarkResult(SchemaModel):
    """Serialized benchmark result artifact."""

    run_id: str = Field(
        description="Stable run identifier.",
        json_schema_extra={
            "why": "Lets external tools associate metrics and traces with the same run.",
            "producer": "BenchmarkScheduler",
            "consumer": "ResultReader",
            "visibility": "public",
            "stability": "stable",
            "example": "run_001",
        },
    )
    benchmark_name: str = Field(
        description="Benchmark preset name.",
        json_schema_extra={
            "why": "Makes result artifacts self-describing when moved out of the repo.",
            "producer": "BenchmarkScheduler",
            "consumer": "ResultReader",
            "visibility": "public",
            "stability": "stable",
            "example": "both_random_pick_place",
        },
    )
    policy_name: str = Field(
        description="Resolved policy adapter name.",
        json_schema_extra={
            "why": "Supports cross-policy comparisons without out-of-band metadata.",
            "producer": "BenchmarkScheduler",
            "consumer": "ResultReader",
            "visibility": "public",
            "stability": "stable",
            "example": "pi05_cpu",
        },
    )
    aggregate_metrics: dict[str, float] = Field(
        description="Aggregate benchmark metrics.",
        json_schema_extra={
            "why": "Provides one stable summary object for dashboards and acceptance checks.",
            "producer": "BenchmarkScheduler",
            "consumer": "ResultReader",
            "visibility": "public",
            "stability": "stable",
            "example": {"success_rate": 0.5},
        },
    )
    episode_rows: list[dict[str, Any]] = Field(
        description="Per-episode metrics rows.",
        json_schema_extra={
            "why": "Retains the detailed evidence behind aggregate metrics.",
            "producer": "BenchmarkScheduler",
            "consumer": "ResultReader",
            "visibility": "public",
            "stability": "stable",
            "example": [{"success": 1.0, "object_family": "block"}],
        },
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Paths to emitted artifacts such as trace files or videos.",
        json_schema_extra={
            "why": "Avoids coupling result consumers to fixed output directory conventions.",
            "producer": "BenchmarkScheduler",
            "consumer": "ResultReader",
            "visibility": "public",
            "stability": "stable",
            "example": {"trace_jsonl": "benchmark_results/run_001/trace.jsonl"},
        },
    )
