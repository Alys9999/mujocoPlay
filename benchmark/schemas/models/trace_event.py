from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .action_packet import ActionPacket, SchemaModel


class TraceEvent(SchemaModel):
    """Trace event emitted by the scheduler and runtime."""

    event_type: Literal["session.start", "episode.start", "step", "episode.end", "session.end"] = Field(
        description="Stable event type.",
        json_schema_extra={
            "why": "Allows sinks to filter events without coupling to scheduler branches.",
            "producer": "BenchmarkScheduler",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": "step",
        },
    )
    run_id: str = Field(
        description="Stable run identifier.",
        json_schema_extra={
            "why": "Ties all session and episode events to one benchmark invocation.",
            "producer": "SessionManager",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": "run_001",
        },
    )
    session_id: str = Field(
        description="Session identifier for the active scheduler run.",
        json_schema_extra={
            "why": "Allows later multi-session scheduling without changing sink format.",
            "producer": "SessionManager",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": "session_001",
        },
    )
    episode_id: str | None = Field(
        default=None,
        description="Episode identifier when the event is episode-scoped.",
        json_schema_extra={
            "why": "Step and episode events need a stable link back to the episode lifecycle.",
            "producer": "BenchmarkScheduler",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": "ep_003",
        },
    )
    step_idx: int | None = Field(
        default=None,
        ge=0,
        description="Step index for step events.",
        json_schema_extra={
            "why": "Supports step-level replay and validation without parsing info payloads.",
            "producer": "SceneRuntime",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": 27,
        },
    )
    sim_time: float | None = Field(
        default=None,
        ge=0.0,
        description="Simulation time in seconds.",
        json_schema_extra={
            "why": "Makes time-based debugging independent from fixed control step counts.",
            "producer": "SceneRuntime",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": 1.08,
        },
    )
    action_packet: ActionPacket | None = Field(
        default=None,
        description="Canonical action packet for step events.",
        json_schema_extra={
            "why": "Preserves the exact control command that led to a transition.",
            "producer": "PolicyAdapter",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": {
                "schema_id": "cartesian_gripper_v1",
                "arm": {"mode": "delta_pose", "xyz": [0.0, 0.0, 0.01], "rpy": [0.0, 0.0, 0.0]},
                "hand": {"mode": "scalar_close", "value": 0.8},
                "metadata": {"source": "pi05"},
            },
        },
    )
    public_obs_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Public-observation summary without raw image arrays.",
        json_schema_extra={
            "why": "Lets sinks inspect observation availability while avoiding large step payloads.",
            "producer": "BenchmarkScheduler",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": {"has_overview": True},
        },
    )
    info_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Semantic runtime info summary.",
        json_schema_extra={
            "why": "Captures success and failure state without leaking full internal runtime objects.",
            "producer": "SceneRuntime",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": {"success": False, "slip": True},
        },
    )
    privileged_context: dict[str, Any] | None = Field(
        default=None,
        description="Optional hidden context reserved for privileged sinks.",
        json_schema_extra={
            "why": "Keeps hidden labels out of the public observation while still allowing offline analysis.",
            "producer": "SceneRuntime",
            "consumer": "TraceSink",
            "visibility": "privileged",
            "stability": "stable",
            "example": {"body": {"reach_scale": 1.02}, "env": {"mass": 0.12}},
        },
    )
