from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SchemaModel(BaseModel):
    """Base model for exported benchmark contracts."""

    model_config = ConfigDict(extra="forbid")


class ArmCommand(SchemaModel):
    """Arm-control sub-command inside one action packet."""

    mode: Literal["delta_pose", "absolute_pose"] = Field(
        description="Arm control mode.",
        json_schema_extra={
            "why": "Separates delta and absolute Cartesian commands without changing the scheduler loop.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "stable",
            "example": "delta_pose",
        },
    )
    xyz: list[float] = Field(
        min_length=3,
        max_length=3,
        description="Cartesian translation command in meters.",
        json_schema_extra={
            "why": "The canonical runtime contract always carries a Cartesian target or delta.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "stable",
            "example": [0.0, 0.0, 0.01],
        },
    )
    rpy: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="XYZ Euler orientation command in radians.",
        json_schema_extra={
            "why": "Allows wrist orientation control without forcing all adapters to use quaternions.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "stable",
            "example": [0.0, 0.0, 0.0],
        },
    )
    quat: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Quaternion orientation command when absolute pose is used.",
        json_schema_extra={
            "why": "Keeps compatibility with policies that emit absolute orientation as quaternions.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "experimental",
            "example": [1.0, 0.0, 0.0, 0.0],
        },
    )

    @model_validator(mode="after")
    def validate_orientation(self) -> "ArmCommand":
        if self.mode == "absolute_pose" and self.rpy is None and self.quat is None:
            raise ValueError("absolute_pose arm commands require `rpy` or `quat`.")
        if self.mode == "delta_pose" and self.rpy is None:
            self.rpy = [0.0, 0.0, 0.0]
        return self


class HandCommand(SchemaModel):
    """Hand-control sub-command inside one action packet."""

    mode: Literal["scalar_close", "finger_vector"] = Field(
        description="Hand control mode.",
        json_schema_extra={
            "why": "Decouples different gripper topologies from the scheduler.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "stable",
            "example": "scalar_close",
        },
    )
    value: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Single close command in the unit interval.",
        json_schema_extra={
            "why": "Supports current two-finger policies with one stable scalar interface.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "stable",
            "example": 0.8,
        },
    )
    values: list[float] | None = Field(
        default=None,
        description="Per-finger close commands in the unit interval.",
        json_schema_extra={
            "why": "Pre-reserves a vector hand command for non-two-finger topologies.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "experimental",
            "example": [0.75, 0.75, 0.75],
        },
    )

    @model_validator(mode="after")
    def validate_values(self) -> "HandCommand":
        if self.mode == "scalar_close" and self.value is None:
            raise ValueError("scalar_close hand commands require `value`.")
        if self.mode == "finger_vector":
            if not self.values:
                raise ValueError("finger_vector hand commands require `values`.")
            if any(value < 0.0 or value > 1.0 for value in self.values):
                raise ValueError("finger_vector values must stay in [0, 1].")
        return self


class ActionPacket(SchemaModel):
    """Canonical action packet shared by all policy adapters."""

    schema_id: str = Field(
        description="Control schema identifier.",
        json_schema_extra={
            "why": "Allows policy adapters and robot profiles to negotiate a stable control contract.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "stable",
            "example": "cartesian_gripper_v1",
        },
    )
    arm: ArmCommand = Field(
        description="Arm-control sub-command.",
        json_schema_extra={
            "why": "Keeps arm semantics explicit instead of hiding them in a flat action vector.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "stable",
            "example": {"mode": "delta_pose", "xyz": [0.0, 0.0, 0.01], "rpy": [0.0, 0.0, 0.0]},
        },
    )
    hand: HandCommand = Field(
        description="Hand-control sub-command.",
        json_schema_extra={
            "why": "Lets different gripper topologies evolve without changing the scheduler loop.",
            "producer": "PolicyAdapter",
            "consumer": "ControlSchemaAdapter",
            "visibility": "public",
            "stability": "stable",
            "example": {"mode": "scalar_close", "value": 0.8},
        },
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional producer metadata for debugging and compatibility tracing.",
        json_schema_extra={
            "why": "Captures adapter provenance without polluting control semantics.",
            "producer": "PolicyAdapter",
            "consumer": "TraceSink",
            "visibility": "public",
            "stability": "stable",
            "example": {"source": "pi05"},
        },
    )

    @model_validator(mode="after")
    def validate_schema_id(self) -> "ActionPacket":
        if not self.schema_id.strip():
            raise ValueError("schema_id must be a non-empty string.")
        return self
