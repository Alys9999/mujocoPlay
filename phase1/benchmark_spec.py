from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectFamilySpec:
    """Store reference grasp geometry for one object family.
    
    Args:
        name: Value for the name used by this routine.
        rest_height: Height used for rest.
        approach_height: Height used for approach.
        grasp_height: Height used for grasp.
        release_height: Height used for release.
    """
    name: str
    rest_height: float
    approach_height: float
    grasp_height: float
    release_height: float


@dataclass(frozen=True)
class TaskVariantSpec:
    """Store task tolerances and nominal transport geometry for one task variant.
    
    Args:
        name: Value for the name used by this routine.
        task_mode: High-level task family. This trimmed benchmark only uses `pick_place`.
        instruction: Natural-language instruction shown to benchmarked policies.
        target_radius: Value for the target radius used by this routine.
        lift_height: Height used for lift.
        transport_height: Height used for transport.
        place_height: Height used for place.
        success_height: Height threshold used by success checks.
        grasp_hold_steps: Number of steps to use for grasp hold.
        settle_hold_steps: Number of steps to use for settle hold.
        release_hold_steps: Number of steps to use for release hold.
    """
    name: str
    task_mode: str
    instruction: str
    target_radius: float
    lift_height: float
    transport_height: float
    place_height: float
    success_height: float
    grasp_hold_steps: int
    settle_hold_steps: int
    release_hold_steps: int


OBJECT_FAMILY_SPECS: dict[str, ObjectFamilySpec] = {
    "block": ObjectFamilySpec(
        name="block",
        rest_height=0.015,
        approach_height=0.080,
        grasp_height=0.030,
        release_height=0.090,
    ),
    "cylinder": ObjectFamilySpec(
        name="cylinder",
        rest_height=0.018,
        approach_height=0.085,
        grasp_height=0.032,
        release_height=0.095,
    ),
    "small_box": ObjectFamilySpec(
        name="small_box",
        rest_height=0.010,
        approach_height=0.078,
        grasp_height=0.030,
        release_height=0.088,
    ),
}


TASK_VARIANT_SPECS: dict[str, TaskVariantSpec] = {
    "pick_place": TaskVariantSpec(
        name="pick_place",
        task_mode="pick_place",
        instruction="Pick the object and place it in the green target.",
        target_radius=0.050,
        lift_height=0.120,
        transport_height=0.120,
        place_height=0.050,
        success_height=0.105,
        grasp_hold_steps=28,
        settle_hold_steps=12,
        release_hold_steps=20,
    ),
}


def get_object_family_spec(name: str) -> ObjectFamilySpec:
    """Return object family spec.
    
    Args:
        name: Value for the name used by this routine.
    """
    try:
        return OBJECT_FAMILY_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown object family: {name}") from exc


def get_task_variant_spec(name: str) -> TaskVariantSpec:
    """Return task variant spec.
    
    Args:
        name: Value for the name used by this routine.
    """
    try:
        return TASK_VARIANT_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown task variant: {name}") from exc
