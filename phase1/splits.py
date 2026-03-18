from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


OBJECT_FAMILIES: tuple[str, ...] = ("block", "cylinder", "small_box")
TASK_VARIANTS: tuple[str, ...] = ("pick_place",)

SEEN_OBJECT_FAMILIES: tuple[str, ...] = ("block", "cylinder")
UNSEEN_OBJECT_FAMILIES: tuple[str, ...] = ("small_box",)
FAMILY_SPLITS = {
    "all": OBJECT_FAMILIES,
    "seen": SEEN_OBJECT_FAMILIES,
    "heldout": UNSEEN_OBJECT_FAMILIES,
}


@dataclass(frozen=True)
class ScalarInterval:
    """Describe one continuous interval for a hidden scalar parameter.

    Args:
        low: Inclusive lower bound of the interval.
        high: Inclusive upper bound of the interval.
    """

    low: float
    high: float

    @property
    def width(self) -> float:
        """Return the interval width used for weighted sampling."""
        return max(float(self.high - self.low), 1e-6)

    def sample(self, rng: np.random.Generator) -> float:
        """Sample one scalar value from the interval.

        Args:
            rng: Random-number generator used for sampling.
        """
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class DiscreteChoice:
    """Describe a categorical latent parameter support.

    Args:
        values: Candidate categorical values for the hidden parameter.
    """

    values: tuple[Any, ...]

    def sample(self, rng: np.random.Generator) -> Any:
        """Sample one categorical value.

        Args:
            rng: Random-number generator used for sampling.
        """
        return self.values[int(rng.integers(len(self.values)))]


def _sample_from_intervals(intervals: tuple[ScalarInterval, ...], rng: np.random.Generator) -> float:
    """Sample from a union of scalar intervals.

    Args:
        intervals: Candidate intervals that define the support of the latent parameter.
        rng: Random-number generator used for sampling.
    """
    widths = np.array([interval.width for interval in intervals], dtype=float)
    weights = widths / widths.sum()
    interval = intervals[int(rng.choice(len(intervals), p=weights))]
    return interval.sample(rng)


def _sample_parameter(support: tuple[ScalarInterval, ...] | DiscreteChoice, rng: np.random.Generator) -> Any:
    """Sample one hidden parameter from continuous or categorical support.

    Args:
        support: Parameter support specification.
        rng: Random-number generator used for sampling.
    """
    if isinstance(support, DiscreteChoice):
        return support.sample(rng)
    return _sample_from_intervals(support, rng)


FULL_SPLIT = {
    "body": {
        "reach_scale": (ScalarInterval(0.93, 1.07),),
        "arm_mass_scale": (ScalarInterval(0.80, 1.20),),
        "payload_scale": (ScalarInterval(0.80, 1.35),),
        "joint_damping_scale": (ScalarInterval(0.70, 1.30),),
        "actuator_gain_scale": (ScalarInterval(0.75, 1.25),),
        "fingertip_friction_scale": (ScalarInterval(0.50, 1.50),),
        "damage_joint_index": DiscreteChoice((0, 1, 2, 3, 4, 5, 6)),
        "damage_gain_scale": (ScalarInterval(0.45, 1.00),),
        "damage_damping_scale": (ScalarInterval(1.00, 2.40),),
        "local_finger_wear_side": DiscreteChoice(("left", "right")),
        "local_finger_friction_scale": (ScalarInterval(0.45, 1.05),),
    },
    "env": {
        "mass": (ScalarInterval(0.06, 0.18),),
        "friction": (ScalarInterval(0.35, 0.95),),
        "stiffness": (ScalarInterval(0.55, 0.95),),
    },
}

SEEN_SPLIT = {
    "body": {
        "reach_scale": (ScalarInterval(0.97, 1.03),),
        "arm_mass_scale": (ScalarInterval(0.90, 1.10),),
        "payload_scale": (ScalarInterval(0.92, 1.12),),
        "joint_damping_scale": (ScalarInterval(0.85, 1.15),),
        "actuator_gain_scale": (ScalarInterval(0.88, 1.12),),
        "fingertip_friction_scale": (ScalarInterval(0.70, 1.30),),
        "damage_joint_index": DiscreteChoice((0, 1, 2, 3, 4, 5, 6)),
        "damage_gain_scale": (ScalarInterval(0.72, 1.00),),
        "damage_damping_scale": (ScalarInterval(1.00, 1.55),),
        "local_finger_wear_side": DiscreteChoice(("left", "right")),
        "local_finger_friction_scale": (ScalarInterval(0.68, 1.00),),
    },
    "env": {
        "mass": (ScalarInterval(0.085, 0.155),),
        "friction": (ScalarInterval(0.47, 0.83),),
        "stiffness": (ScalarInterval(0.63, 0.87),),
    },
}

UNSEEN_SPLIT = {
    "body": {
        "reach_scale": (ScalarInterval(0.93, 0.97), ScalarInterval(1.03, 1.07)),
        "arm_mass_scale": (ScalarInterval(0.80, 0.90), ScalarInterval(1.10, 1.20)),
        "payload_scale": (ScalarInterval(0.80, 0.92), ScalarInterval(1.12, 1.35)),
        "joint_damping_scale": (ScalarInterval(0.70, 0.85), ScalarInterval(1.15, 1.30)),
        "actuator_gain_scale": (ScalarInterval(0.75, 0.88), ScalarInterval(1.12, 1.25)),
        "fingertip_friction_scale": (ScalarInterval(0.50, 0.70), ScalarInterval(1.30, 1.50)),
        "damage_joint_index": DiscreteChoice((0, 1, 2, 3, 4, 5, 6)),
        "damage_gain_scale": (ScalarInterval(0.45, 0.72),),
        "damage_damping_scale": (ScalarInterval(1.55, 2.40),),
        "local_finger_wear_side": DiscreteChoice(("left", "right")),
        "local_finger_friction_scale": (ScalarInterval(0.45, 0.68),),
    },
    "env": {
        "mass": (ScalarInterval(0.06, 0.085), ScalarInterval(0.155, 0.18)),
        "friction": (ScalarInterval(0.35, 0.47), ScalarInterval(0.83, 0.95)),
        "stiffness": (ScalarInterval(0.55, 0.63), ScalarInterval(0.87, 0.95)),
    },
}

SPLITS = {
    "seen": SEEN_SPLIT,
    "unseen": UNSEEN_SPLIT,
    "full": FULL_SPLIT,
}


def sample_hidden_context(split_name: str, rng: np.random.Generator) -> dict[str, dict[str, Any]]:
    """Sample one hidden body+environment context from a split distribution.

    Args:
        split_name: Name of the split distribution to sample from.
        rng: Random-number generator used for sampling.
    """
    split = SPLITS[split_name]
    body = {
        name: _sample_parameter(support, rng)
        for name, support in split["body"].items()
    }
    env = {
        name: _sample_parameter(support, rng)
        for name, support in split["env"].items()
    }
    return {"body": body, "env": env}


def enumerate_eval_cases() -> tuple[tuple[str, str], ...]:
    """Enumerate all object-family and task-variant evaluation pairs."""
    return tuple(product(OBJECT_FAMILIES, TASK_VARIANTS))


def resolve_object_families(family_split: str) -> tuple[str, ...]:
    """Resolve the object-family subset used by a family split.

    Args:
        family_split: Family split mode such as `all`, `seen`, or `heldout`.
    """
    try:
        return FAMILY_SPLITS[family_split]
    except KeyError as exc:
        raise ValueError(f"Unknown family split: {family_split}") from exc
