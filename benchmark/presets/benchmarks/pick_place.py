from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def _default_cameras() -> list[dict[str, Any]]:
    return [
        {
            "name": "overview",
            "mount_type": "world_fixed",
            "preset": "overview",
            "pose_mode": "preset",
            "width": 320,
            "height": 240,
        }
    ]


def _build_preset_payload(
    *,
    name: str,
    description: str,
    randomize_body: bool,
    randomize_env: bool,
) -> dict[str, Any]:
    return {
        "benchmark": {
            "name": name,
            "seed": 0,
            "episodes": 24,
            "output_dir": Path("benchmark_results") / name,
            "family_split": "all",
            "hidden_split": "unseen",
        },
        "runtime": {
            "backend": "mujoco",
            "max_steps": 0,
        },
        "robot": {
            "profile_pool": ["franka_panda_2f_v1"],
            "selection_strategy": "first",
            "randomization": {
                "enabled": randomize_body,
                "topology": {
                    "enabled": False,
                    "candidates": ["franka_panda_2f_v1"],
                },
                "parameters": {},
            },
        },
        "task": {
            "name": "pick_place",
        },
        "objects": {
            "set": "hidden_physics_blocks_v1",
            "randomization": {
                "enabled": randomize_env,
                "parameters": {},
            },
        },
        "cameras": {
            "rigs": _default_cameras(),
        },
        "policy": {
            "adapter": "random",
            "kwargs": {},
        },
        "tracing": {
            "sinks": [
                {
                    "type": "jsonl",
                    "include": [
                        "session.start",
                        "episode.start",
                        "step",
                        "episode.end",
                        "session.end",
                    ],
                    "include_privileged_context": False,
                }
            ]
        },
    }


_PRESETS: dict[str, dict[str, Any]] = {
    "normal_pick_place": _build_preset_payload(
        name="normal_pick_place",
        description="Nominal pick-and-place benchmark without robot or object randomization.",
        randomize_body=False,
        randomize_env=False,
    ),
    "body_random_pick_place": _build_preset_payload(
        name="body_random_pick_place",
        description="Pick-and-place benchmark with robot-body randomization enabled.",
        randomize_body=True,
        randomize_env=False,
    ),
    "object_random_pick_place": _build_preset_payload(
        name="object_random_pick_place",
        description="Pick-and-place benchmark with object hidden-physics randomization enabled.",
        randomize_body=False,
        randomize_env=True,
    ),
    "both_random_pick_place": _build_preset_payload(
        name="both_random_pick_place",
        description="Pick-and-place benchmark with robot and object randomization enabled.",
        randomize_body=True,
        randomize_env=True,
    ),
}

BENCHMARK_PRESET_NAMES = tuple(_PRESETS)


def get_benchmark_preset_payload(name: str) -> dict[str, Any]:
    """Return one built-in preset payload."""
    try:
        return deepcopy(_PRESETS[name])
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark preset '{name}'.") from exc
