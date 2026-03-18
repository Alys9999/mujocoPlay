from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

import mujoco
import numpy as np

from .adaptation_env import FrankaLatentAdaptationEnv
from .benchmark_spec import OBJECT_FAMILY_SPECS, TASK_VARIANT_SPECS
from .cli_utils import parse_mapping_arg
from .pi05_policy import PI05SequentialPolicy
from .pipeline_config import (
    DEFAULT_PIPELINE_CONFIG_PATH,
    PIPELINE_SPECS,
    build_episode_setup,
    load_pipeline_name,
    resolve_pipeline_spec,
)
from .splits import SPLITS, resolve_object_families
from .task_language import build_instruction


@dataclass
class FailureEvent:
    """Store one detected failure event.

    Args:
        type: Failure type label.
        onset_step: Step at which the event first appeared.
        onset_time_sec: Simulation time at which the event first appeared.
        recovered_step: Step at which the event was recovered, if any.
        recovered_time_sec: Simulation time at which the event was recovered, if any.
    """

    type: str
    onset_step: int
    onset_time_sec: float
    recovered_step: int | None = None
    recovered_time_sec: float | None = None


class EpisodeEventTracker:
    """Track benchmark failure events over the course of one episode."""

    def __init__(self) -> None:
        """Initialize EpisodeEventTracker."""
        self.reset()

    def reset(self) -> None:
        """Reset the event tracker for a fresh episode."""
        self.events: list[FailureEvent] = []
        self._seen_nonrecoverable: set[str] = set()
        self._active_recoverable: dict[str, FailureEvent] = {}

    def observe(self, info: dict[str, Any]) -> None:
        """Update failure-event state from the latest simulator info.

        Args:
            info: Semantic episode info produced by the environment.
        """
        step = int(info["step_count"])
        time_sec = float(info["episode_duration_sec"])
        if info.get("object_broken", False) and "break" not in self._seen_nonrecoverable:
            self.events.append(FailureEvent("break", step, time_sec))
            self._seen_nonrecoverable.add("break")

        if info.get("object_dropped", False) and "drop" not in self._active_recoverable:
            event = FailureEvent("drop", step, time_sec)
            self.events.append(event)
            self._active_recoverable["drop"] = event
        elif (
            info.get("slip_detected", False)
            and not info.get("object_dropped", False)
            and "slip" not in self._active_recoverable
        ):
            event = FailureEvent("slip", step, time_sec)
            self.events.append(event)
            self._active_recoverable["slip"] = event

        if self._active_recoverable and (info.get("object_in_gripper", False) or info.get("success", False)):
            for key, event in list(self._active_recoverable.items()):
                if event.recovered_step is None:
                    event.recovered_step = step
                    event.recovered_time_sec = time_sec
                del self._active_recoverable[key]

    def finalize(self, info: dict[str, Any]) -> None:
        """Add a terminal failure label when the episode ends unsuccessfully.

        Args:
            info: Final semantic episode info produced by the environment.
        """
        if info.get("success", False):
            return
        event_types = {event.type for event in self.events}
        step = int(info["step_count"])
        time_sec = float(info["episode_duration_sec"])
        if not info.get("ever_lifted", False) and "break" not in event_types:
            self.events.append(FailureEvent("grasp_miss", step, time_sec))
            return
        if not any(name in event_types for name in ("break", "drop", "slip")):
            self.events.append(FailureEvent("placement_failure", step, time_sec))


class SequentialPolicy(Protocol):
    """Protocol that all benchmarked sequential policies must satisfy."""

    name: str
    requires_image: bool
    benchmark_interface: str

    def reset(self) -> None:
        """Reset any policy-side recurrent state before a new episode."""

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        """Return the next control action for the active benchmark interface."""


class RandomPolicy:
    """Simple random baseline for smoke tests and environment sanity checks.

    Args:
        seed: Random seed used for policy sampling.
        action_scale: Scale applied to the translational action components.
        close_bias: Mean close command before clipping.
    """

    name = "random"
    requires_image = False
    benchmark_interface = "default"

    def __init__(self, seed: int = 0, action_scale: float = 1.0, close_bias: float = 0.5) -> None:
        """Initialize RandomPolicy.

        Args:
            seed: Random seed used for policy sampling.
            action_scale: Scale applied to the translational action components.
            close_bias: Mean close command before clipping.
        """
        self._seed = int(seed)
        self._action_scale = float(action_scale)
        self._close_bias = float(close_bias)
        self._rng = np.random.default_rng(self._seed)

    def reset(self) -> None:
        """Reset the random-number generator for a fresh episode."""
        self._rng = np.random.default_rng(self._seed)

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a random 4D action.

        Args:
            observation: Current non-privileged observation available to the policy.
        """
        del observation
        action = self._rng.uniform(-1.0, 1.0, size=4).astype(float)
        action[:3] *= self._action_scale
        action[3] = float(np.clip(self._close_bias + 0.35 * self._rng.standard_normal(), 0.0, 1.0))
        return action


class AdaptiveBenchmarkSession:
    """Manage one sequential benchmark rollout around a latent adaptation environment.

    Args:
        object_family: Object family to simulate.
        task_variant: Task variant to simulate.
        render_width: Width of the optional overview render.
        render_height: Height of the optional overview render.
    """

    def __init__(
        self,
        object_family: str,
        task_variant: str,
        render_width: int = 320,
        render_height: int = 240,
    ) -> None:
        """Initialize AdaptiveBenchmarkSession.

        Args:
            object_family: Object family to simulate.
            task_variant: Task variant to simulate.
            render_width: Width of the optional overview render.
            render_height: Height of the optional overview render.
        """
        self.env = FrankaLatentAdaptationEnv(object_family=object_family, task_variant=task_variant)
        self.renderer = mujoco.Renderer(self.env.model, width=render_width, height=render_height)
        self.event_tracker = EpisodeEventTracker()
        self._instruction = build_instruction(object_family, task_variant)
        self._latest_obs: dict[str, np.ndarray] | None = None
        self._latest_info: dict[str, Any] | None = None
        self._previous_action = np.zeros(4, dtype=float)
        self._action_history: list[np.ndarray] = []
        self._info_history: list[dict[str, Any]] = []
        self._first_contact_step: int | None = None
        self._first_lift_step: int | None = None
        self._success_step: int | None = None
        self._initial_gripper_object_distance = 0.0
        self._min_gripper_object_distance = 0.0
        self.env.set_step_callback(self._on_step)

    def close(self) -> None:
        """Release renderer resources."""
        self.renderer.close()

    def reset(
        self,
        seed: int | None = None,
        hidden_context: dict[str, dict[str, Any]] | None = None,
        target_xy: np.ndarray | None = None,
        include_image: bool = False,
    ) -> dict[str, Any]:
        """Reset the session for a new hidden context.

        Args:
            seed: Random seed for reproducible sampling or resets.
            hidden_context: Optional hidden body/env context override.
            target_xy: Optional XY target override for the episode.
            include_image: Whether to include the overview image in the returned observation.
        """
        self.event_tracker.reset()
        self._previous_action = np.zeros(4, dtype=float)
        self._action_history = []
        self._info_history = []
        self._first_contact_step = None
        self._first_lift_step = None
        self._success_step = None
        obs, info = self.env.reset(seed=seed, hidden_context=hidden_context, target_xy=target_xy)
        self._latest_obs = obs
        self._latest_info = info
        self._initial_gripper_object_distance = float(info.get("gripper_object_distance", 0.0))
        self._min_gripper_object_distance = self._initial_gripper_object_distance
        return self.get_public_observation(include_image=include_image)

    def step(
        self,
        action: np.ndarray,
        include_image: bool = False,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the session by one low-level control step.

        Args:
            action: Policy action to apply in the environment.
            include_image: Whether to include the overview image in the returned observation.
        """
        action = np.asarray(action, dtype=float).reshape(4)
        self._previous_action = np.array(action, copy=True)
        self._action_history.append(np.array(action, copy=True))
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._latest_obs = obs
        self._latest_info = info
        public_obs = self.get_public_observation(include_image=include_image)
        return public_obs, reward, terminated, truncated, info

    def render_overview(self) -> np.ndarray:
        """Render the overview camera image."""
        self.renderer.update_scene(self.env.data, camera="overview")
        return self.renderer.render().copy()

    def get_public_observation(self, include_image: bool = False) -> dict[str, Any]:
        """Build the non-privileged observation exposed to benchmarked policies.

        Args:
            include_image: Whether to include the overview image in the observation.
        """
        if self._latest_obs is None or self._latest_info is None:
            raise RuntimeError("Call reset() before requesting observations.")
        observation = {
            "arm_qpos": np.array(self._latest_obs["arm_qpos"], copy=True),
            "gripper_pos": np.array(self._latest_obs["gripper_pos"], copy=True),
            "ee_quat": np.array(self._latest_obs["ee_quat"], copy=True),
            "gripper_aperture": np.array(self._latest_obs["gripper_aperture"], copy=True),
            "wrist_force": np.array(self._latest_obs["wrist_force"], copy=True),
            "wrist_torque": np.array(self._latest_obs["wrist_torque"], copy=True),
            "target_pos": np.array(self._latest_obs["target_pos"], copy=True),
            "mocap_target": np.array(self._latest_obs["mocap_target"], copy=True),
            "time_sec": float(self._latest_info["episode_duration_sec"]),
            "step_count": int(self._latest_info["step_count"]),
            "instruction": self._instruction,
            "object_family": self.env.object_family,
            "task_variant": self.env.task_variant,
            "previous_action": np.array(self._previous_action, copy=True),
        }
        if include_image:
            observation["overview_rgb"] = self.render_overview()
        return observation

    def summarize_episode(self) -> dict[str, Any]:
        """Summarize the completed episode into benchmark metrics."""
        if self._latest_info is None:
            raise RuntimeError("No episode info available.")
        self.event_tracker.finalize(self._latest_info)
        success = bool(self._latest_info["success"])
        first_attempt_success = bool(success and not self.event_tracker.events)
        recoverable = [event for event in self.event_tracker.events if event.type in {"slip", "drop"}]
        recovered = [event for event in recoverable if event.recovered_time_sec is not None]
        recovered_success = bool(success and recovered)
        recovery_times = [
            float(event.recovered_time_sec - event.onset_time_sec)
            for event in recovered
            if event.recovered_time_sec is not None
        ]
        event_counts = {name: 0 for name in ("break", "drop", "slip", "grasp_miss", "placement_failure")}
        for event in self.event_tracker.events:
            if event.type in event_counts:
                event_counts[event.type] += 1

        mean_close = mean(float(action[3]) for action in self._action_history) if self._action_history else 0.0
        mean_motion = mean(float(np.linalg.norm(action[:3])) for action in self._action_history) if self._action_history else 0.0
        adaptation_horizon = int(self._latest_info["adaptation_success_horizon_steps"])
        success_step = self._success_step or int(self._latest_info["step_count"])
        return {
            "success": float(success),
            "first_attempt_success": float(first_attempt_success),
            "recovered_success": float(recovered_success),
            "adapted_success": float(success and success_step <= adaptation_horizon),
            "failure_rate": float(bool(self.event_tracker.events)),
            "failure_count": int(len(self.event_tracker.events)),
            "first_failure_type": self.event_tracker.events[0].type if self.event_tracker.events else "none",
            "recovery_rate": float(len(recovered) / len(recoverable)) if recoverable else 0.0,
            "mean_recovery_time_sec": float(mean(recovery_times)) if recovery_times else 0.0,
            "lift_rate": float(self._latest_info.get("ever_lifted", False)),
            "break_rate": float(bool(event_counts["break"])),
            "drop_rate": float(bool(event_counts["drop"])),
            "slip_rate": float(bool(event_counts["slip"])),
            "grasp_miss_rate": float(bool(event_counts["grasp_miss"])),
            "placement_failure_rate": float(bool(event_counts["placement_failure"])),
            "placement_error": float(self._latest_info["placement_error"]),
            "initial_gripper_object_distance": float(self._initial_gripper_object_distance),
            "final_gripper_object_distance": float(self._latest_info.get("gripper_object_distance", 0.0)),
            "min_gripper_object_distance": float(self._min_gripper_object_distance),
            "episode_duration_sec": float(self._latest_info["episode_duration_sec"]),
            "step_count": int(self._latest_info["step_count"]),
            "success_step": int(success_step),
            "first_contact_step": int(self._first_contact_step or -1),
            "first_lift_step": int(self._first_lift_step or -1),
            "mean_close_command": float(mean_close),
            "mean_motion_command": float(mean_motion),
            "events": [asdict(event) for event in self.event_tracker.events],
            "object_family": self.env.object_family,
            "task_variant": self.env.task_variant,
        }

    def _on_step(self, _env, obs: dict[str, np.ndarray], info: dict[str, Any]) -> None:
        """Capture per-step benchmark state.

        Args:
            _env: Unused environment instance supplied by the step callback.
            obs: Full internal environment observation.
            info: Semantic episode info produced by the environment.
        """
        del _env
        self._latest_obs = obs
        self._latest_info = info
        self._info_history.append(dict(info))
        self.event_tracker.observe(info)
        self._min_gripper_object_distance = min(
            float(self._min_gripper_object_distance),
            float(info.get("gripper_object_distance", self._min_gripper_object_distance)),
        )
        if self._first_contact_step is None and (info.get("left_contact", False) or info.get("right_contact", False)):
            self._first_contact_step = int(info["step_count"])
        if self._first_lift_step is None and info.get("ever_lifted", False):
            self._first_lift_step = int(info["step_count"])
        if self._success_step is None and info.get("success", False):
            self._success_step = int(info["step_count"])


def build_policy(name: str, kwargs: dict[str, Any], seed: int) -> SequentialPolicy:
    """Build a benchmark policy from a builtin name or `module:Class` path.

    Args:
        name: Builtin policy name or `module:Class` import path.
        kwargs: Keyword arguments passed to the policy constructor.
        seed: Random seed for policies that support stochastic sampling.
    """
    if name == "random":
        return RandomPolicy(seed=seed, **kwargs)
    if ":" not in name:
        raise ValueError(f"Unknown policy spec: {name}. Use 'random' or 'package.module:ClassName'.")
    module_name, class_name = name.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    policy_cls = getattr(module, class_name)
    policy = policy_cls(**kwargs)
    if not hasattr(policy, "name"):
        policy.name = class_name
    if not hasattr(policy, "requires_image"):
        policy.requires_image = False
    if not hasattr(policy, "benchmark_interface"):
        policy.benchmark_interface = "default"
    return policy


def make_benchmark_session(
    object_family: str,
    task_variant: str,
    interface_name: str = "default",
    policy: SequentialPolicy | None = None,
):
    """Instantiate the benchmark session that matches the requested interface.

    Args:
        object_family: Object family to simulate.
        task_variant: Task variant to simulate.
        interface_name: Benchmark interface name. This trimmed repo only uses `default`.
        policy: Optional policy instance whose interface-specific metadata should be used.
    """
    if interface_name == "default":
        return AdaptiveBenchmarkSession(object_family=object_family, task_variant=task_variant)
    raise ValueError(f"Unknown benchmark interface: {interface_name}")


def rollout_policy(
    policy: SequentialPolicy,
    session: AdaptiveBenchmarkSession,
    seed: int,
    hidden_context: dict[str, dict[str, Any]],
    target_xy: np.ndarray,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Roll out one sequential policy episode.

    Args:
        policy: Policy instance to evaluate.
        session: Benchmark session that owns the environment and renderer.
        seed: Random seed for the episode reset.
        hidden_context: Hidden body/env context applied to the episode.
        target_xy: Target XY location applied to the episode.
        max_steps: Optional benchmark-side step cap used for smoke tests.
    """
    policy.reset()
    observation = session.reset(
        seed=seed,
        hidden_context=hidden_context,
        target_xy=target_xy,
        include_image=policy.requires_image,
    )
    terminated = False
    truncated = False
    external_step_cap_reached = False
    step_limit = None if max_steps is None or int(max_steps) <= 0 else int(max_steps)
    steps_taken = 0
    while not (terminated or truncated):
        if step_limit is not None and steps_taken >= step_limit:
            external_step_cap_reached = True
            truncated = True
            break
        action = np.asarray(policy.act(observation), dtype=float)
        next_observation, reward, terminated, truncated, info = session.step(
            action,
            include_image=policy.requires_image,
        )
        observe_transition = getattr(policy, "observe_transition", None)
        if callable(observe_transition):
            observe_transition(observation, action, reward, next_observation, terminated, truncated, info)
        observation = next_observation
        steps_taken += 1
    summary = session.summarize_episode()
    summary["benchmark_step_cap_reached"] = bool(external_step_cap_reached)
    summary["benchmark_max_steps"] = int(step_limit) if step_limit is not None else -1
    return summary


def evaluate_policy(
    policy: SequentialPolicy,
    family: str,
    task: str,
    split_name: str,
    pipeline_name: str,
    seed: int,
    episodes: int,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one sequential policy over the selected latent split.

    Args:
        policy: Policy instance to evaluate.
        family: Object family to simulate.
        task: Task variant to simulate.
        split_name: Name of the latent split to evaluate.
        pipeline_name: Name of the benchmark pipeline to evaluate.
        seed: Base random seed for deterministic episode construction.
        episodes: Number of episodes to sample from the split distribution.
        max_steps: Optional benchmark-side step cap used for smoke tests.
    """
    session = make_benchmark_session(
        object_family=family,
        task_variant=task,
        interface_name=getattr(policy, "benchmark_interface", "default"),
        policy=policy,
    )
    rows: list[dict[str, Any]] = []
    try:
        if split_name not in SPLITS:
            raise ValueError(f"Unknown split: {split_name}")
        rng = np.random.default_rng(seed)
        for episode_idx in range(episodes):
            hidden_context, target_xy = build_episode_setup(
                pipeline_name=pipeline_name,
                split_name=split_name,
                config=session.env.config,
                rng=rng,
            )
            row = rollout_policy(
                policy=policy,
                session=session,
                seed=seed + episode_idx,
                hidden_context=hidden_context,
                target_xy=target_xy,
                max_steps=max_steps,
            )
            row["policy"] = policy.name
            row["split"] = split_name
            row["pipeline"] = pipeline_name
            rows.append(row)
    finally:
        session.close()
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode rows into benchmark summary metrics.

    Args:
        rows: Episode rows collected for one family-task-policy setting.
    """
    return {
        "episodes": len(rows),
        "success_rate": mean(float(row["success"]) for row in rows),
        "first_attempt_success_rate": mean(float(row["first_attempt_success"]) for row in rows),
        "recovered_success_rate": mean(float(row["recovered_success"]) for row in rows),
        "adapted_success_rate": mean(float(row["adapted_success"]) for row in rows),
        "failure_rate": mean(float(row["failure_rate"]) for row in rows),
        "mean_failure_count": mean(float(row["failure_count"]) for row in rows),
        "recovery_rate": mean(float(row["recovery_rate"]) for row in rows),
        "mean_recovery_time_sec": mean(float(row["mean_recovery_time_sec"]) for row in rows),
        "lift_rate": mean(float(row["lift_rate"]) for row in rows),
        "break_rate": mean(float(row["break_rate"]) for row in rows),
        "drop_rate": mean(float(row["drop_rate"]) for row in rows),
        "slip_rate": mean(float(row["slip_rate"]) for row in rows),
        "grasp_miss_rate": mean(float(row["grasp_miss_rate"]) for row in rows),
        "placement_failure_rate": mean(float(row["placement_failure_rate"]) for row in rows),
        "mean_placement_error": mean(float(row["placement_error"]) for row in rows),
        "mean_initial_gripper_object_distance": mean(float(row["initial_gripper_object_distance"]) for row in rows),
        "mean_final_gripper_object_distance": mean(float(row["final_gripper_object_distance"]) for row in rows),
        "mean_min_gripper_object_distance": mean(float(row["min_gripper_object_distance"]) for row in rows),
        "mean_duration_sec": mean(float(row["episode_duration_sec"]) for row in rows),
        "mean_success_step": mean(float(row["success_step"]) for row in rows),
        "mean_first_contact_step": mean(float(row["first_contact_step"]) for row in rows),
        "mean_first_lift_step": mean(float(row["first_lift_step"]) for row in rows),
        "mean_close_command": mean(float(row["mean_close_command"]) for row in rows),
        "mean_motion_command": mean(float(row["mean_motion_command"]) for row in rows),
        "step_cap_rate": mean(float(row.get("benchmark_step_cap_reached", False)) for row in rows),
    }


def render_markdown(
    aggregates: list[dict[str, Any]],
    split_name: str,
    family_split: str,
    pipeline_name: str,
) -> str:
    """Render benchmark summary metrics as markdown.

    Args:
        aggregates: Aggregated rows to render.
        split_name: Name of the latent split that was evaluated.
        family_split: Name of the object-family split that was evaluated.
        pipeline_name: Name of the benchmark pipeline that was evaluated.
    """
    lines = [
        "# Pick-and-Place Benchmark",
        "",
        "- env: `franka_latent_adaptation`",
        f"- split: `{split_name}`",
        f"- family_split: `{family_split}`",
        f"- pipeline: `{pipeline_name}`",
        "",
        "| family | task | policy | success | first_try | recovered_success | adapted_success | fail | recover | recover_s | break | drop | slip | miss | place_fail | capped | success_step | place_err | grip_obj_min | grip_obj_final | duration_s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in aggregates:
        lines.append(
            "| {family} | {task} | {policy} | {success_rate:.3f} | {first_attempt_success_rate:.3f} | {recovered_success_rate:.3f} | {adapted_success_rate:.3f} | "
            "{failure_rate:.3f} | {recovery_rate:.3f} | {mean_recovery_time_sec:.3f} | "
            "{break_rate:.3f} | {drop_rate:.3f} | {slip_rate:.3f} | {grasp_miss_rate:.3f} | "
            "{placement_failure_rate:.3f} | {step_cap_rate:.3f} | {mean_success_step:.1f} | {mean_placement_error:.3f} | "
            "{mean_min_gripper_object_distance:.3f} | {mean_final_gripper_object_distance:.3f} | {mean_duration_sec:.2f} |".format(**item)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the sequential latent-adaptation benchmark CLI."""
    parser = argparse.ArgumentParser(description="Evaluate sequential policies on the latent-adaptation benchmark.")
    parser.add_argument("--family", choices=tuple(OBJECT_FAMILY_SPECS) + ("all",), default="all")
    parser.add_argument("--family-split", choices=("all", "seen", "heldout"), default="all")
    parser.add_argument("--task", choices=tuple(TASK_VARIANT_SPECS) + ("all",), default="pick_place")
    parser.add_argument("--split", choices=tuple(SPLITS), default="unseen")
    parser.add_argument("--pipeline", choices=tuple(PIPELINE_SPECS) + ("all",), default=None)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG_PATH)
    parser.add_argument("--policies", type=str, default="random")
    parser.add_argument("--policy-kwargs", type=str, default="{}")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/phase1_policy_benchmark.md"))
    args = parser.parse_args()

    pipeline_names = tuple(PIPELINE_SPECS) if args.pipeline == "all" else (args.pipeline or load_pipeline_name(args.pipeline_config),)
    for pipeline_name in pipeline_names:
        resolve_pipeline_spec(pipeline_name)
    policy_kwargs = parse_mapping_arg(args.policy_kwargs)
    policy_specs = [item.strip() for item in args.policies.split(",") if item.strip()]
    if args.family == "all":
        families = resolve_object_families(args.family_split)
    else:
        families = (args.family,)
        if args.family not in resolve_object_families(args.family_split):
            raise ValueError(f"Family {args.family} is not part of family split {args.family_split}.")
    tasks = tuple(TASK_VARIANT_SPECS) if args.task == "all" else (args.task,)

    all_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for pipeline_name in pipeline_names:
        for policy_index, policy_spec in enumerate(policy_specs):
            policy = build_policy(policy_spec, kwargs=policy_kwargs, seed=args.seed + policy_index)
            for family in families:
                for task in tasks:
                    rows = evaluate_policy(
                        policy=policy,
                        family=family,
                        task=task,
                        split_name=args.split,
                        pipeline_name=pipeline_name,
                        seed=args.seed,
                        episodes=args.episodes,
                        max_steps=args.max_steps,
                    )
                    for row in rows:
                        row["family_split"] = args.family_split
                    aggregate = aggregate_rows(rows)
                    aggregate.update(
                        {
                            "family": family,
                            "family_split": args.family_split,
                            "task": task,
                            "policy": policy.name,
                            "pipeline": pipeline_name,
                        }
                    )
                    aggregates.append(aggregate)
                    all_rows.extend(rows)
                    print(
                        f"finished family={family} task={task} pipeline={pipeline_name} policy={policy.name} "
                        f"success={aggregate['success_rate']:.3f} adapted={aggregate['adapted_success_rate']:.3f}"
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(aggregates, split_name=args.split, family_split=args.family_split, pipeline_name=",".join(pipeline_names)),
        encoding="utf-8",
    )
    args.output.with_suffix(".json").write_text(
        json.dumps({"aggregates": aggregates, "episodes": all_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"saved={args.output}")
    print(f"saved={args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
