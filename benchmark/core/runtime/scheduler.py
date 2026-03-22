from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from phase1.policy_benchmark import aggregate_rows

from benchmark.core.config.models import BenchmarkConfig
from benchmark.core.interfaces.observation import ObservationBundle
from benchmark.core.registry.registry import BenchmarkRegistry, create_default_registry
from benchmark.schemas.models.action_packet import ActionPacket
from benchmark.schemas.models.benchmark_result import BenchmarkResult
from benchmark.schemas.models.trace_event import TraceEvent
from .episode_builder import EpisodeBuilder, EvaluationCase
from .session_manager import SessionManager


def _summarize_public_observation(observation: ObservationBundle) -> dict[str, Any]:
    return {
        "step_count": observation.step_count,
        "time_sec": observation.time_sec,
        "object_family": observation.object_family,
        "task_variant": observation.task_variant,
        "available_cameras": tuple(sorted(observation.images)),
        "has_overview": "overview" in observation.images,
    }


class BenchmarkScheduler:
    """Config-driven benchmark scheduler with decoupled runtime and tracing."""

    def __init__(self, registry: BenchmarkRegistry | None = None) -> None:
        self.registry = create_default_registry() if registry is None else registry

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        builder = EpisodeBuilder(config, self.registry)
        session = SessionManager(config=config, registry=self.registry)
        policy = builder.build_policy()
        rng = np.random.default_rng(config.benchmark.seed)
        rows: list[dict[str, Any]] = []
        session.emit_session_start()
        try:
            for case_index, case in enumerate(builder.resolve_eval_cases()):
                runtime = builder.build_runtime(case)
                try:
                    for episode_index in range(config.benchmark.episodes):
                        episode_id = session.episode_id(case_index, episode_index)
                        episode_setup = builder.sample_episode(episode_index=episode_index, rng=rng)
                        policy.reset()
                        observation = runtime.reset(
                            seed=episode_setup.seed,
                            hidden_context=episode_setup.hidden_context,
                            target_xy=episode_setup.target_xy,
                        )
                        session.emitter.emit(
                            TraceEvent(
                                event_type="episode.start",
                                run_id=session.run_id,
                                session_id=session.session_id,
                                episode_id=episode_id,
                                info_summary={
                                    "object_family": case.object_family,
                                    "task_variant": case.task_variant,
                                    "seed": episode_setup.seed,
                                },
                            )
                        )
                        rows.append(
                            self._rollout_episode(
                                config=config,
                                session=session,
                                runtime=runtime,
                                policy=policy,
                                case=case,
                                episode_id=episode_id,
                                observation=observation,
                            )
                        )
                finally:
                    runtime.close()
        finally:
            session.emit_session_end()
            session.close()
        aggregate_metrics = aggregate_rows(rows) if rows else self._empty_aggregate()
        result = BenchmarkResult(
            run_id=session.run_id,
            benchmark_name=config.benchmark.name,
            policy_name=policy.name,
            aggregate_metrics=aggregate_metrics,
            episode_rows=rows,
            artifacts=session.artifacts,
        )
        self._write_result(config.benchmark.output_dir, result)
        return result

    def _rollout_episode(
        self,
        *,
        config: BenchmarkConfig,
        session: SessionManager,
        runtime,
        policy,
        case: EvaluationCase,
        episode_id: str,
        observation: ObservationBundle,
    ) -> dict[str, Any]:
        terminated = False
        truncated = False
        step_limit = None if config.runtime.max_steps <= 0 else int(config.runtime.max_steps)
        steps_taken = 0
        while not (terminated or truncated):
            if step_limit is not None and steps_taken >= step_limit:
                truncated = True
                break
            action_packet = policy.act(observation)
            action_packet = ActionPacket.model_validate(action_packet)
            step_result = runtime.step(action_packet)
            session.emitter.emit(
                TraceEvent(
                    event_type="step",
                    run_id=session.run_id,
                    session_id=session.session_id,
                    episode_id=episode_id,
                    step_idx=step_result.observation.step_count,
                    sim_time=step_result.observation.time_sec,
                    action_packet=action_packet,
                    public_obs_summary=_summarize_public_observation(step_result.observation),
                    info_summary={
                        "success": bool(step_result.info.get("success", False)),
                        "slip": bool(step_result.info.get("slip_detected", False)),
                        "drop": bool(step_result.info.get("object_dropped", False)),
                    },
                    privileged_context=runtime.get_privileged_context(),
                )
            )
            observe_transition = getattr(policy, "observe_transition", None)
            if callable(observe_transition):
                observe_transition(
                    observation,
                    action_packet,
                    step_result.reward,
                    step_result.observation,
                    step_result.terminated,
                    step_result.truncated,
                    step_result.info,
                )
            observation = step_result.observation
            terminated = bool(step_result.terminated)
            truncated = bool(step_result.truncated)
            steps_taken += 1
        summary = runtime.summarize_episode()
        summary.update(
            {
                "policy": policy.name,
                "pipeline": config.benchmark.name,
                "family_split": config.benchmark.family_split,
                "split": config.benchmark.hidden_split,
                "object_family": case.object_family,
                "task_variant": case.task_variant,
            }
        )
        session.emitter.emit(
            TraceEvent(
                event_type="episode.end",
                run_id=session.run_id,
                session_id=session.session_id,
                episode_id=episode_id,
                info_summary={
                    "success": bool(summary.get("success", False)),
                    "failure_count": int(summary.get("failure_count", 0)),
                    "step_count": int(summary.get("step_count", 0)),
                },
            )
        )
        return summary

    def _write_result(self, output_dir: str | Path, result: BenchmarkResult) -> None:
        result_path = Path(output_dir) / "benchmark-result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _empty_aggregate(self) -> dict[str, float]:
        return {
            "episodes": 0.0,
            "success_rate": 0.0,
            "first_attempt_success_rate": 0.0,
            "recovered_success_rate": 0.0,
            "adapted_success_rate": 0.0,
            "failure_rate": 0.0,
            "mean_failure_count": 0.0,
            "recovery_rate": 0.0,
            "mean_recovery_time_sec": 0.0,
            "lift_rate": 0.0,
            "break_rate": 0.0,
            "drop_rate": 0.0,
            "slip_rate": 0.0,
            "grasp_miss_rate": 0.0,
            "placement_failure_rate": 0.0,
            "mean_placement_error": 0.0,
            "mean_initial_gripper_object_distance": 0.0,
            "mean_final_gripper_object_distance": 0.0,
            "mean_min_gripper_object_distance": 0.0,
            "mean_duration_sec": 0.0,
            "mean_success_step": 0.0,
            "mean_first_contact_step": 0.0,
            "mean_first_lift_step": 0.0,
            "mean_close_command": 0.0,
            "mean_motion_command": 0.0,
            "step_cap_rate": 0.0,
        }
