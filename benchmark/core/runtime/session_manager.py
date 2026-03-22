from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from benchmark.components.tracing.emitter import TraceEmitter, build_trace_settings
from benchmark.core.config.models import BenchmarkConfig
from benchmark.core.registry.registry import BenchmarkRegistry
from benchmark.schemas.models.trace_event import TraceEvent


@dataclass(slots=True)
class SessionManager:
    """Own the session lifecycle and trace-sink wiring for one scheduler run."""

    config: BenchmarkConfig
    registry: BenchmarkRegistry
    output_dir: Path = field(init=False)
    run_id: str = field(init=False)
    session_id: str = field(init=False)
    emitter: TraceEmitter = field(init=False)
    artifacts: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.config.benchmark.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = f"run_{self.config.benchmark.name}_{self.config.benchmark.seed}"
        self.session_id = "session_001"
        self.emitter, self.artifacts = self._build_trace_emitter()

    def _build_trace_emitter(self) -> tuple[TraceEmitter, dict[str, str]]:
        sink_configs = self.config.tracing.sinks
        settings = build_trace_settings(sink_configs)
        sinks = []
        artifacts: dict[str, str] = {}
        for sink_config in sink_configs:
            sink_factory = self.registry.category("trace_sinks").get(sink_config.type)
            output_path = sink_config.output_path or (self.output_dir / f"trace.{sink_config.type}")
            sink = sink_factory(output_path=output_path)
            sinks.append(sink)
            if hasattr(sink, "output_path"):
                artifacts[f"trace_{sink_config.type}"] = str(getattr(sink, "output_path"))
        return TraceEmitter(settings=settings, sinks=sinks), artifacts

    def emit_session_start(self) -> None:
        self.emitter.emit(
            TraceEvent(
                event_type="session.start",
                run_id=self.run_id,
                session_id=self.session_id,
                info_summary={"benchmark_name": self.config.benchmark.name},
            )
        )

    def emit_session_end(self) -> None:
        self.emitter.emit(
            TraceEvent(
                event_type="session.end",
                run_id=self.run_id,
                session_id=self.session_id,
                info_summary={"benchmark_name": self.config.benchmark.name},
            )
        )

    def episode_id(self, case_index: int, episode_index: int) -> str:
        return f"ep_{case_index:02d}_{episode_index:04d}"

    def close(self) -> None:
        self.emitter.close()
