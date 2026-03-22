from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from benchmark.core.interfaces.tracing import TraceSettings, TraceSink
from benchmark.schemas.models.trace_event import TraceEvent


@dataclass(slots=True)
class TraceEmitter:
    """Emit trace events to one or more sinks."""

    settings: TraceSettings
    sinks: list[TraceSink]

    def emit(self, event: TraceEvent) -> None:
        """Emit one already-built event when enabled by tracing settings."""
        if event.event_type not in self.settings.include_event_types:
            return
        if not self.settings.include_privileged_context and event.privileged_context is not None:
            event = event.model_copy(update={"privileged_context": None})
        for sink in self.sinks:
            sink.emit(event)

    def close(self) -> None:
        """Close all sinks."""
        for sink in self.sinks:
            sink.close()


def build_trace_settings(sink_configs: Iterable[Any]) -> TraceSettings:
    """Resolve sink configs into one effective tracing policy."""
    included_events: set[str] = set()
    include_privileged_context = False
    for sink_config in sink_configs:
        included_events.update(sink_config.include)
        include_privileged_context = include_privileged_context or bool(sink_config.include_privileged_context)
    return TraceSettings(include_event_types=frozenset(included_events), include_privileged_context=include_privileged_context)
