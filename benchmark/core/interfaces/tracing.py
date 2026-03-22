from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from benchmark.schemas.models.trace_event import TraceEvent


@dataclass(frozen=True, slots=True)
class TraceSettings:
    """Resolved tracing policy for a scheduler run."""

    include_event_types: frozenset[str]
    include_privileged_context: bool = False


class TraceSink(Protocol):
    """Consumable sink for trace events."""

    def emit(self, event: TraceEvent) -> None:
        """Consume one trace event."""

    def close(self) -> None:
        """Flush and close any sink resources."""
