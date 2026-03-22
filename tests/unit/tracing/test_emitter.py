import pytest

from benchmark.components.tracing.emitter import TraceEmitter
from benchmark.core.interfaces.tracing import TraceSettings
from benchmark.schemas.models.trace_event import TraceEvent


class CollectSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def close(self):
        pass


@pytest.mark.fast
def test_trace_emitter_filters_privileged_context():
    sink = CollectSink()
    emitter = TraceEmitter(
        settings=TraceSettings(include_event_types=frozenset({"step"}), include_privileged_context=False),
        sinks=[sink],
    )
    emitter.emit(
        TraceEvent(
            event_type="step",
            run_id="run_001",
            session_id="session_001",
            privileged_context={"body": {"reach_scale": 1.0}},
        )
    )
    assert len(sink.events) == 1
    assert sink.events[0].privileged_context is None
