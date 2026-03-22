import json

import pytest

from benchmark.components.tracing.sinks import JSONLTraceSink
from benchmark.schemas.models.trace_event import TraceEvent


@pytest.mark.io
def test_jsonl_trace_sink_writes_one_line_per_event(tmp_path):
    output_path = tmp_path / "trace.jsonl"
    sink = JSONLTraceSink(output_path=output_path)
    sink.emit(TraceEvent(event_type="session.start", run_id="run_001", session_id="session_001"))
    sink.close()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "session.start"
