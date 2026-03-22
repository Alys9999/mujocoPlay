from __future__ import annotations

import json
from pathlib import Path

from benchmark.schemas.models.trace_event import TraceEvent


class JSONLTraceSink:
    """Append trace events to a JSONL file."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("a", encoding="utf-8")

    def emit(self, event: TraceEvent) -> None:
        self._handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
