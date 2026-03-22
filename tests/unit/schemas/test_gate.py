from pathlib import Path

import pytest

from benchmark.schemas.gate import run_schema_gate


@pytest.mark.io
def test_run_schema_gate_exports_registered_artifacts():
    run_schema_gate(changed_only=False)
    assert Path("contracts/jsonschema/action-packet.schema.json").exists()
    assert Path("docs/schemas/action-packet.md").exists()
