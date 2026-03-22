import json

import pytest

from benchmark.schemas.exporter import export_all_json_schemas


@pytest.mark.io
def test_export_all_json_schemas_writes_files(tmp_path):
    written_paths = export_all_json_schemas(output_dir=tmp_path)
    assert written_paths
    payload = json.loads(written_paths[0].read_text(encoding="utf-8"))
    assert "$defs" in payload or "properties" in payload
