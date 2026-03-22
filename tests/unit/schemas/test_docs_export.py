import pytest

from benchmark.schemas.docs import export_all_markdown_docs


@pytest.mark.io
def test_export_all_markdown_docs_writes_tables(tmp_path):
    written_paths = export_all_markdown_docs(output_dir=tmp_path)
    assert written_paths
    content = written_paths[0].read_text(encoding="utf-8")
    assert "| field | type | required |" in content
