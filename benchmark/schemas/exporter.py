from __future__ import annotations

import json
from pathlib import Path

from .registry import SchemaDefinition, iter_schema_definitions


def export_json_schema(definition: SchemaDefinition) -> dict:
    """Build one JSON schema document from its Pydantic model."""
    return definition.model.model_json_schema()


def export_all_json_schemas(output_dir: Path | None = None) -> list[Path]:
    """Export all registered JSON schemas to disk."""
    written_paths: list[Path] = []
    for definition in iter_schema_definitions():
        target_path = definition.jsonschema_path if output_dir is None else output_dir / definition.jsonschema_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        schema_payload = export_json_schema(definition)
        target_path.write_text(json.dumps(schema_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written_paths.append(target_path)
    return written_paths
