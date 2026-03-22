from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, get_args, get_origin

from pydantic import BaseModel

from .registry import SchemaDefinition, iter_schema_definitions


def _format_annotation(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", repr(annotation)).replace("typing.", "")
    args = ", ".join(_format_annotation(arg) for arg in get_args(annotation))
    return f"{getattr(origin, '__name__', repr(origin))}[{args}]"


def _iter_field_rows(model: type[BaseModel], prefix: str = "") -> Iterable[dict[str, str]]:
    for field_name, field_info in model.model_fields.items():
        path = f"{prefix}{field_name}"
        extras = field_info.json_schema_extra or {}
        yield {
            "field": path,
            "type": _format_annotation(field_info.annotation),
            "required": "yes" if field_info.is_required() else "no",
            "description": field_info.description or "",
            "why": str(extras.get("why", "")),
            "producer": str(extras.get("producer", "")),
            "consumer": str(extras.get("consumer", "")),
            "visibility": str(extras.get("visibility", "")),
            "stability": str(extras.get("stability", "")),
            "example": repr(extras.get("example", "")),
        }
        annotation = field_info.annotation
        nested_model = None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested_model = annotation
        else:
            for arg in get_args(annotation):
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    nested_model = arg
                    break
        if nested_model is not None:
            yield from _iter_field_rows(nested_model, prefix=f"{path}.")


def render_markdown(definition: SchemaDefinition) -> str:
    """Render one registered schema into Markdown documentation."""
    lines = [
        f"# {definition.name}",
        "",
        "| field | type | required | description | why | producer | consumer | visibility | stability | example |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in _iter_field_rows(definition.model):
        lines.append(
            "| {field} | {type} | {required} | {description} | {why} | {producer} | {consumer} | {visibility} | {stability} | {example} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def export_all_markdown_docs(output_dir: Path | None = None) -> list[Path]:
    """Export Markdown docs for all registered schemas."""
    written_paths: list[Path] = []
    for definition in iter_schema_definitions():
        target_path = definition.markdown_path if output_dir is None else output_dir / definition.markdown_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(render_markdown(definition), encoding="utf-8")
        written_paths.append(target_path)
    return written_paths
