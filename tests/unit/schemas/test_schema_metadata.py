import pytest
from pydantic import BaseModel

from benchmark.schemas.registry import iter_schema_definitions


REQUIRED_METADATA_KEYS = {"why", "producer", "consumer", "visibility", "stability", "example"}


def _iter_fields(model: type[BaseModel]):
    for field_name, field_info in model.model_fields.items():
        yield field_name, field_info
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield from _iter_fields(annotation)


@pytest.mark.fast
def test_schema_fields_expose_required_metadata():
    for definition in iter_schema_definitions():
        for _, field_info in _iter_fields(definition.model):
            extras = field_info.json_schema_extra or {}
            assert REQUIRED_METADATA_KEYS.issubset(extras.keys())
