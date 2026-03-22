from __future__ import annotations

from typing import Any

from .registry import get_schema_definition

try:
    from openapi_schema_validator import OAS31Validator
except ImportError:  # pragma: no cover - optional dependency in local environments
    OAS31Validator = None


def validate_contract_instance(schema_name: str, instance: dict[str, Any]) -> None:
    """Validate one instance against an exported contract schema."""
    definition = get_schema_definition(schema_name)
    if OAS31Validator is None:
        definition.model.model_validate(instance)
        return
    schema_payload = definition.model.model_json_schema()
    OAS31Validator.check_schema(schema_payload)
    OAS31Validator(schema_payload).validate(instance)
