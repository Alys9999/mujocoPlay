from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .models import BenchmarkConfig
from benchmark.presets.benchmarks import get_benchmark_preset_payload


def _load_raw_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(payload, dict):
        raise ValueError("Benchmark config root must be a mapping.")
    return payload


def _merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_config_from_preset(name: str, overrides: dict[str, Any] | None = None) -> BenchmarkConfig:
    """Build a typed config from one registered preset payload."""
    payload = get_benchmark_preset_payload(name)
    if overrides:
        payload = _merge_dicts(payload, overrides)
    return BenchmarkConfig.model_validate(payload)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load a typed config from YAML or JSON, optionally applying a preset."""
    config_path = Path(path)
    payload = _load_raw_payload(config_path)
    benchmark_section = payload.get("benchmark", {})
    if not isinstance(benchmark_section, dict):
        raise ValueError("`benchmark` section must be a mapping when present.")
    preset_name = benchmark_section.get("name")
    if isinstance(preset_name, str) and preset_name:
        return build_config_from_preset(preset_name, overrides=payload)
    return BenchmarkConfig.model_validate(payload)
