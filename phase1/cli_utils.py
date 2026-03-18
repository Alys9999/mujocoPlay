from __future__ import annotations

import ast
import json
from typing import Any


def _split_top_level_items(text: str) -> list[str]:
    """Split a comma-separated mapping body while respecting nesting and quotes.

    Args:
        text: Mapping body without the surrounding braces.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escape = False

    for char in text:
        if quote is not None:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
            current.append(char)
            continue
        if char in ")]}":
            depth = max(0, depth - 1)
            current.append(char)
            continue
        if char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_scalar(value_text: str) -> Any:
    """Parse one scalar or structured value from a CLI mapping argument.

    Args:
        value_text: Raw value fragment extracted from the CLI argument.
    """
    stripped = value_text.strip()
    if stripped == "":
        return ""
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        quote = stripped[0]
        return stripped[1:-1].replace(f"\\{quote}", quote)
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(stripped)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


def parse_mapping_arg(text: str) -> dict[str, Any]:
    """Parse a CLI mapping argument with JSON and PowerShell-friendly fallbacks.

    Args:
        text: Raw command-line value such as `'{}'`, strict JSON, or a relaxed
            PowerShell-shaped mapping like `{checkpoint_path: D:\\file.pt, num_layers: 1}`.
    """
    stripped = text.strip()
    if stripped in {"", "{}"}:
        return {}

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    else:
        if not isinstance(value, dict):
            raise ValueError("Expected a mapping for the CLI argument.")
        return value

    if stripped.startswith("{") and stripped.endswith("}"):
        result: dict[str, Any] = {}
        for item in _split_top_level_items(stripped[1:-1].strip()):
            if ":" in item:
                key_text, value_text = item.split(":", maxsplit=1)
            elif "=" in item:
                key_text, value_text = item.split("=", maxsplit=1)
            else:
                raise ValueError(f"Could not parse mapping entry: {item}")
            key = key_text.strip().strip("'\"")
            if not key:
                raise ValueError(f"Empty mapping key in: {item}")
            result[key] = _parse_scalar(value_text)
        return result

    try:
        value = ast.literal_eval(stripped)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Expected a mapping-like value, got: {text}") from error
    if not isinstance(value, dict):
        raise ValueError("Expected a mapping for the CLI argument.")
    return value
