from __future__ import annotations

import subprocess
from pathlib import Path

from .docs import export_all_markdown_docs
from .exporter import export_all_json_schemas
from .registry import iter_schema_definitions
from .validator import validate_contract_instance

_SCHEMA_RELATED_PREFIXES = (
    "benchmark/schemas/",
    "contracts/jsonschema/",
    "docs/schemas/",
    "benchmark/components/policies/",
    "benchmark/components/robots/control_adapters/",
)


def _changed_paths(repo_root: Path) -> tuple[str, ...]:
    try:
        output = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=repo_root,
            text=True,
        )
    except Exception:
        return ()
    return tuple(path.strip().replace("\\", "/") for path in output.splitlines() if path.strip())


def run_schema_gate(*, changed_only: bool = False) -> None:
    """Run schema export, docs export, and representative instance validation."""
    repo_root = Path(__file__).resolve().parents[2]
    if changed_only:
        changed_paths = _changed_paths(repo_root)
        if changed_paths and not any(
            path.startswith(prefix) for prefix in _SCHEMA_RELATED_PREFIXES for path in changed_paths
        ):
            return
    export_all_json_schemas()
    export_all_markdown_docs()
    for definition in iter_schema_definitions():
        validate_contract_instance(definition.name, definition.example_factory())


def main() -> None:
    """CLI entrypoint for the local schema gate."""
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark schema export and validation gate.")
    parser.add_argument("--changed", action="store_true", help="Run only when schema-related files changed.")
    args = parser.parse_args()
    run_schema_gate(changed_only=args.changed)


if __name__ == "__main__":
    main()
