from __future__ import annotations

from .benchmark_spec import get_task_variant_spec


def build_instruction(family: str, task: str) -> str:
    """Build the text instruction exposed to benchmarked policies.

    Args:
        family: Object family being manipulated in the current episode.
        task: Task variant being evaluated in the current episode.
    """
    task_text = get_task_variant_spec(task).instruction
    family_text = {
        "block": "The object is a red block.",
        "cylinder": "The object is a red cylinder.",
        "small_box": "The object is a red small box.",
    }[family]
    return f"{family_text} {task_text}"
