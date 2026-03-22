from __future__ import annotations

from phase1.task_language import build_instruction


class PickPlaceTaskDefinition:
    """Task-definition wrapper for the phase1 pick-and-place benchmark."""

    name = "pick_place"
    variants = ("pick_place",)

    def resolve_variants(self) -> tuple[str, ...]:
        """Return the supported task variants."""
        return self.variants

    def build_instruction(self, object_family: str, task_variant: str) -> str:
        """Return the public language instruction for one evaluation case."""
        return build_instruction(object_family, task_variant)
