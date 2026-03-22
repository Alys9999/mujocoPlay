from __future__ import annotations

from phase1.splits import OBJECT_FAMILIES, resolve_object_families


class HiddenPhysicsBlocksV1:
    """Object-set wrapper for the phase1 hidden-physics families."""

    name = "hidden_physics_blocks_v1"
    families = OBJECT_FAMILIES

    def resolve_families(self, family_split: str = "all") -> tuple[str, ...]:
        """Resolve the object-family subset used by this benchmark run."""
        return resolve_object_families(family_split)
