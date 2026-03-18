"""Phase 1 benchmark package for sequential latent-physics adaptation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import Phase1Config

if TYPE_CHECKING:
    from .adaptation_env import FrankaLatentAdaptationEnv
    from .franka_env import FrankaHiddenPhysicsPickPlaceEnv

__all__ = ["FrankaHiddenPhysicsPickPlaceEnv", "FrankaLatentAdaptationEnv", "Phase1Config"]


def __getattr__(name: str):
    if name == "FrankaHiddenPhysicsPickPlaceEnv":
        from .franka_env import FrankaHiddenPhysicsPickPlaceEnv

        return FrankaHiddenPhysicsPickPlaceEnv
    if name == "FrankaLatentAdaptationEnv":
        from .adaptation_env import FrankaLatentAdaptationEnv

        return FrankaLatentAdaptationEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
