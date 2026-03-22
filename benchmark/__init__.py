"""Config-driven benchmark framework for MuJoCo policy evaluation."""

__all__ = [
    "BenchmarkScheduler",
    "build_config_from_preset",
    "create_default_registry",
    "load_benchmark_config",
]


def __getattr__(name: str):
    if name == "BenchmarkScheduler":
        from .core.runtime.scheduler import BenchmarkScheduler

        return BenchmarkScheduler
    if name == "build_config_from_preset":
        from .core.config.loader import build_config_from_preset

        return build_config_from_preset
    if name == "load_benchmark_config":
        from .core.config.loader import load_benchmark_config

        return load_benchmark_config
    if name == "create_default_registry":
        from .core.registry.registry import create_default_registry

        return create_default_registry
    raise AttributeError(f"module 'benchmark' has no attribute {name!r}")
