import pytest

from benchmark.core.config.loader import build_config_from_preset
from benchmark.presets.benchmarks import BENCHMARK_PRESET_NAMES


@pytest.mark.fast
def test_all_presets_build_into_typed_config():
    for preset_name in BENCHMARK_PRESET_NAMES:
        config = build_config_from_preset(preset_name)
        assert config.benchmark.name == preset_name
        assert config.robot.profile_pool
        assert config.tracing.sinks
