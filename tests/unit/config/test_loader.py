import pytest

from benchmark.core.config.loader import build_config_from_preset, load_benchmark_config


@pytest.mark.fast
def test_build_config_from_preset_applies_overrides():
    config = build_config_from_preset(
        "normal_pick_place",
        overrides={"benchmark": {"episodes": 2, "output_dir": "benchmark_results/test_loader"}},
    )
    assert config.benchmark.name == "normal_pick_place"
    assert config.benchmark.episodes == 2


@pytest.mark.fast
def test_load_benchmark_config_from_yaml(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        """
benchmark:
  name: body_random_pick_place
  episodes: 3
policy:
  adapter: random
  kwargs:
    seed: 11
""".strip(),
        encoding="utf-8",
    )
    config = load_benchmark_config(config_path)
    assert config.benchmark.name == "body_random_pick_place"
    assert config.benchmark.episodes == 3
    assert config.policy.adapter == "random"
    assert config.policy.kwargs["seed"] == 11
