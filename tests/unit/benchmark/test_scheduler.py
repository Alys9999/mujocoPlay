import pytest

from benchmark.core.config.loader import build_config_from_preset
from benchmark.core.runtime.scheduler import BenchmarkScheduler


@pytest.mark.compat
def test_scheduler_smoke_run_writes_result(tmp_path):
    config = build_config_from_preset(
        "normal_pick_place",
        overrides={
            "benchmark": {"episodes": 1, "output_dir": str(tmp_path / "run")},
            "runtime": {"max_steps": 1},
        },
    )
    result = BenchmarkScheduler().run(config)
    assert result.aggregate_metrics["episodes"] == pytest.approx(3.0)
    assert (tmp_path / "run" / "benchmark-result.json").exists()
