import pytest

from benchmark.schemas.models import ActionPacket, BenchmarkResult, TraceEvent


@pytest.mark.fast
def test_schema_models_validate_examples():
    packet = ActionPacket(
        schema_id="cartesian_gripper_v1",
        arm={"mode": "delta_pose", "xyz": [0.0, 0.0, 0.01], "rpy": [0.0, 0.0, 0.0]},
        hand={"mode": "scalar_close", "value": 0.7},
    )
    event = TraceEvent(event_type="step", run_id="run_001", session_id="session_001", action_packet=packet)
    result = BenchmarkResult(
        run_id="run_001",
        benchmark_name="normal_pick_place",
        policy_name="random",
        aggregate_metrics={"success_rate": 0.0},
        episode_rows=[{"success": 0.0}],
    )
    assert packet.schema_id == "cartesian_gripper_v1"
    assert event.action_packet is not None
    assert result.benchmark_name == "normal_pick_place"
