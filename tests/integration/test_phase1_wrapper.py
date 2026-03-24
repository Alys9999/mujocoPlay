import pytest

from benchmark.components.robots.franka_panda_2f_v1 import build_phase1_franka_runtime
from benchmark.schemas.models.action_packet import ActionPacket


@pytest.mark.compat
def test_phase1_runtime_wrapper_reset_and_step():
    runtime = build_phase1_franka_runtime(object_family="block", task_variant="pick_place", camera_names=("overview",))
    try:
        observation = runtime.reset(seed=0)
        assert observation.object_family == "block"
        assert "overview" in observation.images
        assert "wrist_left" in observation.images
        assert "wrist_right" in observation.images
        result = runtime.step(
            ActionPacket(
                schema_id="cartesian_gripper_v1",
                arm={"mode": "delta_pose", "xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
                hand={"mode": "scalar_close", "value": 0.5},
            )
        )
        assert result.observation.step_count >= 1
    finally:
        runtime.close()
