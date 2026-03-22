import numpy as np
import pytest

from benchmark.components.policies.lerobot_pi05 import LeRobotPI05PolicyAdapter


@pytest.mark.fast
def test_pi05_raw_action_maps_to_action_packet():
    adapter = LeRobotPI05PolicyAdapter(model_path="external/models/pi05", image_key_mapping={"observation.images.base_0_rgb": "overview"})
    packet = adapter._raw_action_to_packet(np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.4], dtype=np.float32))
    assert packet.schema_id == "cartesian_gripper_v1"
    assert packet.arm.mode == "delta_pose"
    assert 0.0 <= packet.hand.value <= 1.0
