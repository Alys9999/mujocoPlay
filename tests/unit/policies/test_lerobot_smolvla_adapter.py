import numpy as np
import pytest

from benchmark.components.policies.lerobot_smolvla import LeRobotSmolVLAAdapter


@pytest.mark.fast
def test_smolvla_raw_action_maps_to_action_packet():
    adapter = LeRobotSmolVLAAdapter(model_path="external/models/smolvla")
    raw_action = np.zeros(6, dtype=np.float32)
    raw_action[0:3] = [0.2, -0.1, 0.3]
    raw_action[5] = 0.75
    packet = adapter._raw_action_to_packet(raw_action)
    assert packet.schema_id == "cartesian_gripper_v1"
    assert packet.hand.value == pytest.approx(0.5 * (np.tanh(0.75) + 1.0))
