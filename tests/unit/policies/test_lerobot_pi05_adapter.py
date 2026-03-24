import numpy as np
import pytest

from benchmark.components.policies.lerobot_pi05 import LeRobotPI05PolicyAdapter
from benchmark.core.interfaces.observation import ObservationBundle


@pytest.mark.fast
def test_pi05_raw_action_maps_to_action_packet():
    adapter = LeRobotPI05PolicyAdapter(
        model_path="external/models/pi05",
        image_key_mapping={"observation.images.base_0_rgb": "overview"},
    )
    adapter._latest_observation = ObservationBundle(
        joint_positions=np.zeros(7, dtype=np.float32),
        ee_position=np.asarray([0.47, 0.0, 0.3175], dtype=np.float32),
        ee_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        gripper_aperture=0.02,
        wrist_force=np.zeros(3, dtype=np.float32),
        wrist_torque=np.zeros(3, dtype=np.float32),
        target_position=np.zeros(3, dtype=np.float32),
        mocap_target=np.asarray([0.47, 0.0, 0.3175], dtype=np.float32),
        previous_action_vector=np.zeros(4, dtype=np.float32),
        time_sec=0.0,
        step_count=0,
        instruction="pick and place",
        object_family="block",
        task_variant="pick_place",
        images={"overview": np.zeros((4, 5, 3), dtype=np.uint8)},
    )
    packet = adapter._raw_action_to_packet(np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.4], dtype=np.float32))
    assert packet.schema_id == "cartesian_gripper_v1"
    assert packet.arm.mode == "absolute_pose"
    assert packet.arm.rpy is not None
    assert len(packet.arm.xyz) == 3
    assert len(packet.arm.rpy) == 3
    assert 0.0 <= packet.hand.value <= 1.0


@pytest.mark.fast
def test_pi05_build_batch_uses_libero_style_state_and_zero_padded_right_wrist():
    adapter = LeRobotPI05PolicyAdapter(model_path="external/models/pi05")
    overview = np.full((4, 5, 3), 17, dtype=np.uint8)
    wrist_left = np.full((4, 5, 3), 99, dtype=np.uint8)
    observation = ObservationBundle(
        joint_positions=np.zeros(7, dtype=np.float32),
        ee_position=np.asarray([0.47, 0.0, 0.3175], dtype=np.float32),
        ee_quaternion=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        gripper_aperture=0.02,
        wrist_force=np.zeros(3, dtype=np.float32),
        wrist_torque=np.zeros(3, dtype=np.float32),
        target_position=np.zeros(3, dtype=np.float32),
        mocap_target=np.zeros(3, dtype=np.float32),
        previous_action_vector=np.zeros(4, dtype=np.float32),
        time_sec=0.0,
        step_count=0,
        instruction="pick and place",
        object_family="block",
        task_variant="pick_place",
        images={"overview": overview, "wrist_left": wrist_left},
    )

    batch = adapter._build_batch(observation)

    np.testing.assert_allclose(
        batch["observation.state"].numpy()[:8],
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(batch["observation.state"].numpy()[8:], 0.0)
    np.testing.assert_array_equal(batch["observation.images.base_0_rgb"].permute(1, 2, 0).numpy(), overview)
    np.testing.assert_array_equal(
        batch["observation.images.left_wrist_0_rgb"].permute(1, 2, 0).numpy(),
        wrist_left,
    )
    np.testing.assert_array_equal(
        batch["observation.images.right_wrist_0_rgb"].permute(1, 2, 0).numpy(),
        np.zeros_like(overview),
    )
