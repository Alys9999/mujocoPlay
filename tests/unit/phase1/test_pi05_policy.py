import numpy as np
import pytest

from phase1.pi05_policy import PI05SequentialPolicy, benchmark_observation_to_pi05_state
from phase1.policy_benchmark import AdaptiveBenchmarkSession


@pytest.mark.fast
def test_benchmark_observation_to_pi05_state_uses_libero_style_prefix_and_zero_padding():
    observation = {
        "gripper_pos": np.asarray([0.47, 0.0, 0.3175], dtype=np.float32),
        "ee_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gripper_aperture": np.asarray([0.02], dtype=np.float32),
    }

    state = benchmark_observation_to_pi05_state(observation)

    np.testing.assert_allclose(
        state[:8],
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(state[8:], 0.0)


@pytest.mark.fast
def test_pi05_build_batch_zero_pads_right_wrist():
    policy = PI05SequentialPolicy(duplicate_overview_to_all_cameras=False)
    overview = np.full((4, 5, 3), 11, dtype=np.uint8)
    wrist_left = np.full((4, 5, 3), 22, dtype=np.uint8)
    observation = {
        "overview_rgb": overview,
        "base_rgb": overview,
        "left_wrist_rgb": wrist_left,
        "gripper_pos": np.asarray([0.47, 0.0, 0.3175], dtype=np.float32),
        "ee_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gripper_aperture": np.asarray([0.02], dtype=np.float32),
        "instruction": "pick and place",
    }

    batch = policy._build_batch(observation)

    np.testing.assert_array_equal(batch["observation.images.base_0_rgb"].permute(1, 2, 0).numpy(), overview)
    np.testing.assert_array_equal(
        batch["observation.images.left_wrist_0_rgb"].permute(1, 2, 0).numpy(),
        wrist_left,
    )
    np.testing.assert_array_equal(
        batch["observation.images.right_wrist_0_rgb"].permute(1, 2, 0).numpy(),
        np.zeros_like(overview),
    )


@pytest.mark.fast
def test_pi05_to_benchmark_action_returns_absolute_7d_command():
    policy = PI05SequentialPolicy(duplicate_overview_to_all_cameras=False)
    observation = {
        "mocap_target": np.asarray([0.47, 0.0, 0.3175], dtype=np.float32),
        "ee_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }

    action = policy._to_benchmark_action(
        np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1, 0.4], dtype=np.float32),
        observation,
    )

    assert action.shape == (7,)
    assert 0.0 <= action[6] <= 1.0
    assert not np.allclose(action[:3], observation["mocap_target"])


@pytest.mark.compat
def test_phase1_session_accepts_absolute_7d_action():
    session = AdaptiveBenchmarkSession(object_family="block", task_variant="pick_place")
    try:
        observation = session.reset(seed=0, include_image=False)
        absolute_action = np.concatenate(
            [
                np.asarray(observation["mocap_target"], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.asarray([0.5], dtype=np.float64),
            ]
        )
        next_observation, reward, terminated, truncated, info = session.step(absolute_action, include_image=False)
        assert "mocap_target" in next_observation
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        session.close()
