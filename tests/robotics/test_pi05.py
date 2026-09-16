import numpy as np
import pytest

from djepa.robotics.pi05 import (
    ActionConvention,
    from_pi05_actions,
    make_training_example,
    to_pi05_actions,
    to_pi05_observation,
)
from djepa.robotics.rlds import EpisodeIdentity, RoboTwinEpisode, RoboTwinStep


def frozen(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result.flags.writeable = False
    return result


def fake_step(index: int = 0) -> RoboTwinStep:
    image = np.full((240, 320, 3), index, dtype=np.uint8)
    return RoboTwinStep(
        base_rgb=frozen(image),
        low_rgb=frozen(image + 1),
        left_wrist_rgb=frozen(image + 2),
        right_wrist_rgb=frozen(image + 3),
        joint_state=frozen(np.linspace(0.1, 0.9, 14, dtype=np.float32)),
        eef_state=frozen(np.linspace(-0.5, 0.5, 20, dtype=np.float32)),
        joint_action=frozen(np.linspace(0.2, 0.8, 14, dtype=np.float32)),
        eef_action=frozen(np.linspace(-0.2, 0.2, 20, dtype=np.float32)),
        instruction="grab the roller",
        reward=0.0,
        is_first=index == 0,
        is_last=False,
        is_terminal=False,
    )


def test_pi05_observation_maps_expected_cameras() -> None:
    step = fake_step()

    observation = to_pi05_observation(step)

    assert set(observation.images) == {
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    assert observation.images["cam_high"].shape == (3, 240, 320)
    assert observation.state.shape == (14,)
    assert observation.prompt == "grab the roller"
    assert not observation.images["cam_high"].flags.writeable
    assert not observation.state.flags.writeable
    np.testing.assert_array_equal(step.base_rgb, np.zeros((240, 320, 3), dtype=np.uint8))


def test_pi05_action_round_trip_is_exact_for_identity_convention() -> None:
    actions = np.linspace(-0.5, 0.5, 70, dtype=np.float32).reshape(5, 14)

    internal = to_pi05_actions(actions, ActionConvention.IDENTITY)
    restored = from_pi05_actions(internal, ActionConvention.IDENTITY)

    np.testing.assert_array_equal(restored, actions)
    assert not restored.flags.writeable


def test_pi05_action_round_trip_matches_openpi_aloha_equations() -> None:
    actions = np.linspace(-0.3, 0.3, 70, dtype=np.float32).reshape(5, 14)
    actions[:, [6, 13]] = np.asarray([0.2, 0.8], dtype=np.float32)

    internal = to_pi05_actions(actions, ActionConvention.OPENPI_ALOHA)
    restored = from_pi05_actions(internal, ActionConvention.OPENPI_ALOHA)

    np.testing.assert_allclose(restored, actions, rtol=0.0, atol=1e-6)
    assert not np.shares_memory(internal, actions)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((4, 13), dtype=np.float32),
        np.full((4, 14), np.nan, dtype=np.float32),
        np.zeros((4, 14), dtype=np.float64),
    ],
)
def test_pi05_actions_reject_invalid_arrays(actions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="actions"):
        to_pi05_actions(actions, ActionConvention.IDENTITY)


def test_training_example_uses_fixed_horizon_without_mutating_episode() -> None:
    steps = tuple(fake_step(index) for index in range(4))
    episode = RoboTwinEpisode(EpisodeIdentity("grab_roller", 3), steps)

    example = make_training_example(
        episode,
        start=1,
        action_horizon=3,
        convention=ActionConvention.IDENTITY,
    )

    assert example.identity == episode.identity
    assert example.start == 1
    assert example.actions.shape == (3, 14)
    np.testing.assert_array_equal(example.actions[0], steps[1].joint_action)


def test_training_example_rejects_insufficient_horizon() -> None:
    episode = RoboTwinEpisode(EpisodeIdentity("grab_roller", 0), (fake_step(),))

    with pytest.raises(ValueError, match="action horizon"):
        make_training_example(
            episode,
            start=0,
            action_horizon=2,
            convention=ActionConvention.IDENTITY,
        )
