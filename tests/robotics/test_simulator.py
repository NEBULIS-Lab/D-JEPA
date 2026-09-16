from types import SimpleNamespace

import numpy as np
import pytest

from djepa.robotics.simulator import (
    BranchOutcome,
    GrabRollerReference,
    fixed_action_window,
    grab_roller_progress,
    multiview_pixel_distance,
    summarize_branch_outcomes,
    to_pi05_simulator_observation,
    unique_consecutive_actions,
)


def _fake_observation() -> SimpleNamespace:
    sensors = {
        name: {"rgb": np.full((240, 320, 3), index, dtype=np.uint8)}
        for index, name in enumerate(
            ("head_camera", "front_camera", "left_camera", "right_camera")
        )
    }
    return SimpleNamespace(
        state=tuple(np.linspace(-0.5, 0.5, 14)),
        sensors=sensors,
        instruction="  Hold   the roller. ",
    )


def test_simulator_observation_maps_four_views_and_state() -> None:
    transformed = to_pi05_simulator_observation(
        _fake_observation(), task="grab_roller", context_ordinal=100000
    )

    assert transformed.images["cam_high"].shape == (3, 240, 320)
    assert transformed.images["cam_low"][0, 0, 0] == 1
    assert transformed.images["cam_left_wrist"][0, 0, 0] == 2
    assert transformed.images["cam_right_wrist"][0, 0, 0] == 3
    assert transformed.state.shape == (14,)
    assert transformed.state.dtype == np.float32
    assert transformed.prompt == "Hold the roller."


def test_simulator_observation_rejects_missing_view() -> None:
    observation = _fake_observation()
    del observation.sensors["left_camera"]

    with pytest.raises(ValueError, match="left_camera"):
        to_pi05_simulator_observation(
            observation, task="grab_roller", context_ordinal=100000
        )


def test_grab_roller_progress_rewards_approach_then_lift() -> None:
    reference = GrabRollerReference(roller_z=0.74, mean_tcp_distance=0.40)

    initial = grab_roller_progress(
        reference,
        roller_position=np.asarray([0.0, 0.0, 0.74]),
        left_tcp=np.asarray([-0.4, 0.0, 0.74]),
        right_tcp=np.asarray([0.4, 0.0, 0.74]),
        left_closed=False,
        right_closed=False,
        success=False,
    )
    approached = grab_roller_progress(
        reference,
        roller_position=np.asarray([0.0, 0.0, 0.74]),
        left_tcp=np.asarray([-0.1, 0.0, 0.74]),
        right_tcp=np.asarray([0.1, 0.0, 0.74]),
        left_closed=True,
        right_closed=True,
        success=False,
    )
    lifted = grab_roller_progress(
        reference,
        roller_position=np.asarray([0.0, 0.0, 0.82]),
        left_tcp=np.asarray([-0.05, 0.0, 0.82]),
        right_tcp=np.asarray([0.05, 0.0, 0.82]),
        left_closed=True,
        right_closed=True,
        success=True,
    )

    assert initial == pytest.approx(0.0)
    assert 0.0 < approached < lifted
    assert lifted == pytest.approx(1.0)


def test_grab_roller_progress_is_bounded_for_adverse_motion() -> None:
    reference = GrabRollerReference(roller_z=0.74, mean_tcp_distance=0.40)

    score = grab_roller_progress(
        reference,
        roller_position=np.asarray([0.0, 0.0, 0.60]),
        left_tcp=np.asarray([-2.0, 0.0, 0.74]),
        right_tcp=np.asarray([2.0, 0.0, 0.74]),
        left_closed=True,
        right_closed=True,
        success=False,
    )

    assert 0.0 <= score <= 1.0


def test_branch_summary_preserves_native_and_finds_oracle() -> None:
    outcomes = tuple(
        BranchOutcome(
            candidate_index=index,
            noise_seed=index,
            success=index == 2,
            progress=progress,
            roller_position=(0.0, 0.0, 0.74 + progress * 0.1),
            mean_tcp_distance=0.4 - progress * 0.2,
        )
        for index, progress in enumerate((0.1, 0.3, 0.9))
    )

    summary = summarize_branch_outcomes(outcomes)

    assert summary["native_index"] == 0
    assert summary["native_success"] is False
    assert summary["oracle_index"] == 2
    assert summary["oracle_success"] is True
    assert summary["success_candidates"] == 1
    assert summary["progress_span"] == pytest.approx(0.8)


def test_multiview_pixel_distance_prefers_matching_scene() -> None:
    query = {
        "cam_high": np.zeros((3, 8, 8), dtype=np.uint8),
        "cam_left_wrist": np.full((3, 8, 8), 10, dtype=np.uint8),
        "cam_right_wrist": np.full((3, 8, 8), 20, dtype=np.uint8),
    }
    close = {name: value.copy() for name, value in query.items()}
    far = {name: np.full_like(value, 255) for name, value in query.items()}

    assert multiview_pixel_distance(query, close) == pytest.approx(0.0)
    assert multiview_pixel_distance(query, far) > 0.8


def test_fixed_action_window_pads_short_demonstration() -> None:
    actions = np.arange(70 * 14, dtype=np.float32).reshape(70, 14)

    window = fixed_action_window(actions, start=50, horizon=50)

    assert window.shape == (50, 14)
    np.testing.assert_array_equal(window[:20], actions[50:])
    np.testing.assert_array_equal(window[20:], np.repeat(actions[-1:], 30, axis=0))
    assert window.dtype == np.float32


def test_unique_consecutive_actions_removes_only_adjacent_duplicates() -> None:
    actions = np.asarray([[0.0] * 14, [0.0] * 14, [1.0] * 14, [0.0] * 14], dtype=np.float32)

    result = unique_consecutive_actions(actions)

    assert result.shape == (3, 14)
    np.testing.assert_array_equal(result[[0, 2]], np.zeros((2, 14), dtype=np.float32))
