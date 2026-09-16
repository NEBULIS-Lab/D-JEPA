from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from djepa.robotics.smoke import (
    SmokeObservation,
    load_candidate_bundle,
    load_observation_bundle,
    make_flow_noises,
    publish_candidate_bundle,
    publish_observation_bundle,
    standardize_policy_actions,
    summarize_action_diversity,
)


def _observation() -> SmokeObservation:
    images = {
        name: np.zeros((3, 240, 320), dtype=np.uint8)
        for name in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
    }
    return SmokeObservation(
        task="grab_roller",
        episode_ordinal=3,
        images=images,
        state=np.arange(14, dtype=np.float32),
        prompt="grab the roller",
    )


def test_flow_noises_are_seeded_reproducible_and_distinct() -> None:
    first = make_flow_noises(tuple(range(17)))
    second = make_flow_noises(tuple(range(17)))

    assert first.shape == (17, 50, 32)
    assert first.dtype == np.float32
    assert first.tobytes() == second.tobytes()
    assert not np.array_equal(first[0], first[1])
    assert not first.flags.writeable


def test_observation_bundle_round_trip_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "observation"
    publish_observation_bundle(output, _observation(), provenance={"source": "fixture"})

    loaded = load_observation_bundle(output)
    assert loaded.task == "grab_roller"
    assert loaded.episode_ordinal == 3
    assert loaded.prompt == "grab the roller"
    assert loaded.state.dtype == np.float32
    assert set(loaded.images) == {
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    assert (output / "completion.json").exists()
    with pytest.raises(FileExistsError):
        publish_observation_bundle(output, _observation(), provenance={})


def test_diversity_summary_counts_complete_link_clusters() -> None:
    actions = np.zeros((17, 50, 14), dtype=np.float32)
    for index in range(17):
        actions[index, :, 0] = float(index)

    summary = summarize_action_diversity(actions, executed_prefix=16, cluster_distance=0.25)

    assert summary.candidate_count == 17
    assert summary.cluster_count >= 4
    assert summary.maximum_pairwise_rms > summary.minimum_pairwise_rms > 0


def test_policy_actions_are_safely_standardized_to_float32() -> None:
    source = np.linspace(-2.0, 2.0, 50 * 14, dtype=np.float64).reshape(50, 14)

    converted, maximum_error = standardize_policy_actions(source)

    assert converted.shape == (50, 14)
    assert converted.dtype == np.float32
    assert np.isfinite(converted).all()
    assert maximum_error < 1e-6
    assert not converted.flags.writeable


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((49, 14), dtype=np.float64),
        np.full((50, 14), np.nan, dtype=np.float64),
        np.full((50, 14), 1e40, dtype=np.float64),
    ],
)
def test_policy_action_standardization_rejects_invalid_values(actions: np.ndarray) -> None:
    with pytest.raises(ValueError):
        standardize_policy_actions(actions)


def test_candidate_bundle_requires_full_fixed_unfiltered_set(tmp_path: Path) -> None:
    actions = np.zeros((17, 50, 14), dtype=np.float32)
    actions[:, :, 0] = np.arange(17, dtype=np.float32)[:, None]
    output = tmp_path / "candidates"

    publish_candidate_bundle(
        output,
        actions=actions,
        seeds=tuple(range(17)),
        native_index=0,
        executed_prefix=16,
        cluster_distance=0.25,
        post_outcome_selected=False,
        provenance={"checkpoint": "fixture"},
    )

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["candidate_count"] == 17
    assert metrics["native_index"] == 0
    assert metrics["post_outcome_selected"] is False
    assert metrics["cluster_count"] >= 4
    loaded_actions, loaded_seeds, loaded_metrics = load_candidate_bundle(output)
    np.testing.assert_array_equal(loaded_actions, actions)
    np.testing.assert_array_equal(loaded_seeds, np.arange(17))
    assert loaded_metrics["native_index"] == 0
    with pytest.raises(ValueError, match="post-outcome"):
        publish_candidate_bundle(
            tmp_path / "forbidden",
            actions=actions,
            seeds=tuple(range(17)),
            native_index=0,
            executed_prefix=16,
            cluster_distance=0.25,
            post_outcome_selected=True,
            provenance={},
        )


def test_candidate_bundle_detects_tampering(tmp_path: Path) -> None:
    actions = np.zeros((17, 50, 14), dtype=np.float32)
    output = tmp_path / "candidates"
    publish_candidate_bundle(
        output,
        actions=actions,
        seeds=tuple(range(17)),
        native_index=0,
        executed_prefix=16,
        cluster_distance=0.25,
        post_outcome_selected=False,
        provenance={},
    )
    with (output / "candidates.npz").open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_candidate_bundle(output)
