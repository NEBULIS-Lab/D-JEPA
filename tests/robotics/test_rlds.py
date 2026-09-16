import importlib
import sys

import numpy as np
import pytest

from djepa.robotics.rlds import EpisodeIdentity, iter_episodes, validate_episode


def fake_step(index: int, count: int) -> dict[str, object]:
    observation = {
        "image": np.full((240, 320, 3), index, dtype=np.uint8),
        "low_cam_image": np.full((240, 320, 3), index + 1, dtype=np.uint8),
        "left_wrist_image": np.full((240, 320, 3), index + 2, dtype=np.uint8),
        "right_wrist_image": np.full((240, 320, 3), index + 3, dtype=np.uint8),
        "joint_state": np.linspace(0, 1, 14, dtype=np.float32),
        "eef_state": np.linspace(0, 1, 20, dtype=np.float32),
    }
    return {
        "observation": observation,
        "joint_action": np.linspace(-1, 1, 14, dtype=np.float32),
        "eef_action": np.linspace(-1, 1, 20, dtype=np.float32),
        "language_instruction": np.asarray([b"grab the roller"]),
        "reward": np.float32(index == count - 1),
        "is_first": index == 0,
        "is_last": index == count - 1,
        "is_terminal": index == count - 1,
    }


def fake_raw_episode(steps: int = 3) -> dict[str, object]:
    return {"steps": [fake_step(index, steps) for index in range(steps)]}


def test_import_does_not_load_tensorflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    module = importlib.reload(importlib.import_module("djepa.robotics.rlds"))
    assert hasattr(module, "RoboTwinEpisode")


def test_validate_episode_preserves_four_views_and_actions() -> None:
    episode = validate_episode(
        fake_raw_episode(steps=3),
        EpisodeIdentity("grab_roller", 7),
    )

    assert episode.steps[0].base_rgb.shape == (240, 320, 3)
    assert episode.steps[0].joint_action.shape == (14,)
    assert episode.steps[0].eef_action.shape == (20,)
    assert episode.steps[0].instruction == "grab the roller"
    assert episode.steps[-1].is_last
    assert not episode.steps[0].base_rgb.flags.writeable


def test_validate_episode_uses_first_of_stable_prompt_alternatives() -> None:
    raw = fake_raw_episode(steps=3)
    for step in raw["steps"]:
        step["language_instruction"] = np.asarray(
            [b"grab the roller", b"pick up the cylindrical roller"]
        )

    episode = validate_episode(raw, EpisodeIdentity("grab_roller", 7))

    assert {step.instruction for step in episode.steps} == {"grab the roller"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["steps"][0]["observation"].__setitem__("image", np.zeros((2, 2, 3), dtype=np.uint8)), "base_rgb"),
        (lambda raw: raw["steps"][0].__setitem__("joint_action", np.full(14, np.nan, dtype=np.float32)), "joint_action"),
        (lambda raw: raw["steps"][1].__setitem__("language_instruction", [b""]), "instruction"),
        (lambda raw: raw["steps"][-1].__setitem__("is_last", False), "is_last"),
    ],
)
def test_validate_episode_rejects_invalid_records(mutation, message: str) -> None:
    raw = fake_raw_episode()
    mutation(raw)

    with pytest.raises(ValueError, match=message):
        validate_episode(raw, EpisodeIdentity("grab_roller", 0))


def test_iter_episodes_uses_injected_builder_and_requested_ordinals() -> None:
    class Builder:
        def as_dataset(self, *, split: str, shuffle_files: bool):
            assert split == "train"
            assert not shuffle_files
            return [fake_raw_episode(), fake_raw_episode(), fake_raw_episode()]

    calls = []

    def factory(path):
        calls.append(path)
        return Builder()

    inventory = type(
        "Inventory",
        (),
        {
            "task": type("Task", (), {"name": "grab_roller", "dataset_name": "robotwin_grab_roller"})(),
            "version": "1.0.0",
            "episode_count": 50,
        },
    )()

    episodes = list(iter_episodes("/source", inventory, [2, 0], builder_factory=factory))

    assert [episode.identity.ordinal for episode in episodes] == [2, 0]
    assert len(calls) == 1
