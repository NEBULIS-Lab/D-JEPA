"""Explicit RoboTwin2-to-pi0.5 ALOHA interface transforms."""

from __future__ import annotations

from dataclasses import dataclass
import enum
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .rlds import EpisodeIdentity, RoboTwinEpisode, RoboTwinStep


JOINT_FLIP_MASK = np.asarray(
    [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1],
    dtype=np.float32,
)


class ActionConvention(enum.Enum):
    IDENTITY = "identity"
    OPENPI_ALOHA = "openpi_aloha"


@dataclass(frozen=True)
class Pi05Observation:
    images: Mapping[str, np.ndarray]
    state: np.ndarray
    prompt: str


@dataclass(frozen=True)
class Pi05TrainingExample:
    identity: EpisodeIdentity
    start: int
    observation: Pi05Observation
    actions: np.ndarray


def _read_only(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result.flags.writeable = False
    return result


def _validated_actions(actions: np.ndarray) -> np.ndarray:
    value = np.asarray(actions)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(f"actions must have shape [horizon,14], got {value.shape}")
    if value.dtype != np.float32:
        raise ValueError(f"actions must have dtype float32, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError("actions must contain only finite values")
    return value.copy()


def _normalize(value: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return (value - minimum) / (maximum - minimum)


def _unnormalize(value: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return value * (maximum - minimum) + minimum


def _gripper_from_angular(value: np.ndarray) -> np.ndarray:
    return _normalize(value + 0.5476, -0.6213, 1.4910)


def _gripper_from_angular_inverse(value: np.ndarray) -> np.ndarray:
    return _unnormalize(value, -0.6213, 1.4910) - 0.5476


def to_pi05_observation(step: RoboTwinStep) -> Pi05Observation:
    """Map one validated RoboTwin step to the published ALOHA input schema."""

    images = {
        "cam_high": _read_only(np.moveaxis(step.base_rgb, -1, 0)),
        "cam_low": _read_only(np.moveaxis(step.low_rgb, -1, 0)),
        "cam_left_wrist": _read_only(np.moveaxis(step.left_wrist_rgb, -1, 0)),
        "cam_right_wrist": _read_only(np.moveaxis(step.right_wrist_rgb, -1, 0)),
    }
    return Pi05Observation(
        images=MappingProxyType(images),
        state=_read_only(step.joint_state),
        prompt=step.instruction,
    )


def to_pi05_actions(actions: np.ndarray, convention: ActionConvention) -> np.ndarray:
    """Convert source actions into the explicit selected training convention."""

    value = _validated_actions(actions)
    if convention is ActionConvention.OPENPI_ALOHA:
        value *= JOINT_FLIP_MASK
        value[:, (6, 13)] = _gripper_from_angular_inverse(value[:, (6, 13)])
    elif convention is not ActionConvention.IDENTITY:
        raise ValueError(f"unsupported action convention: {convention!r}")
    value.flags.writeable = False
    return value


def from_pi05_actions(actions: np.ndarray, convention: ActionConvention) -> np.ndarray:
    """Invert :func:`to_pi05_actions` for interface admission."""

    value = _validated_actions(actions)
    if convention is ActionConvention.OPENPI_ALOHA:
        value *= JOINT_FLIP_MASK
        value[:, (6, 13)] = _gripper_from_angular(value[:, (6, 13)])
    elif convention is not ActionConvention.IDENTITY:
        raise ValueError(f"unsupported action convention: {convention!r}")
    value.flags.writeable = False
    return value


def make_training_example(
    episode: RoboTwinEpisode,
    *,
    start: int,
    action_horizon: int,
    convention: ActionConvention,
) -> Pi05TrainingExample:
    """Construct one fixed-horizon π0.5 example without mutating the episode."""

    if start < 0 or action_horizon <= 0 or start + action_horizon > len(episode.steps):
        raise ValueError("requested action horizon is outside the episode")
    source_actions = np.stack(
        [step.joint_action for step in episode.steps[start : start + action_horizon]],
        axis=0,
    )
    return Pi05TrainingExample(
        identity=episode.identity,
        start=start,
        observation=to_pi05_observation(episode.steps[start]),
        actions=to_pi05_actions(source_actions, convention),
    )
