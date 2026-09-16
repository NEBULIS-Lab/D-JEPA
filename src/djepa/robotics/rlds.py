"""Lazy, read-only RoboTwin2 RLDS episode validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .inventory import SourceInventory


@dataclass(frozen=True)
class EpisodeIdentity:
    task: str
    ordinal: int


@dataclass(frozen=True)
class RoboTwinStep:
    base_rgb: np.ndarray
    low_rgb: np.ndarray
    left_wrist_rgb: np.ndarray
    right_wrist_rgb: np.ndarray
    joint_state: np.ndarray
    eef_state: np.ndarray
    joint_action: np.ndarray
    eef_action: np.ndarray
    instruction: str
    reward: float
    is_first: bool
    is_last: bool
    is_terminal: bool


@dataclass(frozen=True)
class RoboTwinEpisode:
    identity: EpisodeIdentity
    steps: tuple[RoboTwinStep, ...]


def _numpy(value: object) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    finite: bool,
) -> np.ndarray:
    result = _numpy(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if result.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {result.dtype}")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result = result.copy()
    result.flags.writeable = False
    return result


def _scalar_bool(value: object, name: str) -> bool:
    result = _numpy(value)
    if result.shape != () or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a scalar bool")
    return bool(result)


def _instruction(value: object) -> str:
    array = _numpy(value)
    values = array.reshape(-1).tolist()
    if not values:
        raise ValueError("instruction must be nonempty")
    decoded: list[str] = []
    for item in values:
        if isinstance(item, bytes):
            try:
                item = item.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("instruction must be valid UTF-8") from error
        if not isinstance(item, str) or not item.strip():
            raise ValueError("instruction must be nonempty")
        decoded.append(" ".join(item.split()))
    # RoboTwin2 stores a stable ordered bank of task paraphrases at every step.
    # The first entry is the preregistered canonical prompt; episode validation
    # still rejects any change to that canonical entry over time.
    return decoded[0]


def _steps(raw: Mapping[str, object]) -> Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]]:
    if "steps" not in raw:
        raise ValueError("episode must contain steps")
    steps = raw["steps"]
    if isinstance(steps, Mapping) or isinstance(steps, (str, bytes)):
        raise ValueError("steps must be an iterable of mappings")
    try:
        return iter(steps)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("steps must be iterable") from error


def validate_episode(raw: Mapping[str, object], identity: EpisodeIdentity) -> RoboTwinEpisode:
    """Validate and detach one raw episode from TensorFlow-backed storage."""

    if identity.ordinal < 0:
        raise ValueError("episode ordinal must be nonnegative")
    validated: list[RoboTwinStep] = []
    for index, raw_step in enumerate(_steps(raw)):
        if not isinstance(raw_step, Mapping):
            raise ValueError("every step must be a mapping")
        observation = raw_step.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError("step observation must be a mapping")
        reward_array = _numpy(raw_step.get("reward"))
        if reward_array.shape != () or not np.issubdtype(reward_array.dtype, np.number):
            raise ValueError("reward must be a numeric scalar")
        reward = float(reward_array)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        validated.append(
            RoboTwinStep(
                base_rgb=_array(
                    observation.get("image"),
                    name="base_rgb",
                    shape=(240, 320, 3),
                    dtype=np.dtype(np.uint8),
                    finite=False,
                ),
                low_rgb=_array(
                    observation.get("low_cam_image"),
                    name="low_rgb",
                    shape=(240, 320, 3),
                    dtype=np.dtype(np.uint8),
                    finite=False,
                ),
                left_wrist_rgb=_array(
                    observation.get("left_wrist_image"),
                    name="left_wrist_rgb",
                    shape=(240, 320, 3),
                    dtype=np.dtype(np.uint8),
                    finite=False,
                ),
                right_wrist_rgb=_array(
                    observation.get("right_wrist_image"),
                    name="right_wrist_rgb",
                    shape=(240, 320, 3),
                    dtype=np.dtype(np.uint8),
                    finite=False,
                ),
                joint_state=_array(
                    observation.get("joint_state"),
                    name="joint_state",
                    shape=(14,),
                    dtype=np.dtype(np.float32),
                    finite=True,
                ),
                eef_state=_array(
                    observation.get("eef_state"),
                    name="eef_state",
                    shape=(20,),
                    dtype=np.dtype(np.float32),
                    finite=True,
                ),
                joint_action=_array(
                    raw_step.get("joint_action"),
                    name="joint_action",
                    shape=(14,),
                    dtype=np.dtype(np.float32),
                    finite=True,
                ),
                eef_action=_array(
                    raw_step.get("eef_action"),
                    name="eef_action",
                    shape=(20,),
                    dtype=np.dtype(np.float32),
                    finite=True,
                ),
                instruction=_instruction(raw_step.get("language_instruction")),
                reward=reward,
                is_first=_scalar_bool(raw_step.get("is_first"), "is_first"),
                is_last=_scalar_bool(raw_step.get("is_last"), "is_last"),
                is_terminal=_scalar_bool(raw_step.get("is_terminal"), "is_terminal"),
            )
        )
    if not validated:
        raise ValueError("episode must contain at least one step")
    if [step.is_first for step in validated] != [True] + [False] * (len(validated) - 1):
        raise ValueError("is_first must be true only on the first step")
    if [step.is_last for step in validated] != [False] * (len(validated) - 1) + [True]:
        raise ValueError("is_last must be true only on the final step")
    terminal_indices = [index for index, step in enumerate(validated) if step.is_terminal]
    if terminal_indices not in ([], [len(validated) - 1]):
        raise ValueError("is_terminal may be true only on the final step")
    instructions = {step.instruction for step in validated}
    if len(instructions) != 1:
        raise ValueError("instruction must remain constant across an episode")
    return RoboTwinEpisode(identity=identity, steps=tuple(validated))


def _default_builder(version_root: Path):
    try:
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise RuntimeError("TensorFlow Datasets is required to read production RLDS records") from error
    return tfds.builder_from_directory(builder_dir=str(version_root))


def iter_episodes(
    dataset_root: Path | str,
    inventory: SourceInventory,
    ordinals: Iterable[int],
    *,
    builder_factory: Callable[[Path], object] | None = None,
) -> Iterator[RoboTwinEpisode]:
    """Yield only predeclared episode ordinals, preserving requested order."""

    requested = tuple(int(value) for value in ordinals)
    if len(set(requested)) != len(requested):
        raise ValueError("requested episode ordinals must be unique")
    if any(value < 0 or value >= inventory.episode_count for value in requested):
        raise ValueError("requested episode ordinal is outside the inventory")
    if not requested:
        return
    version_root = Path(dataset_root) / inventory.task.dataset_name / inventory.version
    factory = builder_factory or _default_builder
    builder = factory(version_root)
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    wanted = set(requested)
    found: dict[int, RoboTwinEpisode] = {}
    for ordinal, raw in enumerate(dataset):
        if ordinal in wanted:
            found[ordinal] = validate_episode(
                raw,
                EpisodeIdentity(task=inventory.task.name, ordinal=ordinal),
            )
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - found.keys())
    if missing:
        raise ValueError(f"dataset ended before requested episodes were found: {missing}")
    for ordinal in requested:
        yield found[ordinal]
