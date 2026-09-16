"""Reproducible observation and action-candidate smoke artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .candidates import trajectory_cluster_count


_IMAGE_NAMES = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
_CANDIDATES = 17
_HORIZON = 50
_MODEL_ACTION_DIM = 32
_ROBOT_ACTION_DIM = 14


@dataclass(frozen=True)
class SmokeObservation:
    task: str
    episode_ordinal: int
    images: Mapping[str, np.ndarray]
    state: np.ndarray
    prompt: str

    def __post_init__(self) -> None:
        if not self.task or not self.prompt.strip() or self.episode_ordinal < 0:
            raise ValueError("observation identity and prompt must be valid")
        if set(self.images) != set(_IMAGE_NAMES):
            raise ValueError(f"images must contain exactly {_IMAGE_NAMES}")
        images: dict[str, np.ndarray] = {}
        for name in _IMAGE_NAMES:
            value = np.asarray(self.images[name])
            if value.shape != (3, 240, 320) or value.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8 with shape (3,240,320)")
            detached = value.copy()
            detached.flags.writeable = False
            images[name] = detached
        state = np.asarray(self.state)
        if state.shape != (14,) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise ValueError("state must be finite float32 with shape (14,)")
        detached_state = state.copy()
        detached_state.flags.writeable = False
        object.__setattr__(self, "images", MappingProxyType(images))
        object.__setattr__(self, "state", detached_state)
        object.__setattr__(self, "prompt", " ".join(self.prompt.split()))


@dataclass(frozen=True)
class ActionDiversity:
    candidate_count: int
    cluster_count: int
    minimum_pairwise_rms: float
    median_pairwise_rms: float
    maximum_pairwise_rms: float
    cluster_distance: float
    executed_prefix: int


def make_flow_noises(
    seeds: tuple[int, ...],
    *,
    action_horizon: int = _HORIZON,
    action_dim: int = _MODEL_ACTION_DIM,
) -> np.ndarray:
    """Create explicit standard-normal flow noise without global RNG state."""

    if len(seeds) != _CANDIDATES or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must contain exactly {_CANDIDATES} unique values")
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("noise seeds must be nonnegative integers")
    if action_horizon <= 0 or action_dim <= 0:
        raise ValueError("noise shape must be positive")
    result = np.stack(
        [
            np.random.default_rng(seed).standard_normal(
                (action_horizon, action_dim), dtype=np.float32
            )
            for seed in seeds
        ]
    )
    result.flags.writeable = False
    return result


def _validated_actions(actions: np.ndarray) -> np.ndarray:
    value = np.asarray(actions)
    expected = (_CANDIDATES, _HORIZON, _ROBOT_ACTION_DIM)
    if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"candidate actions must be finite float32 with shape {expected}")
    return value


def standardize_policy_actions(actions: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert one π0.5 output chunk to float32 with an explicit error bound."""

    value = np.asarray(actions)
    if value.shape != (_HORIZON, _ROBOT_ACTION_DIM):
        raise ValueError(
            f"policy actions must have shape ({_HORIZON},{_ROBOT_ACTION_DIM}), got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("policy actions must contain only finite floating-point values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = value.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("policy actions are outside the finite float32 range")
    maximum_error = float(np.max(np.abs(value.astype(np.float64) - converted.astype(np.float64))))
    if maximum_error > 1e-5:
        raise ValueError(f"float32 conversion error {maximum_error} exceeds 1e-5")
    converted.flags.writeable = False
    return converted, maximum_error


def summarize_action_diversity(
    actions: np.ndarray,
    *,
    executed_prefix: int,
    cluster_distance: float,
) -> ActionDiversity:
    """Summarize normalized action-prefix separation before any rollout outcome."""

    value = _validated_actions(actions)
    if executed_prefix <= 0 or executed_prefix > _HORIZON:
        raise ValueError("executed_prefix is outside the action horizon")
    if not math.isfinite(cluster_distance) or cluster_distance <= 0:
        raise ValueError("cluster_distance must be positive and finite")
    prefixes = value[:, :executed_prefix].astype(np.float64)
    scale = prefixes.reshape(-1, _ROBOT_ACTION_DIM).std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    standardized = (prefixes - prefixes.mean(axis=(0, 1), keepdims=True)) / scale
    distances = np.asarray(
        [
            np.sqrt(np.mean(np.square(standardized[left] - standardized[right])))
            for left in range(_CANDIDATES)
            for right in range(left + 1, _CANDIDATES)
        ],
        dtype=np.float64,
    )
    return ActionDiversity(
        candidate_count=_CANDIDATES,
        cluster_count=trajectory_cluster_count(standardized, cluster_distance),
        minimum_pairwise_rms=float(distances.min()),
        median_pairwise_rms=float(np.median(distances)),
        maximum_pairwise_rms=float(distances.max()),
        cluster_distance=float(cluster_distance),
        executed_prefix=executed_prefix,
    )


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _publish(output: Path, files: Mapping[str, bytes], schema: str) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f"{output.name}.partial-{os.getpid()}")
    staging.mkdir()
    try:
        hashes: dict[str, str] = {}
        for name, data in files.items():
            target = staging / name
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(descriptor)
            hashes[name] = hashlib.sha256(data).hexdigest()
        completion = _canonical_bytes(
            {"schema": schema, "status": "complete", "artifact_sha256": hashes}
        )
        descriptor = os.open(
            staging / "completion.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(completion)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        directory = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging)
        raise


def publish_observation_bundle(
    output: Path | str,
    observation: SmokeObservation,
    *,
    provenance: Mapping[str, object],
) -> None:
    arrays = {name: observation.images[name] for name in _IMAGE_NAMES}
    arrays["state"] = observation.state
    metadata = {
        "schema": "D-JEPA-RoboTwin2-pi05-observation-v1",
        "task": observation.task,
        "episode_ordinal": observation.episode_ordinal,
        "prompt": observation.prompt,
        "provenance": dict(provenance),
    }
    _publish(
        Path(output),
        {
            "observation.npz": _npz_bytes(**arrays),
            "metadata.json": _canonical_bytes(metadata),
        },
        "D-JEPA-RoboTwin2-pi05-observation-completion-v1",
    )


def load_observation_bundle(output: Path | str) -> SmokeObservation:
    root = Path(output)
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError("observation bundle is incomplete")
    for name, expected in completion.get("artifact_sha256", {}).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"observation artifact hash mismatch: {name}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    with np.load(root / "observation.npz", allow_pickle=False) as archive:
        images = {name: archive[name] for name in _IMAGE_NAMES}
        state = archive["state"]
    return SmokeObservation(
        task=metadata["task"],
        episode_ordinal=int(metadata["episode_ordinal"]),
        images=images,
        state=state,
        prompt=metadata["prompt"],
    )


def load_candidate_bundle(
    output: Path | str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load and authenticate one fixed π0.5 candidate bundle."""

    root = Path(output)
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError("candidate bundle is incomplete")
    for name, expected in completion.get("artifact_sha256", {}).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"candidate artifact hash mismatch: {name}")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    with np.load(root / "candidates.npz", allow_pickle=False) as archive:
        actions = _validated_actions(archive["actions"]).copy()
        seeds = np.asarray(archive["seeds"], dtype=np.int64).copy()
    if seeds.shape != (_CANDIDATES,) or len(set(seeds.tolist())) != _CANDIDATES:
        raise ValueError("candidate seeds must contain 17 unique values")
    if int(metrics.get("native_index", -1)) != 0:
        raise ValueError("candidate bundle native index must be zero")
    actions.flags.writeable = False
    seeds.flags.writeable = False
    return actions, seeds, metrics


def publish_candidate_bundle(
    output: Path | str,
    *,
    actions: np.ndarray,
    seeds: tuple[int, ...],
    native_index: int,
    executed_prefix: int,
    cluster_distance: float,
    post_outcome_selected: bool,
    provenance: Mapping[str, object],
) -> ActionDiversity:
    if post_outcome_selected:
        raise ValueError("post-outcome candidate selection is forbidden")
    if native_index != 0:
        raise ValueError("native candidate index must be zero")
    if len(seeds) != _CANDIDATES or len(set(seeds)) != _CANDIDATES:
        raise ValueError("candidate seeds must contain 17 unique values")
    value = _validated_actions(actions)
    summary = summarize_action_diversity(
        value,
        executed_prefix=executed_prefix,
        cluster_distance=cluster_distance,
    )
    metrics = {
        "schema": "D-JEPA-RoboTwin2-pi05-candidate-smoke-v1",
        **asdict(summary),
        "native_index": native_index,
        "noise_seeds": list(seeds),
        "post_outcome_selected": False,
        "provenance": dict(provenance),
    }
    _publish(
        Path(output),
        {
            "candidates.npz": _npz_bytes(
                actions=value,
                seeds=np.asarray(seeds, dtype=np.int64),
            ),
            "metrics.json": _canonical_bytes(metrics),
        },
        "D-JEPA-RoboTwin2-pi05-candidate-smoke-completion-v1",
    )
    return summary
