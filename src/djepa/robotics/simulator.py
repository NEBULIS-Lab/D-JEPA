"""Pure RoboTwin simulator interface and task-progress contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .smoke import SmokeObservation


_CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_low": "front_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def multiview_pixel_distance(
    query: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    stride: int = 4,
) -> float:
    """Deterministic normalized L1 scene distance over the three policy views."""

    names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    if stride <= 0:
        raise ValueError("pixel stride must be positive")
    distances: list[float] = []
    for name in names:
        if name not in query or name not in reference:
            raise ValueError(f"both image sets must contain {name}")
        left = np.asarray(query[name])
        right = np.asarray(reference[name])
        if left.shape != right.shape or left.ndim != 3:
            raise ValueError(f"{name} images must have equal three-dimensional shapes")
        sample_left = left[:, ::stride, ::stride].astype(np.float32)
        sample_right = right[:, ::stride, ::stride].astype(np.float32)
        distances.append(float(np.mean(np.abs(sample_left - sample_right)) / 255.0))
    return float(np.mean(distances))


def fixed_action_window(actions: np.ndarray, *, start: int, horizon: int = 50) -> np.ndarray:
    """Return a fixed action window, padding a short tail with its final command."""

    value = np.asarray(actions)
    if value.ndim != 2 or value.shape[1] != 14 or value.dtype != np.float32:
        raise ValueError("demonstration actions must be float32 with shape [time,14]")
    if not np.isfinite(value).all() or start < 0 or horizon <= 0 or start >= len(value):
        raise ValueError("requested action window is invalid")
    window = value[start : start + horizon].copy()
    if len(window) < horizon:
        window = np.concatenate(
            [window, np.repeat(window[-1:], horizon - len(window), axis=0)], axis=0
        )
    window.flags.writeable = False
    return window


def unique_consecutive_actions(actions: np.ndarray) -> np.ndarray:
    """Remove redundant adjacent qpos records while preserving trajectory order."""

    value = np.asarray(actions)
    if value.ndim != 2 or value.shape[1] != 14 or value.dtype != np.float32:
        raise ValueError("actions must be float32 with shape [time,14]")
    if not len(value) or not np.isfinite(value).all():
        raise ValueError("actions must be nonempty and finite")
    keep = np.ones(len(value), dtype=bool)
    keep[1:] = np.any(value[1:] != value[:-1], axis=1)
    result = value[keep].copy()
    result.flags.writeable = False
    return result


def to_pi05_simulator_observation(
    observation: Any,
    *,
    task: str,
    context_ordinal: int,
) -> SmokeObservation:
    """Map a CoWAM RoboTwin observation into the admitted π0.5 schema."""

    images: dict[str, np.ndarray] = {}
    for destination, source in _CAMERA_MAP.items():
        payload = observation.sensors.get(source)
        if not isinstance(payload, dict) or "rgb" not in payload:
            raise ValueError(f"RoboTwin observation does not contain {source}.rgb")
        rgb = np.asarray(payload["rgb"])
        if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"{source}.rgb must be uint8 with shape (240,320,3)")
        images[destination] = np.moveaxis(rgb, -1, 0)
    state = np.asarray(observation.state, dtype=np.float32)
    return SmokeObservation(
        task=task,
        episode_ordinal=context_ordinal,
        images=images,
        state=state,
        prompt=str(observation.instruction),
    )


@dataclass(frozen=True)
class GrabRollerReference:
    roller_z: float
    mean_tcp_distance: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.roller_z):
            raise ValueError("reference roller height must be finite")
        if not np.isfinite(self.mean_tcp_distance) or self.mean_tcp_distance <= 0.0:
            raise ValueError("reference TCP distance must be positive and finite")


@dataclass(frozen=True)
class BranchOutcome:
    candidate_index: int
    noise_seed: int
    success: bool
    progress: float
    roller_position: tuple[float, float, float]
    mean_tcp_distance: float

    def __post_init__(self) -> None:
        if self.candidate_index < 0 or self.noise_seed < 0:
            raise ValueError("candidate identity must be nonnegative")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("branch progress must lie in [0,1]")
        values = (*self.roller_position, self.mean_tcp_distance)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("branch geometry must be finite")


def summarize_branch_outcomes(outcomes: tuple[BranchOutcome, ...]) -> dict[str, object]:
    """Summarize a fixed unfiltered branch set with candidate zero as native."""

    if not outcomes or outcomes[0].candidate_index != 0:
        raise ValueError("branch outcomes must begin with native candidate zero")
    if [row.candidate_index for row in outcomes] != list(range(len(outcomes))):
        raise ValueError("branch outcomes must preserve consecutive candidate order")
    oracle = max(outcomes, key=lambda row: (int(row.success), row.progress, -row.candidate_index))
    progress_values = [row.progress for row in outcomes]
    return {
        "candidate_count": len(outcomes),
        "native_index": 0,
        "native_success": outcomes[0].success,
        "native_progress": outcomes[0].progress,
        "oracle_index": oracle.candidate_index,
        "oracle_success": oracle.success,
        "oracle_progress": oracle.progress,
        "success_candidates": sum(row.success for row in outcomes),
        "progress_minimum": min(progress_values),
        "progress_maximum": max(progress_values),
        "progress_span": max(progress_values) - min(progress_values),
        "outcomes": [asdict(row) for row in outcomes],
    }


def mean_tcp_distance(
    roller_position: np.ndarray,
    left_tcp: np.ndarray,
    right_tcp: np.ndarray,
) -> float:
    roller = np.asarray(roller_position, dtype=np.float64)
    left = np.asarray(left_tcp, dtype=np.float64)
    right = np.asarray(right_tcp, dtype=np.float64)
    if roller.shape != (3,) or left.shape != (3,) or right.shape != (3,):
        raise ValueError("roller and TCP positions must be three-dimensional")
    if not all(np.isfinite(value).all() for value in (roller, left, right)):
        raise ValueError("roller and TCP positions must be finite")
    return float((np.linalg.norm(left - roller) + np.linalg.norm(right - roller)) / 2.0)


def grab_roller_progress(
    reference: GrabRollerReference,
    *,
    roller_position: np.ndarray,
    left_tcp: np.ndarray,
    right_tcp: np.ndarray,
    left_closed: bool,
    right_closed: bool,
    success: bool,
) -> float:
    """Continuous preregistered progress for early grab-roller prefixes."""

    if success:
        return 1.0
    distance = mean_tcp_distance(roller_position, left_tcp, right_tcp)
    approach = np.clip(1.0 - distance / reference.mean_tcp_distance, 0.0, 1.0)
    bilateral_close = float(bool(left_closed) and bool(right_closed))
    useful_close = bilateral_close * float(approach)
    roller_z = float(np.asarray(roller_position, dtype=np.float64)[2])
    lift_denominator = max(0.8 - reference.roller_z, 1e-6)
    lift = np.clip((roller_z - reference.roller_z) / lift_denominator, 0.0, 1.0)
    return float(np.clip(0.35 * approach + 0.15 * useful_close + 0.50 * lift, 0.0, 1.0))
