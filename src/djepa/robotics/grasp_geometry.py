"""Small scene-conditioned geometry model for RoboTwin grasp proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def colored_tool_geometry(image: np.ndarray) -> np.ndarray:
    """Extract color-invariant line geometry for the saturated tabletop tool."""

    value = np.asarray(image)
    _require(value.shape == (240, 320, 3) and value.dtype == np.uint8,
             "image must be uint8 with shape (240,320,3)")
    channels = value.astype(np.float32)
    chroma = channels.max(axis=-1) - channels.min(axis=-1)
    yy, xx = np.indices(chroma.shape)
    mask = (chroma > 25.0) & (yy > 105) & (xx > 25) & (xx < 318)
    rows, columns = np.nonzero(mask)
    _require(len(columns) >= 64, "fewer than 64 colored tool pixels were detected")
    points = np.stack((columns, rows), axis=1).astype(np.float64)
    center = points.mean(axis=0)
    _, _, right = np.linalg.svd(points - center, full_matrices=False)
    direction = right[0]
    if direction[0] < 0.0:
        direction = -direction
    coordinates = (points - center) @ direction
    low, high = np.quantile(coordinates, (0.02, 0.98))
    first = center + low * direction
    second = center + high * direction
    return np.asarray(
        [*center, *direction, high - low, *first, *second], dtype=np.float32
    )


def _design(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 9,
             "geometry features must have shape (N,9)")
    return np.column_stack(
        (
            np.ones(len(values)),
            values[:, 0:2] / 320.0,
            values[:, 2:4],
            values[:, 4:5] / 200.0,
            values[:, 5:7] / 320.0,
            values[:, 7:9] / 320.0,
        )
    )


@dataclass(frozen=True)
class GeometryRidge:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    lambda_value: float
    calibration_rmse: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        design = _design(features)
        standardized = (design - self.mean) / self.scale
        standardized[:, 0] = 1.0
        return np.asarray(standardized @ self.weights, dtype=np.float32)


@dataclass(frozen=True)
class PreservationGate:
    """A calibrated rule that preserves native behavior when it is aligned."""

    threshold: float
    development_success: int
    correction_count: int

    def choose_correction(self, alignment_cost: np.ndarray) -> np.ndarray:
        values = np.asarray(alignment_cost, dtype=np.float64)
        _require(np.isfinite(values).all(), "alignment costs must be finite")
        return values > self.threshold


@dataclass(frozen=True)
class EarlyGraspGate:
    """Correct native futures whose closest synchronized grasp occurs too early."""

    threshold: float
    development_success: int
    correction_count: int

    def choose_correction(self, closest_step_fraction: np.ndarray) -> np.ndarray:
        values = np.asarray(closest_step_fraction, dtype=np.float64)
        _require(np.isfinite(values).all(), "closest-step fractions must be finite")
        return values < self.threshold


def native_target_relations(
    actions: np.ndarray, predicted_target: np.ndarray
) -> dict[str, float]:
    """Summarize native grasp timing and bimanual geometry relative to a target."""

    values = np.asarray(actions, dtype=np.float64)
    target = np.asarray(predicted_target, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 16 and len(values) > 0,
             "EE actions must have shape (T,16)")
    _require(np.isfinite(values).all(), "EE actions must be finite")
    _require(target.shape == (6,) and np.isfinite(target).all(),
             "predicted target must be a finite six-vector")
    positions = np.stack((values[:, 0:3], values[:, 8:11]), axis=1)
    closed = (values[:, 7] <= 0.5) & (values[:, 15] <= 0.5)
    closed_fraction = float(np.mean(closed))
    indices = np.flatnonzero(closed)
    if np.any(closed):
        positions = positions[closed]
    else:
        indices = np.arange(len(values))
    per_arm = np.linalg.norm(positions - target.reshape(2, 3), axis=-1)
    mean_distance = per_arm.mean(axis=1)
    closest_local = int(np.argmin(mean_distance))
    closest_step = int(indices[closest_local])
    native_span = positions[closest_local, 1] - positions[closest_local, 0]
    target_span = target[3:6] - target[0:3]
    return {
        "alignment_cost_m": float(mean_distance[closest_local]),
        "left_alignment_m": float(per_arm[closest_local, 0]),
        "right_alignment_m": float(per_arm[closest_local, 1]),
        "arm_imbalance_m": float(abs(per_arm[closest_local, 0] - per_arm[closest_local, 1])),
        "relative_geometry_error_m": float(np.linalg.norm(native_span - target_span)),
        "closest_closed_step_fraction": float(closest_step / len(values)),
        "closed_fraction": closed_fraction,
    }


def native_target_alignment(actions: np.ndarray, predicted_target: np.ndarray) -> float:
    """Measure how closely a native EE trace reaches the predicted grasp target."""

    return native_target_relations(actions, predicted_target)["alignment_cost_m"]


def calibrate_preservation_gate(
    alignment_costs: np.ndarray,
    *,
    native_success: np.ndarray,
    correction_success: np.ndarray,
) -> PreservationGate:
    """Choose a development-only threshold, breaking ties toward preservation."""

    costs = np.asarray(alignment_costs, dtype=np.float64)
    native = np.asarray(native_success, dtype=bool)
    correction = np.asarray(correction_success, dtype=bool)
    _require(costs.ndim == native.ndim == correction.ndim == 1,
             "gate inputs must be one-dimensional")
    _require(len(costs) > 0 and len(costs) == len(native) == len(correction),
             "gate inputs must be nonempty and have equal length")
    _require(np.isfinite(costs).all(), "alignment costs must be finite")
    unique = np.unique(costs)
    thresholds = [float(np.nextafter(unique[0], -np.inf))]
    thresholds.extend(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:]))
    thresholds.append(float(unique[-1]))
    best: tuple[int, int, float, int] | None = None
    for threshold in thresholds:
        use_correction = costs > threshold
        successes = np.where(use_correction, correction, native)
        candidate = (
            int(successes.sum()),
            -int(use_correction.sum()),
            threshold,
            int(use_correction.sum()),
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    return PreservationGate(
        threshold=best[2],
        development_success=best[0],
        correction_count=best[3],
    )


def calibrate_early_grasp_gate(
    closest_step_fractions: np.ndarray,
    *,
    native_success: np.ndarray,
    correction_success: np.ndarray,
) -> EarlyGraspGate:
    """Calibrate an early-grasp boundary with preservation-oriented tie breaks."""

    fractions = np.asarray(closest_step_fractions, dtype=np.float64)
    native = np.asarray(native_success, dtype=bool)
    correction = np.asarray(correction_success, dtype=bool)
    _require(fractions.ndim == native.ndim == correction.ndim == 1,
             "gate inputs must be one-dimensional")
    _require(len(fractions) > 0 and len(fractions) == len(native) == len(correction),
             "gate inputs must be nonempty and have equal length")
    _require(np.isfinite(fractions).all(), "closest-step fractions must be finite")
    _require(bool(np.all((fractions >= 0.0) & (fractions < 1.0))),
             "closest-step fractions must lie in [0,1)")
    unique = np.unique(fractions)
    thresholds = [float(unique[0])]
    thresholds.extend(
        float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:])
    )
    thresholds.append(float(np.nextafter(unique[-1], np.inf)))
    best: tuple[int, int, float, int, float] | None = None
    for threshold in thresholds:
        use_correction = fractions < threshold
        successes = np.where(use_correction, correction, native)
        candidate = (
            int(successes.sum()),
            -int(use_correction.sum()),
            -threshold,
            int(use_correction.sum()),
            threshold,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    return EarlyGraspGate(
        threshold=best[4],
        development_success=best[0],
        correction_count=best[3],
    )


def fit_geometry_ridge(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    *,
    lambdas: Iterable[float] = (0.0001, 0.001, 0.01, 0.1, 1.0),
) -> GeometryRidge:
    """Fit on train rows and choose ridge strength solely on calibration rows."""

    train_design = _design(train_features)
    calibration_design = _design(calibration_features)
    train_y = np.asarray(train_targets, dtype=np.float64)
    calibration_y = np.asarray(calibration_targets, dtype=np.float64)
    _require(train_y.shape == (len(train_design), 6),
             "training targets must have shape (N,6)")
    _require(calibration_y.shape == (len(calibration_design), 6),
             "calibration targets must have shape (N,6)")
    _require(len(train_design) > 0 and len(calibration_design) > 0,
             "train and calibration sets must both be nonempty")
    mean = train_design.mean(axis=0)
    scale = train_design.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_x = (train_design - mean) / scale
    calibration_x = (calibration_design - mean) / scale
    train_x[:, 0] = 1.0
    calibration_x[:, 0] = 1.0
    best: tuple[float, float, np.ndarray] | None = None
    for raw_lambda in lambdas:
        lambda_value = float(raw_lambda)
        _require(np.isfinite(lambda_value) and lambda_value >= 0.0,
                 "ridge strengths must be finite and nonnegative")
        penalty = np.eye(train_x.shape[1], dtype=np.float64) * lambda_value
        penalty[0, 0] = 0.0
        weights = np.linalg.pinv(train_x.T @ train_x + penalty) @ train_x.T @ train_y
        prediction = calibration_x @ weights
        rmse = float(np.sqrt(np.mean(np.square(prediction - calibration_y))))
        candidate = (rmse, lambda_value, weights)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return GeometryRidge(
        mean=mean,
        scale=scale,
        weights=best[2],
        lambda_value=best[1],
        calibration_rmse=best[0],
    )


def grasp_target(actions: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the bimanual grasp pose at the first synchronized close event."""

    values = np.asarray(actions)
    _require(values.ndim == 2 and values.shape[1] == 20,
             "RoboTwin eef actions must have shape (T,20)")
    _require(len(values) > 0 and np.isfinite(values).all(),
             "RoboTwin eef actions must be nonempty and finite")
    closing_steps = []
    for index in (9, 19):
        closed = np.flatnonzero(values[:, index] <= 0.5)
        _require(len(closed) > 0, "trajectory has no gripper-close event")
        closing_steps.append(int(closed[0]))
    step = int(round(float(np.mean(closing_steps))))
    target = np.concatenate((values[step, 0:3], values[step, 10:13]))
    return np.asarray(target, dtype=np.float32), step


def normalize_target_span(target: np.ndarray, *, span: float) -> np.ndarray:
    value = np.asarray(target, dtype=np.float64)
    _require(value.shape == (6,) and np.isfinite(value).all(),
             "target must be a finite six-vector")
    _require(np.isfinite(span) and span > 0.0, "span must be positive")
    pair = value.reshape(2, 3)
    center = pair.mean(axis=0)
    direction = pair[1] - pair[0]
    norm = float(np.linalg.norm(direction))
    _require(norm > 1e-8, "target endpoints must be distinct")
    half = 0.5 * span * direction / norm
    return np.asarray(np.concatenate((center - half, center + half)), dtype=np.float32)


def spatial_target_grid(
    target: np.ndarray,
    *,
    along: Iterable[float] = (0.0, 0.02, 0.04, 0.06),
    perpendicular: Iterable[float] = (-0.03, -0.01, 0.01, 0.03),
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    value = np.asarray(target, dtype=np.float64)
    _require(value.shape == (6,) and np.isfinite(value).all(),
             "target must be a finite six-vector")
    direction = value[3:6] - value[0:3]
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    _require(norm > 1e-8, "target must define a tabletop direction")
    direction /= norm
    normal = np.asarray((-direction[1], direction[0], 0.0))
    candidates: list[np.ndarray] = []
    rows: list[dict[str, float | int]] = []
    for along_value in along:
        for perpendicular_value in perpendicular:
            shift = float(along_value) * direction + float(perpendicular_value) * normal
            shifted = value.reshape(2, 3) + shift[None]
            candidates.append(shifted.reshape(-1))
            rows.append(
                {
                    "candidate_index": len(candidates),
                    "along_shift": float(along_value),
                    "perpendicular_shift": float(perpendicular_value),
                }
            )
    _require(len(candidates) > 0, "spatial target grid must be nonempty")
    return np.asarray(candidates, dtype=np.float32), rows


def warp_ee_trajectory(
    actions: np.ndarray,
    source_target: np.ndarray,
    destination_target: np.ndarray,
    *,
    grasp_step: int,
) -> np.ndarray:
    """Smoothly align an EE demonstration to a predicted target geometry."""

    value = np.asarray(actions, dtype=np.float64)
    source = np.asarray(source_target, dtype=np.float64)
    destination = np.asarray(destination_target, dtype=np.float64)
    _require(value.ndim == 2 and value.shape[1] == 16 and len(value) > 0,
             "EE actions must have shape (T,16)")
    _require(source.shape == (6,) and destination.shape == (6,),
             "source and destination targets must be six-vectors")
    _require(0 < grasp_step < len(value), "grasp step must lie inside the trajectory")
    result = value.copy()
    progress = np.minimum(np.arange(len(value), dtype=np.float64) / grasp_step, 1.0)
    progress = progress * progress * (3.0 - 2.0 * progress)
    delta = (destination - source).reshape(2, 3)
    result[:, 0:3] += progress[:, None] * delta[0]
    result[:, 8:11] += progress[:, None] * delta[1]
    for rotation_slice in (slice(3, 7), slice(11, 15)):
        quaternion = result[:, rotation_slice]
        norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        _require(bool(np.all(norm > 1e-8)), "trajectory contains a zero quaternion")
        result[:, rotation_slice] = quaternion / norm
    _require(np.isfinite(result).all(), "warped trajectory contains non-finite values")
    return np.asarray(result, dtype=np.float32)
