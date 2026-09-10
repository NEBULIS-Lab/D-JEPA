"""Explicit Cartesian contract matching V-JEPA2-AC's public planning utilities.

This is a prediction-space contract, NOT a calibrated PiPER actuator interface.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

SEMANTICS = "eef_delta_xyz_extrinsic_xyz_gripper"


def _seven(values, name):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != 7 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite [...,7]")
    return x


def _pose(values):
    x = _seven(values, "pose")
    if np.any((x[..., -1] < 0) | (x[..., -1] > 1)):
        raise ValueError("pose gripper closedness must be normalized to [0,1]")
    return x


def pose_delta(start, end):
    start, end = np.broadcast_arrays(_pose(start), _pose(end))
    s, e = start.reshape(-1, 7), end.reshape(-1, 7)
    rotation = Rotation.from_euler("xyz", e[:, 3:6]) * Rotation.from_euler("xyz", s[:, 3:6]).inv()
    result = e - s
    result[:, 3:6] = rotation.as_euler("xyz")
    return result.reshape(start.shape)


def advance_pose(start, delta):
    start, delta = np.broadcast_arrays(_pose(start), _seven(delta, "delta"))
    s, d = start.reshape(-1, 7), delta.reshape(-1, 7)
    result = s + d
    result[:, 3:6] = (Rotation.from_euler("xyz", d[:, 3:6]) *
                      Rotation.from_euler("xyz", s[:, 3:6])).as_euler("xyz")
    result[:, -1] = np.clip(result[:, -1], 0, 1)
    return result.reshape(start.shape)


def validate_limits(lower, upper):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.shape != (7,) or upper.shape != (7,) or not np.isfinite([lower, upper]).all():
        raise ValueError("bounds must be finite seven-vectors")
    if np.any(lower > upper):
        raise ValueError("lower bounds exceed upper bounds")
    return lower, upper


@dataclass(frozen=True)
class CandidateBatch:
    ids: np.ndarray
    actions: np.ndarray
    semantics: str
    dt: float

    def __post_init__(self):
        ids, actions = np.asarray(self.ids), np.asarray(self.actions, dtype=np.float64)
        if self.semantics != SEMANTICS:
            raise ValueError("joint/native policy commands require calibrated conversion; wrong semantics")
        if ids.ndim != 1 or len(ids) == 0 or ids.dtype.kind not in "iu":
            raise ValueError("candidate IDs must be a nonempty integer vector")
        if np.any(ids < 0) or np.any(ids > np.iinfo(np.int64).max) or len(np.unique(ids)) != len(ids):
            raise ValueError("candidate IDs must be distinct nonnegative int64 values")
        if actions.ndim != 3 or actions.shape[0] != len(ids) or actions.shape[1] < 1 or actions.shape[2] != 7:
            raise ValueError("actions must be [candidates,horizon>=1,7]")
        if not np.isfinite(actions).all() or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("actions and positive dt must be finite")
        ids, actions = ids.astype(np.int64, copy=True), actions.copy()
        ids.flags.writeable = actions.flags.writeable = False
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "actions", actions)

    def validate_bounds(self, lower, upper):
        lower, upper = validate_limits(lower, upper)
        if np.any(self.actions < lower) or np.any(self.actions > upper):
            raise ValueError("candidate actions exceed explicit bounds")
