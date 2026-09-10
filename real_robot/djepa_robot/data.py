"""Read recorded states/commands and audit timing, without decoding RGB streams."""

import json
from pathlib import Path

import h5py
import numpy as np


def _timestamps(values, length, name):
    values = np.asarray(values)
    if values.shape != (length,) or values.dtype.kind not in "iu":
        raise ValueError(f"{name}: expected integer nanosecond timestamps [{length}]")
    values = values.astype(np.int64)
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name}: timestamps must be strictly increasing")
    return values


def read_episode(path):
    path = Path(path).resolve(strict=True)
    with h5py.File(path, "r") as f:
        required = ["left_arm/joint", "left_arm/gripper", "left_arm/action",
                    "left_arm/timestamp"]
        required += [f"{cam}/{key}" for cam in ("cam_head", "cam_wrist")
                     for key in ("color", "timestamp")]
        missing = [key for key in required if key not in f]
        if missing:
            raise ValueError(f"{path}: missing measured fields {missing}; no inferred commands allowed")
        joints = f["left_arm/joint"][:]
        if joints.ndim != 2 or joints.shape[1] != 6 or len(joints) < 2:
            raise ValueError("joint feedback must have shape [T>=2,6]")
        n = len(joints)
        gripper, commands = f["left_arm/gripper"][:], f["left_arm/action"][:]
        if gripper.shape not in ((n,), (n, 1)) or commands.shape != (n, 7):
            raise ValueError("gripper or command stream length/shape mismatch")
        state = np.concatenate((joints, gripper.reshape(n, 1)), axis=-1)
        if not np.isfinite(state).all() or not np.isfinite(commands).all():
            raise ValueError("nonfinite state or command")
        timestamps = _timestamps(f["left_arm/timestamp"][:], n, "left_arm")
        image_shapes, camera_timestamps = {}, {}
        for cam in ("cam_head", "cam_wrist"):
            shape = tuple(f[f"{cam}/color"].shape)
            if len(shape) != 4 or shape[0] != n or shape[-1] != 3:
                raise ValueError(f"{cam}: RGB stream must have matching shape [T,H,W,3]")
            if f[f"{cam}/color"].dtype != np.uint8:
                raise ValueError(f"{cam}: RGB stream must be uint8")
            image_shapes[cam] = shape
            camera_timestamps[cam] = _timestamps(f[f"{cam}/timestamp"][:], n, cam)
    return {"path": str(path), "state": state, "commands": commands,
            "action_semantics": "joint_position_target_plus_gripper",
            "timestamp_ns": timestamps,
            "time_s": (timestamps - timestamps[0]).astype(np.float64) * 1e-9,
            "camera_timestamp_ns": camera_timestamps, "image_shapes": image_shapes}


def valid_window_starts(timestamp_ns, horizon, min_dt, max_dt):
    """Indices with H consecutive measured transitions inside the requested dt range."""
    if not isinstance(horizon, int) or horizon < 1 or not 0 < min_dt <= max_dt:
        raise ValueError("invalid horizon or dt limits")
    stamps = _timestamps(timestamp_ns, len(timestamp_ns), "window")
    dt = np.diff(stamps).astype(np.float64) * 1e-9
    if len(dt) < horizon:
        return np.empty(0, dtype=np.int64)
    valid = (dt >= min_dt) & (dt <= max_dt)
    return np.flatnonzero(np.lib.stride_tricks.sliding_window_view(valid, horizon).all(axis=1))


def summarize_episode(episode, nominal_fps=10, gap_factor=2.0):
    if not np.isfinite(nominal_fps) or nominal_fps <= 0 or gap_factor <= 1:
        raise ValueError("invalid nominal timing")
    stamps = episode["timestamp_ns"]
    dt = np.diff(stamps).astype(np.float64) * 1e-9
    gaps = np.flatnonzero(dt > gap_factor / nominal_fps)
    skew = max(float(np.abs(ts - stamps).max()) * 1e-9
               for ts in episode["camera_timestamp_ns"].values())
    return {"path": episode["path"], "frames": len(stamps),
            "duration_s": float(episode["time_s"][-1]),
            "dt_min_s": float(dt.min()), "dt_median_s": float(np.median(dt)),
            "dt_max_s": float(dt.max()), "gap_count": len(gaps),
            "gap_after_indices": gaps.tolist(), "gap_threshold_s": gap_factor / nominal_fps,
            "camera_skew_max_s": skew, "image_shapes": episode["image_shapes"],
            "commands_equal_feedback": bool(np.array_equal(episode["commands"], episode["state"]))}


def audit_dataset(lerobot_dir, raw_dir):
    lerobot_dir, raw_dir = Path(lerobot_dir).resolve(strict=True), Path(raw_dir).resolve(strict=True)
    with (lerobot_dir / "meta/info.json").open() as f:
        info = json.load(f)
    files = sorted(raw_dir.glob("*.hdf5"))
    reports, invalid = [], []
    for path in files:
        try:
            reports.append(summarize_episode(read_episode(path), float(info["fps"])))
        except (ValueError, OSError, KeyError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    frames = sum(r["frames"] for r in reports)
    return {"lerobot_dir": str(lerobot_dir), "raw_dir": str(raw_dir),
            "scope": "raw numeric streams and RGB shapes; Parquet pixels not cross-compared",
            "nominal_fps": info["fps"], "expected_episodes": info["total_episodes"],
            "expected_frames": info["total_frames"], "raw_files": len(files),
            "valid_episodes": len(reports), "raw_frames": frames,
            "metadata_counts_match": not invalid and len(files) == info["total_episodes"]
                                     and frames == info["total_frames"],
            "episodes_with_gaps": sum(r["gap_count"] > 0 for r in reports),
            "gap_count": sum(r["gap_count"] for r in reports),
            "duration_s": sum(r["duration_s"] for r in reports),
            "invalid": invalid, "episodes": reports}
