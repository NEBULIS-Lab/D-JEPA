"""Audit recorded six-joint motion with an explicitly supplied robot description."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from djepa_robot.data import read_episode, valid_window_starts
from djepa_robot.kinematics import URDFChain, motion_arrays
from djepa_robot.motion_audit import tracking_scan
from .common import new_output, write_json


def source_receipt(path):
    # Numeric auditing must not import the large torch runtime just for file metadata.
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "identity_check": "path/stat only; not a content hash"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="your URDF, joint ordering and gripper-unit configuration")
    p.add_argument("--raw-dir", required=True, help="directory containing numbered .hdf5 episodes")
    p.add_argument("--output", required=True)
    p.add_argument("--export-episodes", nargs="*", default=["0"])
    args = p.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    fk = URDFChain(config["urdf"], base=config["base"], tip=config["tip"], joint_names=config["joint_names"])
    output = new_output(args.output)
    raw_files = sorted(Path(args.raw_dir).resolve(strict=True).glob("*.hdf5"), key=lambda x: int(x.stem))
    if not raw_files:
        raise ValueError("no raw recordings found")
    grip = {"record_to_m": config["gripper_record_to_m"], "closed_m": config["gripper_closed_m"],
            "open_m": config["gripper_open_m"]}
    rows, invalid, joint_rows = [], [], []
    for path in raw_files:
        ep = read_episode(path)
        state, command, timestamps = ep["state"], ep["commands"], ep["timestamp_ns"]
        # Arm tracking remains informative even when gripper calibration is unresolved.
        joint_rows.append({"episode": path.stem, "frames": len(state),
                           "gripper_feedback_record_range": [float(state[:,6].min()), float(state[:,6].max())],
                           "gripper_command_record_range": [float(command[:,6].min()), float(command[:,6].max())],
                           "state_limit_violation_frames": int(fk.limit_violations(state[:,:6]).any(axis=1).sum()),
                           "command_limit_violation_frames": int(fk.limit_violations(command[:,:6]).any(axis=1).sum()),
                           "tracking": tracking_scan(state[:,:6], command[:,:6], timestamps,
                                                      min_dt=config["window_min_dt_s"], max_dt=config["window_max_dt_s"])})
        try:
            arrays = motion_arrays(fk, state[:, :6], command[:, :6], state[:, 6], command[:, 6], **grip)
        except ValueError as exc:
            invalid.append({"source": str(path), "error": str(exc)})
            continue
        valid = valid_window_starts(timestamps, 1, config["window_min_dt_s"], config["window_max_dt_s"])
        translation_error = np.linalg.norm(arrays["nominal_command_target_pose"][valid, :3] -
                                           arrays["nominal_flange_pose"][valid+1, :3], axis=-1)
        row = {"source": source_receipt(path), "episode": path.stem, "frames": len(state),
               "gripper_record_range": [float(state[:,6].min()), float(state[:,6].max())],
               "valid_one_step_windows": len(valid),
               "state_joint_limit_violation_frames": int(arrays["state_joint_limit_violations"].any(axis=1).sum()),
               "command_joint_limit_violation_frames": int(arrays["command_joint_limit_violations"].any(axis=1).sum()),
               "next_observation_vs_command_target_translation_median_m": float(np.median(translation_error)) if len(valid) else None,
               "tracking": tracking_scan(state[:, :6], command[:, :6], timestamps,
                                          min_dt=config["window_min_dt_s"], max_dt=config["window_max_dt_s"])}
        rows.append(row)
        if path.stem in args.export_episodes:
            with (output / f"episode_{path.stem}_nominal_motion.npz").open("xb") as f:
                np.savez_compressed(f, **arrays, timestamp_ns=timestamps, joint_state=state,
                                    joint_commands=command, valid_one_step_starts=valid,
                                    calibration_status=np.array(config["calibration_status"]),
                                    planner_ready=np.array(False), source_episode=np.array(str(path)))
    report = {"artifact_scope": "offline_nominal_motion_audit_not_robot_evaluation", "config": config,
              "hardware_execution": False, "hardware_calibrated": False,
              "urdf_sha256": hashlib.sha256(Path(config["urdf"]).read_bytes()).hexdigest(),
              "episodes": rows, "invalid": invalid, "joint_only_episodes": joint_rows,
              "summary": {"recordings": len(raw_files), "converted": len(rows),
                          "joint_frames_audited": sum(r["frames"] for r in joint_rows),
                          "all_recording_state_limit_violation_frames": sum(r["state_limit_violation_frames"] for r in joint_rows),
                          "all_recording_command_limit_violation_frames": sum(r["command_limit_violation_frames"] for r in joint_rows),
                          "frames": sum(r["frames"] for r in rows),
                          "valid_one_step_windows": sum(r["valid_one_step_windows"] for r in rows),
                          "state_limit_violation_frames": sum(r["state_joint_limit_violation_frames"] for r in rows),
                          "command_limit_violation_frames": sum(r["command_joint_limit_violation_frames"] for r in rows)}}
    write_json(output / "motion_audit.json", report)
    print(json.dumps(report["summary"]), flush=True)
    print(output / "motion_audit.json", flush=True)
    return 0 if not invalid else 2


if __name__ == "__main__":
    raise SystemExit(main())
