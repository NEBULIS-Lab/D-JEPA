"""Prepare selected recorded RGB or encode it; never fabricate Cartesian poses."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from djepa_robot.training import source_receipt
from djepa_robot.vision import encode_rgb_frames, official_ac_transform, read_rgb_frames
from djepa_robot.world_model import load_local_vjepa2_ac
from .common import new_output, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", required=True)
    p.add_argument("--camera", choices=("cam_head", "cam_wrist"), required=True)
    p.add_argument("--indices", required=True, help="literal comma-separated recorded frame indices")
    p.add_argument("--checkout", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--model-dt", type=float, default=0.1)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error("output exists; choose a new directory")
    if not args.prepare_only and not args.checkpoint:
        p.error("encoding requires an explicit local checkpoint")
    try:
        indices = np.array([int(s) for s in args.indices.split(",")], dtype=np.int64)
    except ValueError:
        p.error("indices must be literal comma-separated integers")
    selected = read_rgb_frames(args.episode, args.camera, indices)
    transform = official_ac_transform(args.checkout)
    transformed = transform(selected["rgb"])
    if not np.isfinite(transformed.numpy()).all():
        raise ValueError("nonfinite preprocessed pixels")
    checkout = Path(args.checkout).resolve(strict=True)
    git = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_scope": "recorded_RGB_preparation" if args.prepare_only else "recorded_visual_latents",
                "hardware_execution": False, "model_inference": not args.prepare_only,
                "planner_ready": False, "missing_planner_fields": ["calibrated_eef_pose", "verified_action_semantics"],
                "episode": source_receipt(args.episode), "camera": args.camera,
                "frame_indices": indices.tolist(), "rgb_shape": list(selected["rgb"].shape),
                "selected_rgb_sha256": hashlib.sha256(selected["rgb"].tobytes()).hexdigest(),
                "selected_camera_intervals_s": (np.diff(selected["camera_timestamp_ns"]) * 1e-9).tolist(),
                "color_order": "RGB", "checkout": str(checkout), "git_commit": git.stdout.strip(),
                "preprocessing": "official AC demo: fixed full-scale square crop, bilinear 256, ImageNet normalize",
                "preprocessed_shape_C_N_H_W": list(transformed.shape),
                "preprocessed_channel_means": transformed.mean(dim=(1, 2, 3)).tolist(),
                "preprocessor_file_sha256": hashlib.sha256(
                    (checkout / "app/vjepa_droid/transforms.py").read_bytes()).hexdigest(),
                "encoding": "each physical frame duplicated into a separate two-frame tubelet; final feature layernorm",
                "world_model": None}
    features = None
    if not args.prepare_only:
        encoder, predictor, receipt = load_local_vjepa2_ac(args.checkout, args.checkpoint,
                                                        expected_dt=args.model_dt, device=args.device)
        del predictor
        features = encode_rgb_frames(encoder, selected["rgb"], transform,
                                      batch_size=args.batch_size, device=args.device)
        metadata["world_model"] = receipt
        metadata["latent_shape"] = list(features.shape)
    output = new_output(args.output)
    with (output / "recorded_rgb.npz").open("xb") as f:
        np.savez_compressed(f, **selected)
    if features is not None:
        with (output / "visual_latents.npz").open("xb") as f:
            np.savez_compressed(f, latents=features.numpy(), frame_indices=indices,
                                camera_timestamp_ns=selected["camera_timestamp_ns"],
                                robot_timestamp_ns=selected["robot_timestamp_ns"],
                                joint_state=selected["joint_state"], joint_commands=selected["joint_commands"],
                                camera=selected["camera"], data_origin=np.array("recorded_real"),
                                source_model=np.array(str(Path(args.checkpoint).resolve(strict=True))))
    write_json(output / "cache.json", metadata)
    print(output / "cache.json", flush=True)


if __name__ == "__main__":
    main()
