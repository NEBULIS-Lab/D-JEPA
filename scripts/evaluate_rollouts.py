#!/usr/bin/env python3
"""Execute D-JEPA-selected PushT controls in the recorded native physics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets

import numpy as np

from djepa.native import evaluate_pusht_rollouts, load_native_config, make_official_pusht_environment, sha256_file
from djepa.native.io import load_npz, prepend_python_roots, save_json, save_npz


def _subset(arrays, rows, population):
    result = {}
    for key, value in arrays.items():
        raw = np.asarray(value)
        result[key] = raw[rows] if raw.ndim > 0 and raw.shape[0] == population else raw
    return result


def _require_raw_split(inputs, expected: str) -> None:
    if "split" not in inputs:
        raise ValueError("native raw inputs must carry their explicit split")
    raw = np.asarray(inputs["split"])
    if raw.shape != np.asarray(inputs.get("start_ids")).shape:
        raise ValueError("native raw split must provide one value per start")
    decoded = {item.decode() if isinstance(item, bytes) else str(item) for item in raw.reshape(-1)}
    if decoded != {expected}:
        raise ValueError(f"native raw inputs are not exclusively split {expected}")


def _validate_feature_receipt(path: Path, inputs_path: Path, config) -> None:
    sidecar = path.with_suffix(path.suffix + ".json")
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"prepared feature sidecar is unavailable or invalid: {sidecar}") from error
    if value.get("schema") != "djepa_native_pusht_features_v1" or value.get("mode") != "native_predictor_execution":
        raise ValueError("prepared features are not a native predictor artifact")
    if value.get("output_sha256") != sha256_file(path):
        raise ValueError("prepared feature artifact SHA-256 differs from its sidecar")
    if value.get("source_inputs_sha256") != sha256_file(inputs_path):
        raise ValueError("prepared features were built from different raw inputs")
    if value.get("evaluation_split") != config["evaluation_split"]:
        raise ValueError("prepared feature split differs from native config")
    if value.get("checkpoint") != config["checkpoint"]:
        raise ValueError("prepared feature checkpoint differs from native config")


def _write_videos(directory: Path, frames: np.ndarray, start_ids: np.ndarray, selected_ids: np.ndarray) -> list[dict[str, object]]:
    try:
        import imageio.v3 as iio
    except ImportError as error:
        raise RuntimeError("MP4 export requires `pip install imageio imageio-ffmpeg`") from error
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (start_id, selected_id) in enumerate(zip(start_ids, selected_ids)):
        final = directory / f"pusht-start-{int(start_id)}-candidate-{int(selected_id)}.mp4"
        if final.exists():
            raise FileExistsError(f"refusing to replace existing video: {final}")
        stage = final.with_name(f".{final.stem}.stage.{os.getpid()}.{secrets.token_hex(8)}.mp4")
        try:
            iio.imwrite(stage, frames[index], fps=10, codec="libx264", pixelformat="yuv420p")
            if not stage.is_file() or stage.stat().st_size == 0:
                raise RuntimeError("video encoder produced no output")
            os.link(stage, final)
        finally:
            if stage.exists():
                stage.unlink()
        records.append({"start_id": int(start_id), "selected_id": int(selected_id), "path": str(final), "sha256": sha256_file(final), "frame_count": 25, "fps": 10})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Raw initial/goal states and the exact candidate controls")
    parser.add_argument("--features", type=Path, required=True, help="Output of prepare_features.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-root", action="append", default=[], help="Root containing the recorded stable_worldmodel package")
    parser.add_argument("--start-id", action="append", type=int, default=[], help="Evaluate only these literal starts")
    parser.add_argument("--video-dir", type=Path, help="Also encode one complete 25-frame MP4 per selected rollout")
    args = parser.parse_args()
    output_sidecar = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or output_sidecar.exists():
        raise FileExistsError(f"refusing to replace native output or sidecar: {args.output}")
    config = load_native_config(args.config)
    inputs, decisions = load_npz(args.inputs), load_npz(args.features)
    _require_raw_split(inputs, config["evaluation_split"])
    _validate_feature_receipt(args.features, args.inputs, config)
    starts = np.asarray(inputs.get("start_ids"))
    if starts.ndim != 1:
        raise ValueError("raw inputs lack one-dimensional start_ids")
    if args.start_id:
        wanted = set(args.start_id)
        if len(wanted) != len(args.start_id):
            raise ValueError("duplicate --start-id")
        index = {int(value): row for row, value in enumerate(starts.tolist())}
        if not wanted <= set(index):
            raise ValueError("requested start ID is unavailable")
        rows = np.asarray([index[value] for value in args.start_id], dtype=np.int64)
        inputs = _subset(inputs, rows, len(starts))
        decision_index = {int(value): row for row, value in enumerate(np.asarray(decisions["start_ids"]).tolist())}
        if not wanted <= set(decision_index):
            raise ValueError("requested start ID is absent from prepared features")
        decisions = _subset(decisions, np.asarray([decision_index[value] for value in args.start_id]), len(decision_index))
    prepend_python_roots(args.python_root)
    env = make_official_pusht_environment()
    try:
        arrays = evaluate_pusht_rollouts(env, inputs, decisions, capture_frames=args.video_dir is not None)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    frames = arrays.pop("frames", None)
    videos = _write_videos(args.video_dir, frames, arrays["start_ids"], arrays["selected_ids"]) if args.video_dir else []
    digest = save_npz(args.output, arrays)
    metadata = {
        "schema": "djepa_native_pusht_rollouts_v1",
        "mode": "native_physics_selected_rollouts",
        "task": "pusht",
        "evaluation_split": config["evaluation_split"],
        "count": int(len(arrays["start_ids"])),
        "control_horizon": 25,
        "integrator_steps_per_control": 10,
        "physics_steps_per_rollout": 250,
        "physical_seconds_per_rollout": 2.5,
        "trajectory_state_count": 26,
        "source_inputs_sha256": sha256_file(args.inputs),
        "feature_artifact_sha256": sha256_file(args.features),
        "output_sha256": digest,
        "videos": videos,
    }
    save_json(output_sidecar, metadata)
    print(args.output)


if __name__ == "__main__":
    main()
