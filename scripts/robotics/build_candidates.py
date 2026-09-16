#!/usr/bin/env python3
"""Learn grasp geometry from admitted demos and build test-time EE candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


import numpy as np

from djepa.robotics.grasp_geometry import (
    colored_tool_geometry,
    fit_geometry_ridge,
    grasp_target,
    normalize_target_span,
    spatial_target_grid,
    warp_ee_trajectory,
)
from djepa.robotics.inventory import TaskSpec, inspect_task
from djepa.robotics.rlds import iter_episodes
from djepa.robotics.smoke import _canonical_bytes, _npz_bytes, _publish


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticate(root: Path) -> dict[str, object]:
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError(f"incomplete artifact bundle: {root}")
    for name, expected in completion.get("artifact_sha256", {}).items():
        if _sha256(root / name) != expected:
            raise ValueError(f"artifact hash mismatch: {root / name}")
    return completion


def _revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sixd_to_quat(value: np.ndarray) -> np.ndarray:
    first = value[:3] / (np.linalg.norm(value[:3]) + 1e-8)
    second = value[3:6] - np.dot(first, value[3:6]) * first
    second /= np.linalg.norm(second) + 1e-8
    third = np.cross(first, second)
    matrix = np.stack((first, second, third), axis=1)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return np.asarray(quaternion, dtype=np.float32)


def _eef20_to_ee16(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all():
        raise ValueError("RoboTwin eef actions must be finite with shape (T,20)")
    rows = []
    for action in values:
        rows.append(
            np.concatenate(
                (
                    action[0:3],
                    _sixd_to_quat(action[3:9]),
                    action[9:10],
                    action[10:13],
                    _sixd_to_quat(action[13:19]),
                    action[19:20],
                )
            )
        )
    return np.asarray(rows, dtype=np.float32)


def _pad(actions: np.ndarray, horizon: int) -> np.ndarray:
    if len(actions) > horizon:
        return np.asarray(actions[:horizon], dtype=np.float32)
    return np.concatenate(
        (actions, np.repeat(actions[-1:], horizon - len(actions), axis=0)), axis=0
    ).astype(np.float32, copy=False)


def _compose_candidates(
    native: np.ndarray,
    demonstration: np.ndarray,
    *,
    source_target: np.ndarray,
    target_hypotheses: np.ndarray,
    grasp_step: int,
) -> np.ndarray:
    native_value = np.asarray(native, dtype=np.float32)
    if native_value.ndim != 2 or native_value.shape[1] != 16:
        raise ValueError("native actions must have shape (T,16)")
    demonstration_value = np.asarray(demonstration, dtype=np.float32)
    horizon = max(len(native_value), len(demonstration_value))
    native_value = _pad(native_value, horizon)
    demo = _pad(demonstration_value, horizon)
    rows = [native_value]
    for target in np.asarray(target_hypotheses):
        rows.append(
            warp_ee_trajectory(
                demo, source_target, target, grasp_step=grasp_step
            )
        )
    result = np.stack(rows).astype(np.float32, copy=False)
    if result.shape[0] != 17 or not np.isfinite(result).all():
        raise ValueError("candidate construction must yield 17 finite trajectories")
    return result


def _retrieval_score(source: np.ndarray, query: np.ndarray) -> float:
    source_pair = source.reshape(2, 3)
    query_pair = query.reshape(2, 3)
    center = float(np.linalg.norm(source_pair.mean(0) - query_pair.mean(0)))
    source_direction = source_pair[1] - source_pair[0]
    query_direction = query_pair[1] - query_pair[0]
    cosine = float(
        np.dot(source_direction, query_direction)
        / (np.linalg.norm(source_direction) * np.linalg.norm(query_direction) + 1e-8)
    )
    return center + 0.05 * (1.0 - np.clip(cosine, -1.0, 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--native-rollout", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="grab_roller")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    _authenticate(args.admission)
    _authenticate(args.native_rollout)
    inventory_doc = json.loads(
        (args.admission / "source_inventory.json").read_text(encoding="utf-8")
    )
    split_doc = json.loads(
        (args.admission / "episode_splits.json").read_text(encoding="utf-8")
    )
    inventory_row = next(
        row for row in inventory_doc["tasks"] if row["task"]["name"] == args.task
    )
    split_row = next(
        row for row in split_doc["tasks"] if row["task"]["name"] == args.task
    )
    inventory = inspect_task(
        args.dataset_root,
        TaskSpec(name=args.task, dataset_name=inventory_row["task"]["dataset_name"]),
    )
    if inventory.source_sha256 != inventory_row["source_sha256"]:
        raise ValueError("live dataset does not match admitted source")
    native_metrics = json.loads(
        (args.native_rollout / "metrics.json").read_text(encoding="utf-8")
    )
    if native_metrics["task"] != args.task or native_metrics["seed"] != args.seed:
        raise ValueError("native rollout identity differs from requested context")
    with np.load(args.native_rollout / "frames.npz", allow_pickle=False) as archive:
        native = np.asarray(archive["actions"], dtype=np.float32)
        query_image = np.asarray(archive["frames"][0], dtype=np.uint8)

    train_ordinals = tuple(int(value) for value in split_row["train"])
    calibration_ordinals = tuple(int(value) for value in split_row["calibration"])
    requested = train_ordinals + calibration_ordinals
    records: list[dict[str, object]] = []
    for episode in iter_episodes(args.dataset_root, inventory, requested):
        raw_actions = np.stack([step.eef_action for step in episode.steps]).astype(
            np.float32
        )
        target, closing_step = grasp_target(raw_actions)
        records.append(
            {
                "ordinal": episode.identity.ordinal,
                "feature": colored_tool_geometry(episode.steps[0].base_rgb),
                "target": target,
                "closing_step": closing_step,
                "actions": raw_actions,
            }
        )
    by_ordinal = {int(record["ordinal"]): record for record in records}
    train = [by_ordinal[value] for value in train_ordinals]
    calibration = [by_ordinal[value] for value in calibration_ordinals]
    model = fit_geometry_ridge(
        np.stack([record["feature"] for record in train]),
        np.stack([record["target"] for record in train]),
        np.stack([record["feature"] for record in calibration]),
        np.stack([record["target"] for record in calibration]),
    )
    query_feature = colored_tool_geometry(query_image)
    predicted_raw = model.predict(query_feature[None])[0]
    train_spans = np.asarray(
        [
            np.linalg.norm(record["target"][3:6] - record["target"][0:3])
            for record in train
        ]
    )
    mean_span = float(np.mean(train_spans))
    predicted_target = normalize_target_span(predicted_raw, span=mean_span)
    retrieved = min(
        train,
        key=lambda record: (
            _retrieval_score(
                normalize_target_span(record["target"], span=mean_span),
                predicted_target,
            ),
            int(record["ordinal"]),
        ),
    )
    source_target = np.asarray(retrieved["target"], dtype=np.float32)
    hypotheses, hypothesis_rows = spatial_target_grid(predicted_target)
    candidates = _compose_candidates(
        native,
        _eef20_to_ee16(np.asarray(retrieved["actions"])),
        source_target=source_target,
        target_hypotheses=hypotheses,
        grasp_step=int(retrieved["closing_step"]),
    )
    candidate_sources = [
        {"candidate_index": 0, "role": "twinvla_native"},
        *(
            {**row, "role": "learned_geometry_warp"}
            for row in hypothesis_rows
        ),
    ]
    report = {
        "schema": "D-JEPA-RoboTwin2-learned-grasp-geometry-candidates-v1",
        "task": args.task,
        "simulator_seed": args.seed,
        "candidate_count": int(len(candidates)),
        "native_index": 0,
        "post_outcome_selected": False,
        "candidate_sources": candidate_sources,
        "supervision": {
            "train_episode_count": len(train_ordinals),
            "calibration_episode_count": len(calibration_ordinals),
            "lambda_selected_on_calibration": model.lambda_value,
            "calibration_target_rmse_m": model.calibration_rmse,
        },
        "proposal": {
            "predicted_target_raw": predicted_raw.tolist(),
            "predicted_target_normalized": predicted_target.tolist(),
            "training_mean_grasp_span_m": mean_span,
            "retrieved_training_ordinal": int(retrieved["ordinal"]),
            "retrieved_closing_step": int(retrieved["closing_step"]),
            "candidate_sources": candidate_sources,
        },
        "provenance": {
            "code_revision": _revision(args.repo.resolve()),
            "source_sha256": inventory.source_sha256,
            "episode_splits_sha256": _sha256(args.admission / "episode_splits.json"),
            "native_completion_sha256": _sha256(args.native_rollout / "completion.json"),
        },
    }
    _publish(
        args.output,
        {
            "candidates.npz": _npz_bytes(actions=candidates, target_hypotheses=hypotheses),
            "model.npz": _npz_bytes(
                mean=model.mean,
                scale=model.scale,
                weights=model.weights,
                lambda_value=np.asarray(model.lambda_value),
            ),
            "metrics.json": _canonical_bytes(report),
        },
        "D-JEPA-RoboTwin2-learned-grasp-geometry-candidates-completion-v1",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
