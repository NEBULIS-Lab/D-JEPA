#!/usr/bin/env python3
"""Execute a fixed RoboTwin end-effector candidate pool from one scene snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from djepa.robotics.simulator import (
    GrabRollerReference,
    grab_roller_progress,
    mean_tcp_distance,
)
from djepa.robotics.smoke import _canonical_bytes, _npz_bytes, _publish


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    marker = root.parent / f"{root.name}_commit.txt"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"cannot establish source revision for {root}")


def _validate_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[2] != 16:
        raise ValueError("EE candidate actions must have shape [N,T,16]")
    if not np.isfinite(actions).all():
        raise ValueError("EE candidate actions must be finite")
    return actions


def _oracle_index(outcomes: list[dict[str, object]]) -> int:
    return max(
        range(len(outcomes)),
        key=lambda index: (
            int(bool(outcomes[index]["success"])),
            float(outcomes[index]["progress"]),
            -int(outcomes[index]["candidate_index"]),
        ),
    )


def _candidate_sources(pool: dict[str, object]) -> list[dict[str, object]]:
    sources = pool.get("candidate_sources")
    if sources is None:
        proposal = pool.get("proposal")
        if isinstance(proposal, dict):
            sources = proposal.get("candidate_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("candidate sources are absent from the candidate manifest")
    return sources


def _geometry(task: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roller = np.asarray(task.roller.get_pose().p, dtype=np.float64)
    left = np.asarray(task.robot.get_left_tcp_pose()[:3], dtype=np.float64)
    right = np.asarray(task.robot.get_right_tcp_pose()[:3], dtype=np.float64)
    return roller, left, right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--cowam-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="grab_roller")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--planner", default="curobo")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    completion = json.loads(
        (args.candidates / "completion.json").read_text(encoding="utf-8")
    )
    for name, expected in completion["artifact_sha256"].items():
        if _sha256(args.candidates / name) != expected:
            raise ValueError(f"candidate artifact hash mismatch: {name}")
    pool = json.loads((args.candidates / "metrics.json").read_text(encoding="utf-8"))
    if pool["task"] != args.task or pool["simulator_seed"] != args.seed:
        raise ValueError("candidate context does not match simulator request")
    with np.load(args.candidates / "candidates.npz", allow_pickle=False) as archive:
        actions = _validate_actions(archive["actions"])
    if len(actions) != int(pool["candidate_count"]):
        raise ValueError("candidate count disagrees with manifest")

    sys.path.insert(0, str(args.cowam_root.resolve()))
    from cowam.environments import RoboTwinEEAdapter, load_robotwin_task

    task = load_robotwin_task(
        args.robotwin_root.resolve(),
        args.task,
        task_config=args.task_config,
        seed=args.seed,
        planner=args.planner,
        instruction=args.prompt,
    )
    try:
        adapter = RoboTwinEEAdapter(task, episode_id=f"{args.task}-seed-{args.seed}")
        initial = adapter.observe()
        roller, left, right = _geometry(task)
        reference = GrabRollerReference(
            roller_z=float(roller[2]),
            mean_tcp_distance=mean_tcp_distance(roller, left, right),
        )
        snapshot = adapter.snapshot()
        endpoints: list[np.ndarray] = []
        outcomes: list[dict[str, object]] = []
        for index, candidate in enumerate(actions):
            adapter.restore(snapshot)
            endpoint = adapter.observe()
            first_success = None
            execution_error = None
            steps_executed = 0
            try:
                for step, action in enumerate(candidate, start=1):
                    endpoint = adapter.step(action)
                    steps_executed = step
                    if adapter.get_success():
                        first_success = step
                        break
            except Exception as error:  # Preserve all preregistered candidate outcomes.
                execution_error = f"{type(error).__name__}: {error}"
            success = bool(adapter.get_success()) and execution_error is None
            roller, left, right = _geometry(task)
            progress = grab_roller_progress(
                reference,
                roller_position=roller,
                left_tcp=left,
                right_tcp=right,
                left_closed=task.is_left_gripper_close(),
                right_closed=task.is_right_gripper_close(),
                success=success,
            )
            outcome = {
                "candidate_index": index,
                "success": success,
                "first_success_step": first_success,
                "steps_executed": steps_executed,
                "progress": float(progress),
                "execution_error": execution_error,
            }
            outcomes.append(outcome)
            endpoints.append(
                np.asarray(endpoint.sensors["head_camera"]["rgb"], dtype=np.uint8)
            )
            print(
                f"candidate={index:02d} success={int(success)} "
                f"first_success={first_success} progress={progress:.6f} "
                f"error={execution_error!r}",
                flush=True,
            )
        oracle_index = _oracle_index(outcomes)
        report = {
            "schema": "D-JEPA-RoboTwin2-EE-full-branch-v1",
            "task": args.task,
            "seed": args.seed,
            "planner": args.planner,
            "candidate_count": len(outcomes),
            "horizon": int(actions.shape[1]),
            "native_index": int(pool["native_index"]),
            "native_success": outcomes[int(pool["native_index"])]["success"],
            "oracle_index": oracle_index,
            "oracle_success": outcomes[oracle_index]["success"],
            "success_candidates": sum(bool(row["success"]) for row in outcomes),
            "execution_error_candidates": sum(
                row["execution_error"] is not None for row in outcomes
            ),
            "outcomes": outcomes,
            "candidate_sources": _candidate_sources(pool),
            "provenance": {
                "code_revision": _revision(args.repo.resolve()),
                "cowam_revision": _revision(args.cowam_root.resolve()),
                "candidate_completion_sha256": _sha256(
                    args.candidates / "completion.json"
                ),
                "post_outcome_candidate_replacement": False,
            },
        }
        _publish(
            args.output,
            {
                "metrics.json": _canonical_bytes(report),
                "frames.npz": _npz_bytes(
                    initial=np.asarray(
                        initial.sensors["head_camera"]["rgb"], dtype=np.uint8
                    ),
                    endpoints=np.stack(endpoints),
                ),
            },
            "D-JEPA-RoboTwin2-EE-full-branch-completion-v1",
        )
        print(
            json.dumps(
                {
                    "native_success": report["native_success"],
                    "oracle_index": report["oracle_index"],
                    "oracle_success": report["oracle_success"],
                    "success_candidates": report["success_candidates"],
                },
                indent=2,
            )
        )
        return 0
    finally:
        task.close_env(clear_cache=True)


if __name__ == "__main__":
    raise SystemExit(main())
