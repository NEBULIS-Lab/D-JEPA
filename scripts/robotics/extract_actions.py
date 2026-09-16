#!/usr/bin/env python3
"""Extract one authenticated full demonstration action sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


import numpy as np

from djepa.robotics.inventory import TaskSpec, inspect_task
from djepa.robotics.rlds import iter_episodes
from djepa.robotics.smoke import _canonical_bytes, _npz_bytes, _publish


def _revision() -> str:
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--task", default="grab_roller")
    parser.add_argument("--episode-ordinal", type=int, required=True)
    parser.add_argument("--simulator-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
    if args.episode_ordinal not in split_row["train"]:
        raise ValueError("episode must belong to the admitted training split")
    inventory = inspect_task(
        args.dataset_root,
        TaskSpec(name=args.task, dataset_name=inventory_row["task"]["dataset_name"]),
    )
    episode = next(
        iter_episodes(args.dataset_root, inventory, (args.episode_ordinal,))
    )
    actions = np.stack([step.joint_action for step in episode.steps]).astype(np.float32)
    metadata = {
        "schema": "D-JEPA-RoboTwin2-full-demonstration-actions-v1",
        "task": args.task,
        "episode_ordinal": args.episode_ordinal,
        "simulator_seed": args.simulator_seed,
        "split": "train",
        "steps": len(actions),
        "source_sha256": inventory.source_sha256,
        "episode_splits_sha256": _sha256(args.admission / "episode_splits.json"),
        "code_revision": _revision(),
    }
    _publish(
        args.output,
        {
            "actions.npz": _npz_bytes(
                actions=actions,
                initial_base_rgb=np.asarray(episode.steps[0].base_rgb, dtype=np.uint8),
            ),
            "metadata.json": _canonical_bytes(metadata),
        },
        "D-JEPA-RoboTwin2-full-demonstration-actions-completion-v1",
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
