#!/usr/bin/env python3
"""Extract one preregistered RoboTwin2 observation for the π0.5 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from djepa.robotics.inventory import TaskSpec, inspect_task
from djepa.robotics.pi05 import to_pi05_observation
from djepa.robotics.rlds import iter_episodes
from djepa.robotics.smoke import SmokeObservation, publish_observation_bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> str:
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--task", default="grab_roller")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    completion = json.loads((args.admission / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError("source admission is incomplete")
    inventory_path = args.admission / "source_inventory.json"
    split_path = args.admission / "episode_splits.json"
    inventory_document = json.loads(inventory_path.read_text(encoding="utf-8"))
    split_document = json.loads(split_path.read_text(encoding="utf-8"))
    inventory_row = next(
        row for row in inventory_document["tasks"] if row["task"]["name"] == args.task
    )
    split_row = next(row for row in split_document["tasks"] if row["task"]["name"] == args.task)
    ordinal = int(split_row["train"][0])
    inventory = inspect_task(
        args.dataset_root,
        TaskSpec(name=args.task, dataset_name=inventory_row["task"]["dataset_name"]),
    )
    if inventory.source_sha256 != inventory_row["source_sha256"]:
        raise ValueError("live source does not match admitted source")
    episode = next(iter_episodes(args.dataset_root, inventory, (ordinal,)))
    transformed = to_pi05_observation(episode.steps[0])
    publish_observation_bundle(
        args.output,
        SmokeObservation(
            task=args.task,
            episode_ordinal=ordinal,
            images=transformed.images,
            state=transformed.state,
            prompt=transformed.prompt,
        ),
        provenance={
            "code_revision": _revision(),
            "source_sha256": inventory.source_sha256,
            "source_inventory_sha256": _sha256(inventory_path),
            "episode_splits_sha256": _sha256(split_path),
            "split": "train",
            "step": 0,
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
