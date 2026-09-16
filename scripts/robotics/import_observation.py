#!/usr/bin/env python3
"""Import one RoboTwin simulator observation into the π0.5 bundle contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from djepa.robotics.simulator import to_pi05_simulator_observation
from djepa.robotics.smoke import publish_observation_bundle


def _revision() -> str:
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-observation", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--context-ordinal", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.raw_observation, allow_pickle=False) as archive:
        sensors = {
            name: {"rgb": archive[f"{name}_rgb"]}
            for name in ("head_camera", "front_camera", "left_camera", "right_camera")
        }
        observation = SimpleNamespace(
            state=tuple(float(value) for value in archive["state"]),
            sensors=sensors,
            instruction=args.prompt,
        )
    transformed = to_pi05_simulator_observation(
        observation, task=args.task, context_ordinal=args.context_ordinal
    )
    publish_observation_bundle(
        args.output,
        transformed,
        provenance={
            "code_revision": _revision(),
            "simulator_seed": args.context_ordinal,
            "raw_observation_sha256": _sha256(args.raw_observation),
            "source": "RoboTwin2 simulator",
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
