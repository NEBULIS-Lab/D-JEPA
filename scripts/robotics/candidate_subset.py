#!/usr/bin/env python3
"""Materialize an authenticated subset of a fixed RoboTwin EE candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from djepa.robotics.smoke import _canonical_bytes, _npz_bytes, _publish


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authenticate(root: Path) -> None:
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError("source candidate bundle is incomplete")
    for name, expected in completion.get("artifact_sha256", {}).items():
        if _sha256(root / name) != expected:
            raise ValueError(f"source candidate hash mismatch: {name}")


def _sources(pool: dict[str, object]) -> list[dict[str, object]]:
    value = pool.get("candidate_sources")
    if value is None and isinstance(pool.get("proposal"), dict):
        value = pool["proposal"].get("candidate_sources")
    if not isinstance(value, list):
        raise ValueError("candidate sources are missing")
    return value


def _subset(
    actions: np.ndarray,
    sources: list[dict[str, object]],
    indices: tuple[int, ...],
    *,
    native_index: int,
) -> tuple[np.ndarray, list[dict[str, object]], int]:
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("candidate subset indices must be nonempty and unique")
    if native_index not in indices:
        raise ValueError("candidate subset must retain the native candidate")
    if min(indices) < 0 or max(indices) >= len(actions):
        raise ValueError("candidate subset index is outside the source pool")
    by_index = {int(row["candidate_index"]): row for row in sources}
    rows = []
    for new_index, original_index in enumerate(indices):
        source = dict(by_index[original_index])
        source["original_candidate_index"] = original_index
        source["candidate_index"] = new_index
        rows.append(source)
    return (
        np.asarray(actions[list(indices)], dtype=np.float32),
        rows,
        indices.index(native_index),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    _authenticate(args.input)
    pool = json.loads((args.input / "metrics.json").read_text(encoding="utf-8"))
    with np.load(args.input / "candidates.npz", allow_pickle=False) as archive:
        actions = np.asarray(archive["actions"], dtype=np.float32)
    indices = tuple(args.indices)
    selected, sources, native_index = _subset(
        actions, _sources(pool), indices, native_index=int(pool["native_index"])
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": "D-JEPA-RoboTwin2-EE-candidate-subset-v1",
        "task": pool["task"],
        "simulator_seed": pool["simulator_seed"],
        "candidate_count": len(selected),
        "native_index": native_index,
        "candidate_sources": sources,
        "post_outcome_selected": False,
        "provenance": {
            "code_revision": revision,
            "source_completion_sha256": _sha256(args.input / "completion.json"),
            "source_indices": list(indices),
        },
    }
    _publish(
        args.output,
        {
            "candidates.npz": _npz_bytes(actions=selected),
            "metrics.json": _canonical_bytes(report),
        },
        "D-JEPA-RoboTwin2-EE-candidate-subset-completion-v1",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
