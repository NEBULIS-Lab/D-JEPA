#!/usr/bin/env python3
"""Lock a preservation-gate decision without reading RoboTwin outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from djepa.robotics.grasp_geometry import native_target_relations
from djepa.robotics.smoke import _canonical_bytes, _publish


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection(model_path: Path, candidate_root: Path) -> dict[str, object]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    metrics_path = candidate_root / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    native_index = int(metrics["native_index"])
    with np.load(candidate_root / "candidates.npz", allow_pickle=False) as archive:
        actions = np.asarray(archive["actions"], dtype=np.float32)
    if len(actions) <= 2:
        raise ValueError("candidate pool must contain fixed correction candidate 2")
    target = np.asarray(
        metrics["proposal"]["predicted_target_normalized"], dtype=np.float32
    )
    relations = native_target_relations(actions[native_index], target)
    feature_name = str(model.get("feature_name", "alignment_cost_m"))
    if feature_name not in relations:
        raise ValueError(f"unknown preservation feature: {feature_name}")
    feature_value = relations[feature_name]
    threshold = float(model.get("threshold", model.get("threshold_m")))
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("preservation threshold must be finite and nonnegative")
    operator = str(model.get("operator", ">"))
    if operator == ">":
        use_correction = feature_value > threshold
    elif operator == "<":
        use_correction = feature_value < threshold
    else:
        raise ValueError("preservation operator must be '>' or '<'")
    return {
        "seed": int(metrics["simulator_seed"]),
        **relations,
        "feature_name": feature_name,
        "feature_value": feature_value,
        "operator": operator,
        "threshold": threshold,
        "decision": "correction" if use_correction else "native",
        "selected_original_candidate_index": 2 if use_correction else native_index,
        "post_outcome_selected": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    selection = _selection(args.model, args.candidates)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    selection.update(
        {
            "schema": "D-JEPA-RoboTwin2-preservation-gate-selection-v1",
            "provenance": {
                "code_revision": revision,
                "model_sha256": _sha256(args.model),
                "candidate_metrics_sha256": _sha256(args.candidates / "metrics.json"),
                "candidates_sha256": _sha256(args.candidates / "candidates.npz"),
            },
        }
    )
    _publish(
        args.output,
        {"selection.json": _canonical_bytes(selection)},
        "D-JEPA-RoboTwin2-preservation-gate-selection-completion-v1",
    )
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
