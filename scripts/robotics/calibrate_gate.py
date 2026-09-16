#!/usr/bin/env python3
"""Calibrate a native-preservation gate from matched RoboTwin outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from djepa.robotics.grasp_geometry import (
    calibrate_early_grasp_gate,
    calibrate_preservation_gate,
    native_target_relations,
)
from djepa.robotics.smoke import _canonical_bytes, _publish


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _development_row(seed_root: Path) -> dict[str, object]:
    candidate_root = seed_root / "full_candidates_v2"
    outcome_root = seed_root / "selected_outcomes_v2"
    candidate_metrics = json.loads(
        (candidate_root / "metrics.json").read_text(encoding="utf-8")
    )
    outcome_metrics = json.loads(
        (outcome_root / "metrics.json").read_text(encoding="utf-8")
    )
    native_index = int(candidate_metrics["native_index"])
    with np.load(candidate_root / "candidates.npz", allow_pickle=False) as archive:
        native_actions = np.asarray(archive["actions"][native_index], dtype=np.float32)
    target = np.asarray(
        candidate_metrics["proposal"]["predicted_target_normalized"],
        dtype=np.float32,
    )
    by_index = {
        int(row["candidate_index"]): row for row in outcome_metrics["outcomes"]
    }
    if set(by_index) != {0, 1}:
        raise ValueError("matched outcomes must contain candidates 0 and 1")
    return {
        "seed": int(candidate_metrics["simulator_seed"]),
        **native_target_relations(native_actions, target),
        "native_success": bool(by_index[0]["success"]),
        "correction_success": bool(by_index[1]["success"]),
        "candidate_metrics_sha256": _sha256(candidate_root / "metrics.json"),
        "candidates_sha256": _sha256(candidate_root / "candidates.npz"),
        "outcomes_sha256": _sha256(outcome_root / "metrics.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--feature",
        choices=("alignment_cost_m", "closest_closed_step_fraction"),
        default="alignment_cost_m",
    )
    args = parser.parse_args()

    rows = [_development_row(path) for path in args.seed_dir]
    values = np.asarray([row[args.feature] for row in rows], dtype=np.float64)
    native = np.asarray([row["native_success"] for row in rows], dtype=bool)
    correction = np.asarray([row["correction_success"] for row in rows], dtype=bool)
    if args.feature == "alignment_cost_m":
        gate = calibrate_preservation_gate(
            values,
            native_success=native,
            correction_success=correction,
        )
        operator = ">"
        method = "native-preservation-gated learned geometry correction"
    else:
        gate = calibrate_early_grasp_gate(
            values,
            native_success=native,
            correction_success=correction,
        )
        operator = "<"
        method = "relational timing-gated learned geometry correction"
    selected = gate.choose_correction(values)
    gated_success = np.where(selected, correction, native)
    for row, choose, success in zip(rows, selected, gated_success):
        row["gate_choice"] = "correction" if bool(choose) else "native"
        row["gated_success"] = bool(success)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    model = {
        "schema": "D-JEPA-RoboTwin2-native-preservation-gate-v1",
        "feature_name": args.feature,
        "operator": operator,
        "threshold": gate.threshold,
        "decision_rule": f"use correction iff {args.feature} {operator} threshold",
    }
    if args.feature == "alignment_cost_m":
        model["threshold_m"] = gate.threshold
    report = {
        "schema": "D-JEPA-RoboTwin2-native-preservation-gate-report-v1",
        "method": method,
        "development_seed_count": len(rows),
        "development_native_success": int(native.sum()),
        "development_correction_success": int(correction.sum()),
        "development_pair_oracle_success": int(np.logical_or(native, correction).sum()),
        "development_gated_success": int(gated_success.sum()),
        "development_gate_correction_count": gate.correction_count,
        "model": model,
        "rows": rows,
        "protocol": {
            "correction_candidate_original_index": 2,
            "correction_candidate_prelocked_before_outcomes": True,
            "threshold_selected_on_development_outcomes": True,
            "requires_separate_sealed_evaluation": True,
        },
        "provenance": {"code_revision": revision},
    }
    _publish(
        args.output,
        {
            "model.json": _canonical_bytes(model),
            "metrics.json": _canonical_bytes(report),
        },
        "D-JEPA-RoboTwin2-native-preservation-gate-completion-v1",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
