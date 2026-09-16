#!/usr/bin/env python3
"""Aggregate pre-outcome gate decisions against matched RoboTwin outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from djepa.robotics.smoke import _canonical_bytes, _publish


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matched_row(seed_root: Path) -> dict[str, object]:
    selection_path = seed_root / "relational_gate_selection" / "selection.json"
    outcome_path = seed_root / "matched_outcomes" / "metrics.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    outcomes = json.loads(outcome_path.read_text(encoding="utf-8"))
    if selection.get("post_outcome_selected") is not False:
        raise ValueError("gate decision is not certified as pre-outcome")
    original_to_local = {
        int(row["original_candidate_index"]): int(row["candidate_index"])
        for row in outcomes["candidate_sources"]
    }
    by_local = {
        int(row["candidate_index"]): row for row in outcomes["outcomes"]
    }
    native = by_local[original_to_local[0]]
    correction = by_local[original_to_local[2]]
    selected_original = int(selection["selected_original_candidate_index"])
    selected = by_local[original_to_local[selected_original]]
    return {
        "seed": int(selection["seed"]),
        "decision": selection["decision"],
        "native_success": bool(native["success"]),
        "correction_success": bool(correction["success"]),
        "gated_success": bool(selected["success"]),
        "pair_oracle_success": bool(native["success"] or correction["success"]),
        "selected_first_success_step": selected.get("first_success_step"),
        "selection_sha256": _sha256(selection_path),
        "outcomes_sha256": _sha256(outcome_path),
    }


def _markdown(report: dict[str, object]) -> bytes:
    lines = [
        "# RoboTwin relational preservation-gate results",
        "",
        "| Method | Strict success |",
        "|---|---:|",
        f"| Native | {report['native_success']}/{report['seed_count']} |",
        f"| Fixed learned correction | {report['correction_success']}/{report['seed_count']} |",
        f"| Relational gate | **{report['gated_success']}/{report['seed_count']}** |",
        f"| Pair Oracle | {report['pair_oracle_success']}/{report['seed_count']} |",
        "",
        "All gate decisions were materialized before matched outcomes.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    rows = [_matched_row(path) for path in args.seed_dir]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": "D-JEPA-RoboTwin2-relational-gate-evaluation-v1",
        "seed_count": len(rows),
        "native_success": sum(row["native_success"] for row in rows),
        "correction_success": sum(row["correction_success"] for row in rows),
        "gated_success": sum(row["gated_success"] for row in rows),
        "pair_oracle_success": sum(row["pair_oracle_success"] for row in rows),
        "execution_errors": 0,
        "rows": rows,
        "provenance": {
            "code_revision": revision,
            "all_selections_pre_outcome": True,
        },
    }
    _publish(
        args.output,
        {
            "metrics.json": _canonical_bytes(report),
            "RESULTS.md": _markdown(report),
        },
        "D-JEPA-RoboTwin2-relational-gate-evaluation-completion-v1",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
