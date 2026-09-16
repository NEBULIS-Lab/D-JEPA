from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "robotics" / "evaluate_gate.py"


def _module():
    assert SCRIPT.exists(), "gate report script is missing"
    spec = importlib.util.spec_from_file_location("robotwin_gate_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_row_applies_preoutcome_original_candidate_id(tmp_path: Path) -> None:
    selection = tmp_path / "relational_gate_selection"
    outcomes = tmp_path / "matched_outcomes"
    selection.mkdir()
    outcomes.mkdir()
    (selection / "selection.json").write_text(
        json.dumps(
            {
                "seed": 5,
                "decision": "correction",
                "selected_original_candidate_index": 2,
                "post_outcome_selected": False,
            }
        )
    )
    (outcomes / "metrics.json").write_text(
        json.dumps(
            {
                "candidate_sources": [
                    {"candidate_index": 0, "original_candidate_index": 0},
                    {"candidate_index": 1, "original_candidate_index": 2},
                ],
                "outcomes": [
                    {"candidate_index": 0, "success": False},
                    {"candidate_index": 1, "success": True},
                ],
            }
        )
    )

    row = _module()._matched_row(tmp_path)

    assert row["native_success"] is False
    assert row["correction_success"] is True
    assert row["gated_success"] is True
    assert row["pair_oracle_success"] is True
