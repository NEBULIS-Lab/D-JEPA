from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "robotics" / "calibrate_gate.py"


def _module():
    assert SCRIPT.exists(), "preservation-gate calibration script is missing"
    spec = importlib.util.spec_from_file_location("robotwin_preservation_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_development_row_reads_matched_native_and_correction(tmp_path: Path) -> None:
    root = tmp_path / "seed-123"
    candidates = root / "full_candidates_v2"
    outcomes = root / "selected_outcomes_v2"
    candidates.mkdir(parents=True)
    outcomes.mkdir()
    target = np.asarray([0.0, 0.0, 0.9, 0.2, 0.0, 0.9], dtype=np.float32)
    actions = np.zeros((2, 5, 16), dtype=np.float32)
    actions[:, :, 3] = actions[:, :, 11] = 1.0
    actions[:, :, 7] = actions[:, :, 15] = 0.0
    actions[0, 2, 0:3] = target[0:3]
    actions[0, 2, 8:11] = target[3:6]
    np.savez(candidates / "candidates.npz", actions=actions)
    (candidates / "metrics.json").write_text(
        json.dumps(
            {
                "simulator_seed": 123,
                "native_index": 0,
                "proposal": {"predicted_target_normalized": target.tolist()},
            }
        )
    )
    (outcomes / "metrics.json").write_text(
        json.dumps(
            {
                "outcomes": [
                    {"candidate_index": 0, "success": False},
                    {"candidate_index": 1, "success": True},
                ]
            }
        )
    )

    row = _module()._development_row(root)

    assert row["seed"] == 123
    assert row["alignment_cost_m"] < 1e-6
    assert row["closest_closed_step_fraction"] == 0.4
    assert row["native_success"] is False
    assert row["correction_success"] is True
