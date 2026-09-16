from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "robotics" / "select_candidates.py"


def _module():
    assert SCRIPT.exists(), "preservation-gate application script is missing"
    spec = importlib.util.spec_from_file_location("robotwin_apply_preservation_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_uses_native_alignment_without_outcome_access(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    actions = np.zeros((17, 4, 16), dtype=np.float32)
    actions[:, :, 3] = actions[:, :, 11] = 1.0
    actions[:, :, 7] = actions[:, :, 15] = 0.0
    np.savez(candidate_root / "candidates.npz", actions=actions)
    (candidate_root / "metrics.json").write_text(
        json.dumps(
            {
                "simulator_seed": 7,
                "native_index": 0,
                "proposal": {
                    "predicted_target_normalized": [0.1, 0, 0, 0.1, 0, 0]
                },
            }
        )
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"threshold_m": 0.05}))

    selection = _module()._selection(model_path, candidate_root)

    assert np.isclose(selection["alignment_cost_m"], 0.1)
    assert selection["selected_original_candidate_index"] == 2
    assert selection["decision"] == "correction"


def test_selection_applies_relational_early_grasp_gate(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    actions = np.zeros((17, 10, 16), dtype=np.float32)
    actions[:, :, 3] = actions[:, :, 11] = 1.0
    actions[:, :, 7] = actions[:, :, 15] = 1.0
    actions[:, 2:, 7] = actions[:, 2:, 15] = 0.0
    np.savez(candidate_root / "candidates.npz", actions=actions)
    (candidate_root / "metrics.json").write_text(
        json.dumps(
            {
                "simulator_seed": 9,
                "native_index": 0,
                "proposal": {
                    "predicted_target_normalized": [0, 0, 0, 0, 0, 0]
                },
            }
        )
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "feature_name": "closest_closed_step_fraction",
                "operator": "<",
                "threshold": 0.4,
            }
        )
    )

    selection = _module()._selection(model_path, candidate_root)

    assert selection["feature_name"] == "closest_closed_step_fraction"
    assert selection["feature_value"] == 0.2
    assert selection["selected_original_candidate_index"] == 2
