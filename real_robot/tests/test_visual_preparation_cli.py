import importlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_cli_writes_real_rgb_identity_without_planner_pose(tmp_path, monkeypatch, upstream_checkout):
    script = importlib.import_module("scripts.cache_rgb")
    episode = tmp_path / "0.hdf5"
    with h5py.File(episode, "w") as f:
        f["cam_head/color"] = np.zeros((3, 8, 12, 3), dtype=np.uint8)
        f["cam_head/timestamp"] = np.arange(3, dtype=np.int64) * 100000000 + 300
        f["left_arm/timestamp"] = np.arange(3, dtype=np.int64) * 100000000
        f["left_arm/joint"] = np.zeros((3, 6))
        f["left_arm/gripper"] = np.zeros(3)
        f["left_arm/action"] = np.ones((3, 7))
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["cache_rgb", "--episode", str(episode), "--camera", "cam_head",
                       "--indices", "0,2", "--checkout", str(upstream_checkout),
                       "--prepare-only", "--output", str(output)])
    script.main()
    with np.load(output / "recorded_rgb.npz") as cache:
        assert cache["frame_indices"].tolist() == [0, 2]
        assert "pose" not in cache.files
        assert "latents" not in cache.files
    metadata = json.loads((output / "cache.json").read_text())
    assert metadata["hardware_execution"] is False
    assert metadata["model_inference"] is False
    assert metadata["camera"] == "cam_head"


def test_reference_smoke_keeps_counterfactual_outcomes_unknown():
    script = importlib.import_module("scripts.smoke_ac")
    from djepa_robot.world_model import ACPredictor
    class TinyPredictor(torch.nn.Module):
        def forward(self, x, actions, states):
            return x + actions[:, :, 0].repeat_interleave(4, dim=1)[..., None]
    model = ACPredictor(TinyPredictor(), expected_dt=0.1, normalize_reps=False)
    context, goal = torch.zeros(1, 4, 8), torch.ones(1, 4, 8) * 0.1
    states = np.zeros((2, 7), dtype=np.float32)
    states[1, 0] = 0.1
    result = script.reference_diagnostics(model, context, goal, states)
    assert result["action_response_l1"] > 0
    assert result["recorded_transition_prediction_l1"] < 1e-6
    assert result["counterfactual_outcomes"] == "unknown"
    assert result["robot_success_rate"] is None
