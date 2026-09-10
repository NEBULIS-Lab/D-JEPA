import importlib

import numpy as np
import pytest
import torch
from torch import nn


class ActionSensitivePredictor(nn.Module):
    """Shape-faithful synthetic predictor, not a pretrained robotics model."""
    def __init__(self, patches=4):
        super().__init__()
        self.patches = patches
        self.calls = []

    def forward(self, x, actions, states):
        n, steps = actions.shape[:2]
        assert x.shape[1] == steps * self.patches
        assert states.shape == actions.shape
        self.calls.append((x.detach().clone(), actions.detach().clone(), states.detach().clone()))
        return x + actions[:, :, 0].repeat_interleave(self.patches, dim=1)[..., None]


def test_rollout_preserves_causal_action_state_history():
    wm = importlib.import_module("djepa_robot.world_model")
    a = importlib.import_module("djepa_robot.actions")
    predictor = ActionSensitivePredictor()
    model = wm.ACPredictor(predictor, expected_dt=0.1, max_history=4, normalize_reps=False)
    actions = np.zeros((2, 3, 7))
    actions[0, :, 0] = 0.01
    actions[1, :, 0] = 0.02
    batch = a.CandidateBatch(np.array([4, 2]), actions, a.SEMANTICS, 0.1)
    future = model.rollout(torch.zeros(1, 4, 8), torch.zeros(1, 7), batch)
    assert future.shape == (2, 3, 4, 8)
    torch.testing.assert_close(future[0, :, 0, 0], torch.tensor([0.01, 0.02, 0.03]))
    torch.testing.assert_close(future[1, :, 0, 0], torch.tensor([0.02, 0.04, 0.06]))
    assert [call[1].shape[1] for call in predictor.calls] == [1, 2, 3]
    torch.testing.assert_close(predictor.calls[2][2][0, :, 0], torch.tensor([0, 0.01, 0.02]))


@pytest.mark.parametrize("kind", ["time", "history", "nan", "gripper"])
def test_rollout_rejects_invalid_contract(kind):
    wm = importlib.import_module("djepa_robot.world_model")
    a = importlib.import_module("djepa_robot.actions")
    model = wm.ACPredictor(ActionSensitivePredictor(), expected_dt=0.1, max_history=2)
    context, pose = torch.zeros(1, 4, 8), torch.zeros(1, 7)
    if kind == "nan": context[0, 0, 0] = float("nan")
    if kind == "gripper": pose[0, -1] = 10
    batch = a.CandidateBatch(np.array([0]), np.zeros((1, 3 if kind == "history" else 1, 7)),
                             a.SEMANTICS, 0.2 if kind == "time" else 0.1)
    with pytest.raises(ValueError):
        model.rollout(context, pose, batch)


def test_zero_init_and_permutation_equivariance():
    alignment = importlib.import_module("djepa_robot.alignment")
    torch.manual_seed(7)
    model = alignment.RelationalAlignment(dim=8, geometries=2).eval()
    terminal, goal = torch.randn(2, 5, 2, 8), torch.randn(2, 2, 8)
    ids = torch.tensor([[9, 3, 2, 7, 8], [1, 7, 3, 2, 9]])
    costs = torch.rand(2, 5, 2)
    out = model(terminal, goal, ids, native_costs=costs)
    assert torch.equal(out["scores"], out["base_scores"])
    assert torch.count_nonzero(out["correction"]) == 0
    nn.init.normal_(model.correction_up.weight, std=0.1)
    out = model(terminal, goal, ids, native_costs=costs)
    perm = torch.tensor([3, 1, 4, 0, 2])
    shuffled = model(terminal[:, perm], goal, ids[:, perm], native_costs=costs[:, perm])
    torch.testing.assert_close(shuffled["scores"], out["scores"][:, perm], atol=1e-6, rtol=1e-5)
    assert out["correction"].abs().max() <= 0.2
    out["scores"].sum().backward()
    assert model.correction_up.weight.grad is not None


def test_literal_id_ties_and_single_candidate():
    alignment = importlib.import_module("djepa_robot.alignment")
    rank = alignment.ordinal_ranks(torch.zeros(1, 3), torch.tensor([[7, 1, 4]]))
    torch.testing.assert_close(rank, torch.tensor([[1.0, 0, 0.5]]))
    assert alignment.ordinal_ranks(torch.zeros(1, 1), torch.tensor([[4]])).item() == 0
    with pytest.raises(ValueError, match="unique"):
        alignment.ordinal_ranks(torch.zeros(1, 2), torch.tensor([[4, 4]]))


def test_patch_features_and_native_distance_are_separate():
    wm = importlib.import_module("djepa_robot.world_model")
    x, goal = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2), torch.zeros(1, 16, 2)
    assert wm.spatial_features(x).shape == (1, 8)
    assert wm.native_cost(x, goal).item() == pytest.approx(15.5)


def test_checkpoint_loader_rejects_missing_or_extra_weights():
    wm = importlib.import_module("djepa_robot.world_model")
    layer = nn.Linear(2, 3)
    state = {"module.backbone." + k: v.clone() for k, v in layer.state_dict().items()}
    receipt = wm.load_checked_state(layer, state, role="test")
    assert receipt["missing"] == []
    state.pop("module.backbone.bias")
    with pytest.raises(ValueError, match="missing"):
        wm.load_checked_state(layer, state, role="test")


def test_local_loader_refuses_absent_dependency_before_hub(tmp_path, monkeypatch):
    wm = importlib.import_module("djepa_robot.world_model")
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: pytest.fail("must not invoke hub"))
    with pytest.raises(FileNotFoundError):
        wm.load_local_vjepa2_ac(tmp_path / "missing", tmp_path / "missing.pt", expected_dt=0.1)
