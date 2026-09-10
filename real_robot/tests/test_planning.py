import importlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_pose_roundtrip_uses_rotation_composition():
    a = importlib.import_module("djepa_robot.actions")
    start = np.array([0.2, 0, 0.3, 0.5, 0.4, -0.3, 0.2])
    end = np.array([0.21, 0.01, 0.29, -0.2, 0.3, 0.7, 0.8])
    delta = a.pose_delta(start, end)
    assert not np.allclose(delta[3:6], end[3:6] - start[3:6])
    actual = a.advance_pose(start, delta)
    np.testing.assert_allclose(actual[[0, 1, 2, 6]], end[[0, 1, 2, 6]])
    np.testing.assert_allclose(Rotation.from_euler("xyz", actual[3:6]).as_matrix(),
                               Rotation.from_euler("xyz", end[3:6]).as_matrix(), atol=1e-10)


@pytest.mark.parametrize("kind", ["joint", "duplicate", "nan", "float_id", "dt"])
def test_bad_candidate_contracts_rejected(kind):
    a = importlib.import_module("djepa_robot.actions")
    ids, actions, semantics, dt = np.array([3, 1]), np.zeros((2, 3, 7)), a.SEMANTICS, 0.1
    if kind == "joint": semantics = "joint_position_target_plus_gripper"
    if kind == "duplicate": ids = np.array([1, 1])
    if kind == "float_id": ids = np.array([1.1, 2.2])
    if kind == "nan": actions[0, 0, 0] = np.nan
    if kind == "dt": dt = 0
    with pytest.raises(ValueError):
        a.CandidateBatch(ids, actions, semantics, dt)


def test_bounds_and_immutability():
    a = importlib.import_module("djepa_robot.actions")
    raw = np.zeros((2, 3, 7))
    batch = a.CandidateBatch(np.array([3, 1]), raw, a.SEMANTICS, 0.1)
    raw[0, 0, 0] = 99
    assert batch.actions[0, 0, 0] == 0
    with pytest.raises(ValueError):
        batch.actions[0, 0, 0] = 1
    with pytest.raises(ValueError, match="bounds"):
        batch.validate_bounds(np.ones(7), np.ones(7) * 2)


def test_cem_improves_and_returns_scored_literal_candidate():
    planning = importlib.import_module("djepa_robot.planning")
    calls = {}
    def objective(batch):
        costs = ((batch.actions - 0.025) ** 2).mean(axis=(1, 2))
        for i, actions, cost in zip(batch.ids, batch.actions, costs):
            calls[int(i)] = (actions.copy(), float(cost))
        return costs
    kwargs = dict(lower=np.full(7, -0.05), upper=np.full(7, 0.05),
                  horizon=2, samples=48, elites=8, iterations=7, seed=42, dt=0.1)
    result = planning.cem(objective, **kwargs)
    actual, cost = calls[result.candidate_id]
    np.testing.assert_array_equal(result.actions, actual)
    assert result.cost == cost
    assert cost < 0.025**2 / 3
    other = planning.cem(objective, **kwargs)
    np.testing.assert_array_equal(result.actions, other.actions)
    assert result.candidate_id == other.candidate_id


def test_cem_ties_and_nonfinite_cost():
    planning = importlib.import_module("djepa_robot.planning")
    kwargs = dict(lower=np.zeros(7), upper=np.ones(7), horizon=1,
                  samples=4, elites=2, iterations=2, seed=1, dt=0.1)
    result = planning.cem(lambda b: np.zeros(len(b.ids)), **kwargs)
    assert result.candidate_id == 0
    with pytest.raises(ValueError, match="finite"):
        planning.cem(lambda b: np.full(len(b.ids), np.nan), **kwargs)
