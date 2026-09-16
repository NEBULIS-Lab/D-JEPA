"""CPU-only API tests on synthetic tensors, not task-performance measurements."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from djepa.driving.alignment import Aligner
from djepa.driving.risk import RiskAligner, boundary_weights, log_folds, select
from djepa.driving.contract import validate_rows
from djepa.driving.score import features, predict


@pytest.mark.parametrize('kind', ['relation', 'mlp', 'fusion'])
def test_zero_head_keeps_native(kind):
    model = Aligner(10, kind).eval()
    native = torch.tensor([[.2, .7, .1]])
    torch.testing.assert_close(model(torch.ones(1, 3, 10), native), native, atol=0, rtol=0)


def test_relative_anchor_bound_and_permutation():
    torch.manual_seed(7)
    model = RiskAligner(10).eval()
    with torch.no_grad():
        model.gain_head.weight.normal_()
    x, native = torch.randn(2, 5, 10), torch.tensor([[.1, .2, .9, .3, .4]] * 2)
    order = [2, 0, 4, 1, 3]
    gain, risk = model(x, native)
    pg, pr = model(x[:, order], native[:, order])
    torch.testing.assert_close(gain[:, 2], torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(pg, gain[:, order], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pr, risk[:, order], rtol=1e-5, atol=1e-6)
    assert float(gain.detach().abs().max()) <= .200001


def test_risk_veto_and_margin():
    native, gain = np.array([[.90, .89]]), np.array([[0., .15]])
    risk = np.array([[[.01, .01], [.9, .9]]])
    assert select(native, gain, risk, 0, 0, 0).tolist() == [0]
    assert select(native, gain, risk, 1, 0, 0).tolist() == [1]
    assert select(native, gain, risk, 1, 1, 0).tolist() == [0]
    assert select(native, gain, risk, 1, 0, .2).tolist() == [0]


def test_group_isolation():
    groups = ['a', 'b', 'a', 'c']
    for train, validation in log_folds(groups):
        assert not {groups[i] for i in train} & {groups[i] for i in validation}
    with pytest.raises(ValueError, match='leakage'):
        validate_rows([{'token': 'a', 'role': 'fit', 'log_name': 'log_00001_00100'},
                       {'token': 'b', 'role': 'test', 'log_name': 'log_00100_00200'}])


def test_safety_weighting():
    w = boundary_weights(torch.tensor([[.9, .89, .88]]), torch.tensor([[1., 0., .95]]),
                         torch.tensor([[[0., 0.], [1., 1.], [0., 0.]]]))
    assert w[0, 1] > w[0, 2]


@pytest.mark.parametrize('relative', [False, True])
def test_label_free_readout(relative):
    prediction = {'candidate_id': np.arange(4), 'query_feature': np.zeros((4, 256)),
                  'trajectory': np.zeros((4, 8, 3)), 'native_score': np.array([.1, .2, .9, .3]),
                  'predicted_factors': np.zeros((4, 6))}
    x, _, _ = features(prediction, relative)
    model = RiskAligner(x.shape[-1]) if relative else Aligner(x.shape[-1], 'relation')
    state = {'state': model.state_dict(), 'dim': x.shape[-1],
             'mean': np.zeros(x.shape[-1]), 'std': np.ones(x.shape[-1]), 'alpha': 1.}
    bundle = ({'model': state, 'selection': {'alpha': 1., 'risk_weight': 0., 'margin': 0.}}
              if relative else {'models': {'relation': state}})
    assert predict(bundle, prediction)['candidate_id'] == 2
    assert predict(bundle, prediction)['labels_used'] is False
    assert x.shape[-1] == (282 if relative else 288)


def test_recipe_is_argument_list_not_shell():
    path = Path(__file__).resolve().parents[2] / 'scripts/run_task.py'
    spec = importlib.util.spec_from_file_location('task_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.command({'entrypoint': 'djepa.driving.learn',
                             'arguments': {'epochs': 30, 'device': 'cpu'}}, ['--epochs', '1'])
    assert result[-2:] == ['--epochs', '1']
    with pytest.raises(ValueError):
        module.command({'entrypoint': '../outside.py', 'arguments': {}}, [])
