"""Load full model partitions without depending on private research paths."""
import json
from pathlib import Path
import torch
from .exact_realization import ExactRealizationWorldModel
from .reacher_contract import ReacherPlanningProfile
from .reacher_world_model import ReacherWorldModel
from .temporal_transport import ReacherFutureResidualHead


def load_exact_world_model(directory, *, tdjepa, lewm):
    """Call with instantiated upstream models matching the exported architecture."""
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text())
    if config['architecture']!='exact_realization_world_model':
        raise ValueError('not an exact realization profile')
    prep=config['preprocessing'];decision=config['decision']
    model=ExactRealizationWorldModel(tdjepa=tdjepa,lewm=lewm,
             fusion_alpha=decision['fusion_alpha'],gate_threshold=decision['gate_threshold'],
             action_mean=torch.tensor(prep['action_mean']),action_std=torch.tensor(prep['action_std']))
    model.load_state_dict(torch.load(directory/'model.pt',map_location='cpu',weights_only=True),strict=True)
    return model.eval().requires_grad_(False)


def load_reacher_world_model(directory, *, tdjepa, lewm):
    """Return trained feature-level model and its strict-loaded native backbones."""
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text())
    if config['architecture']!='reacher_world_model':raise ValueError('not a Reacher full profile')
    state=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    for name,model in [('tdjepa',tdjepa),('lewm',lewm)]:
        model.load_state_dict({k.removeprefix(name+'.'):v for k,v in state.items() if k.startswith(name+'.')},strict=True)
        model.eval().requires_grad_(False)
    model=ReacherWorldModel(task_name='DMC-Reacher',planning_profile=ReacherPlanningProfile.reviewed(),
          fusion_alpha=config['fusion_alpha'],gate_threshold=config['gate_threshold'],gate_advantage_decimals=config['gate_decimals'])
    model.load_state_dict({k.removeprefix('adapters.'):v for k,v in state.items() if k.startswith('adapters.')},strict=True)
    return model.eval().requires_grad_(False),tdjepa,lewm


def load_temporal_transport(directory):
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text())
    model=ReacherFutureResidualHead(hidden_dim=config['hidden_dim'],radius=config['radius'])
    model.load_state_dict(torch.load(directory/'model.pt',map_location='cpu',weights_only=True),strict=True)
    return model.eval()
