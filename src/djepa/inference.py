"""Portable, label-free inference over cached predictive features."""
from pathlib import Path
import json

import numpy as np
import torch


def ordinal_ranks(scores, candidate_ids):
    scores, ids = np.asarray(scores), np.asarray(candidate_ids)
    if (scores.ndim != 2 or scores.shape != ids.shape or scores.shape[1] < 2
            or not np.isfinite(scores).all() or ids.dtype.kind not in 'iu'
            or any(len(set(row.tolist())) != len(row) for row in ids)):
        raise ValueError('finite score matrix and unique integer candidate IDs required')
    result = np.empty(scores.shape, dtype=np.float64)
    for row in range(len(scores)):
        order = np.lexsort((ids[row], scores[row]))
        result[row, order] = np.arange(scores.shape[1]) / (scores.shape[1]-1)
    return result


def gated_positions(base, refined, ids, threshold, mode, decimals=None):
    base, refined, ids = np.asarray(base), np.asarray(refined), np.asarray(ids)
    b = ordinal_ranks(base, ids).argmin(1)
    r = ordinal_ranks(refined, ids).argmin(1)
    rows = np.arange(len(b))
    if mode == 'base_minus_refined':
        advantage = base[rows, b] - refined[rows, r]
    elif mode == 'refined_gap':
        advantage = refined[rows, b] - refined[rows, r]
    else:
        raise ValueError('unknown gate mode')
    if decimals is not None:
        advantage = np.round(advantage, int(decimals))
    return np.where(advantage > threshold, r, b)


def exact_lift(future, goal, scores, candidate_ids):
    """Published terminal radius construction; earlier times unchanged."""
    if future.ndim != 4 or future.shape[:2] != scores.shape or goal.shape != (future.shape[0], future.shape[-1]):
        raise ValueError('future, goal and scores must align')
    ranks = ordinal_ranks(scores.detach().cpu().numpy(), candidate_ids.detach().cpu().numpy())
    integer = torch.as_tensor(ranks*(scores.shape[1]-1), device=future.device, dtype=torch.float64).round()
    direction = future.double()[:, :, -1] - goal.double()[:, None]
    rms = direction.square().mean(-1, keepdim=True).sqrt()
    unit = torch.where((rms > 1e-12).expand_as(direction), direction/rms.clamp_min(1e-12), torch.ones_like(direction))
    lifted = future.clone()
    lifted[:, :, -1] = (goal.double()[:, None]+((integer+1)/64)[..., None]*unit).to(future.dtype)
    return lifted


def compose_decisions(base, refined, candidate_ids, relational_positions, adapted_refined_scores, threshold):
    """Calibrated proposal gate with the original relational winner encoding.

    `adapted_refined_scores` are the relation head's scores after substituting
    adapted predictor futures, NOT raw adapted-predictor MSE.
    """
    base, refined, ids = map(np.asarray, (base, refined, candidate_ids))
    positions = np.asarray(relational_positions)
    rows = np.arange(len(base))
    if positions.shape != (len(base),):
        raise ValueError('one relational position required per candidate set')
    gated = base.copy()
    gated[rows, positions] = np.minimum(base.min(1)-1e-6, refined[rows, positions])
    chosen = gated_positions(gated, adapted_refined_scores, ids, threshold, 'base_minus_refined')
    return ids[rows, chosen]


def load_profile(directory):
    """Load only tensor payloads; metadata lives in JSON, not arbitrary pickle."""
    from .ordinal import DecisionAligner
    from .granular import GranularSetRanker
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    state = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
    if config['architecture'] == 'granular':
        model = GranularSetRanker(**config['model'])
    elif config['architecture'] == 'set_aligner':
        model = DecisionAligner(config['input_dim'], config.get('max_correction', .2))
    else:
        raise ValueError('this profile is not a cached-feature relational model; see its card')
    model.load_state_dict(state, strict=True)
    return model.eval(), config


@torch.no_grad()
def predict(directory, inputs, batch_size=16):
    """Consumes only features, base scores and IDs; never outcome labels."""
    model, config = load_profile(directory)
    features, base, ids = (np.asarray(inputs[k]) for k in ('features', 'base_scores', 'candidate_ids'))
    if features.shape[:2] != base.shape or ids.shape != base.shape or len(features) == 0 or batch_size < 1:
        raise ValueError('unaligned or empty candidate features')
    ordinal_ranks(base, ids)
    values = []
    for start in range(0, len(features), batch_size):
        # Original forward paths cast base scores to float32 before correction.
        values.append(model(torch.tensor(features[start:start+batch_size], dtype=torch.float32),
                            torch.tensor(base[start:start+batch_size], dtype=torch.float32))[0].numpy())
    refined = np.concatenate(values).astype(np.float64)
    positions = gated_positions(base.astype(np.float64), refined, ids, config['gate_threshold'],
                                config['gate_mode'], config.get('gate_decimals'))
    return {'selected_ids': ids[np.arange(len(ids)), positions], 'selected_positions': positions,
            'refined_scores': refined}
