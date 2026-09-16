"""Label-free trajectory selection from a prediction NPZ and a trusted model file."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .alignment import Aligner
from .risk import RiskAligner, select


def features(prediction, relative=False):
    native = np.asarray(prediction['native_score'], dtype=np.float32)
    query = np.asarray(prediction['query_feature'], dtype=np.float32)
    trajectory = np.asarray(prediction['trajectory'], dtype=np.float32)
    ids = np.asarray(prediction['candidate_id'])
    k = len(native)
    if native.shape != (k,) or k < 2 or query.shape != (k, 256) or trajectory.shape != (k, 8, 3):
        raise ValueError('Expected native [K], query [K,256], trajectory [K,8,3]')
    if not np.array_equal(ids, np.arange(k)):
        raise ValueError('Preserve the exported candidate ordering')
    rank = ((native[:, None] > native[None, :]).sum(1) +
            .5 * (native[:, None] == native[None, :]).sum(1)) / k
    parts = [query]
    fusion = None
    if not relative:
        factors = np.asarray(prediction['predicted_factors'], dtype=np.float32)
        if factors.shape != (k, 6):
            raise ValueError('The score-correction interface requires six native factor channels')
        parts.append(factors)
        fusion = np.column_stack((factors, native)).astype('float32')
    x = np.column_stack((*parts, native, rank, trajectory.reshape(k, -1))).astype('float32')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite prediction features')
    return x, fusion, native


def predict(bundle, prediction, kind='relation', device='cpu'):
    relative = 'models' not in bundle
    x, fusion, native = features(prediction, relative)
    state = bundle['model'] if relative else bundle['models'][kind]
    values = fusion if not relative and kind == 'fusion' else x
    if values.shape[-1] != state['dim']:
        raise ValueError('Prediction feature dimensions do not match the model')
    mean, std = np.asarray(state['mean']), np.asarray(state['std'])
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid training normalization')
    tx = torch.as_tensor((values - mean) / std, dtype=torch.float32, device=device)[None]
    tn = torch.as_tensor(native, device=device)[None]
    model = (RiskAligner(state['dim']) if relative else Aligner(state['dim'], kind)).to(device).eval()
    model.load_state_dict(state['state'], strict=True)
    with torch.inference_mode():
        if relative:
            gain, risk = model(tx, tn)
            setting = bundle['selection']
            selected = select(native[None], gain.cpu().numpy(), risk.sigmoid().cpu().numpy(),
                              setting['alpha'], setting['risk_weight'], setting['margin'])[0]
        else:
            score = tn + state['alpha'] * (model(tx, tn) - tn)
            selected = int(score.argmax(-1).item())
    selected = int(selected)
    return {'candidate_id': selected, 'native_candidate_id': int(native.argmax()),
            'trajectory': prediction['trajectory'][selected].tolist(),
            'labels_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['relation', 'mlp', 'fusion'], default='relation')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    # Original training bundles contain NumPy statistics. Load only trusted files.
    bundle = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    with np.load(args.prediction, allow_pickle=False) as prediction:
        result = predict(bundle, prediction, args.kind, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
