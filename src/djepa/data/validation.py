"""CPU-only release preflight; no simulator execution or device discovery."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(directory, checkpoint=None):
    directory = Path(directory)
    with np.load(directory / 'inputs.npz', allow_pickle=False) as data:
        ids = data['candidate_ids']
        if ids.ndim != 2 or min(ids.shape) == 0 or ids.dtype.kind not in 'iu':
            raise ValueError('candidate_ids must be a nonempty integer matrix')
        if any(len(np.unique(row)) != len(row) for row in ids):
            raise ValueError('duplicate candidate_ids within a candidate set')
        n, k = ids.shape
        for key in data.files:
            a = data[key]
            if a.dtype.hasobject:
                raise ValueError(f'object array is not permitted: {key}')
            if a.dtype.kind in 'fc' and not np.isfinite(a).all():
                raise ValueError(f'nonfinite input: {key}')
        dimension = None
        if 'features' in data:
            features = data['features']
            if features.ndim != 3 or features.shape[:2] != (n, k):
                raise ValueError('features must have shape (starts, candidates, coordinates)')
            dimension = features.shape[-1]
        if 'base_scores' in data and data['base_scores'].shape != (n, k):
            raise ValueError('base_scores do not align with candidate_ids')
        for key in ('start_ids', 'base_ids', 'episode_ids'):
            if key in data and data[key].shape != (n,):
                raise ValueError(f'{key} must have one identity per candidate set')
    report = {'count': n, 'candidates': k, 'feature_dim': dimension, 'labels': False}
    labels = directory / 'labels.npz'
    if labels.exists():
        with np.load(labels, allow_pickle=False) as data:
            success = data['success']
            if success.shape != (n, k) or success.dtype.kind != 'b':
                raise ValueError('success must be a boolean candidate-aligned matrix')
            mask = data['supervision_mask'] if 'supervision_mask' in data else np.ones((n, k), bool)
            if mask.shape != (n, k) or mask.dtype.kind != 'b':
                raise ValueError('supervision_mask must be a boolean candidate-aligned matrix')
            for key in ('state_distance', 'distance'):
                if key in data and (data[key].shape != (n, k) or not np.isfinite(data[key][mask]).all()):
                    raise ValueError(f'{key} is invalid on observed candidates')
            report.update(labels=True, observed_candidates=int(mask.sum()), total_candidates=n*k)
    if checkpoint is not None:
        config = json.loads((Path(checkpoint) / 'config.json').read_text())
        expected = config.get('input_dim', config.get('model', {}).get('input_dim'))
        if expected is not None and dimension != expected:
            raise ValueError(f'checkpoint input dimension {expected} does not match features {dimension}')
    return report


def validate_manifest(directory):
    """Verify only files explicitly named by the extracted archive manifest."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    checked = 0
    for entry in manifest['files']:
        path = (directory / entry['path']).resolve()
        if not path.is_relative_to(directory) or path == directory:
            raise ValueError('manifest path escapes dataset root')
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'manifest mismatch: {entry["path"]}')
        checked += 1
    if not checked:
        raise ValueError('empty dataset manifest')
    return {'verified_files': checked}
