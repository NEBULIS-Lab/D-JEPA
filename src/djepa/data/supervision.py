"""Training/calibration inputs and identity checks for the ordinal trainer."""
from pathlib import Path
import numpy as np


def require_disjoint(train_ids, calibration_ids):
    left, right = set(np.asarray(train_ids).tolist()), set(np.asarray(calibration_ids).tolist())
    if not left or not right or left & right:
        raise ValueError('training and calibration base identities must be nonempty and disjoint')


def load_split(directory):
    directory = Path(directory)
    with np.load(directory/'inputs.npz', allow_pickle=False) as a, np.load(directory/'labels.npz', allow_pickle=False) as b:
        data = {k: a[k] for k in a.files}
        if 'supervision_mask' in b and not b['supervision_mask'].all():
            raise ValueError('ordinal trainer requires dense supervision; use granular masked objective for sparse data')
        data['success'] = b['success']
        data['distance'] = b['state_distance']
    if data['features'].shape[-1] != 3:
        raise ValueError('this trainer is the task-local three-coordinate ordinal instance')
    return data
