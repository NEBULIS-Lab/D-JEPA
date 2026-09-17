"""Export a verified matched candidate pair from an existing pool (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('pool', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    case_file = args.case_dir / 'case.json'
    case = json.loads(case_file.read_text())
    start = case['start_index']
    candidates, offsets = [], []
    with h5py.File(args.pool) as pool:
        ids = pool['source_candidate_id'][start]
        goal = pool['lewm/goal_z'][start].astype(np.float64)
        for label, cid in [('A', 79), ('B', 23)]:
            position = int(np.flatnonzero(ids == cid).item())
            cost = float(pool['lewm/predicted_cost'][start, position])
            success = bool(pool['success'][start, position])
            method = next(m for m in case['methods'] if m['candidate_id'] == cid)
            assert cost == method['lewm_predicted_cost']
            assert success == method['physical_success']
            future = pool['lewm/predicted_z'][start, position, -1].astype(np.float64)
            offset = future - goal
            offsets.append(offset)
            assert np.isclose(np.mean(offset**2), cost, rtol=2e-3)
            trace_path = args.case_dir / f'candidate_{cid}_trace.npz'
            with np.load(trace_path) as trace:
                assert trace['states'].shape[0] == 126
                candidates.append(dict(label=label, id=cid, cost=cost,
                    rms_distance=float(np.sqrt(cost)), rank=method['lewm_rank'],
                    success=success, trace_file=trace_path.name,
                    trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest()))
    assert not candidates[0]['success'] and candidates[1]['success']
    assert candidates[0]['cost'] < candidates[1]['cost']
    cosine = float(np.dot(*offsets) / np.prod([np.linalg.norm(x) for x in offsets]))
    result = dict(start=start, model='LeWM', latent_dimension=192,
        metric='mean squared terminal future-to-goal distance',
        radius='sqrt(native cost), i.e. RMS distance',
        candidates=candidates, angle_rad=float(np.arccos(np.clip(cosine, -1, 1))),
        geometry_note='Radii use original predicted costs. The opening angle is reconstructed from stored float16 goal-relative latents. Camera orientation is illustrative; screen distances are projections.',
        source=dict(case_file=case_file.name, case_sha256=hashlib.sha256(case_file.read_bytes()).hexdigest(),
            pool_file=args.pool.name, start_index=start,
            score_key='lewm/predicted_cost', future_key='lewm/predicted_z', goal_key='lewm/goal_z',
            candidate_key='source_candidate_id', outcome_key='success'),
        selection='Curated opposing-outcome pair within the same model, start and candidate pool; not an aggregate estimate.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
