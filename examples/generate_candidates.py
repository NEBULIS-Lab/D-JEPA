"""Exercise all candidate CLIs on synthetic controls; no model, labels or simulator.

The example's squared-control costs only demonstrate the retention file format.
Use actual TD-JEPA native predictive costs for an experimental PushT pool.
"""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = np.linspace(-.8,.8,50).reshape(25,2).astype(np.float32)
    np.save(output/'control-reference.npy', reference)
    np.save(output/'pixel-reference.npy', reference*100)
    np.save(output/'granular-replay.npy', np.linspace(-1,1,80).reshape(20,4).astype(np.float32))

    def run(*arguments):
        subprocess.run([sys.executable, str(ROOT/'scripts/generate_candidates.py'),
                        *map(str,arguments)], check=True)

    for profile in ('pusht-confirmation','reference-control','visual-shape','granular'):
        source = 'control-reference.npy'
        extra = ['--seed','123']
        if profile == 'visual-shape':
            source, extra = 'pixel-reference.npy', []
        elif profile == 'granular':
            source, extra = 'granular-replay.npy', ['--goal-id','example-goal','--offset','2']
        run('generate','--config',ROOT/f'configs/candidates/{profile}.yaml',
            '--reference-actions',output/source,'--start-id','42',*extra,'--output',output/profile)
    pool_path = output/'pusht-confirmation/candidates.npz'
    with np.load(pool_path, allow_pickle=False) as pool:
        scores = {key: pool[key] for key in ('start_ids','candidate_ids','candidate_action_sha256')}
        scores['native_costs'] = np.square(pool['candidate_actions'].astype(np.float64)).mean(axis=(-1,-2))
    np.savez_compressed(output/'illustrative-costs.npz', **scores)
    run('retain','--pool',pool_path,'--scores',output/'illustrative-costs.npz',
        '--output',output/'pusht-selected')
    print('Synthetic interface example only; illustrative costs are not model predictions or paper results.')


if __name__ == '__main__':
    main()
