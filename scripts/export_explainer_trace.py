"""Export existing PushT state traces for CPU-only browser replay; no simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    case = json.loads((args.case_dir / 'case.json').read_text())
    result = {
        'description': 'Recorded simulator state replay, not decoded latent predictions.',
        'start': case['start_index'], 'duration': case['physical_duration_s'],
        'protocol': 'Matched start and goal; complete fixed 25-control-action sequences.',
        'viewport': case['common_viewport'],
        'geometry': {'tee': [[-60, 0], [60, 0], [60, 30], [15, 30], [15, 120], [-15, 120], [-15, 30], [-60, 30]], 'pusher_radius': 15},
        'candidates': [],
    }
    for method in case['methods']:
        source = args.case_dir / f"candidate_{method['candidate_id']}_trace.npz"
        with np.load(source) as trace:
            result['goal'] = trace['goal'][:5].tolist()
            result['candidates'].append({
                'id': method['candidate_id'], 'method': method['method'],
                'success': method['physical_success'],
                'ranks': [method['lewm_rank'], method['td_rank'], method['ours_rank']],
                'score': method['decision_score'],
                'states': trace['states'][:, :5].round(5).tolist(),
                'times': trace['timestamps'].round(5).tolist(),
                'source': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            })
    assert all(np.allclose(result['candidates'][0]['states'][0], c['states'][0]) for c in result['candidates'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(f"Exported {len(result['candidates'])} full traces: {args.output.stat().st_size} bytes")


if __name__ == '__main__':
    main()
