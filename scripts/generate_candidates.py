#!/usr/bin/env python3
"""Generate recorded-protocol candidates or retain PushT proposals using native costs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from djepa.candidates import CandidatePool, control_candidates, granular_candidates, retain_pusht_candidates


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path, required, optional=()):
    with np.load(path, allow_pickle=False) as data:
        if not set(required) <= set(data.files) or set(data.files) - set(required) - set(optional):
            raise ValueError(f'{path.name}: expected only {sorted(required | set(optional))}')
        return {key: data[key] for key in data.files}


def publish(output, pool, start_id, metadata):
    # Single-start files use the same leading batch dimension as native inference.
    arrays = dict(start_ids=np.asarray([start_id]), candidate_ids=pool.ids[None],
                  candidate_actions=pool.actions[None], candidate_types=pool.types[None],
                  candidate_action_sha256=pool.action_hashes()[None])
    if pool.seed is not None:
        arrays['generator_seed'] = np.asarray([pool.seed], dtype=np.uint64)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'candidates.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    record = dict(schema='djepa_candidate_pool_v1', profile=pool.profile,
                  candidate_count=len(pool.ids), action_shape=list(pool.actions.shape[1:]),
                  action_dtype=str(pool.actions.dtype), reference_id=pool.reference_id,
                  generator_seed=pool.seed, labels_consumed=False,
                  candidate_sha256=sha256(output/'candidates.npz'), **metadata)
    with (output/'metadata.json').open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    print(output/'candidates.npz')


def generate(args):
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or 'profile' not in config or set(config)-{'profile', 'difficulty'}:
        raise ValueError('candidate config requires profile and optional Granular difficulty')
    reference = np.load(args.reference_actions, allow_pickle=False)
    if not isinstance(reference, np.ndarray):
        reference.close()
        raise ValueError('reference actions must be one NPY array, not a supervision archive')
    profile = config['profile']
    if profile == 'granular':
        if args.seed is not None or not args.goal_id:
            raise ValueError('Granular requires --goal-id and uses identity hashing, not --seed')
        pool = granular_candidates(reference, start_id=args.start_id, goal_id=args.goal_id,
                                   offset=args.offset, difficulty=config.get('difficulty', 'standard'))
        start_id = args.start_id
    else:
        if 'difficulty' in config or args.offset != 0 or args.goal_id:
            raise ValueError('difficulty, offset and goal-id apply only to Granular')
        start_id = int(args.start_id)
        if start_id < 0:
            raise ValueError('control start ID must be nonnegative')
        seed = args.seed
        if seed is None and profile == 'visual-shape':
            seed = 20260902 + start_id
        if seed is None:
            raise ValueError('provide the recorded --seed for this control profile')
        pool = control_candidates(reference, seed=seed, profile=profile)
        if profile != 'pusht-confirmation':
            pool = pool.without_reference()
    publish(args.output, pool, start_id,
            dict(stage='proposals' if profile == 'pusht-confirmation' else 'selectable',
                 reference_actions_sha256=sha256(args.reference_actions),
                 config=config, requested_seed=args.seed, offset=args.offset, goal_id=args.goal_id))


def retain(args):
    required = {'start_ids', 'candidate_ids', 'candidate_actions', 'candidate_types',
                'candidate_action_sha256'}
    arrays = read_npz(args.pool, required, {'generator_seed'})
    metadata = json.loads((args.pool.parent/'metadata.json').read_text())
    if metadata.get('candidate_sha256') != sha256(args.pool):
        raise ValueError('pool differs from its metadata fingerprint')
    if (arrays['candidate_ids'].shape != (1, 256)
            or arrays['candidate_actions'].shape != (1, 256, 25, 2)
            or arrays['candidate_types'].shape != (1, 256) or arrays['start_ids'].shape != (1,)):
        raise ValueError('retention requires a single-start, 256-action PushT proposal file')
    pool = CandidatePool(arrays['candidate_ids'][0], arrays['candidate_actions'][0],
                         arrays['candidate_types'][0], metadata['profile'],
                         metadata['generator_seed'], metadata['reference_id'])
    if not np.array_equal(arrays['candidate_action_sha256'].astype('U64'),
                          pool.action_hashes()[None].astype('U64')):
        raise ValueError('action hashes do not match the candidate controls')
    scores = read_npz(args.scores, {'start_ids', 'candidate_ids', 'candidate_action_sha256', 'native_costs'})
    for key in ('start_ids', 'candidate_ids', 'candidate_action_sha256'):
        left, right = arrays[key], scores[key]
        if key.endswith('sha256'):
            left, right = left.astype('U64'), right.astype('U64')
        if not np.array_equal(left, right):
            raise ValueError(f'prediction/pool identity mismatch: {key}')
    if scores['native_costs'].shape != (1, 256):
        raise ValueError('native_costs must have shape (1,256)')
    selected = retain_pusht_candidates(pool, scores['native_costs'][0])
    publish(args.output, selected, int(arrays['start_ids'][0]),
            dict(stage='selectable', source_pool_sha256=sha256(args.pool),
                 source_scores_sha256=sha256(args.scores), native_cost_source='TD-JEPA'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generation = commands.add_parser('generate', help='Construct candidates from reference/replay actions')
    generation.add_argument('--config', type=Path, required=True)
    generation.add_argument('--reference-actions', type=Path, required=True, help='NPY: [25,D] controls or [20,4] Granular replay')
    generation.add_argument('--start-id', required=True)
    generation.add_argument('--seed', type=int)
    generation.add_argument('--goal-id')
    generation.add_argument('--offset', type=int, default=0)
    generation.add_argument('--output', type=Path, required=True, help='New output directory')
    retention = commands.add_parser('retain', help='Retain 63 PushT actions from native predictive costs')
    retention.add_argument('--pool', type=Path, required=True)
    retention.add_argument('--scores', type=Path, required=True)
    retention.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    (generate if args.command == 'generate' else retain)(args)


if __name__ == '__main__':
    main()
