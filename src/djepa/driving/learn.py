#!/usr/bin/env python3
"""Train small heads on fixed caches, evaluate paired selection.

Candidate truth is supervision/evaluation only, never a feature. Calibration
selects checkpoint and residual scale; held-out data never enters training.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import os
import random
import sys
import traceback

from .contract import ROOT, read_rows, write_csv, write_json, validate_rows
from .runtime import log_group

DEVICE = 'cpu'

FEATURE_SCHEMA = 'query256_predicted_factors_native_tieaware_rank_trajectory_v1'


def load_cache(path, role):
    import numpy as np
    path = Path(path)
    summary = json.loads((path / 'summary.json').read_text())
    if summary['state'] != 'complete' or summary['role'] != role:
        raise ValueError('Only completed cache of expected role may be consumed')
    rows = read_rows(path / 'processed_manifest.csv')
    if len(rows) != summary['count'] or any(r['role'] != role for r in rows):
        raise ValueError('Cache role/count mismatch')
    examples = []
    for row in rows:
        file = path / 'predictions' / f'{row["token"]}.npz'
        labels = json.loads((path / 'labels' / f'{row["token"]}.json').read_text())
        if labels['prediction_sha256'] != hashlib.sha256(file.read_bytes()).hexdigest():
            raise ValueError('Prediction/label identity mismatch')
        with np.load(file, allow_pickle=False) as a:
            native = a['native_score'].copy()
            factors = a['predicted_factors'].copy()
            k = len(native)
            if not np.array_equal(a['candidate_id'], np.arange(k)):
                raise ValueError('Candidate order changed')
            # Tied values share the same rank; no arbitrary candidate-index signal.
            rank = ((native[:, None] > native[None, :]).sum(1) +
                    .5 * (native[:, None] == native[None, :]).sum(1)) / k
            geometry = a['trajectory'].reshape(k, -1).copy()
            x = np.column_stack((a['query_feature'], factors, native, rank, geometry)).astype('float32')
            fusion = np.column_stack((factors, native)).astype('float32')
        if labels['token'] != row['token'] or labels['role'] != role:
            raise ValueError('Label role/token mismatch')
        if [r['candidate_id'] for r in labels['labels']] != list(range(k)):
            raise ValueError('Incomplete or reordered candidate labels')
        y = np.asarray([r['score'] for r in labels['labels']], dtype='float32')
        if not all(np.isfinite(v).all() for v in (x, fusion, native, y)) or np.any((y < 0) | (y > 1)):
            raise ValueError('Invalid cache values')
        examples.append(dict(row=row, x=x, fusion=fusion, native=native, y=y, labels=labels['labels']))
    identity = {'manifest_sha256': hashlib.sha256((path / 'manifest.csv').read_bytes()).hexdigest(),
                'config': json.loads((path / 'config.json').read_text()),
                'provenance': json.loads((path / 'provenance.json').read_text())}
    return examples, identity


def assert_identity(a, b):
    if a['manifest_sha256'] != b['manifest_sha256']:
        raise ValueError('Caches use different full split manifests')
    for key in ('git_commit', 'git_status', 'source_sha256', 'checkpoint', 'checkpoint_bytes', 'selection', 'protocol'):
        if a['provenance'][key] != b['provenance'][key]:
            raise ValueError(f'Cache source mismatch: {key}')
    if a['config'] != b['config']:
        raise ValueError('Caches use different configuration')


def feature_stats(examples, key):
    import numpy as np
    x = np.concatenate([e[key] for e in examples])
    return x.mean(0), np.maximum(x.std(0), 1e-4)


def tensors(examples, key, stats):
    import numpy as np
    import torch
    mean, std = stats
    return (torch.as_tensor(np.stack([(e[key]-mean)/std for e in examples]), device=DEVICE),
            torch.as_tensor(np.stack([e['native'] for e in examples]), device=DEVICE),
            torch.as_tensor(np.stack([e['y'] for e in examples]), device=DEVICE))


def train(args):
    import numpy as np
    import torch
    from .alignment import Aligner
    fit, identity = load_cache(args.fit_cache, 'fit')
    cal, cal_identity = load_cache(args.calibration_cache, 'calibration')
    assert_identity(identity, cal_identity)
    validate_rows([e['row'] for e in fit+cal])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    bundle = {'feature_schema': FEATURE_SCHEMA, 'identity': identity, 'seed': args.seed,
              'models': {}, 'fit_count': len(fit), 'calibration_count': len(cal),
              'protocol': identity['config']['protocol'], 'test_used': False}
    history = []
    for kind in ('fusion', 'mlp', 'relation'):
        key = 'fusion' if kind == 'fusion' else 'x'
        stats = feature_stats(fit, key)  # never normalize on calibration or test
        x, native, y = tensors(fit, key, stats)
        cx, cn, cy = tensors(cal, key, stats)
        model = Aligner(x.shape[-1], kind).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best_score = float(cy.gather(1, cn.argmax(-1, keepdim=True)).mean())
        best = {'state': copy.deepcopy(model.cpu().state_dict()), 'alpha': 0.0, 'epoch': 0}
        model.to(DEVICE)
        for epoch in range(1, args.epochs+1):
            model.train()
            losses = []
            for ids in torch.randperm(len(x), device=DEVICE).split(args.batch_size):
                pred = model(x[ids], native[ids])
                target = y[ids]
                gap = target[:, :, None]-target[:, None, :]
                diff = pred[:, :, None]-pred[:, None, :]
                near = native[ids] >= native[ids].amax(-1, keepdim=True)-0.2
                mask = (gap.abs() > 0.02) & (near[:, :, None] | near[:, None, :])
                pair = torch.nn.functional.softplus(-gap.sign()*diff/0.1)
                ranking = pair[mask].mean() if mask.any() else pred.sum()*0
                loss = ((pred-target)**2).mean() + 0.1*ranking + 0.1*((pred-native[ids])**2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                cp = model(cx, cn)
                for alpha in (0.25, 0.5, 1.0):
                    score = float(cy.gather(1, (cn+alpha*(cp-cn)).argmax(-1, keepdim=True)).mean())
                    if score > best_score + 1e-8:
                        best_score = score
                        best = {'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                'alpha': alpha, 'epoch': epoch}
            history.append({'kind': kind, 'epoch': epoch, 'train_loss': float(np.mean(losses)),
                            'best_calibration_PDMS_0_to_1': best_score})
        bundle['models'][kind] = dict(best, dim=x.shape[-1], mean=stats[0], std=stats[1],
                                     calibration_PDMS_0_to_1=best_score)
    torch.save(bundle, args.output / 'alignment.pt')
    write_csv(args.output / 'training.csv', history)
    write_json(args.output / 'summary.json', {'state': 'complete', 'seed': args.seed,
        'fit_count': len(fit), 'calibration_count': len(cal), 'test_used': False,
        'models': {k: {v: m[v] for v in ('alpha', 'epoch', 'calibration_PDMS_0_to_1')}
                   for k, m in bundle['models'].items()}, 'protocol': bundle['protocol']})


def evaluate(args):
    import numpy as np
    import torch
    from .alignment import Aligner
    examples, identity = load_cache(args.cache, 'test')
    bundle = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if bundle['feature_schema'] != FEATURE_SCHEMA:
        raise ValueError('Feature schema mismatch')
    assert_identity(bundle['identity'], identity)
    selectors = {'native': np.stack([e['native'] for e in examples]).argmax(-1)}
    for kind, state in bundle['models'].items():
        model = Aligner(state['dim'], kind).to(DEVICE).eval()
        model.load_state_dict(state['state'], strict=True)
        x, native, _ = tensors(examples, 'fusion' if kind == 'fusion' else 'x', (state['mean'], state['std']))
        with torch.inference_mode():
            pred = native+state['alpha']*(model(x, native)-native)
        selectors[kind] = pred.argmax(-1).cpu().numpy()
    raw, paired = [], []
    for i, e in enumerate(examples):
        for method, indices in selectors.items():
            candidate = int(indices[i])
            raw.append({**e['row'], 'method': method, **e['labels'][candidate],
                        'oracle_regret': float(e['y'].max()-e['y'][candidate])})
        n, o = int(selectors['native'][i]), int(selectors['relation'][i])
        delta = float(e['y'][o]-e['y'][n])
        paired.append({**e['row'], 'baseline_candidate': n, 'ours_candidate': o,
                       'baseline_PDMS': float(e['y'][n]), 'ours_PDMS': float(e['y'][o]), 'delta': delta,
                       'baseline_NC': e['labels'][n]['no_at_fault_collisions'],
                       'ours_NC': e['labels'][o]['no_at_fault_collisions'],
                       'category': 'gain' if delta > 1e-6 else 'loss' if delta < -1e-6 else 'tie'})
    write_csv(args.output / 'results.csv', raw)
    write_csv(args.output / 'paired.csv', paired)
    metric_keys = [k for k in examples[0]['labels'][0] if k != 'candidate_id']
    table = []
    for method in selectors:
        method_rows = [r for r in raw if r['method'] == method]
        table.append({'method': method, 'N': len(method_rows),
                      **{k: float(np.mean([r[k] for r in method_rows]))*100 for k in metric_keys},
                      'oracle_regret': float(np.mean([r['oracle_regret'] for r in method_rows]))*100})
    write_csv(args.output / 'table.csv', table)
    groups = sorted({log_group(r['log_name']) for r in paired})
    by_group = {g: [r['delta'] for r in paired if log_group(r['log_name']) == g] for g in groups}
    rng = np.random.default_rng(bundle['seed'])
    boot = [np.mean([d for g in rng.choice(groups, len(groups), replace=True) for d in by_group[g]])
            for _ in range(2000)]
    write_json(args.output / 'summary.json', {'state': 'complete', 'N': len(examples), 'logs': len(groups),
        'protocol': bundle['protocol'], 'seed': bundle['seed'],
        'mean_delta_PDMS_points': float(np.mean([r['delta'] for r in paired]))*100,
        'cluster_bootstrap_95_interval_points': (np.quantile(boot, [.025, .975])*100).tolist(),
        'paired_counts': {k: sum(r['category'] == k for r in paired) for k in ('gain', 'loss', 'tie')},
        'table_scale': '0..100 PDMS/subscores, NOT task success rate',
        'cache': str(Path(args.cache).resolve()), 'checkpoint': str(Path(args.checkpoint).resolve())})
    with open(args.output / 'table.md', 'x') as f:
        f.write('Log-disjoint driving evaluation. Metrics are on a 0–100 scale; PDMS is not task success.\n\n')
        f.write('| Method | N | PDMS | NC | DAC | TTC | EP | Comfort |\n|---|---:|---:|---:|---:|---:|---:|---:|\n')
        names = {'native': 'Drive-JEPA', 'fusion': 'Calibrated score fusion', 'mlp': 'Independent MLP', 'relation': 'D-JEPA'}
        for r in table:
            values = [r[k] for k in ('score', 'no_at_fault_collisions', 'drivable_area_compliance',
                                      'time_to_collision_within_bound', 'ego_progress', 'comfort')]
            f.write(f'| {names[r["method"]]} | {r["N"]} | '+' | '.join(f'{v:.2f}' for v in values)+' |\n')


def main():
    global DEVICE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['train', 'evaluate'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fit-cache', type=Path)
    p.add_argument('--calibration-cache', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--seed', type=int, default=20260910)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    DEVICE = args.device
    required = ('fit_cache', 'calibration_cache') if args.stage == 'train' else ('cache', 'checkpoint')
    if any(getattr(args, key) is None for key in required):
        p.error(f'Required: {required}')
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error('Positive training parameters required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'arguments.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(args.output / 'runtime.json', {'device': DEVICE, 'python': sys.version})
    try:
        (train if args.stage == 'train' else evaluate)(args)
    except Exception:
        write_json(args.output / 'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
