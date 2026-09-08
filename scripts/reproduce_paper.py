"""Replay the released cached-feature matrix with checked inputs and safe resume.

This is the cached-decision subset of paper reproduction, not a simulator run.
Numerical outputs and receipts are written locally, never into source directories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch
import yaml

from djepa.cli.config import UniqueLoader
from djepa.data.validation import sha256, validate_split


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f'path escapes root: {relative}')
    return path


def source_hash():
    import djepa
    root = Path(djepa.__file__).parent
    records = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*.py'))}
    records['reproduce_paper.py'] = sha256(__file__)
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def reference_check(run, data_root, split, report):
    reference = run.get('reference')
    if reference is None:
        return {'status': 'not_configured'}
    authority = json.loads(inside(data_root, reference['file']).read_text())
    method = reference['method']
    if reference['kind'] == 'selected_ids':
        expected = authority['methods'][method]['selected_ids']
    elif reference['kind'] == 'media_rows':
        rows = {str(row['start_id']): row['methods'][method]['selected_candidate_id']
                for row in authority['media_rows']}
        with np.load(split/'inputs.npz', allow_pickle=False) as inputs:
            expected = [rows[str(s)] for s in inputs['start_ids']]
    else:
        raise ValueError('unsupported reference kind')
    if report['selected_ids'] != expected:
        raise ValueError(f'reference selected-ID mismatch for {run["id"]}')
    return {'status': 'exact_selected_id_match', 'count': len(expected), 'file': reference['file']}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--matrix', type=Path, default=Path('configs/reproduction/paper.yaml'))
    parser.add_argument('--data-root', type=Path, default=Path('data/D-JEPA-supervision-v1'))
    parser.add_argument('--checkpoint-root', type=Path, default=Path('checkpoints'))
    parser.add_argument('--output', type=Path, default=Path('outputs/paper'))
    parser.add_argument('--only', action='append', help='Run ID; repeat to select several rows')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Preflight and show commands without writing outputs')
    args = parser.parse_args()
    matrix = yaml.load(args.matrix.read_text(), Loader=UniqueLoader)
    if not isinstance(matrix, dict) or set(matrix) != {'schema', 'runs'} or matrix['schema'] != 1:
        parser.error('matrix must contain schema: 1 and runs')
    runs = matrix['runs']
    if not isinstance(runs, list) or not runs:
        parser.error('runs must be a nonempty list')
    ids = []
    for run in runs:
        if not isinstance(run, dict) or not {'id','task','split','profile','labels'} <= set(run):
            parser.error('each run requires id, task, split, profile and labels')
        if set(run) - {'id','task','split','profile','labels','reference'}:
            parser.error('unknown run field')
        for field in ('id','task','split','profile'):
            if not isinstance(run[field], str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', run[field]):
                parser.error(f'invalid {field}')
        if type(run['labels']) is not bool:
            parser.error('labels must be a YAML boolean')
        if 'reference' in run:
            reference = run['reference']
            if not isinstance(reference, dict) or set(reference) != {'file', 'kind', 'method'}:
                parser.error('reference requires exactly file, kind and method')
            if not all(isinstance(value, str) and value for value in reference.values()):
                parser.error('reference fields must be nonempty strings')
            if reference['kind'] not in ('selected_ids', 'media_rows'):
                parser.error('unsupported reference kind')
        ids.append(run['id'])
    if len(set(ids)) != len(ids) or (args.only and set(args.only)-set(ids)):
        parser.error('duplicate or unknown run IDs')
    code = source_hash()
    for run in runs:
        if args.only and run['id'] not in args.only:
            continue
        split = inside(args.data_root, f'{run["task"]}/{run["split"]}')
        checkpoint = inside(args.checkpoint_root, run['profile'])
        validate_split(split, checkpoint)
        files = {'inputs': split/'inputs.npz', 'config': checkpoint/'config.json', 'weights': checkpoint/'model.pt'}
        if run['labels']:
            files['labels'] = split/'labels.npz'
        if run.get('reference'):
            files['reference'] = inside(args.data_root, run['reference']['file'])
        hashes = {name: sha256(path) for name, path in files.items()}
        config = json.loads(files['config'].read_text())
        if config.get('weights_sha256') and hashes['weights'] != config['weights_sha256']:
            raise ValueError(f'checkpoint checksum mismatch: {run["id"]}')
        fingerprint = {'run': run, 'files': hashes, 'source': code,
                       'runtime': {'python': sys.version, 'numpy': np.__version__,
                                   'torch': str(torch.__version__), 'yaml': yaml.__version__}}
        folder = inside(args.output, run['id'])
        report_path = folder/'report.json'
        receipt_path = folder/'receipt.json'
        command = [sys.executable, '-m', 'djepa.evaluate', '--checkpoint', str(checkpoint),
                   '--inputs', str(files['inputs']), '--output', str(report_path), '--batch-size', '16']
        if run['labels']:
            command += ['--labels', str(files['labels'])]
        if args.dry_run:
            print(json.dumps({'id': run['id'], 'command': command, 'preflight': 'passed'}))
            continue
        if folder.exists():
            if not args.resume or not receipt_path.is_file():
                raise ValueError(f'existing or incomplete run: {folder}; use a new output directory')
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('fingerprint') != fingerprint or receipt.get('status') != 'complete':
                raise ValueError(f'inputs/config/code changed: {folder}')
            for name, digest in receipt['outputs'].items():
                path = folder/name
                if not path.is_file() or sha256(path) != digest:
                    raise ValueError(f'output changed: {path}')
            print(json.dumps({'id': run['id'], 'status': 'verified_resume_skip'}))
            continue
        folder.mkdir(parents=True)
        with (folder/'run.log').open('x') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        if process.returncode:
            raise RuntimeError(f'evaluation failed; inspect {folder / "run.log"}')
        report = json.loads(report_path.read_text())
        comparison = reference_check(run, args.data_root, split, report)
        output_hashes = {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}
        with receipt_path.open('x') as stream:
            json.dump({'status': 'complete', 'mode': 'cached-feature replay', 'fingerprint': fingerprint,
                       'outputs': output_hashes, 'reference': comparison, 'command': command}, stream, indent=2)
            stream.write('\n')
        print(json.dumps({'id': run['id'], 'status': 'complete', 'reference': comparison}))


if __name__ == '__main__':
    main()
