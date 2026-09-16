"""Drive-JEPA small-scale contracts. Standard library only; no GPU imports."""
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import hashlib
import subprocess

from .runtime import log_group, safe_relative

ROOT = Path.cwd()
PATHS = ('repo', 'assets', 'checkpoint', 'python', 'logs', 'sensors', 'maps',
         'manifest', 'metric_index', 'runs')


def source_identity(repo):
    repo = Path(repo)
    digest = hashlib.sha256()
    for path in sorted(repo.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.json', '.npy') and '.git' not in path.parts:
            digest.update(str(path.relative_to(repo)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    if (repo/'.git').exists():
        commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        status = subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    else:
        commit, status = None, 'exported source archive; no git metadata supplied'
    return {'git_commit': commit, 'git_status': status, 'source_sha256': digest.hexdigest()}


def config(path):
    cfg = json.loads(Path(path).read_text())
    for key in PATHS:
        p = Path(cfg[key])
        cfg[key] = os.path.abspath(p if p.is_absolute() else ROOT / p)
    check_checkpoint_name(cfg['checkpoint'])
    if cfg['selection'] != 'pinned_v1_code_native_argmax':
        raise ValueError('Unsupported selector; do not silently invent a momentum implementation')
    return cfg


def check_checkpoint_name(path):
    if 'perception_free' in Path(path).name:
        raise ValueError('Need the perception-based multi-proposal checkpoint')


def camera_paths(frames, media=False):
    if len(frames) < 4:
        raise ValueError('Four history frames required')
    requests = [(i, 'cam_f0') for i in range(len(frames))] if media else [
        (2, 'cam_f0'), (3, 'cam_b0'), (3, 'cam_f0'), (3, 'cam_l0'), (3, 'cam_r0')]
    return [safe_relative({k.lower(): v for k, v in frames[i]['cams'].items()}[cam]['data_path'])
            for i, cam in requests]


def validate_rows(rows):
    if not rows:
        raise ValueError('Empty manifest')
    tokens, roles = set(), {}
    for r in rows:
        token, role = r['token'], r['role']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', token) or token in tokens:
            raise ValueError('Unsafe or duplicate token')
        if role not in ('fit', 'calibration', 'test'):
            raise ValueError('Unknown role')
        tokens.add(token)
        group = log_group(r['log_name'])
        if group in roles and roles[group] != role:
            raise ValueError(f'Log leakage: {group}')
        roles[group] = role


def partition_rows(rows, counts, seed):
    if set(counts) != {'fit', 'calibration', 'test'} or min(counts.values()) < 1:
        raise ValueError('Need three positive scene budgets')
    groups = sorted({log_group(r['log_name']) for r in rows})
    if len(groups) < 3:
        raise ValueError('Need at least three independent driving logs')
    rng = random.Random(seed)
    rng.shuffle(groups)
    remaining, result = groups, []
    for i, role in enumerate(('fit', 'calibration', 'test')):
        remaining_roles = ('fit', 'calibration', 'test')[i:]
        n = len(remaining) if i == 2 else max(1, min(len(remaining) - (2-i),
            round(len(remaining) * counts[role] / sum(counts[k] for k in remaining_roles))))
        selected, remaining = set(remaining[:n]), remaining[n:]
        pool = sorted([r for r in rows if log_group(r['log_name']) in selected], key=lambda r: r['token'])
        rng.shuffle(pool)
        if len(pool) < counts[role]:
            raise ValueError(f'{role}: need {counts[role]}, have {len(pool)} in fixed log partition')
        result += [dict(r, role=role) for r in pool[:counts[role]]]
    validate_rows(result)
    return result


def read_rows(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    validate_rows(rows)
    return rows


def write_json(path, obj):
    with open(path, 'x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty table')
    with open(path, 'x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def admission(cfg):
    required = {
        'model_code': Path(cfg['repo']) / 'navsim_v1/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py',
        'checkpoint': Path(cfg['checkpoint']),
        'encoder': Path(cfg['assets']) / 'vitl_merge_3dataset_e50.pt',
        'anchors': Path(cfg['repo']) / 'navsim_v1/data/8192.npy',
        'python': Path(cfg['python']),
    }
    checks = [{'name': k, 'path': str(p), 'present': p.is_file() and p.stat().st_size > 0}
              for k, p in required.items()]
    checks += [{'name': k, 'path': cfg[k], 'present': Path(cfg[k]).is_dir()}
               for k in ('logs', 'sensors', 'maps')]
    return {'checks': checks, 'paths_present': all(r['present'] for r in checks),
            'scope': 'existence only, not integrity or runtime verification'}
