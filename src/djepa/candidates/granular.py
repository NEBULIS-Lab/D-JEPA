"""The recorded 63-member, five-control Granular transformation grid."""
import hashlib
import math

import numpy as np

from .pool import CandidatePool

# Immutable hash namespace from the recorded protocol, not a method name.
HASH_NAMESPACE = 'dajepa-granular-formal-pool-v1'
DIFFICULTIES = {'conservative': .75, 'standard': 1., 'broad': 1.25}


def _rotate(displacement, degrees):
    theta = math.radians(degrees)
    matrix = np.array([[math.cos(theta), -math.sin(theta)],
                       [math.sin(theta), math.cos(theta)]], dtype=np.float64)
    return displacement @ matrix.T


def _shift(actions, shift):
    positions = (np.arange(len(actions), dtype=np.float64) - shift) % len(actions)
    lower = np.floor(positions).astype(np.int64)
    upper = (lower + 1) % len(actions)
    weight = (positions - lower)[:, None]
    return (1. - weight) * actions[lower] + weight * actions[upper]


def granular_candidates(replay_actions, *, start_id, goal_id, offset=0, difficulty='standard'):
    """Transform a five-control window from a [20,4] replay; retain anchor ID 1."""
    raw = np.asarray(replay_actions)
    if raw.shape != (20, 4) or raw.dtype.kind != 'f' or not np.isfinite(raw).all():
        raise ValueError('replay actions must be finite floating point [20,4]')
    if type(offset) is not int or not 0 <= offset <= 15:
        raise ValueError('offset must locate a complete five-control replay window')
    if not isinstance(start_id, str) or not start_id or not isinstance(goal_id, str) or not goal_id:
        raise ValueError('start and goal IDs must be nonempty strings')
    if difficulty not in DIFFICULTIES:
        raise ValueError('unknown difficulty tier')
    multiplier = DIFFICULTIES[difficulty]
    base = raw[offset:offset+5].astype(np.float64)
    actions, types = [], []

    def add(value, family):
        actions.append(np.clip(value, -4., 4.))
        types.append(family)

    for scale in (1., .70, .85, 1.15, 1.30):
        for rotation in (0., -10., 10., -20., 20., -30., 30.):
            value = base.copy()
            value[:, 2:] = base[:, :2] + (1. + multiplier*(scale-1.)) * _rotate(
                base[:, 2:] - base[:, :2], multiplier*rotation)
            add(value, 'scale_rotation')
    for shift in (-1.75, -1.25, -.75, -.25, .25, .75, 1.25, 1.75):
        add(_shift(base, multiplier*shift), 'temporal')
    for radius in (.15, .30):
        for degrees in (0., 45., 90., 135., 180., 225., 270., 315.):
            theta = math.radians(degrees)
            value = base.copy()
            value[:, 2:] += multiplier*radius*np.array([math.cos(theta), math.sin(theta)])
            add(value, 'endpoint')
    for name in ('hash-a', 'hash-b', 'hash-c', 'hash-d'):
        key = f'{HASH_NAMESPACE}:{start_id}:{goal_id}:{name}:0'
        digest = hashlib.sha256(key.encode('utf-8')).digest()
        u = np.asarray([int.from_bytes(digest[i:i+8], 'big')/float(2**64)
                        for i in range(0, 32, 8)], dtype=np.float64)
        scale = 1. + multiplier*(2.*u[0]-1.)*.20
        rotation = multiplier*(2.*u[1]-1.)*18.
        temporal = multiplier*(2.*u[2]-1.)*2.5
        theta = 2.*math.pi*u[3]
        radius = multiplier*.12*(.5+.5*u[0])
        value = _shift(base, temporal)
        displacement = value[:, 2:] - value[:, :2]
        value[:, 2:] = value[:, :2] + scale*_rotate(displacement, rotation)
        value[:, 2:] += radius*np.array([math.cos(theta), math.sin(theta)])
        add(value, 'fixed_hash_filler')
    values = np.stack(actions).astype(np.float32)
    if len({a.tobytes() for a in values}) != 63:
        raise ValueError('Granular transformation grid contains duplicate actions')
    return CandidatePool(np.arange(1, 64, dtype=np.int64), values,
                         np.asarray(types, dtype='U24'), 'granular', reference_id=None)
