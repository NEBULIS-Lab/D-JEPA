"""Reference-centred controls with the original three sampling/precision rules."""
from collections import Counter
import math

import numpy as np

from .pool import CandidatePool

PROFILES = ('reference-control', 'pusht-confirmation', 'visual-shape')


def _quotas(large):
    return {'reference': 1, 'noisy_0.05': 32 if large else 8,
            'noisy_0.15': 32 if large else 8, 'noisy_0.3': 32 if large else 8,
            'shuffled': 31 if large else 7, 'smooth_random': 64 if large else 16,
            'uniform_random': 64 if large else 16}


def _draw(reference, family, rng):
    if family == 'reference':
        return reference.copy()
    if family.startswith('noisy_'):
        return np.clip(reference.astype(np.float64) + rng.normal(
            0., float(family.split('_')[1]), size=reference.shape), -1., 1.)
    if family == 'shuffled':
        order = rng.permutation(5)
        if np.array_equal(order, np.arange(5)):
            order = np.roll(order, 1)
        return reference.reshape(5, 5, -1)[order].reshape(reference.shape)
    white = rng.uniform(-1., 1., size=reference.shape)
    if family == 'uniform_random':
        return white
    smooth = np.empty_like(white)
    smooth[0] = white[0]
    for step in range(1, 25):
        smooth[step] = .8 * smooth[step-1] + .2 * white[step]
    return np.clip(smooth, -1., 1.)


def _shuffle_capacity(reference):
    counts = Counter(np.ascontiguousarray(b).tobytes() for b in reference.reshape(5, 5, -1))
    total = math.factorial(5)
    for count in counts.values():
        total //= math.factorial(count)
    return total - 1


def control_candidates(reference_actions, *, seed, profile='reference-control', max_attempts=4096):
    """Construct one pool, including reference ID 0.

    ``reference-control``: 64 byte-unique source-dtype actions, refilled per family.
    ``pusht-confirmation``: 256 float16 actions, refilled from successive seeded banks.
    ``visual-shape``: pixel controls / 100; retry complete 64-action banks until
    float16 quantization is collision-free, then expose float32 controls.
    """
    if profile not in PROFILES:
        raise ValueError(f'unknown candidate profile: {profile}')
    if type(seed) is not int or seed < 0 or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('seed must be a nonnegative integer; max_attempts must be positive')
    raw = np.asarray(reference_actions)
    if (raw.ndim != 2 or raw.shape[0] != 25 or raw.shape[1] < 1
            or raw.dtype.kind != 'f' or not np.isfinite(raw).all()):
        raise ValueError('reference actions must be finite floating point [25,D]')
    if profile != 'reference-control' and raw.shape[1] != 2:
        raise ValueError('PushT/PushObj actions require two control dimensions')
    reference = raw.copy() if profile == 'reference-control' else raw.astype(np.float32)
    if profile == 'visual-shape':
        reference = reference / np.float32(100.)
    if np.any(reference < -1) or np.any(reference > 1):
        raise ValueError('normalized reference actions must lie in [-1,1]')
    quotas = _quotas(profile == 'pusht-confirmation')
    quantized = reference if profile == 'reference-control' else reference.astype(np.float16)
    if _shuffle_capacity(quantized) < quotas['shuffled']:
        raise ValueError('reference cannot supply enough unique shuffled actions')
    actions, types, seen = [], [], set()
    used_seed = seed
    if profile == 'reference-control':
        rng = np.random.default_rng(seed)
        for family, quota in quotas.items():
            accepted = 0
            for _ in range(max_attempts):
                value = np.array(_draw(reference, family, rng), dtype=reference.dtype, order='C')
                key = value.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                actions.append(value)
                types.append(family)
                accepted += 1
                if accepted == quota:
                    break
            if accepted != quota:
                raise ValueError(f'{family} refill exceeded max_attempts')
    else:
        filled = dict.fromkeys(quotas, 0)
        for attempt in range(max_attempts):
            rng = np.random.default_rng(seed + attempt)
            bank = [(_draw(reference, family, rng).astype(np.float32).astype(np.float16), family)
                    for family, quota in quotas.items() for _ in range(quota)]
            if profile == 'visual-shape':
                if len({a.tobytes() for a, _ in bank}) == 64:
                    actions = [a.astype(np.float32) for a, _ in bank]
                    types = [t for _, t in bank]
                    used_seed = seed + attempt
                    break
            else:
                for value, family in bank:
                    key = value.tobytes()
                    if filled[family] == quotas[family] or key in seen:
                        continue
                    seen.add(key)
                    actions.append(value)
                    types.append(family)
                    filled[family] += 1
                if filled == quotas:
                    break
        else:
            raise ValueError('unique candidate construction exceeded max_attempts')
    return CandidatePool(np.arange(len(actions), dtype=np.int64), np.stack(actions),
                         np.asarray(types, dtype='U24'), profile, used_seed)


def retain_pusht_candidates(pool, native_costs):
    """Keep top 16 plus 47 family-covered proposals, then exclude reference ID 0.

    Costs are lower-is-better TD-JEPA native predictions for the original 256
    IDs. Ties use literal IDs. No execution-label argument is accepted.
    """
    if (pool.profile != 'pusht-confirmation' or len(pool.ids) != 256
            or not np.array_equal(pool.ids, np.arange(256)) or pool.reference_id != 0):
        raise ValueError('retention requires the original ordered 256-proposal pool')
    scores = np.asarray(native_costs, dtype=np.float64)
    if scores.shape != (256,) or not np.isfinite(scores).all():
        raise ValueError('native costs must be finite with shape (256,)')
    deployable = np.arange(1, 256)
    selected = deployable[np.lexsort((deployable, scores[deployable]))][:16].tolist()
    used = set(selected)
    coverage = {}
    for i in deployable:
        if i not in used:
            coverage.setdefault(str(pool.types[i]), []).append(int(i))
    for ids in coverage.values():
        ids.sort(key=lambda i: (float(scores[i]), i))
    while len(selected) < 63:
        for family in sorted(coverage):
            if coverage[family]:
                selected.append(coverage[family].pop(0))
                if len(selected) == 63:
                    break
    return pool.subset(selected)
