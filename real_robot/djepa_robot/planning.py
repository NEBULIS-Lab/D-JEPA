"""Policy-free bounded CEM; no actuator or safety certification is implied."""

from dataclasses import dataclass

import numpy as np

from .actions import CandidateBatch, SEMANTICS, validate_limits


@dataclass
class SearchResult:
    candidate_id: int
    actions: np.ndarray
    cost: float
    final_batch: CandidateBatch
    final_costs: np.ndarray
    history: list


def cem(objective, *, lower, upper, horizon, samples=64, elites=8,
        iterations=8, seed=0, dt=0.1):
    """Minimize a *cross-population-comparable* cost with deterministic ID ties.

    Set-wise ranks are NOT comparable across CEM iterations. Generate a pool with
    native latent cost first; apply relational alignment once to the shared pool.
    The returned best sequence was scored verbatim; it is not the final mean.
    """
    lower, upper = validate_limits(lower, upper)
    if any(not isinstance(v, int) for v in (horizon, samples, elites, iterations)):
        raise ValueError("CEM dimensions must be integers")
    if horizon < 1 or samples < 2 or not 1 <= elites <= samples or iterations < 1:
        raise ValueError("invalid CEM dimensions")
    rng = np.random.default_rng(seed)
    mean = np.broadcast_to((lower + upper) / 2, (horizon, 7)).copy()
    std = np.broadcast_to((upper - lower) / 2, (horizon, 7)).copy()
    best_id, best_actions, best_cost, history = None, None, np.inf, []
    for iteration in range(iterations):
        actions = np.clip(rng.normal(mean, std, (samples, horizon, 7)), lower, upper)
        batch = CandidateBatch(np.arange(samples) + iteration * samples, actions, SEMANTICS, dt)
        costs = np.asarray(objective(batch), dtype=np.float64)
        if costs.shape != (samples,) or not np.isfinite(costs).all():
            raise ValueError("objective must return finite [candidates] costs")
        order = np.lexsort((batch.ids, costs))
        first = order[0]
        if costs[first] < best_cost:
            best_id, best_actions, best_cost = int(batch.ids[first]), batch.actions[first].copy(), float(costs[first])
        selected = batch.actions[order[:elites]]
        mean = selected.mean(axis=0)
        std = np.maximum(selected.std(axis=0), (upper - lower) * 1e-3)
        history.append({"iteration": iteration, "best_native_cost": float(costs[first]),
                        "best_native_id": int(batch.ids[first])})
    return SearchResult(best_id, best_actions, best_cost, batch, costs.copy(), history)
