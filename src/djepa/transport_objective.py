"""Decision-aligned objectives and calibration for D-JEPA on Reacher."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _numpy_inputs(
    success: Any, cost: Any, candidate_ids: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ok = np.asarray(success)
    value = np.asarray(cost, dtype=np.float64)
    ids = np.asarray(candidate_ids)
    if (
        ok.ndim != 2
        or ok.dtype.kind != "b"
        or value.shape != ok.shape
        or ids.shape != ok.shape
        or ids.dtype.kind not in "iu"
        or not np.isfinite(value).all()
        or np.any(value < 0)
    ):
        raise ValueError("decision supervision arrays are malformed")
    if any(len(set(row.tolist())) != ok.shape[1] for row in ids):
        raise ValueError("candidate IDs must be unique within every start")
    return ok.astype(bool, copy=False), value, ids.astype(np.int64, copy=False)


def decision_target_ranks(
    physical_success: Any, physical_true_cost: Any, candidate_ids: Any
) -> np.ndarray:
    """Rank candidates by success first, then physical cost, then literal ID."""

    success, cost, ids = _numpy_inputs(
        physical_success, physical_true_cost, candidate_ids
    )
    ranks = np.empty_like(ids, dtype=np.int64)
    for row in range(len(ids)):
        order = np.lexsort((ids[row], cost[row], ~success[row]))
        ranks[row, order] = np.arange(len(order), dtype=np.int64)
    return ranks


@dataclass(frozen=True)
class AlignmentLoss:
    total: Tensor
    refined_listwise: Tensor
    transported_listwise: Tensor
    refined_pairwise: Tensor
    transported_pairwise: Tensor
    preservation: Tensor


def _pairwise(scores: Tensor, target_rank: Tensor, temperature: float) -> Tensor:
    count = scores.shape[1]
    left, right = torch.triu_indices(count, count, offset=1, device=scores.device)
    wanted = torch.where(
        target_rank[:, left] < target_rank[:, right],
        torch.ones((), device=scores.device, dtype=scores.dtype),
        -torch.ones((), device=scores.device, dtype=scores.dtype),
    )
    # wanted=+1 means left should have lower cost than right.
    margin = wanted * (scores[:, right] - scores[:, left])
    return F.softplus(-margin / temperature).mean()


def decision_alignment_loss(
    output: Any,
    *,
    physical_success: Tensor,
    physical_true_cost: Tensor,
    candidate_ids: Tensor,
    base_future: Tensor,
    temperature: float = 0.1,
    pairwise_weight: float = 0.25,
    preservation_weight: float = 0.02,
) -> AlignmentLoss:
    """Align both the set decision and transported criterion to physical order."""

    refined = output.refined_score
    transported = output.criterion_cost
    future = output.future
    if (
        not all(isinstance(value, Tensor) for value in (refined, transported, future))
        or refined.ndim != 2
        or transported.shape != refined.shape
        or physical_success.shape != refined.shape
        or physical_true_cost.shape != refined.shape
        or candidate_ids.shape != refined.shape
        or candidate_ids.dtype != torch.int64
        or physical_success.dtype != torch.bool
        or future.shape != base_future.shape
        or future.shape[:2] != refined.shape
    ):
        raise ValueError("decision-alignment tensor shapes or dtypes are invalid")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (refined, transported, physical_true_cost, future, base_future)
    ):
        raise ValueError("decision-alignment tensors must be finite")
    if temperature <= 0 or pairwise_weight < 0 or preservation_weight < 0:
        raise ValueError("decision-alignment weights must be non-negative and temperature positive")
    ranks_np = decision_target_ranks(
        physical_success.detach().cpu().numpy(),
        physical_true_cost.detach().cpu().numpy(),
        candidate_ids.detach().cpu().numpy(),
    )
    ranks = torch.as_tensor(ranks_np, dtype=torch.int64, device=refined.device)
    oracle = ranks.argmin(dim=1)
    refined_listwise = F.cross_entropy(-refined / temperature, oracle)
    transported_listwise = F.cross_entropy(-transported / temperature, oracle)
    refined_pairwise = _pairwise(refined, ranks, temperature)
    transported_pairwise = _pairwise(transported, ranks, temperature)
    preservation = (future - base_future).float().square().mean()
    total = (
        refined_listwise
        + transported_listwise
        + pairwise_weight * (refined_pairwise + transported_pairwise)
        + preservation_weight * preservation
    )
    return AlignmentLoss(
        total=total,
        refined_listwise=refined_listwise,
        transported_listwise=transported_listwise,
        refined_pairwise=refined_pairwise,
        transported_pairwise=transported_pairwise,
        preservation=preservation,
    )


def _selection_key(
    scores: np.ndarray,
    success: np.ndarray,
    cost: np.ndarray,
    ids: np.ndarray,
) -> tuple[int, float, tuple[int, ...]]:
    chosen = np.asarray(
        [np.lexsort((ids[row], scores[row]))[0] for row in range(len(ids))],
        dtype=np.int64,
    )
    row = np.arange(len(ids))
    literal = tuple(int(ids[index, position]) for index, position in enumerate(chosen))
    return int(success[row, chosen].sum()), -float(cost[row, chosen].mean()), literal


def calibrate_alpha(
    lewm_rank: Any,
    tdjepa_rank: Any,
    physical_success: Any,
    physical_true_cost: Any,
    candidate_ids: Any,
) -> float:
    """Choose one task-specific fusion alpha on calibration rows only."""

    success, cost, ids = _numpy_inputs(
        physical_success, physical_true_cost, candidate_ids
    )
    lewm = np.asarray(lewm_rank, dtype=np.float64)
    tdjepa = np.asarray(tdjepa_rank, dtype=np.float64)
    if lewm.shape != success.shape or tdjepa.shape != success.shape:
        raise ValueError("calibration rank arrays are malformed")
    candidates = np.linspace(0.0, 1.0, 21)
    evaluated = []
    for alpha in candidates:
        fused = alpha * lewm + (1.0 - alpha) * tdjepa
        evaluated.append((_selection_key(fused, success, cost, ids), float(alpha)))
    # Prefer physical success, then lower physical cost, then the larger alpha
    # only as a deterministic final tie break.
    return max(evaluated, key=lambda item: (item[0][0], item[0][1], item[1]))[1]


def calibrate_gate_threshold(
    base_score: Any,
    refined_score: Any,
    physical_success: Any,
    physical_true_cost: Any,
    candidate_ids: Any,
) -> float:
    """Choose a conservative finite strict gate threshold on calibration rows."""

    success, cost, ids = _numpy_inputs(
        physical_success, physical_true_cost, candidate_ids
    )
    base = np.asarray(base_score, dtype=np.float64)
    refined = np.asarray(refined_score, dtype=np.float64)
    if base.shape != success.shape or refined.shape != success.shape:
        raise ValueError("gate calibration score arrays are malformed")
    base_choice = np.asarray(
        [np.lexsort((ids[row], base[row]))[0] for row in range(len(ids))]
    )
    refined_choice = np.asarray(
        [np.lexsort((ids[row], refined[row]))[0] for row in range(len(ids))]
    )
    rows = np.arange(len(ids))
    advantage = base[rows, base_choice] - refined[rows, refined_choice]
    epsilon = max(1e-9, float(np.ptp(advantage)) * 1e-6)
    thresholds = np.unique(
        np.concatenate(([advantage.min() - epsilon], advantage, [advantage.max() + epsilon]))
    )
    evaluated = []
    for threshold in thresholds:
        use_refined = advantage > threshold
        score = np.where(use_refined[:, None], refined, base)
        key = _selection_key(score, success, cost, ids)
        # Higher thresholds are preferred after outcome/cost ties: this keeps
        # more of the frozen/base decision geometry.
        evaluated.append((key, float(threshold)))
    return max(evaluated, key=lambda item: (item[0][0], item[0][1], item[1]))[1]


__all__ = [
    "AlignmentLoss",
    "calibrate_alpha",
    "calibrate_gate_threshold",
    "decision_alignment_loss",
    "decision_target_ranks",
]
