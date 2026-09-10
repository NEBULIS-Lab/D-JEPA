"""Action-conditioned D-JEPA future representations for DMC-Reacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

CANDIDATE_COUNT, FUTURE_STEPS, LATENT_DIM = 63, 5, 192
from djepa.reacher_contract import ReacherContractError, ReacherPlanningProfile
from djepa.temporal_transport import (
    ReacherFutureResidualHead,
    lexicographic_integer_ranks,
)
from djepa.exact_realization import RelationalAligner


CriterionCallable = Callable[[Tensor, Tensor], Tensor]


class ReacherUnifiedModelError(ValueError):
    """Raised when Reacher model identity or feature inputs are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReacherUnifiedModelError(message)


@dataclass(frozen=True)
class UnifiedReacherOutput:
    future: Tensor
    criterion_cost: Tensor
    final_gated_score: Tensor
    base_score: Tensor
    refined_score: Tensor
    gate_active: Tensor
    target_ranks: Tensor
    criterion_ranks: Tensor
    temporal_coefficient: Tensor


class ReacherWorldModel(nn.Module):
    """Learn a same-action future residual and rescore it through a criterion."""

    def __init__(
        self,
        *,
        task_name: str,
        planning_profile: ReacherPlanningProfile,
        fusion_alpha: float,
        gate_threshold: float,
        gate_advantage_decimals: int,
        hidden_dim: int = 64,
        low_rank: int = 8,
        max_correction: float = 0.2,
        transport_hidden_dim: int = 32,
        transport_radius: float = 0.1,
    ) -> None:
        super().__init__()
        _require(task_name == "DMC-Reacher", "task name must be DMC-Reacher")
        try:
            planning_profile.validate_reviewed()
        except ReacherContractError as error:
            raise ReacherUnifiedModelError("planning profile is not the reviewed Reacher planning profile") from error
        alpha = float(fusion_alpha)
        threshold = float(gate_threshold)
        _require(torch.isfinite(torch.tensor(alpha)).item() and 0.0 <= alpha <= 1.0, "fusion alpha must be finite in [0,1]")
        _require(torch.isfinite(torch.tensor(threshold)).item(), "gate threshold must be finite")
        _require(
            isinstance(gate_advantage_decimals, int)
            and 0 <= gate_advantage_decimals <= 8,
            "gate advantage decimals must be an integer in [0,8]",
        )
        self.task_name = task_name
        self.planning_profile = planning_profile
        self.fusion_alpha = alpha
        self.gate_threshold = threshold
        self.gate_advantage_decimals = gate_advantage_decimals
        self.refiner = RelationalAligner(hidden_dim=int(hidden_dim), low_rank=int(low_rank), max_correction=float(max_correction))
        self.future_transport = ReacherFutureResidualHead(hidden_dim=int(transport_hidden_dim), radius=float(transport_radius))

    @property
    def total_adapter_tensor_count(self) -> int:
        return len(self.refiner.state_dict()) + len(self.future_transport.state_dict())

    @staticmethod
    def _validate_features(
        lewm_context: Tensor,
        lewm_future: Tensor,
        lewm_goal: Tensor,
        tdjepa_context: Tensor,
        tdjepa_future: Tensor,
        tdjepa_goal: Tensor,
        candidate_ids: Tensor,
        lewm_criterion: CriterionCallable,
        tdjepa_criterion: CriterionCallable,
    ) -> int:
        _require(isinstance(lewm_future, Tensor) and lewm_future.ndim == 4, "LeWM future must be a tensor")
        batch = int(lewm_future.shape[0])
        future_shape = (batch, CANDIDATE_COUNT, FUTURE_STEPS, LATENT_DIM)
        _require(batch > 0 and tuple(lewm_future.shape) == future_shape and tuple(tdjepa_future.shape) == future_shape, "futures must have shape (B,63,5,192)")
        _require(tuple(lewm_context.shape) == tuple(tdjepa_context.shape) == (batch, 3, LATENT_DIM), "contexts must have shape (B,3,192)")
        _require(tuple(lewm_goal.shape) == tuple(tdjepa_goal.shape) == (batch, LATENT_DIM), "goals must have shape (B,192)")
        values = (lewm_context, lewm_future, lewm_goal, tdjepa_context, tdjepa_future, tdjepa_goal)
        _require(all(torch.is_floating_point(value) and bool(torch.isfinite(value).all()) for value in values), "context, future, and goal tensors must be finite floating point")
        _require(all(value.device == lewm_future.device for value in (*values, candidate_ids)), "feature inputs must share a device")
        _require(candidate_ids.dtype == torch.int64 and candidate_ids.shape == (batch, CANDIDATE_COUNT), "candidate IDs must be int64 (B,63)")
        expected = torch.arange(1, 64, dtype=torch.int64, device=candidate_ids.device).expand(batch, -1)
        _require(torch.equal(torch.sort(candidate_ids, dim=1).values, expected), "candidate IDs must be a permutation of literals 1..63")
        _require(callable(lewm_criterion) and callable(tdjepa_criterion), "both criteria must be callable")
        return batch

    @staticmethod
    def _full_trajectory(context: Tensor, future: Tensor) -> Tensor:
        return torch.cat((context[:, None].expand(-1, future.shape[1], -1, -1), future), dim=2)

    @staticmethod
    def _criterion_cost(
        criterion: CriterionCallable, context: Tensor, future: Tensor, goal: Tensor, label: str
    ) -> Tensor:
        cost = criterion(ReacherWorldModel._full_trajectory(context, future), goal)
        _require(isinstance(cost, Tensor) and cost.shape == future.shape[:2], f"{label} criterion must return (B,63)")
        _require(torch.is_floating_point(cost) and bool(torch.isfinite(cost).all()), f"{label} criterion cost must be finite floating point")
        _require(cost.device == future.device, f"{label} criterion cost must share the model device")
        return cost

    def forward_from_features(
        self,
        *,
        lewm_context: Tensor,
        lewm_future: Tensor,
        lewm_goal: Tensor,
        tdjepa_context: Tensor,
        tdjepa_future: Tensor,
        tdjepa_goal: Tensor,
        candidate_ids: Tensor,
        lewm_criterion: CriterionCallable,
        tdjepa_criterion: CriterionCallable,
    ) -> UnifiedReacherOutput:
        batch = self._validate_features(
            lewm_context, lewm_future, lewm_goal, tdjepa_context, tdjepa_future,
            tdjepa_goal, candidate_ids, lewm_criterion, tdjepa_criterion,
        )
        with torch.autocast(device_type=lewm_future.device.type, enabled=False):
            lewm_cost = self._criterion_cost(lewm_criterion, lewm_context, lewm_future, lewm_goal, "LeWM")
            td_cost = self._criterion_cost(tdjepa_criterion, tdjepa_context, tdjepa_future, tdjepa_goal, "TD-JEPA")
            lewm_rank64 = lexicographic_integer_ranks(lewm_cost.double(), candidate_ids).double() / float(CANDIDATE_COUNT - 1)
            td_rank64 = lexicographic_integer_ranks(td_cost.double(), candidate_ids).double() / float(CANDIDATE_COUNT - 1)
            base64 = self.fusion_alpha * lewm_rank64 + (1.0 - self.fusion_alpha) * td_rank64
            refined32, _ = self.refiner(
                lewm_future.float()[:, :, -1], lewm_goal.float(),
                tdjepa_future.float()[:, :, -1], tdjepa_goal.float(),
                lewm_rank64.float(), td_rank64.float(), base64.float(),
            )
            base_positions = lexicographic_integer_ranks(base64, candidate_ids).argmin(dim=1)
            refined_positions = lexicographic_integer_ranks(refined32.double(), candidate_ids).argmin(dim=1)
            rows = torch.arange(batch, device=candidate_ids.device)
            advantage = base64[rows, base_positions] - refined32.double()[rows, refined_positions]
            scale = float(10**self.gate_advantage_decimals)
            canonical_advantage = torch.round(advantage.double() * scale) / scale
            gate_active = canonical_advantage > self.gate_threshold
            final_score = torch.where(gate_active[:, None], refined32.double(), base64)
            target_ranks = lexicographic_integer_ranks(final_score, candidate_ids)
            transported = self.future_transport(
                tdjepa_future=tdjepa_future,
                lewm_future=lewm_future,
                lewm_rank=lewm_rank64,
                tdjepa_rank=td_rank64,
                target_rank=target_ranks.float() / float(CANDIDATE_COUNT - 1),
                gate_active=gate_active,
            )
            criterion_cost = self._criterion_cost(
                tdjepa_criterion, tdjepa_context, transported.future, tdjepa_goal, "transported TD-JEPA"
            )
            criterion_ranks = lexicographic_integer_ranks(criterion_cost.double(), candidate_ids)
        return UnifiedReacherOutput(
            future=transported.future,
            criterion_cost=criterion_cost,
            final_gated_score=final_score,
            base_score=base64,
            refined_score=refined32,
            gate_active=gate_active,
            target_ranks=target_ranks,
            criterion_ranks=criterion_ranks,
            temporal_coefficient=transported.temporal_coefficient,
        )
