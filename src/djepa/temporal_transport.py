"""Small action-conditioned temporal residual transport for Reacher futures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ReacherTransportError(ValueError):
    """Raised when future residual transport inputs are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReacherTransportError(message)


def lexicographic_integer_ranks(scores: Tensor, candidate_ids: Tensor) -> Tensor:
    """Return zero-based ranks ordered by score and then literal candidate ID."""
    _require(isinstance(scores, Tensor) and scores.ndim == 2, "scores must have shape (B,C)")
    _require(
        isinstance(candidate_ids, Tensor)
        and candidate_ids.shape == scores.shape
        and candidate_ids.dtype == torch.int64,
        "candidate IDs must be int64 with score shape",
    )
    _require(torch.is_floating_point(scores) and bool(torch.isfinite(scores).all()), "scores must be finite floating point")
    _require(scores.device == candidate_ids.device, "scores and candidate IDs must share a device")
    current_score = scores[:, :, None]
    other_score = scores[:, None, :]
    current_id = candidate_ids[:, :, None]
    other_id = candidate_ids[:, None, :]
    precedes = (other_score < current_score) | (
        (other_score == current_score) & (other_id < current_id)
    )
    return precedes.sum(dim=2)


@dataclass(frozen=True)
class ReacherResidualOutput:
    future: Tensor
    temporal_coefficient: Tensor


class ReacherFutureResidualHead(nn.Module):
    """Apply a bounded per-time residual between same-action TD and LeWM futures."""

    def __init__(self, *, hidden_dim: int = 32, radius: float = 0.1) -> None:
        super().__init__()
        _require(isinstance(hidden_dim, int) and hidden_dim > 0, "transport hidden dimension must be positive")
        radius_value = float(radius)
        _require(torch.isfinite(torch.tensor(radius_value)).item() and radius_value > 0.0, "transport radius must be positive and finite")
        self.radius = radius_value
        self.temporal_down = nn.Linear(5, hidden_dim)
        self.temporal_up = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.temporal_up.weight)
        nn.init.zeros_(self.temporal_up.bias)

    def forward(
        self,
        *,
        tdjepa_future: Tensor,
        lewm_future: Tensor,
        lewm_rank: Tensor,
        tdjepa_rank: Tensor,
        target_rank: Tensor,
        gate_active: Tensor,
    ) -> ReacherResidualOutput:
        _require(isinstance(tdjepa_future, Tensor) and tdjepa_future.ndim == 4, "TD future must have shape (B,C,T,D)")
        batch, candidates, steps, latent_dim = tdjepa_future.shape
        _require(batch > 0 and candidates > 1 and steps > 0 and latent_dim > 0, "future dimensions must be positive")
        _require(lewm_future.shape == tdjepa_future.shape, "LeWM and TD futures must align")
        ranks = (lewm_rank, tdjepa_rank, target_rank)
        _require(all(value.shape == (batch, candidates) and torch.is_floating_point(value) for value in ranks), "rank features must be floating (B,C)")
        _require(all(bool(torch.isfinite(value).all()) and bool(((value >= 0.0) & (value <= 1.0)).all()) for value in ranks), "rank features must be finite in [0,1]")
        _require(gate_active.shape == (batch,) and gate_active.dtype == torch.bool, "gate must be bool (B,)")
        values = (tdjepa_future, lewm_future, *ranks, gate_active)
        _require(all(value.device == tdjepa_future.device for value in values), "transport inputs must share a device")
        _require(torch.is_floating_point(tdjepa_future) and torch.is_floating_point(lewm_future), "futures must be floating point")
        _require(bool(torch.isfinite(tdjepa_future).all()) and bool(torch.isfinite(lewm_future).all()), "futures must be finite")

        time = torch.linspace(0.0, 1.0, steps, device=tdjepa_future.device, dtype=torch.float32)
        features = torch.stack(
            (
                lewm_rank.float()[:, :, None].expand(-1, -1, steps),
                tdjepa_rank.float()[:, :, None].expand(-1, -1, steps),
                target_rank.float()[:, :, None].expand(-1, -1, steps),
                gate_active.float()[:, None, None].expand(-1, candidates, steps),
                time[None, None].expand(batch, candidates, -1),
            ),
            dim=-1,
        )
        coefficient = self.radius * torch.tanh(
            self.temporal_up(F.gelu(self.temporal_down(features))).squeeze(-1)
        )
        direction = F.normalize(
            lewm_future.float() - tdjepa_future.float(), p=2.0, dim=-1
        )
        future = tdjepa_future + (coefficient[..., None] * direction).to(
            tdjepa_future.dtype
        )
        _require(bool(torch.isfinite(future).all()), "transported future must be finite")
        return ReacherResidualOutput(future=future, temporal_coefficient=coefficient)
