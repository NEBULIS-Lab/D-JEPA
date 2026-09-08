"""Unified D-JEPA relational alignment decision geometry lifted into TD-JEPA futures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CANDIDATE_COUNT, FUTURE_STEPS, LATENT_DIM = 63, 5, 192
from djepa.native_scoring import native_terminal_cost


FUSION_ALPHA = 0.42
GATE_THRESHOLD = -0.03162526342176623
GATE_ADVANTAGE_DECIMALS = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ExactRealizationOutput:
    """Future/native deployment output plus label-free R diagnostics."""

    future: Tensor
    native_cost: Tensor
    final_gated_score: Tensor
    base_score: Tensor
    refined_score: Tensor
    gate_active: Tensor


class RelationalAligner(nn.Module):
    """Audited local copy of the reviewed dropout-free R set ranker."""

    def __init__(self, *, hidden_dim: int = 64, low_rank: int = 8, max_correction: float = 0.2) -> None:
        super().__init__()
        _require(hidden_dim == 64 and low_rank == 8 and float(max_correction) == 0.2, "reviewed R ranker dimensions changed")
        self.max_correction = float(max_correction)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(2 * LATENT_DIM + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=2 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.correction_down = nn.Linear(hidden_dim, low_rank, bias=False)
        self.correction_up = nn.Linear(low_rank, 1, bias=False)
        nn.init.zeros_(self.correction_up.weight)

    def forward(
        self,
        lewm_terminal: Tensor,
        lewm_goal: Tensor,
        tdjepa_terminal: Tensor,
        tdjepa_goal: Tensor,
        lewm_rank: Tensor,
        tdjepa_rank: Tensor,
        base_score: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lewm_difference = F.layer_norm(lewm_terminal.float() - lewm_goal[:, None].float(), (LATENT_DIM,))
        td_difference = F.layer_norm(tdjepa_terminal.float() - tdjepa_goal[:, None].float(), (LATENT_DIM,))
        features = torch.cat(
            (lewm_difference, td_difference, lewm_rank[..., None].float(), tdjepa_rank[..., None].float()),
            dim=-1,
        )
        encoded = self.set_encoder(self.candidate_encoder(features))
        raw = self.correction_up(torch.tanh(self.correction_down(encoded))).squeeze(-1)
        correction = self.max_correction * torch.tanh(raw)
        return base_score.float() + correction, correction


def _lexicographic_order(scores: Tensor, candidate_ids: Tensor) -> Tensor:
    """Return candidate positions ordered by score then literal ID."""
    ranks = _integer_ranks(scores, candidate_ids)
    return torch.argsort(ranks, dim=1)


def _integer_ranks(scores: Tensor, candidate_ids: Tensor) -> Tensor:
    """Device-sort-independent pairwise lexicographic ranks in ``[0,62]``."""
    current_score, other_score = scores[:, :, None], scores[:, None, :]
    current_id, other_id = candidate_ids[:, :, None], candidate_ids[:, None, :]
    precedes = (other_score < current_score) | ((other_score == current_score) & (other_id < current_id))
    return precedes.sum(dim=2)


def _ordinal_ranks(scores: Tensor, candidate_ids: Tensor, *, dtype: torch.dtype) -> Tensor:
    return _integer_ranks(scores, candidate_ids).to(dtype) / float(CANDIDATE_COUNT - 1)


def _strict_gate(advantage: Tensor, threshold: float) -> Tensor:
    """Reviewed gate semantics, kept explicit for boundary auditing."""
    return advantage > threshold


def _canonical_gate_advantage(advantage: Tensor) -> Tensor:
    """Remove reviewed CPU/CUDA reduction drift before the strict gate."""
    scale = float(10**GATE_ADVANTAGE_DECIMALS)
    return torch.round(advantage.double() * scale) / scale


class ExactRealizationWorldModel(nn.Module):
    """Reproduce reviewed R decisions and expose them only through new futures."""

    def __init__(
        self,
        *,
        fusion_alpha: float = FUSION_ALPHA,
        gate_threshold: float = GATE_THRESHOLD,
        hidden_dim: int = 64,
        low_rank: int = 8,
        max_correction: float = 0.2,
        tdjepa: nn.Module | None = None,
        lewm: nn.Module | None = None,
        action_mean: Tensor | None = None,
        action_std: Tensor | None = None,
    ) -> None:
        super().__init__()
        _require(0.0 <= float(fusion_alpha) <= 1.0, "fusion alpha must be in [0,1]")
        _require(torch.isfinite(torch.tensor(float(gate_threshold))).item(), "gate threshold must be finite")
        _require((tdjepa is None) == (lewm is None), "TD-JEPA and LeWM modules must be supplied together")
        self.fusion_alpha = float(fusion_alpha)
        self.gate_threshold = float(gate_threshold)
        self.tdjepa = tdjepa
        self.lewm = lewm
        self.refiner = RelationalAligner(
            hidden_dim=int(hidden_dim), low_rank=int(low_rank), max_correction=float(max_correction)
        )
        mean = torch.zeros(2, dtype=torch.float32) if action_mean is None else torch.as_tensor(action_mean, dtype=torch.float32)
        std = torch.ones(2, dtype=torch.float32) if action_std is None else torch.as_tensor(action_std, dtype=torch.float32)
        _require(mean.shape == std.shape == (2,) and bool(torch.isfinite(mean).all()) and bool(torch.isfinite(std).all()) and bool((std > 0).all()), "action normalization buffers are invalid")
        self.register_buffer("action_mean", mean.clone(), persistent=False)
        self.register_buffer("action_std", std.clone(), persistent=False)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32), persistent=False)

    @staticmethod
    def _validate(
        lewm_future: Tensor,
        lewm_goal: Tensor,
        tdjepa_future: Tensor,
        tdjepa_goal: Tensor,
        lewm_native_cost: Tensor,
        tdjepa_native_cost: Tensor,
        candidate_ids: Tensor,
    ) -> int:
        _require(isinstance(lewm_future, Tensor) and lewm_future.ndim == 4, "LeWM future must be a tensor")
        batch = lewm_future.shape[0]
        _require(batch > 0, "batch size must be positive")
        future_shape = (batch, CANDIDATE_COUNT, FUTURE_STEPS, LATENT_DIM)
        _require(tuple(lewm_future.shape) == future_shape and tuple(tdjepa_future.shape) == future_shape, "futures must have shape (B,63,5,192)")
        _require(tuple(lewm_goal.shape) == tuple(tdjepa_goal.shape) == (batch, LATENT_DIM), "goals must have shape (B,192)")
        _require(tuple(lewm_native_cost.shape) == tuple(tdjepa_native_cost.shape) == (batch, CANDIDATE_COUNT), "native costs must have shape (B,63)")
        floats = (lewm_future, lewm_goal, tdjepa_future, tdjepa_goal, lewm_native_cost, tdjepa_native_cost)
        _require(all(torch.is_floating_point(value) for value in floats), "unified R latent and cost inputs must be floating point")
        _require(all(bool(torch.isfinite(value).all()) for value in floats), "unified R inputs must be finite")
        device = lewm_future.device
        _require(all(value.device == device for value in (*floats, candidate_ids)), "unified R inputs must share a device")
        _require(candidate_ids.shape == (batch, CANDIDATE_COUNT) and candidate_ids.dtype == torch.int64, "candidate IDs must be int64 with shape (B,63)")
        _require(bool((candidate_ids > 0).all()), "candidate IDs must be positive")
        sorted_ids = torch.sort(candidate_ids, dim=1).values
        _require(bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()), "candidate IDs must be unique per row")
        return batch

    def forward_from_futures(
        self,
        lewm_future: Tensor,
        lewm_goal: Tensor,
        tdjepa_future: Tensor,
        tdjepa_goal: Tensor,
        lewm_native_cost: Tensor,
        tdjepa_native_cost: Tensor,
        candidate_ids: Tensor,
    ) -> ExactRealizationOutput:
        """Lift the exact reviewed gated order into native-score TD futures."""
        batch = self._validate(
            lewm_future, lewm_goal, tdjepa_future, tdjepa_goal,
            lewm_native_cost, tdjepa_native_cost, candidate_ids,
        )
        with torch.autocast(device_type=lewm_future.device.type, enabled=False):
            lewm_rank64 = _ordinal_ranks(lewm_native_cost.double(), candidate_ids, dtype=torch.float64)
            td_rank64 = _ordinal_ranks(tdjepa_native_cost.double(), candidate_ids, dtype=torch.float64)
            base64 = self.fusion_alpha * lewm_rank64 + (1.0 - self.fusion_alpha) * td_rank64
            refined32, _ = self.refiner(
                lewm_future.float()[:, :, -1], lewm_goal.float(),
                tdjepa_future.float()[:, :, -1], tdjepa_goal.float(),
                lewm_rank64.float(), td_rank64.float(), base64.float(),
            )
            base_positions = _lexicographic_order(base64, candidate_ids)[:, 0]
            refined_positions = _lexicographic_order(refined32.double(), candidate_ids)[:, 0]
            rows = torch.arange(batch, device=candidate_ids.device)
            advantage = base64[rows, base_positions] - refined32.double()[rows, refined_positions]
            gate_active = _strict_gate(_canonical_gate_advantage(advantage), self.gate_threshold)
            selected_positions = torch.where(gate_active, refined_positions, base_positions)

            final_score = torch.where(gate_active[:, None], refined32.double(), base64)
            strict_rank = _integer_ranks(final_score, candidate_ids)
            radius64 = (strict_rank.double() + 1.0) / 64.0

            td64 = tdjepa_future.double()
            goal64 = tdjepa_goal.double()
            direction = td64[:, :, -1] - goal64[:, None]
            rms = direction.square().mean(dim=-1, keepdim=True).sqrt()
            unit = direction / rms.clamp_min(1e-12)
            unit = torch.where((rms > 1e-12).expand_as(unit), unit, torch.ones_like(unit))
            lifted_terminal = goal64[:, None] + radius64.unsqueeze(-1) * unit
            future = tdjepa_future.clone()
            future[:, :, -1] = lifted_terminal.to(dtype=tdjepa_future.dtype)
            cost = native_terminal_cost(future.float(), tdjepa_goal.float())
            _require(bool(torch.isfinite(future).all()) and bool(torch.isfinite(cost).all()), "lifted future and native costs must be finite")
            native_ranks = _integer_ranks(cost.double(), candidate_ids)
            native_winner = native_ranks.argmin(dim=1)
            _require(bool(torch.equal(native_ranks, strict_rank)), "lifted native costs do not preserve the exact final R order")
            _require(bool(torch.equal(native_winner, selected_positions)), "lifted native argmin differs from R winner")
            winner_cost = cost[rows, selected_positions]
            others = torch.ones_like(cost, dtype=torch.bool)
            others[rows, selected_positions] = False
            _require(bool((cost[others].reshape(batch, -1) > winner_cost[:, None]).all()), "lifted R winner is not a strict native-cost minimum")
        return ExactRealizationOutput(
            future=future,
            native_cost=cost,
            final_gated_score=final_score,
            base_score=base64,
            refined_score=refined32,
            gate_active=gate_active,
        )

    def _preprocess_pixels(self, pixels: Tensor, *, context: bool) -> Tensor:
        expected_ndim = 5 if context else 4
        _require(isinstance(pixels, Tensor) and pixels.ndim == expected_ndim, "raw pixels have invalid shape")
        value = pixels
        if value.shape[-1] == 3:
            value = value.permute(0, 1, 4, 2, 3) if context else value.permute(0, 3, 1, 2)
        _require(value.shape[-3] == 3, "raw pixels must have three RGB channels")
        value = value.float().div(255.0) if not torch.is_floating_point(pixels) or float(value.max()) > 1.0 else value.float()
        shape = (1, 1, 3, 1, 1) if context else (1, 3, 1, 1)
        return (value - self.image_mean.view(shape)) / self.image_std.view(shape)

    def _assemble_actions(self, history_actions: Tensor, candidate_actions: Tensor) -> Tensor:
        _require(history_actions.ndim == 3 and history_actions.shape[1:] == (10, 2), "history actions must have shape (B,10,2)")
        _require(candidate_actions.shape == (history_actions.shape[0], CANDIDATE_COUNT, 25, 2), "candidate actions must have shape (B,63,25,2)")
        history = (history_actions.float() - self.action_mean) / self.action_std
        candidates = (candidate_actions.float() - self.action_mean) / self.action_std
        history = history.reshape(history.shape[0], 1, 2, 10).expand(-1, CANDIDATE_COUNT, -1, -1)
        candidates = candidates.reshape(candidates.shape[0], CANDIDATE_COUNT, 5, 10)
        return torch.cat((history, candidates), dim=2)

    def forward(
        self,
        context_pixels: Tensor,
        goal_pixels: Tensor,
        history_actions: Tensor,
        candidate_actions: Tensor,
        candidate_ids: Tensor,
    ) -> ExactRealizationOutput:
        """Deploy from raw pixels/actions with no filesystem or external selector."""
        _require(self.tdjepa is not None and self.lewm is not None, "raw deployment requires embedded TD-JEPA and LeWM")
        _require(context_pixels.shape[0] == goal_pixels.shape[0] == history_actions.shape[0] == candidate_actions.shape[0] == candidate_ids.shape[0], "raw deployment batch shapes do not align")
        context = self._preprocess_pixels(context_pixels, context=True)
        goals = self._preprocess_pixels(goal_pixels, context=False)
        actions = self._assemble_actions(history_actions, candidate_actions)
        per_start: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        for index in range(context.shape[0]):
            futures: list[Tensor] = []
            goal_latents: list[Tensor] = []
            for model in (self.lewm, self.tdjepa):
                output = model.rollout({"pixels": context[index : index + 1].unsqueeze(1)}, actions[index : index + 1])
                _require(isinstance(output, dict) and "predicted_emb" in output, "embedded backbone rollout lacks predicted_emb")
                futures.append(output["predicted_emb"][..., -FUTURE_STEPS:, :])
                encoded = model.encode({"pixels": goals[index : index + 1, None]})
                _require(isinstance(encoded, dict) and "emb" in encoded, "embedded backbone goal encoder lacks emb")
                goal_latents.append(encoded["emb"][:, 0])
            lewm_future, tdjepa_future = futures
            lewm_goal, tdjepa_goal = goal_latents
            per_start.append((
                lewm_future.half().float(), lewm_goal.half().float(),
                tdjepa_future.half().float(), tdjepa_goal.half().float(),
                native_terminal_cost(lewm_future.float(), lewm_goal.float()),
                native_terminal_cost(tdjepa_future.float(), tdjepa_goal.float()),
            ))
        lewm_future, lewm_goal, tdjepa_future, tdjepa_goal, lewm_native_cost, tdjepa_native_cost = (
            torch.cat([values[index] for values in per_start], dim=0) for index in range(6)
        )
        return self.forward_from_futures(
            lewm_future, lewm_goal, tdjepa_future, tdjepa_goal,
            lewm_native_cost, tdjepa_native_cost,
            candidate_ids,
        )
