"""Setwise, bounded terminal-representation refinement for D-JEPA relational alignment."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LATENT_DIM = 192
CANDIDATE_COUNT = 63


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class RelationalAligner(nn.Module):
    """Permutation-equivariant full-candidate ranker with bounded low-rank output."""

    def __init__(self, *, hidden_dim: int = 64, low_rank: int = 8, max_correction: float = 0.2) -> None:
        super().__init__()
        _require(hidden_dim % 4 == 0 and low_rank > 0 and max_correction > 0, "invalid DTAIL ranker dimensions")
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

    @staticmethod
    def _check_inputs(
        lewm_terminal: torch.Tensor,
        lewm_goal: torch.Tensor,
        tdjepa_terminal: torch.Tensor,
        tdjepa_goal: torch.Tensor,
        lewm_rank: torch.Tensor,
        tdjepa_rank: torch.Tensor,
        base_score: torch.Tensor,
    ) -> tuple[int, torch.device]:
        _require(lewm_terminal.ndim == 3 and lewm_terminal.shape[1:] == (CANDIDATE_COUNT, LATENT_DIM), "LeWM terminal shape must be (B,63,192)")
        batch = lewm_terminal.shape[0]
        _require(tdjepa_terminal.shape == lewm_terminal.shape, "TD-JEPA terminal shape mismatch")
        _require(lewm_goal.shape == tdjepa_goal.shape == (batch, LATENT_DIM), "goal latent shape mismatch")
        _require(lewm_rank.shape == tdjepa_rank.shape == base_score.shape == (batch, CANDIDATE_COUNT), "rank score shape mismatch")
        tensors = (lewm_terminal, lewm_goal, tdjepa_terminal, tdjepa_goal, lewm_rank, tdjepa_rank, base_score)
        _require(all(torch.is_floating_point(value) and bool(torch.isfinite(value).all()) for value in tensors), "DTAIL ranker inputs must be finite floating point")
        return batch, lewm_terminal.device

    def forward(
        self,
        lewm_terminal: torch.Tensor,
        lewm_goal: torch.Tensor,
        tdjepa_terminal: torch.Tensor,
        tdjepa_goal: torch.Tensor,
        lewm_rank: torch.Tensor,
        tdjepa_rank: torch.Tensor,
        base_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_inputs(lewm_terminal, lewm_goal, tdjepa_terminal, tdjepa_goal, lewm_rank, tdjepa_rank, base_score)
        lewm_difference = F.layer_norm(lewm_terminal.float() - lewm_goal[:, None, :].float(), (LATENT_DIM,))
        td_difference = F.layer_norm(tdjepa_terminal.float() - tdjepa_goal[:, None, :].float(), (LATENT_DIM,))
        features = torch.cat((lewm_difference, td_difference, lewm_rank[..., None].float(), tdjepa_rank[..., None].float()), dim=-1)
        encoded = self.set_encoder(self.candidate_encoder(features))
        raw = self.correction_up(torch.tanh(self.correction_down(encoded))).squeeze(-1)
        correction = self.max_correction * torch.tanh(raw)
        return base_score.float() + correction, correction


def relational_loss(
    scores: torch.Tensor,
    physical_success: torch.Tensor,
    correction: torch.Tensor,
    *,
    tail_k: int,
    temperature: float = 0.05,
    margin: float = 0.02,
    pair_weight: float = 0.25,
    trust_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Full-list success mass plus success/failure ordering and trust region."""
    _require(scores.ndim == 2 and scores.shape[1] == CANDIDATE_COUNT and torch.is_floating_point(scores), "scores must have shape (B,63)")
    labels = physical_success
    _require(labels.dtype == torch.bool and labels.shape == scores.shape, "success labels must be Boolean and score-shaped")
    _require(correction.shape == scores.shape and torch.isfinite(correction).all(), "correction shape mismatch")
    _require(1 <= tail_k <= CANDIDATE_COUNT and temperature > 0, "invalid tail loss settings")
    positive_per_row = labels.sum(dim=1)
    _require(bool(((positive_per_row > 0) & (positive_per_row < CANDIDATE_COUNT)).all()), "each training row requires both success classes")
    scaled = -scores / temperature
    positive_logits = scaled.masked_fill(~labels, -torch.inf)
    listwise = (torch.logsumexp(scaled, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()
    pair_terms: list[torch.Tensor] = []
    tail_order = torch.argsort(scores.detach(), dim=1)[:, :tail_k]
    for row in range(scores.shape[0]):
        tail_mask = torch.zeros(CANDIDATE_COUNT, dtype=torch.bool, device=scores.device)
        tail_mask[tail_order[row]] = True
        success_scores = scores[row][labels[row] & tail_mask]
        failure_scores = scores[row][~labels[row] & tail_mask]
        if not success_scores.numel() or not failure_scores.numel():
            success_scores = scores[row][labels[row]]
            failure_scores = scores[row][~labels[row]]
        pair_terms.append(F.softplus((margin + success_scores[:, None] - failure_scores[None, :]) / temperature).mean())
    pair = torch.stack(pair_terms).mean()
    trust = correction.square().mean()
    total = listwise + pair_weight * pair + trust_weight * trust
    return total, {
        "candidate_count": CANDIDATE_COUNT,
        "positive_count": int(labels.sum().item()),
        "listwise_loss": float(listwise.detach()),
        "pair_loss": float(pair.detach()),
        "trust_loss": float(trust.detach()),
    }


def _selected_positions(scores: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    values, ids = np.asarray(scores), np.asarray(candidate_ids)
    _require(values.ndim == 2 and values.shape[1] == CANDIDATE_COUNT and ids.shape == values.shape, "candidate score rows must align")
    positions = np.empty(values.shape[0], dtype=np.int64)
    for row in range(values.shape[0]):
        positions[row] = np.lexsort((ids[row], values[row]))[0]
    return positions


def gated_candidate_ids(base_scores: np.ndarray, refined_scores: np.ndarray, candidate_ids: np.ndarray, *, threshold: float) -> np.ndarray:
    base, refined, ids = np.asarray(base_scores), np.asarray(refined_scores), np.asarray(candidate_ids)
    _require(base.shape == refined.shape == ids.shape and np.isfinite(base).all() and np.isfinite(refined).all(), "gated score arrays must align and be finite")
    base_positions, refined_positions = _selected_positions(base, ids), _selected_positions(refined, ids)
    rows = np.arange(base.shape[0])
    advantage = base[rows, base_positions] - refined[rows, refined_positions]
    use_refined = advantage > float(threshold)
    positions = np.where(use_refined, refined_positions, base_positions)
    selected = np.array(ids[rows, positions], dtype=np.int64, copy=True)
    selected.flags.writeable = False
    return selected


def calibrate_confidence_threshold(base_scores: np.ndarray, refined_scores: np.ndarray, candidate_ids: np.ndarray, physical_success: np.ndarray) -> float:
    """Choose the most permissive threshold attaining maximal train-calibration success."""
    base, refined, ids, labels = map(np.asarray, (base_scores, refined_scores, candidate_ids, physical_success))
    _require(labels.dtype == np.bool_ and labels.shape == base.shape == refined.shape == ids.shape, "calibration labels must be Boolean and aligned")
    base_positions, refined_positions = _selected_positions(base, ids), _selected_positions(refined, ids)
    rows = np.arange(base.shape[0])
    advantage = base[rows, base_positions] - refined[rows, refined_positions]
    unique = np.unique(advantage)
    thresholds = [math.inf]
    thresholds.extend(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:]))
    thresholds.append(float(np.nextafter(unique[0], -math.inf)))
    best: tuple[int, int, float] | None = None
    for threshold in thresholds:
        use_refined = advantage > threshold
        positions = np.where(use_refined, refined_positions, base_positions)
        success = int(labels[rows, positions].sum())
        changes = int(use_refined.sum())
        record = (success, changes, -float(threshold))
        if best is None or record > best:
            best = record
            selected_threshold = float(threshold)
    return selected_threshold
