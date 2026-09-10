"""Four-geometry rank fusion and bounded D-JEPA set-wise refinement."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


GEOMETRIES = ("lewm", "temporal_distance", "jepa_wm", "dino_wm")
LATENT_DIM = 192
CANDIDATE_COUNT = 63


class MultiGeometryError(ValueError):
    """Raised when the multi-geometry study contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiGeometryError(message)


@dataclass(frozen=True)
class ConvexFusionFit:
    geometry_order: tuple[str, ...]
    weights: tuple[float, ...]
    train_correct: int
    train_count: int
    mean_success_failure_margin: float
    grid_step: float


def _weight_grid(parts: int) -> tuple[tuple[float, float, float, float], ...]:
    rows = []
    for first in range(parts + 1):
        for second in range(parts - first + 1):
            for third in range(parts - first - second + 1):
                fourth = parts - first - second - third
                rows.append((first / parts, second / parts, third / parts, fourth / parts))
    return tuple(rows)


def _selected_columns(scores: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    selected = np.empty(scores.shape[0], dtype=np.int64)
    for row in range(scores.shape[0]):
        selected[row] = np.lexsort((candidate_ids[row], scores[row]))[0]
    return selected


def release_rows_for_pool_branches(
    *,
    pool_branch_rows: np.ndarray,
    pool_candidate_ids: np.ndarray,
    release_episode_ids: np.ndarray,
    release_candidate_ids: np.ndarray,
    authority_branch_rows: np.ndarray,
    authority_episode_ids: np.ndarray,
) -> np.ndarray:
    """Bridge internal branch rows to literal episode IDs without outcome access."""

    branches = np.asarray(pool_branch_rows)
    pool_ids = np.asarray(pool_candidate_ids)
    release_episodes = np.asarray(release_episode_ids)
    release_ids = np.asarray(release_candidate_ids)
    authority_branches = np.asarray(authority_branch_rows)
    authority_episodes = np.asarray(authority_episode_ids)
    _require(branches.dtype == np.int64 and branches.ndim == 1, "pool branch rows are invalid")
    _require(pool_ids.dtype == np.int64 and pool_ids.ndim == 2 and pool_ids.shape[0] == branches.size, "pool candidate IDs are invalid")
    _require(release_episodes.dtype == np.int64 and release_episodes.ndim == 1, "release episode IDs are invalid")
    _require(release_ids.dtype == np.int64 and release_ids.shape == (release_episodes.size, pool_ids.shape[1]), "release candidate IDs are invalid")
    _require(authority_branches.dtype == authority_episodes.dtype == np.int64 and authority_branches.shape == authority_episodes.shape and authority_branches.ndim == 1, "authority ID map is invalid")
    _require(len(set(authority_branches.tolist())) == authority_branches.size and len(set(authority_episodes.tolist())) == authority_episodes.size, "authority ID map must be one-to-one")
    branch_to_episode = {int(branch): int(episode) for branch, episode in zip(authority_branches, authority_episodes, strict=True)}
    release_map = {int(episode): index for index, episode in enumerate(release_episodes.tolist())}
    _require(len(release_map) == release_episodes.size, "release episode IDs must be unique")
    _require(all(int(branch) in branch_to_episode for branch in branches), "pool branch row is absent from authority map")
    episodes = [branch_to_episode[int(branch)] for branch in branches]
    _require(all(episode in release_map for episode in episodes), "release episode IDs do not cover pool")
    rows = np.asarray([release_map[episode] for episode in episodes], dtype=np.int64)
    _require(np.array_equal(release_ids[rows], pool_ids), "release candidate IDs/order mismatch")
    return rows


def fit_convex_rank_fusion(
    ranks: np.ndarray,
    candidate_ids: np.ndarray,
    labels: np.ndarray,
    *,
    grid_step: float,
) -> ConvexFusionFit:
    """Fit non-negative four-way ordinal fusion using training outcomes only."""

    values = np.asarray(ranks)
    ids = np.asarray(candidate_ids)
    outcomes = np.asarray(labels)
    _require(values.dtype == np.float64 and values.ndim == 3 and values.shape[2] == len(GEOMETRIES), "rank tensor must be float64 (N,C,4)")
    _require(ids.dtype == np.int64 and ids.shape == values.shape[:2], "candidate ID alignment is invalid")
    _require(outcomes.dtype == np.bool_ and outcomes.shape == ids.shape, "training outcome alignment is invalid")
    _require(values.shape[0] > 0 and values.shape[1] > 1 and np.isfinite(values).all(), "rank values are invalid")
    _require(np.all(ids > 0) and all(np.unique(row).size == row.size for row in ids), "candidate IDs must be positive and unique")
    positives = outcomes.sum(axis=1)
    _require(bool(np.all((positives > 0) & (positives < outcomes.shape[1]))), "each training row needs both outcome classes")
    _require(type(grid_step) is float and np.isfinite(grid_step) and 0 < grid_step <= 1, "fusion grid step is invalid")
    parts_float = 1.0 / grid_step
    parts = int(round(parts_float))
    _require(parts >= 1 and np.isclose(parts_float, parts, rtol=0.0, atol=1e-12), "fusion grid step must divide one exactly")
    rows = np.arange(values.shape[0])
    best_key: tuple[int, float, float, tuple[float, ...]] | None = None
    best_weights: tuple[float, ...] | None = None
    best_margin = 0.0
    for weights in _weight_grid(parts):
        fused = np.tensordot(values, np.asarray(weights, dtype=np.float64), axes=([2], [0]))
        columns = _selected_columns(fused, ids)
        correct = int(outcomes[rows, columns].sum())
        success_min = np.where(outcomes, fused, np.inf).min(axis=1)
        failure_min = np.where(~outcomes, fused, np.inf).min(axis=1)
        margin = float(np.mean(failure_min - success_min))
        # Prefer success, then robust class separation, then a decisive sparse
        # mixture, then deterministic geometry order.
        concentration = float(np.square(weights).sum())
        stable_margin = round(margin, 12)
        key = (correct, stable_margin, concentration, tuple(weights))
        if best_key is None or key > best_key:
            best_key = key
            best_weights = tuple(float(value) for value in weights)
            best_margin = margin
    assert best_key is not None and best_weights is not None
    return ConvexFusionFit(
        geometry_order=GEOMETRIES,
        weights=best_weights,
        train_correct=best_key[0],
        train_count=values.shape[0],
        mean_success_failure_margin=best_margin,
        grid_step=grid_step,
    )


def build_multi_geometry_features(
    lewm_terminal: torch.Tensor,
    lewm_goal: torch.Tensor,
    temporal_terminal: torch.Tensor,
    temporal_goal: torch.Tensor,
    geometry_ranks: torch.Tensor,
) -> torch.Tensor:
    _require(
        lewm_terminal.ndim == 3
        and lewm_terminal.shape[1:] == (CANDIDATE_COUNT, LATENT_DIM),
        "LeWM terminal shape is invalid",
    )
    batch = lewm_terminal.shape[0]
    _require(temporal_terminal.shape == lewm_terminal.shape, "Temporal-Distance terminal shape is invalid")
    _require(lewm_goal.shape == temporal_goal.shape == (batch, LATENT_DIM), "goal latent shape is invalid")
    _require(geometry_ranks.shape == (batch, CANDIDATE_COUNT, len(GEOMETRIES)), "geometry rank shape is invalid")
    tensors = (lewm_terminal, lewm_goal, temporal_terminal, temporal_goal, geometry_ranks)
    _require(all(torch.is_floating_point(value) and bool(torch.isfinite(value).all()) for value in tensors), "multi-geometry inputs must be finite floating point")
    lewm_difference = F.layer_norm(lewm_terminal.float() - lewm_goal[:, None, :].float(), (LATENT_DIM,))
    temporal_difference = F.layer_norm(temporal_terminal.float() - temporal_goal[:, None, :].float(), (LATENT_DIM,))
    return torch.cat((lewm_difference, temporal_difference, geometry_ranks.float()), dim=-1)


class MultiGeometryRanker(nn.Module):
    """Bounded full-set refiner over two latent and four ordinal geometries."""

    def __init__(self, *, hidden_dim: int = 64, low_rank: int = 8, max_correction: float = 0.2) -> None:
        super().__init__()
        _require(type(hidden_dim) is int and hidden_dim > 0 and hidden_dim % 4 == 0, "hidden dimension is invalid")
        _require(type(low_rank) is int and low_rank > 0 and np.isfinite(max_correction) and max_correction > 0, "correction settings are invalid")
        self.max_correction = float(max_correction)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(2 * LATENT_DIM + len(GEOMETRIES), hidden_dim),
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
        lewm_terminal: torch.Tensor,
        lewm_goal: torch.Tensor,
        temporal_terminal: torch.Tensor,
        temporal_goal: torch.Tensor,
        geometry_ranks: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = build_multi_geometry_features(
            lewm_terminal, lewm_goal, temporal_terminal, temporal_goal, geometry_ranks
        )
        _require(
            base_scores.shape == features.shape[:2]
            and torch.is_floating_point(base_scores)
            and bool(torch.isfinite(base_scores).all()),
            "base scores are invalid",
        )
        encoded = self.set_encoder(self.candidate_encoder(features))
        raw = self.correction_up(torch.tanh(self.correction_down(encoded))).squeeze(-1)
        correction = self.max_correction * torch.tanh(raw)
        return base_scores.float() + correction, correction
