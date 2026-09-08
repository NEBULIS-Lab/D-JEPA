from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
CANDIDATE_COUNT = 63
R_FEATURE_DIM = 2304

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _masked_refinement_loss(
    scores: torch.Tensor, labels: torch.Tensor, correction: torch.Tensor,
    supervision_mask: torch.Tensor, *, tail_k: int,
    temperature: float = 0.05, margin: float = 0.02,
    pair_weight: float = 0.25, trust_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    _require(labels.dtype == supervision_mask.dtype == torch.bool, "labels and supervision mask must be Boolean")
    _require(labels.shape == supervision_mask.shape == scores.shape == correction.shape, "masked refinement tensors must align")
    _require(bool((supervision_mask.sum(dim=1) > 0).all()), "each training row needs supervised candidates")
    positive = labels & supervision_mask
    negative = (~labels) & supervision_mask
    valid = positive.any(dim=1) & negative.any(dim=1)
    trust = correction[supervision_mask].square().mean()
    if not bool(valid.any()):
        zero = scores.sum() * 0.0
        return zero + trust_weight * trust, {
            "candidate_count": CANDIDATE_COUNT, "supervised_count": int(supervision_mask.sum().item()),
            "usable_row_count": 0, "positive_count": int(positive.sum().item()),
            "listwise_loss": 0.0, "pair_loss": 0.0, "trust_loss": float(trust.detach()),
        }
    active_scores = scores[valid]
    active_positive = positive[valid]
    active_mask = supervision_mask[valid]
    scaled = -active_scores / temperature
    denominator = scaled.masked_fill(~active_mask, -torch.inf)
    positive_logits = scaled.masked_fill(~active_positive, -torch.inf)
    listwise = (torch.logsumexp(denominator, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()
    pair_terms: list[torch.Tensor] = []
    masked_order_scores = active_scores.detach().masked_fill(~active_mask, torch.inf)
    tail_order = torch.argsort(masked_order_scores, dim=1)[:, :tail_k]
    for row in range(active_scores.shape[0]):
        tail_mask = torch.zeros(CANDIDATE_COUNT, dtype=torch.bool, device=scores.device)
        tail_mask[tail_order[row]] = True
        success_scores = active_scores[row][active_positive[row] & tail_mask]
        failure_scores = active_scores[row][active_mask[row] & ~active_positive[row] & tail_mask]
        if not success_scores.numel() or not failure_scores.numel():
            success_scores = active_scores[row][active_positive[row]]
            failure_scores = active_scores[row][active_mask[row] & ~active_positive[row]]
        pair_terms.append(F.softplus((margin + success_scores[:, None] - failure_scores[None, :]) / temperature).mean())
    pair = torch.stack(pair_terms).mean()
    total = listwise + pair_weight * pair + trust_weight * trust
    return total, {
        "candidate_count": CANDIDATE_COUNT, "supervised_count": int(supervision_mask.sum().item()),
        "usable_row_count": int(valid.sum().item()), "positive_count": int(positive.sum().item()),
        "listwise_loss": float(listwise.detach()), "pair_loss": float(pair.detach()),
        "trust_loss": float(trust.detach()),
    }


class GranularSetRanker(nn.Module):
    """Permutation-equivariant set ranker with a bounded low-rank correction."""

    def __init__(self, *, input_dim: int, hidden_dim: int = 64, low_rank: int = 8, max_correction: float = 0.2) -> None:
        super().__init__()
        _require(input_dim in {R_FEATURE_DIM + 1, 4}, "Granular ranker input width is invalid")
        _require(hidden_dim > 0 and hidden_dim % 4 == 0 and low_rank > 0 and max_correction > 0, "Granular ranker dimensions are invalid")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.low_rank = int(low_rank)
        self.max_correction = float(max_correction)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
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

    def forward(self, features: torch.Tensor, base_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _require(features.ndim == 3 and features.shape[1:] == (CANDIDATE_COUNT, self.input_dim), "Granular ranker feature shape is invalid")
        _require(base_scores.shape == features.shape[:2], "Granular base score shape is invalid")
        _require(torch.is_floating_point(features) and torch.is_floating_point(base_scores), "Granular ranker inputs must be floating point")
        _require(bool(torch.isfinite(features).all()) and bool(torch.isfinite(base_scores).all()), "Granular ranker inputs must be finite")
        encoded = self.set_encoder(self.candidate_encoder(features.float()))
        raw = self.correction_up(torch.tanh(self.correction_down(encoded))).squeeze(-1)
        correction = self.max_correction * torch.tanh(raw)
        return base_scores.float() + correction, correction
