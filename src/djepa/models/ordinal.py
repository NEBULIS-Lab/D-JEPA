from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DecisionAligner(nn.Module):
    def __init__(self, input_dim, max_correction=0.2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_correction = float(max_correction)
        self.candidate_encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU())
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, activation="gelu",
                                          batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.correction_down = nn.Linear(64, 8, bias=False)
        self.correction_up = nn.Linear(8, 1, bias=False)
        nn.init.zeros_(self.correction_up.weight)

    def forward(self, features, base_rank):
        hidden = self.set_encoder(self.candidate_encoder(features))
        raw = self.correction_up(torch.tanh(self.correction_down(hidden))).squeeze(-1)
        correction = self.max_correction * torch.tanh(raw)
        return base_rank + correction, correction


def decision_loss(scores, correction, successes, temperature=0.05):
    positive = successes.bool()
    valid = positive.any(dim=-1)
    # Full-list successful probability mass. Sets without reachable candidates
    # are retained in the dataset but contribute only the preservation penalty.
    logits = -scores / temperature
    if valid.any():
        normalizer = torch.logsumexp(logits[valid], dim=-1)
        mass = torch.logsumexp(logits[valid].masked_fill(~positive[valid], -torch.inf), dim=-1)
        list_loss = (normalizer-mass).mean()
    else:
        list_loss = scores.sum()*0
    top = scores.detach().argsort(dim=-1)[:, :min(16, scores.shape[-1])]
    boundary_score = scores.gather(1, top)
    boundary_positive = positive.gather(1, top)
    pairs = boundary_positive[:, :, None] & ~boundary_positive[:, None, :]
    differences = boundary_score[:, :, None]-boundary_score[:, None, :]
    pair_loss = F.softplus(differences[pairs]/temperature).mean() if pairs.any() else scores.sum()*0
    return list_loss + 0.25*pair_loss + 0.01*correction.square().mean()
