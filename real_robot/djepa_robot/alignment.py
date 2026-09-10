"""Trainable robot adaptation of the project's existing set-wise relational head.

Uses D-JEPA's set-wise relational architecture with robot-specific training.
Benchmark-specific gate and fusion parameters are not transferred.
"""

import torch
from torch import nn
import torch.nn.functional as F


def ordinal_ranks(costs, candidate_ids):
    if costs.ndim != 2 or candidate_ids.shape != costs.shape or costs.shape[1] < 1:
        raise ValueError("costs and candidate IDs must be nonempty matching [B,N]")
    if candidate_ids.dtype not in (torch.int32, torch.int64) or candidate_ids.device != costs.device:
        raise ValueError("integer IDs on the score device required")
    if not torch.isfinite(costs).all() or torch.any(candidate_ids < 0):
        raise ValueError("finite costs and nonnegative candidate IDs required")
    sorted_ids = candidate_ids.sort(dim=1).values
    if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ValueError("candidate IDs must be unique within each decision")
    precedes = ((costs[:, None, :] < costs[:, :, None]) |
                ((costs[:, None, :] == costs[:, :, None]) &
                 (candidate_ids[:, None, :] < candidate_ids[:, :, None])))
    return precedes.sum(dim=-1).float() / max(costs.shape[1] - 1, 1)


class RelationalAlignment(nn.Module):
    def __init__(self, dim, geometries=1, max_correction=0.2):
        super().__init__()
        if dim < 2 or geometries < 1 or not 0 < max_correction <= 1:
            raise ValueError("invalid relational head dimensions or correction bound")
        self.config = dict(dim=int(dim), geometries=int(geometries), max_correction=float(max_correction))
        self.candidate_encoder = nn.Sequential(nn.Linear(geometries * (dim + 1), 64),
                                               nn.LayerNorm(64), nn.GELU())
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0, activation="gelu",
                                            batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.correction_down, self.correction_up = nn.Linear(64, 8, bias=False), nn.Linear(8, 1, bias=False)
        nn.init.zeros_(self.correction_up.weight)

    def forward(self, terminal, goals, candidate_ids, *, native_costs):
        """Features [B,N,G,D], goals [B,G,D]; raw native costs [B,N,G].

        Geometry ranks are equally weighted in this initial prototype. The full
        native cost must be supplied independently of any feature pooling.
        """
        if terminal.ndim != 4:
            raise ValueError("terminal must be [B,N,G,D]")
        b, n, g, d = terminal.shape
        if (g, d) != (self.config["geometries"], self.config["dim"]) or goals.shape != (b, g, d):
            raise ValueError("feature/goal geometry or dimension mismatch")
        if native_costs.shape != (b, n, g) or candidate_ids.shape != (b, n):
            raise ValueError("native cost or candidate identity shape mismatch")
        if any(not x.is_floating_point() or not torch.isfinite(x).all()
               for x in (terminal, goals, native_costs)):
            raise ValueError("all features/costs must be finite floating tensors")
        if any(x.device != terminal.device for x in (goals, native_costs, candidate_ids)):
            raise ValueError("all inputs must share a device")
        ranks = torch.stack([ordinal_ranks(native_costs[:, :, i], candidate_ids) for i in range(g)], dim=-1)
        difference = F.layer_norm(terminal.float() - goals[:, None].float(), (d,))
        features = torch.cat((difference.flatten(2), ranks), dim=-1)
        encoded = self.set_encoder(self.candidate_encoder(features))
        raw = self.correction_up(torch.tanh(self.correction_down(encoded))).squeeze(-1)
        correction = self.config["max_correction"] * torch.tanh(raw)
        base = ranks.mean(dim=-1)
        return {"scores": base + correction, "base_scores": base, "correction": correction}
