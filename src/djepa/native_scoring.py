from __future__ import annotations
import torch
from torch import Tensor

def _require(condition, message):
    if not condition:
        raise ValueError(message)

def native_terminal_cost(future: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Return TD-JEPA's deployed terminal-latent MSE for each candidate.

    ``future`` is ``(batch, candidate, horizon, latent_dim)`` and ``goal`` is
    ``(batch, latent_dim)``.  Earlier rollout steps intentionally have no
    influence on this deployed score.
    """
    _require(isinstance(future, torch.Tensor) and isinstance(goal, torch.Tensor), "future and goal must be tensors")
    _require(future.ndim == 4 and goal.ndim == 2, "future must be (batch,candidate,horizon,latent) and goal (batch,latent)")
    _require(future.shape[0] == goal.shape[0] and future.shape[-1] == goal.shape[-1] and future.shape[2] >= 1, "future and goal shapes must align")
    _require(torch.is_floating_point(future) and torch.is_floating_point(goal), "future and goal must be floating point")
    _require(bool(torch.isfinite(future).all()) and bool(torch.isfinite(goal).all()), "future and goal must be finite")
    return (future[:, :, -1, :] - goal[:, None, :]).square().mean(dim=-1)
