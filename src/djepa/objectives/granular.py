"""Sparse supervision retains an explicit observation mask."""
from ..models.granular import _masked_refinement_loss as masked_refinement_loss

__all__ = ['masked_refinement_loss']
