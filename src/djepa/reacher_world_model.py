"""Compatibility import; implementation lives in djepa.models.reacher_world_model."""
from .models import reacher_world_model as _implementation
from .models.reacher_world_model import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
