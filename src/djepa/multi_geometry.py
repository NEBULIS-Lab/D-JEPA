"""Compatibility import; implementation lives in djepa.models.multi_geometry."""
from .models import multi_geometry as _implementation
from .models.multi_geometry import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
