"""Compatibility import; implementation lives in djepa.models.plasticity."""
from .models import plasticity as _implementation
from .models.plasticity import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
