"""Compatibility import; implementation lives in djepa.models.exact_realization."""
from .models import exact_realization as _implementation
from .models.exact_realization import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
