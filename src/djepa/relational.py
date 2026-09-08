"""Compatibility import; implementation lives in djepa.models.relational."""
from .models import relational as _implementation
from .models.relational import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
