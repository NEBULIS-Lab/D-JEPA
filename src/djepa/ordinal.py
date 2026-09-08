"""Compatibility import; implementation lives in djepa.models.ordinal."""
from .models import ordinal as _implementation
from .models.ordinal import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
