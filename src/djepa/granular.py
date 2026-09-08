"""Compatibility import; implementation lives in djepa.models.granular."""
from .models import granular as _implementation
from .models.granular import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
