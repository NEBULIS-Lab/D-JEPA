"""Compatibility import; implementation lives in djepa.models.temporal_transport."""
from .models import temporal_transport as _implementation
from .models.temporal_transport import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
