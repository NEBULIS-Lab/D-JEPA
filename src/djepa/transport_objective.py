"""Compatibility import; implementation lives in djepa.objectives.transport_objective."""
from .objectives import transport_objective as _implementation
from .objectives.transport_objective import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
