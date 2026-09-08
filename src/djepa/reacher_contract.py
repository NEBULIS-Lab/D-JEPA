"""Compatibility import; implementation lives in djepa.models.reacher_contract."""
from .models import reacher_contract as _implementation
from .models.reacher_contract import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
