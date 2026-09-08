"""Compatibility import; implementation lives in djepa.evaluation.native_scoring."""
from .evaluation import native_scoring as _implementation
from .evaluation.native_scoring import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
