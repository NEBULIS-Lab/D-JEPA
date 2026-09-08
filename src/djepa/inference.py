"""Compatibility import; implementation lives in djepa.evaluation.inference."""
from .evaluation import inference as _implementation
from .evaluation.inference import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
