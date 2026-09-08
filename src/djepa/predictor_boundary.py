"""Compatibility import; implementation lives in djepa.models.predictor_boundary."""
from .models import predictor_boundary as _implementation
from .models.predictor_boundary import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)
