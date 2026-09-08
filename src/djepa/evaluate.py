"""Compatibility import; implementation lives in djepa.cli.evaluate."""
from .cli import evaluate as _implementation
from .cli.evaluate import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()
