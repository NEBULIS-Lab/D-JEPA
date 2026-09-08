"""Compatibility import; implementation lives in djepa.cli.train."""
from .cli import train as _implementation
from .cli.train import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()
