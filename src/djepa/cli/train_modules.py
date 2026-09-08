"""Train released D-JEPA modules whose public supervision inputs are complete."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import parse_config, save_resolved_config
from ..training.modules import SUPPORTED_MODULES, train_released_module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--module", required=True, choices=SUPPORTED_MODULES)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tail-k", type=int, default=16)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--low-rank", type=int, default=8)
    parser.add_argument("--max-correction", type=float, default=0.2)
    parser.add_argument("--mg-grid-step", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = parse_config(_parser(), argv)
    output = train_released_module(args)
    save_resolved_config(args, output / "resolved_config.json")
    return output


if __name__ == "__main__":
    main()
