import argparse
from pathlib import Path

import torch

from djepa_robot.training import fit_alignment, load_training_cache
from .common import new_output, write_json


def main():
    p = argparse.ArgumentParser(description="Train the relational head on measured candidate outcomes only")
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-synthetic", action="store_true", help="plumbing tests only; permanently labels checkpoint")
    args = p.parse_args()
    if Path(args.output).exists():
        p.error("output exists; choose a new run directory")
    cache = load_training_cache(args.cache, allow_synthetic=args.allow_synthetic)
    checkpoint, receipt = fit_alignment(cache, epochs=args.epochs, batch_size=args.batch_size,
                                        learning_rate=args.learning_rate, seed=args.seed, device=args.device)
    output = new_output(args.output)
    with (output / "alignment.pt").open("xb") as f:
        torch.save(checkpoint, f)
    write_json(output / "training.json", receipt)
    print(f"{receipt['artifact_scope']}: {output / 'alignment.pt'}")


if __name__ == "__main__":
    main()
