#!/usr/bin/env python3
"""Run the full recorded PushT predictors from raw observation/action inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from djepa.native import load_native_config, load_recorded_pusht_model, prepare_pusht_features, sha256_file
from djepa.native.io import load_npz, prepend_python_roots, save_json, save_npz


def _require_split(inputs, expected: str) -> None:
    if "split" not in inputs:
        raise ValueError("native raw inputs must carry their explicit split")
    values = np.asarray(inputs["split"])
    if values.shape != np.asarray(inputs.get("start_ids")).shape:
        raise ValueError("native raw split must provide one value per start")
    decoded = np.asarray([item.decode() if isinstance(item, bytes) else str(item) for item in values.reshape(-1)])
    if len(decoded) == 0 or set(decoded.tolist()) != {expected}:
        raise ValueError(f"native raw inputs are not exclusively split {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Raw context/goal pixels, histories, candidate controls and literal IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tdjepa-template", type=Path, required=True, help="Recorded TD-JEPA weights_epoch_10.pt")
    parser.add_argument("--lewm-template", type=Path, required=True, help="Recorded trusted lewm_object.ckpt")
    parser.add_argument("--python-root", action="append", default=[], help="Explicit upstream import root; repeat as needed")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sidecar_path = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to replace native output or sidecar: {args.output}")
    config = load_native_config(args.config)
    inputs = load_npz(args.inputs)
    _require_split(inputs, config["evaluation_split"])
    prepend_python_roots(args.python_root)
    model = load_recorded_pusht_model(
        config["checkpoint_dir"], tdjepa_template=args.tdjepa_template,
        lewm_template=args.lewm_template, device=args.device,
    )
    arrays = prepare_pusht_features(model, inputs)
    digest = save_npz(args.output, arrays)
    metadata = {
        "schema": "djepa_native_pusht_features_v1",
        "mode": "native_predictor_execution",
        "task": "pusht",
        "evaluation_split": config["evaluation_split"],
        "count": int(len(arrays["start_ids"])),
        "candidate_count": 63,
        "control_horizon": 25,
        "source_inputs_sha256": sha256_file(args.inputs),
        "output_sha256": digest,
        "checkpoint": config["checkpoint"],
        "labels_consumed": False,
    }
    save_json(sidecar_path, metadata)
    print(args.output)


if __name__ == "__main__":
    main()
