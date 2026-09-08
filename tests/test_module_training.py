"""Training entrypoint tests for released, complete module inputs."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


CANDIDATES = 63


def _labels(rows: int, *, flip_unknown: bool = False) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((rows, CANDIDATES), dtype=np.bool_)
    mask[:, :16] = True
    success = np.zeros_like(mask)
    success[:, 0] = True
    if flip_unknown:
        success[:, 16:] = True
    return success, mask


def _write_pusht_pool(root: Path, *, split: str = "train") -> Path:
    root.mkdir()
    rows = 4
    ids = np.tile(np.arange(1, CANDIDATES + 1, dtype=np.int64), (rows, 1))
    starts = np.array([101, 102, 103, 104], dtype=np.int64)
    rng = np.random.default_rng(2)
    np.savez(
        root / "inputs.npz",
        features=rng.normal(size=(rows, CANDIDATES, 386)).astype(np.float32),
        base_scores=np.tile(np.linspace(0, 1, CANDIDATES), (rows, 1)),
        candidate_ids=ids,
        start_ids=starts,
    )
    success = np.zeros((rows, CANDIDATES), dtype=np.bool_)
    success[:, 0] = True
    np.savez(root / "labels.npz", success=success)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "task": "pusht",
                "split": split,
                "count": rows,
                "candidate_count": CANDIDATES,
                "fusion_alpha": 0.42,
                "relational_calibration_ids": [103],
            }
        )
    )
    return root


def _write_granular_split(
    root: Path,
    *,
    split: str,
    start_ids: list[str],
    flip_unknown: bool = False,
) -> Path:
    root.mkdir()
    rows = len(start_ids)
    ids = np.tile(np.arange(1, CANDIDATES + 1, dtype=np.int64), (rows, 1))
    rng = np.random.default_rng(7 if split == "train" else 8)
    ranks = np.empty((rows, CANDIDATES, 4), dtype=np.float64)
    for view in range(4):
        ranks[..., view] = np.tile(
            np.roll(np.linspace(0, 1, CANDIDATES), view), (rows, 1)
        )
    np.savez(
        root / "inputs.npz",
        features=rng.normal(size=(rows, CANDIDATES, 2305)).astype(np.float32),
        base_scores=np.tile(np.linspace(0, 1, CANDIDATES), (rows, 1)),
        candidate_ids=ids,
        multiview_ranks=ranks,
        start_ids=np.asarray(start_ids),
    )
    success, mask = _labels(rows, flip_unknown=flip_unknown)
    np.savez(root / "labels.npz", success=success, supervision_mask=mask)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "task": "granular",
                "split": split,
                "count": rows,
                "candidate_count": CANDIDATES,
                "supervision": "Only mask=True entries are observed.",
            }
        )
    )
    return root


def _train_args(module: str, train: Path, output: Path, calibration: Path | None = None) -> list[str]:
    args = [
        "--module",
        module,
        "--train",
        str(train),
        "--output",
        str(output),
        "--updates",
        "1",
        "--batch-size",
        "2",
        "--evaluate-every",
        "1",
        "--seed",
        "19",
    ]
    if calibration is not None:
        args.extend(["--calibration", str(calibration)])
    return args


class ModuleTrainingTests(unittest.TestCase):
    def test_baseline_only_calibration_uses_finite_globally_disabled_gate(self):
        from djepa.training.modules import finite_gate_threshold
        from djepa.inference import gated_positions

        threshold = finite_gate_threshold(float("inf"), max_correction=0.2)
        self.assertEqual(threshold, np.finfo(np.float64).max)
        self.assertTrue(np.isfinite(threshold))
        json.dumps({"gate_threshold": threshold}, allow_nan=False)
        base = np.array([[0.0, 0.01, 0.02]], dtype=np.float64)
        refined = np.array([[0.2, -0.2, 0.02]], dtype=np.float32).astype(np.float64)
        rounded_advantage = base[0, 0] - refined[0, 1]
        self.assertGreater(rounded_advantage, 0.2)
        ids = np.array([[3, 2, 1]])
        np.testing.assert_array_equal(
            gated_positions(base, refined, ids, threshold, "base_minus_refined"),
            [0],
        )

    def test_pusht_uses_metadata_calibration_ids_and_exports_strict_profile(self):
        from djepa.cli.train_modules import main
        from djepa.inference import load_profile, predict

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = _write_pusht_pool(root / "pool")
            output = root / "trained"
            main(_train_args("pusht-relational", pool, output))

            receipt = json.loads((output / "training_receipt.json").read_text())
            self.assertEqual(receipt["splits"]["calibration_ids"], [103])
            self.assertEqual(receipt["splits"]["fit_ids"], [101, 102, 104])
            self.assertEqual(
                set(receipt["sources"]["pool"]),
                {"inputs.npz", "labels.npz", "metadata.json"},
            )
            state = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
            self.assertTrue(state)
            model, config = load_profile(output)
            model.load_state_dict(state, strict=True)
            with np.load(pool / "inputs.npz", allow_pickle=False) as values:
                inputs = {name: values[name][:1] for name in ("features", "base_scores", "candidate_ids")}
            self.assertEqual(predict(output, inputs)["selected_ids"].shape, (1,))
            self.assertEqual(config["weights_sha256"], receipt["weights_sha256"])

    def test_protected_pusht_split_is_rejected_before_output_creation(self):
        from djepa.cli.train_modules import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = _write_pusht_pool(root / "pool", split="development")
            output = root / "trained"
            with self.assertRaisesRegex(ValueError, "protected|train"):
                main(_train_args("pusht-relational", protected, output))
            self.assertFalse(output.exists())

    def test_sparse_unknown_labels_cannot_change_granular_training(self):
        from djepa.cli.train_modules import main
        from djepa.inference import load_profile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_a = _write_granular_split(root / "train-a", split="train", start_ids=["a", "b"])
            train_b = _write_granular_split(
                root / "train-b", split="train", start_ids=["a", "b"], flip_unknown=True
            )
            calibration = _write_granular_split(
                root / "calibration", split="calibration", start_ids=["c", "d"]
            )
            first, second = root / "first", root / "second"
            main(_train_args("granular-relational", train_a, first, calibration))
            main(_train_args("granular-relational", train_b, second, calibration))
            state_a = torch.load(first / "model.pt", map_location="cpu", weights_only=True)
            state_b = torch.load(second / "model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(state_a.keys(), state_b.keys())
            for name in state_a:
                torch.testing.assert_close(state_a[name], state_b[name], rtol=0, atol=0)
            load_profile(first)[0].load_state_dict(state_a, strict=True)

    def test_granular_multiview_fits_fusion_and_rejects_identity_overlap(self):
        from djepa.cli.train_modules import main
        from djepa.inference import predict

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_granular_split(root / "train", split="train", start_ids=["a", "b"])
            overlap = _write_granular_split(
                root / "overlap", split="calibration", start_ids=["b", "c"]
            )
            with self.assertRaisesRegex(ValueError, "disjoint"):
                main(_train_args("granular-multiview", train, root / "bad", overlap))
            self.assertFalse((root / "bad").exists())

            calibration = _write_granular_split(
                root / "calibration", split="calibration", start_ids=["c", "d"]
            )
            output = root / "trained"
            main(_train_args("granular-multiview", train, output, calibration))
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["architecture"], "granular")
            self.assertEqual(config["model"]["input_dim"], 4)
            self.assertAlmostEqual(sum(config["fusion_weights"]), 1.0)
            torch.testing.assert_close(
                torch.tensor(config["fusion_weights"]) / 0.05,
                (torch.tensor(config["fusion_weights"]) / 0.05).round(),
            )
            with np.load(calibration / "inputs.npz", allow_pickle=False) as values:
                ranks = values["multiview_ranks"][:1]
                inference_inputs = {
                    "features": ranks.astype(np.float32),
                    "base_scores": ranks @ np.asarray(config["fusion_weights"], dtype=np.float64),
                    "candidate_ids": values["candidate_ids"][:1],
                }
            self.assertEqual(predict(output, inference_inputs)["selected_ids"].shape, (1,))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
