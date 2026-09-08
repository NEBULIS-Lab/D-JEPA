"""CPU contracts for source-faithful PushT feature and physics reproduction."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn


CANDIDATES = 63


class _FakeBackbone(nn.Module):
    """Small deterministic stand-in for the external pretrained backbones."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(offset), requires_grad=False)

    def rollout(self, info, action_sequence):
        batch, candidates = action_sequence.shape[:2]
        signal = action_sequence.float().mean(dim=(-1, -2)) + self.anchor
        future = signal[:, :, None, None].expand(batch, candidates, 5, 192).clone()
        return {"predicted_emb": future}

    def encode(self, info):
        batch = info["pixels"].shape[0]
        goal = torch.zeros(batch, 1, 192, device=self.anchor.device) + self.anchor
        return {"emb": goal}


class _FakePushT:
    metadata = {"render_fps": 10}

    def __init__(self) -> None:
        self.state = np.zeros(7, dtype=np.float64)
        self.goal = np.zeros(7, dtype=np.float64)
        self.steps = 0

    def reset(self, *, seed, options):
        assert seed == 0
        self.state = np.asarray(options["state"], dtype=np.float64).copy()
        self.goal = np.asarray(options["goal_state"], dtype=np.float64).copy()
        self.steps = 0
        return {"state": self.state.copy()}, {}

    def step(self, action):
        self.steps += 1
        self.state[:2] += np.asarray(action, dtype=np.float64)
        # Deliberately terminate early; fixed-horizon evaluation must not stop.
        return {"state": self.state.copy()}, -1.0, self.steps == 3, False, {}

    def _get_obs(self):
        return self.state.copy()

    def eval_state(self, goal, state):
        distance = float(np.linalg.norm(np.asarray(goal) - np.asarray(state)))
        return distance < 1e-6, distance

    def render(self):
        return np.full((8, 8, 3), self.steps, dtype=np.uint8)


def _raw_inputs(batch: int = 1) -> dict[str, np.ndarray]:
    candidate_ids = np.tile(np.arange(101, 101 + CANDIDATES, dtype=np.int64), (batch, 1))
    actions = np.zeros((batch, CANDIDATES, 25, 2), dtype=np.float32)
    actions[:, :, :, 0] = np.arange(CANDIDATES, dtype=np.float32)[None, :, None] / 100.0
    return {
        "start_ids": np.arange(9001, 9001 + batch, dtype=np.int64),
        "candidate_ids": candidate_ids,
        "candidate_actions": actions,
        "history_actions": np.zeros((batch, 10, 2), dtype=np.float32),
        "context_pixels": np.zeros((batch, 3, 4, 4, 3), dtype=np.uint8),
        "goal_pixels": np.zeros((batch, 4, 4, 3), dtype=np.uint8),
        "initial_state": np.zeros((batch, 7), dtype=np.float32),
        "goal_state": np.zeros((batch, 7), dtype=np.float32),
    }


class NativeReproductionTests(unittest.TestCase):
    def _model(self):
        from djepa.models.exact_realization import ExactRealizationWorldModel

        return ExactRealizationWorldModel(
            tdjepa=_FakeBackbone(0.2),
            lewm=_FakeBackbone(0.1),
            action_mean=torch.zeros(2),
            action_std=torch.ones(2),
        ).eval()

    def test_feature_preparation_runs_both_predictors_and_keeps_literal_ids(self):
        """Catches cached replay or position-derived IDs replacing real rollouts."""
        from djepa.native import prepare_pusht_features

        raw = _raw_inputs()
        output = prepare_pusht_features(self._model(), raw)

        self.assertEqual(output["lewm_future"].shape, (1, CANDIDATES, 5, 192))
        self.assertEqual(output["tdjepa_future"].shape, (1, CANDIDATES, 5, 192))
        np.testing.assert_array_equal(output["candidate_ids"], raw["candidate_ids"])
        selected_position = int(output["selected_positions"][0])
        self.assertEqual(output["selected_ids"][0], raw["candidate_ids"][0, selected_position])
        self.assertEqual(output["candidate_action_sha256"].dtype.kind, "S")
        self.assertEqual(output["candidate_action_sha256"].shape, (1, CANDIDATES))

    def test_feature_preparation_matches_exact_model_float_cost_ordering(self):
        """Catches half-rounding predictor outputs before the native costs."""
        from djepa.native import prepare_pusht_features

        raw = _raw_inputs()
        raw["history_actions"][:] = 0.00037
        model = self._model()
        prepared = prepare_pusht_features(model, raw)
        direct = model(
            torch.as_tensor(raw["context_pixels"]),
            torch.as_tensor(raw["goal_pixels"]),
            torch.as_tensor(raw["history_actions"]),
            torch.as_tensor(raw["candidate_actions"]),
            torch.as_tensor(raw["candidate_ids"]),
        )
        np.testing.assert_array_equal(
            prepared["selected_ids"],
            raw["candidate_ids"][np.arange(len(raw["start_ids"])), direct.native_cost.argmin(1).numpy()],
        )
        np.testing.assert_allclose(prepared["native_cost"], direct.native_cost.numpy(), rtol=0, atol=0)
        # This literal derives from the fake predictor before half conversion.
        self.assertAlmostEqual(float(prepared["lewm_cost"][0, 0]), 1.1175510204081635e-08, places=12)

    def test_feature_preparation_rejects_duplicate_candidate_identity(self):
        """Catches accidental use of candidate array positions as identities."""
        from djepa.native import prepare_pusht_features

        raw = _raw_inputs()
        raw["candidate_ids"][0, 1] = raw["candidate_ids"][0, 0]
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_pusht_features(self._model(), raw)

    def test_rollout_executes_selected_literal_action_for_all_25_controls(self):
        """Catches argmin-position confusion and early termination shortening."""
        from djepa.native import candidate_action_sha256, evaluate_pusht_rollouts

        raw = _raw_inputs()
        chosen_position = 7
        decisions = {
            "start_ids": raw["start_ids"].copy(),
            "candidate_ids": raw["candidate_ids"].copy(),
            "candidate_action_sha256": candidate_action_sha256(raw["candidate_actions"]),
            "selected_ids": raw["candidate_ids"][:, chosen_position].copy(),
        }
        env = _FakePushT()
        output = evaluate_pusht_rollouts(env, raw, decisions, capture_frames=True)

        self.assertEqual(env.steps, 25)
        self.assertEqual(output["trajectory_states"].shape, (1, 26, 7))
        self.assertEqual(output["frames"].shape, (1, 25, 8, 8, 3))
        self.assertTrue(output["terminated"][0, 2])
        self.assertAlmostEqual(output["trajectory_states"][0, -1, 0], 25 * 0.07, places=6)
        self.assertEqual(output["selected_positions"].tolist(), [chosen_position])
        self.assertEqual(output["selected_action_sha256"].shape, (1,))

    def test_rollout_refuses_actions_that_differ_from_scored_candidates(self):
        """Catches evaluating a mutated action tensor under stale decisions."""
        from djepa.native import candidate_action_sha256, evaluate_pusht_rollouts

        raw = _raw_inputs()
        decisions = {
            "start_ids": raw["start_ids"].copy(),
            "candidate_ids": raw["candidate_ids"].copy(),
            "candidate_action_sha256": candidate_action_sha256(raw["candidate_actions"]),
            "selected_ids": raw["candidate_ids"][:, 0].copy(),
        }
        raw["candidate_actions"][0, 0, 0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "action identity"):
            evaluate_pusht_rollouts(_FakePushT(), raw, decisions)

    def test_frozen_native_config_verifies_checkpoint_and_split(self):
        """Catches silently accepting a different weight file or test split."""
        from djepa.native import load_native_config, sha256_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile"
            profile.mkdir()
            (profile / "model.pt").write_bytes(b"weights")
            (profile / "config.json").write_text(json.dumps({
                "architecture": "exact_realization_world_model",
                "decision": {
                    "fusion_alpha": 0.42,
                    "gate_threshold": -0.03162526342176623,
                    "gate_comparison": "strict_gt",
                    "advantage_quantization": "decimal_half_even_4",
                },
                "preprocessing": {
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                    "context_frames": 3,
                    "action_block": 5,
                    "future_blocks": 5,
                    "action_mean": [-0.007812565192580223, 0.006860686466097832],
                    "action_std": [0.2084674835205078, 0.20674866437911987],
                },
            }), encoding="utf-8")
            cfg = root / "native.json"
            payload = {
                "schema": "djepa_native_pusht_v1",
                "task": "pusht",
                "evaluation_split": "independent_256",
                "checkpoint": {
                    "path": "profile",
                    "weights_sha256": sha256_file(profile / "model.pt"),
                    "config_sha256": sha256_file(profile / "config.json"),
                },
                "protocol": {
                    "candidate_count": 63,
                    "context_frames": 3,
                    "history_controls": 10,
                    "control_horizon": 25,
                    "action_block": 5,
                    "future_blocks": 5,
                    "action_units": "relative_position_delta",
                    "action_low": [-1.0, -1.0],
                    "action_high": [1.0, 1.0],
                    "control_hz": 10,
                    "physics_dt": 0.01,
                    "integrator_steps_per_control": 10,
                },
                "decision": {
                    "fusion_alpha": 0.42,
                    "gate_threshold": -0.03162526342176623,
                    "gate_comparison": "strict_gt",
                    "gate_advantage_decimals": 4,
                },
                "preprocessing": {
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                    "action_mean": [-0.007812565192580223, 0.006860686466097832],
                    "action_std": [0.2084674835205078, 0.20674866437911987],
                },
            }
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_native_config(cfg)
            self.assertEqual(loaded["checkpoint_dir"], profile.resolve())
            payload["evaluation_split"] = "train"
            cfg.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split"):
                load_native_config(cfg)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
