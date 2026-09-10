import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def measured_cache(path, *, provenance="synthetic_test", overlap=False):
    rng = np.random.default_rng(4)
    np.savez(path, terminal=rng.normal(size=(4, 4, 1, 8)).astype("float32"),
             goals=rng.normal(size=(4, 1, 8)).astype("float32"),
             candidate_ids=np.tile(np.arange(4), (4, 1)),
             native_costs=np.tile(np.array([0.1, 0.2, 0.3, 0.4]), (4, 1))[..., None],
             true_costs=np.tile(np.array([1., 0., 1., 0.]), (4, 1)),
             observed=np.ones((4, 4), dtype=bool),
             provenance=np.full((4, 4), provenance),
             evidence_ids=np.array([[f"synthetic-test/{b}/{n}" for n in range(4)] for b in range(4)]),
             start_ids=np.array(["s0", "s1", "s0" if overlap else "s2", "s3"]),
             split=np.array(["train", "train", "val", "val"]),
             feature_spec=np.array("spatial_pool_2x2"), source_model=np.array("synthetic-fixture"))


def run_cli(module, *args):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    return subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=45)


def test_unknown_candidates_and_leakage_refused(tmp_path):
    training = importlib.import_module("djepa_robot.training")
    path = tmp_path / "cache.npz"
    measured_cache(path, provenance="unexecuted")
    with pytest.raises(ValueError, match="provenance"):
        training.load_training_cache(path)
    measured_cache(path, provenance="executed_sim", overlap=True)
    with pytest.raises(ValueError, match="overlap"):
        training.load_training_cache(path)


def test_synthetic_requires_explicit_opt_in(tmp_path):
    training = importlib.import_module("djepa_robot.training")
    path = tmp_path / "cache.npz"
    measured_cache(path)
    with pytest.raises(ValueError, match="synthetic"):
        training.load_training_cache(path)
    cache = training.load_training_cache(path, allow_synthetic=True)
    assert cache["artifact_scope"] == "synthetic_plumbing_test_not_robot_result"


def test_missing_observations_never_become_labels(tmp_path):
    training = importlib.import_module("djepa_robot.training")
    path = tmp_path / "cache.npz"
    measured_cache(path, provenance="executed_sim")
    with np.load(path) as loaded:
        values = dict(loaded)
    values["observed"][0, 0] = False
    np.savez(path, **values)
    with pytest.raises(ValueError, match="observed"):
        training.load_training_cache(path)


def test_training_and_score_cli_record_synthetic_scope_and_no_overwrite(tmp_path):
    cache, output = tmp_path / "cache.npz", tmp_path / "train"
    measured_cache(cache)
    result = run_cli("scripts.train_alignment", "--cache", cache, "--output", output,
                     "--epochs", 3, "--allow-synthetic")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "training.json").read_text())
    assert report["artifact_scope"] == "synthetic_plumbing_test_not_robot_result"
    assert len(report["history"]) == 3
    assert (output / "alignment.pt").is_file()
    score_output = tmp_path / "score"
    result = run_cli("scripts.score_candidates", "--cache", cache, "--checkpoint",
                     output / "alignment.pt", "--output", score_output, "--allow-synthetic")
    assert result.returncode == 0, result.stdout + result.stderr
    scored = json.loads((score_output / "selection.json").read_text())
    assert len(scored["decisions"]) == 4
    assert scored["hardware_execution"] is False
    result = run_cli("scripts.train_alignment", "--cache", cache, "--output", output,
                     "--epochs", 1, "--allow-synthetic")
    assert result.returncode != 0
    assert len(json.loads((output / "training.json").read_text())["history"]) == 3


def test_planning_cli_rejects_hardware_and_joint_input_before_loading(tmp_path):
    target = tmp_path / "plan"
    result = run_cli("scripts.plan_latents", "--execute", "--output", target)
    assert result.returncode != 0
    assert not target.exists()
    cache = tmp_path / "joint.npz"
    np.savez(cache, action_semantics=np.array("joint_position_target_plus_gripper"))
    result = run_cli("scripts.plan_latents", "--cache", cache, "--output", target,
                     "--checkout", tmp_path / "absent", "--checkpoint", tmp_path / "absent.pt")
    assert result.returncode != 0
    assert "semantics" in result.stderr
    assert not target.exists()


def test_latent_planning_cli_selects_from_exact_shared_pool(tmp_path, monkeypatch):
    import torch
    from djepa_robot.actions import SEMANTICS
    from djepa_robot.world_model import ACPredictor
    script = importlib.import_module("scripts.plan_latents")

    class SyntheticPredictor(torch.nn.Module):
        def forward(self, x, actions, states):
            return x + actions[:, :, 0].repeat_interleave(4, dim=1)[..., None]

    fake_checkpoint = tmp_path / "synthetic-only.pt"
    fake_checkpoint.write_bytes(b"test fixture; never loaded as real weights")
    cache = tmp_path / "latents.npz"
    np.savez(cache, context=np.zeros((1, 4, 8), dtype="float32"),
             goal=np.ones((1, 4, 8), dtype="float32") * 0.03,
             pose=np.zeros((1, 7), dtype="float32"), dt=np.array(0.1),
             action_semantics=np.array(SEMANTICS), source_model=np.array(str(fake_checkpoint)),
             data_origin=np.array("synthetic_test"))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"lower": [-0.05] * 7, "upper": [0.05] * 7,
                                 "horizon": 2, "samples": 8, "elites": 2, "iterations": 2,
                                 "seed": 1, "model_dt": 0.1, "chunk_size": 4}))
    monkeypatch.setattr(script, "load_local_vjepa2_ac", lambda *a, **k: (
        None, ACPredictor(SyntheticPredictor(), expected_dt=0.1, normalize_reps=False),
        {"scope": "synthetic_test"}))
    target = tmp_path / "plan"
    monkeypatch.setattr(sys, "argv", ["plan_latents", "--cache", str(cache), "--output", str(target),
                       "--checkpoint", str(fake_checkpoint), "--checkout", str(tmp_path),
                       "--config", str(config), "--allow-synthetic"])
    script.main()
    receipt = json.loads((target / "plan.json").read_text())
    assert receipt["hardware_execution"] is False
    assert receipt["artifact_scope"] == "synthetic_plumbing_test_not_robot_result"
    with np.load(target / "candidate_pool.npz") as pool:
        i = int(np.lexsort((pool["candidate_ids"], pool["native_costs"]))[0])
        assert receipt["native_selected_id"] == int(pool["candidate_ids"][i])
        np.testing.assert_allclose(receipt["selected_prediction_space_actions"], pool["actions"][i])
