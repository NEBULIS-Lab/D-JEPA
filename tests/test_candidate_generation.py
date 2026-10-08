"""CPU regression tests for action proposals, identities and label-free retention."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def reference(dtype=np.float32):
    return np.linspace(-0.8, 0.8, 50).reshape(25, 2).astype(dtype)


class CandidateGenerationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('djepa.candidates'),
                             'The public candidate-generation API is missing')
        return importlib.import_module('djepa.candidates')

    def test_confirmation_matches_original_quantized_actions(self):
        pool = self.api().control_candidates(reference(), seed=123, profile='pusht-confirmation')
        self.assertEqual(pool.actions.shape, (256, 25, 2))
        self.assertEqual(pool.actions.dtype, np.float16)
        # Captured from the original experiment generator, not this implementation.
        self.assertEqual(hashlib.sha256(pool.actions.tobytes()).hexdigest(),
                         '0cc7b7538d15565bc7934dacfc35ebf415a623dbafd0410678f7939760bc892e')
        self.assertEqual(len({row.tobytes() for row in pool.actions}), 256)
        np.testing.assert_array_equal(pool.actions[0], reference().astype(np.float16))

    def test_reference_pool_preserves_dtype_and_family_counts(self):
        for dtype in (np.float32, np.float64):
            pool = self.api().control_candidates(reference(dtype), seed=123)
            self.assertEqual(pool.actions.dtype, dtype)
            expected = {np.float32: '770f0fc35633eee60aacacd3d43a285cbce03d96cec980a2da777059d621648f',
                        np.float64: '72217afebb2f9c6bb19933c988483f25b883d843b18fa85d36a7c7092703a2c2'}
            self.assertEqual(hashlib.sha256(pool.actions.tobytes()).hexdigest(), expected[dtype])
            self.assertEqual(pool.actions.shape, (64, 25, 2))
            self.assertEqual(dict(zip(*np.unique(pool.types, return_counts=True))),
                             {'reference': 1, 'noisy_0.05': 8, 'noisy_0.15': 8,
                              'noisy_0.3': 8, 'shuffled': 7, 'smooth_random': 16,
                              'uniform_random': 16})
            self.assertEqual(len({a.tobytes() for a in pool.actions}), 64)
            np.testing.assert_array_equal(pool.actions[0], reference(dtype))
            deployed = pool.without_reference()
            np.testing.assert_array_equal(deployed.ids, np.arange(1, 64))
            np.testing.assert_array_equal(deployed.actions, pool.actions[1:])

    def test_seed_reproducibility_and_action_bounds(self):
        api = self.api()
        for profile in ('reference-control', 'pusht-confirmation'):
            a = api.control_candidates(reference(), seed=17, profile=profile)
            b = api.control_candidates(reference(), seed=17, profile=profile)
            c = api.control_candidates(reference(), seed=18, profile=profile)
            np.testing.assert_array_equal(a.actions, b.actions)
            self.assertFalse(np.array_equal(a.actions, c.actions))
            self.assertTrue(np.all((a.actions >= -1) & (a.actions <= 1)))

    def test_constant_reference_cannot_supply_unique_shuffles(self):
        for profile in ('reference-control', 'pusht-confirmation', 'visual-shape'):
            with self.assertRaisesRegex(ValueError, 'shuffl'):
                self.api().control_candidates(np.zeros((25, 2), np.float32), seed=0,
                                              profile=profile, max_attempts=2)

    def test_visual_shape_normalization_and_quantization(self):
        pool = self.api().control_candidates(reference() * 100, seed=20260944,
                                             profile='visual-shape')
        self.assertEqual(pool.actions.shape, (64, 25, 2))
        self.assertEqual(pool.actions.dtype, np.float32)
        self.assertGreaterEqual(pool.seed, 20260944)
        np.testing.assert_array_equal(pool.actions, pool.actions.astype(np.float16).astype(np.float32))
        self.assertEqual(len({a.tobytes() for a in pool.actions}), 64)
        np.testing.assert_array_equal(pool.actions[0],
                                      (reference()*100/np.float32(100)).astype(np.float16).astype(np.float32))

    def test_retention_uses_native_cost_and_family_coverage_with_id_ties(self):
        api = self.api()
        pool = api.control_candidates(reference(), seed=123, profile='pusht-confirmation')
        costs = np.zeros(256)
        costs[0] = -1000  # The reference never enters the deployable top 16.
        selected = api.retain_pusht_candidates(pool, costs)
        self.assertEqual(len(selected.ids), 63)
        np.testing.assert_array_equal(selected.ids[:16], np.arange(1, 17))
        self.assertEqual(len(set(selected.ids.tolist())), 63)
        self.assertNotIn(0, selected.ids)
        self.assertEqual(set(selected.types), set(pool.types) - {'reference'})
        np.testing.assert_array_equal(selected.actions, pool.actions[selected.ids])
        np.testing.assert_array_equal(api.retain_pusht_candidates(pool, costs * 5 + 7).ids,
                                      selected.ids)

    def test_retention_rejects_misaligned_or_nonfinite_costs_and_labels(self):
        api = self.api()
        pool = api.control_candidates(reference(), seed=123, profile='pusht-confirmation')
        for costs in (np.zeros(255), np.full(256, np.nan)):
            with self.assertRaises(ValueError):
                api.retain_pusht_candidates(pool, costs)
        with self.assertRaises(TypeError):
            api.retain_pusht_candidates(pool, np.zeros(256), success=np.ones(256))

    def test_invalid_control_inputs_fail_early(self):
        api = self.api()
        for actions in (np.zeros((24, 2)), np.zeros((25, 2), dtype=int),
                        np.full((25, 2), np.nan), reference() * 3):
            with self.assertRaises(ValueError):
                api.control_candidates(actions, seed=0)
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                api.control_candidates(reference(), seed=seed)

    def test_granular_geometry_temporal_shift_and_anchor(self):
        base = np.linspace(-1, 1, 80).reshape(20, 4).astype(np.float32)
        pool = self.api().granular_candidates(base, start_id='s1', goal_id='g1', offset=2)
        self.assertEqual(pool.actions.shape, (63, 5, 4))
        self.assertEqual(hashlib.sha256(pool.actions.tobytes()).hexdigest(),
                         '92e31880ef2926fc4c3438efe35103d0307137434055581a2b3b97332ea16cba')
        np.testing.assert_array_equal(pool.ids, np.arange(1, 64))
        np.testing.assert_array_equal(pool.actions[0], base[2:7])
        self.assertEqual(dict(zip(*np.unique(pool.types, return_counts=True))),
                         {'scale_rotation': 35, 'temporal': 8, 'endpoint': 16, 'fixed_hash_filler': 4})
        self.assertEqual(len({a.tobytes() for a in pool.actions}), 63)
        # Candidate 44 offsets the endpoint by +0.15 in x, leaving the start fixed.
        expected = base[2:7].astype(np.float64)
        expected[:, 2] += .15
        np.testing.assert_array_equal(pool.actions[43], expected.astype(np.float32))
        # Candidate 36 has shift -1.75: position 0 interpolates rows 1 and 2.
        np.testing.assert_allclose(pool.actions[35, 0], .25*base[3]+.75*base[4], atol=1e-7)
        other = self.api().granular_candidates(base, start_id='s2', goal_id='g1', offset=2)
        np.testing.assert_array_equal(pool.actions[:59], other.actions[:59])
        self.assertFalse(np.array_equal(pool.actions[59:], other.actions[59:]))
        self.assertTrue(np.all(np.abs(pool.actions) <= 4))

    def test_granular_invalid_offset_and_duplicate_actions(self):
        api = self.api()
        with self.assertRaises(ValueError):
            api.granular_candidates(np.zeros((20, 4)), start_id='s', goal_id='g', offset=16)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            api.granular_candidates(np.zeros((20, 4)), start_id='s', goal_id='g')

    def cli(self, *args):
        script = ROOT/'scripts/generate_candidates.py'
        self.assertTrue(script.is_file(), 'The public candidate CLI is missing')
        return subprocess.run([sys.executable, str(script), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True)

    def test_cli_generation_retention_identity_checks_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'reference.npy', reference())
            config = ROOT/'configs/candidates/pusht-confirmation.yaml'
            result = self.cli('generate', '--config', config, '--reference-actions', root/'reference.npy',
                              '--seed', 123, '--start-id', 42, '--output', root/'pool')
            self.assertEqual(result.returncode, 0, result.stderr)
            pool_file = root/'pool/candidates.npz'
            with np.load(pool_file) as pool:
                self.assertEqual(pool['candidate_actions'].shape, (1, 256, 25, 2))
                score_data = {name: pool[name] for name in
                              ('candidate_ids', 'candidate_action_sha256', 'start_ids')}
            score_data['native_costs'] = np.zeros((1, 256))
            np.savez(root/'scores.npz', **score_data)
            result = self.cli('retain', '--pool', pool_file, '--scores', root/'scores.npz',
                              '--output', root/'selected')
            self.assertEqual(result.returncode, 0, result.stderr)
            with np.load(root/'selected/candidates.npz') as selected:
                self.assertEqual(selected['candidate_actions'].shape, (1, 63, 25, 2))
                self.assertFalse(np.any(selected['candidate_ids'] == 0))
            metadata = json.loads((root/'selected/metadata.json').read_text())
            self.assertFalse(metadata['labels_consumed'])
            old = pool_file.read_bytes()
            repeat = self.cli('generate', '--config', config, '--reference-actions', root/'reference.npy',
                              '--seed', 123, '--start-id', 42, '--output', root/'pool')
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual(pool_file.read_bytes(), old)
            score_data['candidate_ids'] = score_data['candidate_ids'][:, ::-1]
            np.savez(root/'bad-scores.npz', **score_data)
            bad = self.cli('retain', '--pool', pool_file, '--scores', root/'bad-scores.npz',
                           '--output', root/'bad')
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse((root/'bad').exists())

    def test_cli_rejects_outcome_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root/'labels.npz', success=np.ones(256))
            result = self.cli('retain', '--pool', root/'labels.npz', '--scores', root/'labels.npz',
                              '--output', root/'out')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root/'out').exists())

    def test_example_generates_all_profiles_without_models_or_labels(self):
        script = ROOT/'examples/generate_candidates.py'
        self.assertTrue(script.is_file(), 'The CPU-only candidate example is missing')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'example'
            result = subprocess.run([sys.executable, str(script), '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ('pusht-selected', 'reference-control', 'visual-shape', 'granular'):
                with np.load(output/name/'candidates.npz') as data:
                    self.assertEqual(data['candidate_actions'].shape[1], 63)
                metadata = json.loads((output/name/'metadata.json').read_text())
                self.assertFalse(metadata['labels_consumed'])

    def test_cli_checks_score_hash_and_rejects_additional_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'reference.npy', reference())
            config = ROOT/'configs/candidates/pusht-confirmation.yaml'
            created = self.cli('generate', '--config', config, '--reference-actions', root/'reference.npy',
                               '--seed', 123, '--start-id', 42, '--output', root/'pool')
            self.assertEqual(created.returncode, 0, created.stderr)
            with np.load(root/'pool/candidates.npz') as pool:
                scores = {key: pool[key] for key in
                          ('start_ids','candidate_ids','candidate_action_sha256')}
            scores['native_costs'] = np.zeros((1,256))
            for extra in (dict(success=np.ones((1,256))),
                          dict(candidate_action_sha256=np.full((1,256),'0'*64))):
                np.savez(root/'scores.npz', **(scores | extra))
                result = self.cli('retain', '--pool', root/'pool/candidates.npz',
                                  '--scores', root/'scores.npz', '--output', root/'out')
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
