"""Workflow boundaries: executable configuration, safe exports and replay."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_site_assets_and_navigation_are_resolvable(self):
        run = subprocess.run([sys.executable, str(ROOT/'scripts/check_site.py')],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_evaluation_wrapper_replays_labels_and_refuses_overwrite(self):
        from djepa.ordinal import DecisionAligner
        import os
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            profile = root/'profile'
            profile.mkdir()
            (profile/'config.json').write_text(json.dumps({'architecture':'set_aligner',
                'input_dim':3,'gate_mode':'refined_gap','gate_threshold':.01}))
            torch.save(DecisionAligner(3).state_dict(), profile/'model.pt')
            np.savez(root/'inputs.npz', features=np.zeros((1,3,3)),
                     base_scores=np.array([[.5,.2,.3]]), candidate_ids=np.array([[10,20,30]]))
            np.savez(root/'labels.npz', success=np.array([[False,True,False]]))
            (root/'eval.yaml').write_text('checkpoint: profile\ninputs: inputs.npz\n'
                                         'labels: labels.npz\noutput: evaluated.json\n')
            env = dict(os.environ, DJEPA_PYTHON=sys.executable)
            command = ['bash', str(ROOT/'scripts/evaluate.sh'), '--config', 'eval.yaml']
            run = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((root/'evaluated.json').read_text())
            self.assertEqual(report['selected_ids'], [20])
            self.assertEqual(report['success_count'], 1)
            self.assertEqual(report['full_population_success_rate'], 1.)
            self.assertNotEqual(subprocess.run(command, cwd=root, env=env, capture_output=True).returncode, 0)

    def test_new_package_keeps_old_checkpoint_class_identity(self):
        self.assertIsNotNone(importlib.util.find_spec('djepa.models'))
        from djepa.models.ordinal import DecisionAligner
        from djepa.ordinal import DecisionAligner as Legacy
        self.assertIs(DecisionAligner, Legacy)

    def test_yaml_drives_training_and_cli_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split, ids in [('train', [1, 2]), ('calibration', [3, 4])]:
                p = root / split
                p.mkdir()
                np.savez(p/'inputs.npz', features=np.zeros((2, 3, 3), dtype=np.float32),
                         base_scores=np.array([[.1, .2, .3]]*2),
                         candidate_ids=np.array([[10, 20, 30]]*2), start_ids=ids)
                np.savez(p/'labels.npz', success=np.array([[False, True, False]]*2),
                         state_distance=np.array([[2., 1., 3.]]*2))
            cfg = root/'train.yaml'
            cfg.write_text('train: train\ncalibration: calibration\noutput: trained\n'
                           'epochs: 2\nbatch_size: 1\nlearning_rate: 0.001\nseed: 7\n')
            command = [sys.executable, '-m', 'djepa.train', '--config', str(cfg), '--epochs', '1']
            run = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            resolved = json.loads((root/'trained/resolved_config.json').read_text())
            self.assertEqual(resolved['epochs'], 1)
            self.assertEqual(resolved['batch_size'], 1)
            self.assertEqual(resolved['learning_rate'], .001)
            repeat = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn('exists', repeat.stderr.lower())

    def test_invalid_yaml_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d)/'bad.yaml'
            for body in ['unknown_option: 3\n', 'epochs: true\n', 'epochs: 1\nepochs: 2\n']:
                cfg.write_text(body)
                run = subprocess.run([sys.executable, '-m', 'djepa.train', '--config', str(cfg)],
                                     capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn('configuration', run.stderr.lower())

    def test_result_summary_separates_unlabeled_populations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'a.json').write_text(json.dumps({'mode':'cached-feature checkpoint replay',
                'count':4,'observed_count':3,'unobserved_count':1,'success_count':2,
                'success_rate_on_observed':2/3,'full_population_success_rate':None}))
            (root/'b.json').write_text(json.dumps({'mode':'cached-feature checkpoint replay',
                                                'selected_ids':[10,20]}))
            run = subprocess.run([sys.executable, str(ROOT/'scripts/summarize_results.py'),
                                  str(root/'a.json'), str(root/'b.json'), '--output', str(root/'summary')],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            import csv
            with (root/'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['observed_count'], '3')
            self.assertEqual(rows[0]['full_population_success_rate'], '')
            self.assertEqual(rows[1]['success_count'], '')

    def test_corrupt_download_fails_checksum(self):
        self.assertTrue((ROOT/'scripts/download_artifacts.py').is_file())
        spec = importlib.util.spec_from_file_location('download', ROOT/'scripts/download_artifacts.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'corrupt.pt'
            p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                module.verify_sha256(p, '0'*64)


if __name__ == '__main__':
    unittest.main()
