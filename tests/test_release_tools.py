"""Release preflight and resume protections, using isolated test artifacts."""
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


class ReleaseToolTests(unittest.TestCase):
    def test_native_receipt_binds_raw_goals_and_checkpoint(self):
        spec = importlib.util.spec_from_file_location('native_rollouts', ROOT/'scripts/evaluate_rollouts.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from djepa.data.validation import sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, raw = root/'features.npz', root/'raw.npz'
            features.write_bytes(b'prepared features')
            raw.write_bytes(b'original raw goals and controls')
            config = {'evaluation_split': 'independent_256', 'checkpoint': {'weights_sha256': 'recorded'}}
            receipt = {'schema': 'djepa_native_pusht_features_v1', 'mode': 'native_predictor_execution',
                       'output_sha256': sha256(features), 'source_inputs_sha256': sha256(raw), **config}
            features.with_suffix('.npz.json').write_text(json.dumps(receipt))
            module._validate_feature_receipt(features, raw, config)
            with self.assertRaises(ValueError):
                module._validate_feature_receipt(features, raw, {**config, 'checkpoint': {'weights_sha256': 'different'}})
            raw.write_bytes(b'different raw goal with same candidate controls')
            with self.assertRaisesRegex(ValueError, 'different raw inputs'):
                module._validate_feature_receipt(features, raw, config)

    def test_native_output_collision_precedes_upstream_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'existing.npz'
            output.write_bytes(b'keep')
            for script, extra in [('prepare_features.py', ['--tdjepa-template','missing','--lewm-template','missing']),
                                  ('evaluate_rollouts.py', ['--features','missing'])]:
                command = [sys.executable, str(ROOT/'scripts'/script), '--config','missing',
                           '--inputs','missing','--output',str(output)] + extra
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn('refusing to replace', run.stderr)
                self.assertEqual(output.read_bytes(), b'keep')

    def test_verified_copy_preserves_existing_destination(self):
        spec = importlib.util.spec_from_file_location('download_artifacts', ROOT/'scripts/download_artifacts.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root/'source', root/'destination'
            source.write_bytes(b'complete download')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            module.copy_verified(source, destination, digest)
            module.copy_verified(source, destination, digest)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            destination.write_bytes(b'keep user file')
            with self.assertRaises(ValueError): module.copy_verified(source, destination, digest)
            self.assertEqual(destination.read_bytes(), b'keep user file')

    def split(self, root):
        root.mkdir(parents=True)
        np.savez(root/'inputs.npz', features=np.zeros((2,3,3),np.float32),
                 candidate_ids=np.array([[10,20,30],[10,20,30]]),
                 base_scores=np.array([[.5,.2,.3]]*2),start_ids=np.array([1,2]))
        np.savez(root/'labels.npz',success=np.array([[False,True,False]]*2))

    def test_validator_rejects_duplicate_candidate_ids(self):
        self.assertIsNotNone(importlib.util.find_spec('djepa.data.validation'))
        from djepa.data.validation import validate_split
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'train';self.split(root)
            report=validate_split(root)
            self.assertEqual(report['count'],2)
            with np.load(root/'inputs.npz') as data: arrays=dict(data)
            arrays['candidate_ids'][0]=[10,10,30]
            np.savez(root/'inputs.npz',**arrays)
            with self.assertRaises(ValueError): validate_split(root)

    def test_validator_rejects_sparse_labels_without_boolean_mask(self):
        self.assertIsNotNone(importlib.util.find_spec('djepa.data.validation'))
        from djepa.data.validation import validate_split
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'train';self.split(root)
            np.savez(root/'labels.npz',success=np.zeros((2,3),bool),supervision_mask=np.ones((2,3)))
            with self.assertRaises(ValueError): validate_split(root)

    def test_checkpoint_dimension_mismatch_is_preflight_error(self):
        self.assertIsNotNone(importlib.util.find_spec('djepa.data.validation'))
        from djepa.data.validation import validate_split
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.split(root/'split')
            profile=root/'profile';profile.mkdir()
            (profile/'config.json').write_text(json.dumps({'architecture':'set_aligner','input_dim':7}))
            with self.assertRaises(ValueError): validate_split(root/'split',profile)

    def test_doctor_missing_requested_profile_is_nonzero(self):
        run=subprocess.run([sys.executable,str(ROOT/'scripts/doctor.py'),'--checkpoint','/nonexistent/djepa/profile'],capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0)
        self.assertIn('checkpoint',run.stdout)

    def test_matrix_replay_resumes_and_detects_changed_output(self):
        from djepa.ordinal import DecisionAligner
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.split(root/'data/toy/formal')
            profile=root/'checkpoints/toy';profile.mkdir(parents=True)
            (profile/'config.json').write_text(json.dumps({'architecture':'set_aligner','input_dim':3,
                'gate_mode':'refined_gap','gate_threshold':.01}))
            torch.save(DecisionAligner(3).state_dict(),profile/'model.pt')
            (root/'matrix.yaml').write_text('schema: 1\nruns:\n  - id: toy\n    task: toy\n    split: formal\n    profile: toy\n    labels: true\n')
            command=[sys.executable,str(ROOT/'scripts/reproduce_paper.py'),'--matrix',str(root/'matrix.yaml'),
                '--data-root',str(root/'data'),'--checkpoint-root',str(root/'checkpoints'),'--output',str(root/'outputs')]
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads((root/'outputs/toy/report.json').read_text())
            self.assertEqual(report['selected_ids'],[20,20])
            resume=subprocess.run(command+['--resume'],capture_output=True,text=True)
            self.assertEqual(resume.returncode,0,resume.stderr)
            (root/'outputs/toy/report.json').write_text('{}')
            corrupt=subprocess.run(command+['--resume'],capture_output=True,text=True)
            self.assertNotEqual(corrupt.returncode,0)
            self.assertIn('changed',corrupt.stderr.lower())


if __name__=='__main__': unittest.main()
