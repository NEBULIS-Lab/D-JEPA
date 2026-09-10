"""Public offline entry points must resolve without the simulation package."""

from pathlib import Path
import os
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicEntrypointTests(unittest.TestCase):
    def test_training_entrypoint_runs_from_robot_directory(self):
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-m", "scripts.train_alignment", "--help"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cache", result.stdout)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
