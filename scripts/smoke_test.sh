#!/usr/bin/env bash
set -euo pipefail
"${DJEPA_PYTHON:-python}" -m djepa.train --help
"${DJEPA_PYTHON:-python}" -m djepa.evaluate --help
"${DJEPA_PYTHON:-python}" -m djepa.cli.train_modules --help
"${DJEPA_PYTHON:-python}" scripts/doctor.py
"${DJEPA_PYTHON:-python}" scripts/prepare_features.py --help
"${DJEPA_PYTHON:-python}" scripts/evaluate_rollouts.py --help
"${DJEPA_PYTHON:-python}" -m unittest discover -s tests -v
