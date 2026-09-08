#!/usr/bin/env bash
set -euo pipefail
"${DJEPA_PYTHON:-python}" -m djepa.train --help
"${DJEPA_PYTHON:-python}" -m djepa.evaluate --help
"${DJEPA_PYTHON:-python}" -m unittest discover -s tests -v
