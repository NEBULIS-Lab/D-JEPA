#!/usr/bin/env bash
set -euo pipefail
"${DJEPA_PYTHON:-python}" -m djepa.driving.risk "$@"
