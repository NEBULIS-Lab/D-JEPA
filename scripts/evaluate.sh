#!/usr/bin/env bash
set -euo pipefail
exec "${DJEPA_PYTHON:-python}" -m djepa.evaluate "$@"
