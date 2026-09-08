#!/usr/bin/env bash
set -euo pipefail
exec "${DJEPA_PYTHON:-python}" -m djepa.cli.train_modules "$@"
