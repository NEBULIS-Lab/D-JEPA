#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${DJEPA_PYTHON:-python}" scripts/reproduce_paper.py "$@"
