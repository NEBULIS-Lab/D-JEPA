#!/usr/bin/env bash
set -euo pipefail
# Local-only preview; no external deployment or DNS change.
exec "${DJEPA_PYTHON:-python}" -m http.server "${1:-8000}" --bind 127.0.0.1 --directory docs
