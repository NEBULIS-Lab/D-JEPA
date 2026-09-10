#!/usr/bin/env bash
set -euo pipefail
robot_release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$robot_release_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
"${DJEPA_ROBOT_PYTHON:-python}" -m pytest tests -q "$@"
