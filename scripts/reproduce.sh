#!/usr/bin/env bash
# Cached-feature replay only. Supply evaluation YAMLs, or use the default PushT replay.
set -euo pipefail
if [[ $# -eq 0 ]]; then
  set -- configs/evaluation/pusht-independent.yaml
fi
for config in "$@"; do
  "${DJEPA_PYTHON:-python}" -m djepa.evaluate --config "$config"
done
