# Autonomous driving

`src/djepa/driving/` contains Drive-JEPA candidate export, cached-feature
training, calibration, label-free selection and paired evaluation. The native
predictive model and trajectory generator are unchanged. Both score correction
and anchor-relative risk correction are configurations of D-JEPA.

## Entry points

| Operation | Entry point | Dependency |
|---|---|---|
| Check resources / construct splits / export candidates | `python -m djepa.driving.native` | Drive-JEPA's `navsim_v1`, maps, sensors and model assets |
| Train relation, independent MLP and score-fusion readouts | `scripts/driving/train.sh` | Prediction/label caches, NumPy and PyTorch |
| Calibrate relative correction and risk | `scripts/driving/calibrate_risk.sh` | Fit/calibration caches grouped by source log |
| Score one scene without labels | `python -m djepa.driving.score` | Prediction NPZ and trusted model bundle |
| Evaluate cached selections | `scripts/driving/evaluate.sh` | Test predictions and labels |

Cached workflows default to CPU; `--device cuda` selects CUDA explicitly.
Native perception export uses the upstream CUDA runtime. Model files are loaded
using the original tensor-state layout. Original bundles include NumPy statistics;
load only trusted model files because their format uses Python serialization.

## Prediction and label formats

Each scene has 32 ordered candidate IDs and these prediction arrays:

| Field | Shape | Meaning |
|---|---|---|
| `candidate_id` | `[32]` | Original IDs, in exported order |
| `trajectory` | `[32,8,3]` | Ego-frame x/y/heading; four-second horizon |
| `query_feature` | `[32,256]` | Proposal query summary captured at the native scorer |
| `native_score` | `[32]` | Native predicted total score |
| `predicted_factors` | `[32,6]` | Native auxiliary prediction channels |

Score-correction inputs concatenate query, six channels, native score, tie-aware
rank and flattened trajectory (288 dimensions). Relative/risk correction omits
the six auxiliary channels (282 dimensions). Risk outputs are learned from
official collision and TTC labels; the auxiliary channels are not treated as
calibrated safety probabilities.

The cache layout under each of `fit/`, `calibration/`, and `test/` is:

```text
summary.json                 state, role and completed scene count
manifest.csv                full log-disjoint split manifest
processed_manifest.csv      rows for this role: token, log_name, role
config.json                 common generation/scoring configuration
provenance.json             source and native-model identity
predictions/<token>.npz      prediction arrays above
labels/<token>.json          candidate IDs, official scores and subscores
```

Label files bind the prediction file hash and contain `token`, `role`, and
`labels`. Each label has `candidate_id`, `score`, `no_at_fault_collisions`,
`time_to_collision_within_bound`, and other official subscores. Observed outcomes
are not input features. The label-free scoring command does not open this folder.

## Generate caches

Copy and edit `configs/driving/native.example.json`. Paths are relative to the
working directory. The assets directory must retain the upstream basename
`Drive-JEPA-cache`; install Drive-JEPA's own NAVSIM v1 dependencies separately.

```bash
python -m djepa.driving.native --stage doctor --config configs/driving/native.example.json
python -m djepa.driving.native --stage inventory --config configs/driving/native.example.json \
  --output runs/driving/inventory
bash scripts/driving/export_candidates.sh --config configs/driving/native.example.json \
  --role fit --output data/driving/fit
```

Set `manifest` in the configuration to the generated `scenes.csv`, then export
each role into a new directory. The exporter saves actual official-score
simulator trajectories separately from logged camera observations.

## Train, calibrate and evaluate

```bash
python scripts/run_task.py --config configs/driving/relational.yaml
python scripts/run_task.py --config configs/driving/evaluate.yaml
python scripts/run_task.py --config configs/driving/risk.yaml
python -m djepa.driving.score --checkpoint runs/driving/relational/alignment.pt \
  --prediction data/driving/test/predictions/SCENE_TOKEN.npz \
  --output runs/driving/scene-selection.json
```

Use `--dry-run` on the recipe runner to inspect the command without executing
it; extra CLI arguments override YAML values. Relative/risk calibration produces
`boundary_only.pt` and `risk_relative.pt`, which the same label-free scorer accepts.
The two files represent disabling/enabling the risk objective, not new method names.

The relation recipe uses a 64-dimensional projection, two four-head Transformer
layers, a zero-initialized bounded correction, and training-only normalization.
The risk recipe uses leave-one-source-log-out calibration, then refits on the
development logs. Its existing runner also produces a diagnostic replay after
calibration is locked. Log segments from the same source remain in one group.
Output directories must be new. Reports contain executed candidate IDs, paired
differences and protocol metadata; PDMS is reported on a 0–100 scale, not as
binary task success. Evaluation summarizes the cache supplied to the command,
without hard-coded paper target values or hand-selected scene replacements.

```bash
python -m pytest tests/driving -q
```

Tests use synthetic CPU tensors to check interfaces, native preservation,
permutation behavior, correction bounds, risk vetoes and split isolation. They
are not training runs or new driving measurements.
