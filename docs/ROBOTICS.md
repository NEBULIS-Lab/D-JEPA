# Robotic manipulation

The RoboTwin interface lives in `src/djepa/robotics/`, with runnable tools in
`scripts/robotics/`. Physical-robot code is independently installed from
[`real_robot/`](../real_robot/README.md). Neither subsystem changes the shared
D-JEPA model interfaces or installs the other's simulator/controller stack.

## Components

| Component | Implementation | Input → output |
|---|---|---|
| Demonstration decoding | `rlds.py`, `inventory.py` | RoboTwin RLDS → episodes and fixed episode splits |
| Action conventions | `pi05.py`, `simulator.py` | Joint/EE observations → explicitly converted action arrays |
| Scene-conditioned proposals | `grasp_geometry.py`, `build_candidates.py` | Current scene + training demonstrations → native and warped EE futures |
| Preservation calibration | `calibrate_gate.py` | Paired development outcomes → distance or grasp-timing threshold |
| Pre-execution selection | `select_candidates.py` | Candidate actions + predicted geometry + gate → selected candidate ID |
| Matched execution | `evaluate_candidates.py` | Shared simulator snapshot + candidate pool → per-candidate outcomes |
| Result aggregation | `evaluate_gate.py` | Previously saved selections + paired outcomes → aggregate metrics |

The geometry proposal implementation is task-specific to bimanual grasping.
The RLDS inventory supports the four tasks listed in `configs/robotics/tasks.json`.
Generic candidate relations remain available through the shared D-JEPA models;
the distance and timing gates here are explicit task-specific readouts, not a
new implementation of those neural models.

## Installation and data

Install the main package with `pip install -e .`. Production RLDS decoding uses
`pip install -e '.[robotics-data]'`; the lightweight geometry and gate tests need
only NumPy and pytest. Use a separate environment for the upstream RoboTwin
simulator and policy, following their own dependency instructions.

The admitted RLDS layout is `<dataset-root>/<dataset-name>/1.0.0/`, containing
`dataset_info.json`, `features.json` and complete TFRecord shards. Each task's
50 demonstrations are split deterministically into 35 training, 5 calibration
and 10 offline-validation episodes. These are demonstration splits, not the
number of simulator evaluation starts.

Each decoded step contains four 240×320 RGB views, a 14-dimensional joint
state/action, a 20-dimensional end-effector state/action, language and terminal
flags. The EE execution representation has 16 entries: left position (3),
quaternion (4), gripper (1), then the same fields for the right arm.

```bash
python scripts/robotics/inventory.py --dataset-root data/robotwin \
  --tasks-config configs/robotics/tasks.json --output runs/robotics/inventory
python scripts/run_task.py --config configs/robotics/geometry.yaml \
  --native-rollout data/robotwin/native-rollout --seed 100056
```

The native-rollout bundle supplies `metrics.json`, `frames.npz` and
`completion.json`. Its metrics identify task and simulator seed; the arrays
contain the recorded native actions and frames. The proposal builder reads
the initial frame for scene geometry and the action sequence for the native
alternative. It does not use rollout outcome labels to select a proposal.

`build_candidates.py` fits ridge geometry from the training demonstrations,
selects regularization on calibration demonstrations, retrieves a training
trajectory and warps its positions toward the predicted bimanual grasp target.
It retains the native action sequence and creates sixteen spatial variants.
Outputs include `candidates.npz`, `model.npz`, `metrics.json`, and a completion
marker. Existing output directories are not overwritten.

## Gate calibration and label-free selection

```bash
python scripts/run_task.py --config configs/robotics/gate.yaml \
  --seed-dir data/robotwin/calibration/scene-a data/robotwin/calibration/scene-b
python scripts/robotics/select_candidates.py \
  --model runs/robotics/gate/model.json --candidates runs/robotics/candidates \
  --repo . --output runs/robotics/selection
```

Each calibration scene contains `full_candidates_v2/` and
`selected_outcomes_v2/` bundles, the existing on-disk format retained for
compatibility. Candidate indices in the paired outcomes are 0 (native) and 1
(the fixed corrected candidate), with original-pool IDs recorded separately.
The correction's original candidate index is 2. `candidate_subset.py` creates
such subsets without losing original identities.

The default recipe calibrates `closest_closed_step_fraction`: the time fraction
of the closest synchronized closed-gripper approach to the predicted target.
Correction is chosen only below the fitted threshold. `alignment_cost_m` is
also available with the opposite threshold direction. Selection reads the
geometry and actions, not the paired outcomes. `evaluate_gate.py` joins saved
selections to outcomes afterward. Its legacy bundle names are storage keys,
not names of methods or experimental stages.

## Simulator execution

`evaluate_candidates.py` is the existing complete-trajectory branch runner.
It requires RoboTwin plus an externally supplied CoWAM-compatible environment
adapter (`cowam.environments.RoboTwinEEAdapter` and `load_robotwin_task`), supplied
through `--robotwin-root` and `--cowam-root`. It restores one snapshot for every
candidate and records strict success, progress, first-success step, executed
length and execution errors. This runner's progress geometry is specific to
`grab_roller`; do not silently apply it to a different task.

```bash
python scripts/robotics/evaluate_candidates.py --help
python -m pytest tests/robotics -q
```

Upstream policy weights, datasets, simulator assets and environment-adapter
source are external dependencies, not vendored here. Device-specific physical
robot control remains outside this simulation interface.
