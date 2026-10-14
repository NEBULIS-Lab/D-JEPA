# Release scope

This release consolidates the currently completed D-JEPA study into a stable
code, model and decision-supervision package. Numerical tables and qualitative
media are presented on the [project website](https://nebulis-lab.com/D-JEPA).
Weights and supervision are distributed through their respective Hugging Face
repositories; they are not bundled into Git source history.

## Available workflows

| Workflow | Entry point | Description |
|---|---|---|
| Artifact download | `scripts/download_artifacts.py` | Selected checkpoints and supervision data with automatic integrity checks |
| Installation / data preflight | `scripts/doctor.py`, `scripts/verify_dataset.py` | CPU environment and data-format checks |
| Cached paper matrix | `scripts/reproduce_paper.sh` | Per-task evaluation and resumable runs |
| Task-local ordinal training | `scripts/train.sh` | Dense train/calibration split recipes |
| Relational / spatial training | `scripts/train_modules.sh` | Released-data recipes and module-specific calibration identities |
| Native features and rollouts | `scripts/prepare_features.py`, `scripts/evaluate_rollouts.py` | Explicit upstream models/raw states and the declared fixed control horizon |
| Local summaries | `scripts/summarize_results.py` | Per-run metrics; no pooling of different protocols |
| Project page | `docs/index.html` | Method diagrams, recorded demonstrations and results |
| Offline robotics | `real_robot/README.md` | Independently installed recorded-data, training, scoring and planning tools |
| Robotic manipulation | `ROBOTICS.md` | RLDS interfaces, learned geometry candidates, preservation calibration and matched execution |
| Autonomous driving | `DRIVING.md` | Native export, relation/risk calibration, label-free prediction and cached evaluation |

Follow [reproduction](REPRODUCING.md), [module training](MODULE_TRAINING.md) and
[native execution](NATIVE_REPRODUCING.md) for the exact supported inputs and
dependencies. Cached choices, simulator execution and retraining are distinct
workflows, with commands and dependencies described in each guide.

## Task-specific setup

The offline robotics package lives in `real_robot/`. Robotic manipulation and
autonomous driving have dedicated guides and configurations. Install the
upstream environments and provide the inputs listed in the corresponding guide.
Device-specific control and calibration interfaces remain separate from the
public offline workflows.
