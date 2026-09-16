# Release scope

This release consolidates the currently completed D-JEPA study into a stable
code, model and decision-supervision package. Numerical tables and qualitative
media are presented on the [project website](https://nebulis-lab.com/D-JEPA).
Weights and supervision are distributed through their respective Hugging Face
repositories; they are not bundled into Git source history.

## Available workflows

| Workflow | Entry point | Contract |
|---|---|---|
| Artifact download | `scripts/download_artifacts.py` | Requested profiles, immutable Hub revision receipts and checksums |
| Installation / data preflight | `scripts/doctor.py`, `scripts/verify_dataset.py` | CPU checks, schema and optional manifest validation |
| Cached paper matrix | `scripts/reproduce_paper.sh` | Distinct protocol identities, exact reference checks where available, safe resume |
| Task-local ordinal training | `scripts/train.sh` | Dense train/calibration split recipes |
| Relational / spatial training | `scripts/train_modules.sh` | Released-data recipes and module-specific calibration identities |
| Native features and rollouts | `scripts/prepare_features.py`, `scripts/evaluate_rollouts.py` | Explicit upstream models/raw states and the declared fixed control horizon |
| Local summaries | `scripts/summarize_results.py` | Per-run metrics; no pooling of different protocols |
| Project page | `docs/index.html` | Existing approved media and author-artwork placeholders |
| Offline robotics | `real_robot/README.md` | Independently installed recorded-data, training, scoring and planning tools |
| Robotic manipulation | `ROBOTICS.md` | RLDS interfaces, learned geometry candidates, preservation calibration and matched execution |
| Autonomous driving | `DRIVING.md` | Native export, relation/risk calibration, label-free prediction and cached evaluation |

Follow [reproduction](REPRODUCING.md), [module training](MODULE_TRAINING.md) and
[native execution](NATIVE_REPRODUCING.md) for the exact supported inputs and
dependencies. Cached choices, simulator execution and retraining are distinct
workflows; a successful test of one is not reported as verification of another.

## Next content update

The offline robotics code is now included separately under `real_robot/`.
Device-specific control and calibration interfaces are intentionally not part of
the public release. Publishing this code does not add robot numerical results
or relabel simulation weights as robot checkpoints.

Robotic-manipulation and autonomous-driving interfaces are now included by
application, alongside the shared scientific modules. Environment-specific
dependencies and inputs are listed in their guides. Code integration does not
re-run experiments or alter the numerical authorities. The final paper/video
and author-drawn method figures remain separate content updates.
The Paper and Video badges currently link to explicitly marked placeholders.
Existing numerical authorities, checkpoint identities and evaluation splits
remain unchanged in this release.
