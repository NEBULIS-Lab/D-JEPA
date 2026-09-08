# Release validation

See `VALIDATION.json` for machine-readable checks on the current exported files.

The configuration/script and static-site extension is recorded separately in
[WORKFLOW_VALIDATION.json](WORKFLOW_VALIDATION.json): 39 CPU tests, 12 preserved
scientific implementation hashes, 49 installed-wheel module imports, 12 validated
dataset splits and 47 verified manifest files. The four-run cached matrix passes,
including exact PushT and Granular selected-ID agreement. Historical checks below retain
their original validation scope; the new checks do not represent new experiments.

- CPU unit tests cover ranking ties, strict and task-specific gates, bounded
  corrections, permutation equivariance, training gradients, sparse-supervision
  reporting, checkpoint loading, terminal realization and calibrated composition.
- Public lightweight relation/transport checkpoints strict-load into their models.
- PushT relational decisions match **all 256/256** independent recorded choices.
- Exact realization matches those **256/256** choices and preserves the earlier
  four future steps.
- Granular relational decisions match **all 64/64** formal recorded choices.
- Task-local checkpoints match **256/256 PushObj calibration choices** and
  **441/441 visual-shift calibration choices**. These are calibration replays,
  not new independent benchmark results.

Three new module recipes were also run for one update on the actual released
training data, then strict-loaded and used for inference. This validates the
training/export interface, not bitwise historical retraining or a new result.

The [native interface](NATIVE_REPRODUCING.md) has six CPU contract tests plus
workflow checks for input provenance and output protection. These use deterministic
stand-ins rather than executing external upstream predictors or simulators.
Full upstream execution requires the documented runtime and raw-input bundle.
The lightweight cached-data code does not depend on those environments.
No new experiment results are claimed.
