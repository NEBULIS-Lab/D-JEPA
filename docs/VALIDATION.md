# Release validation

See `VALIDATION.json` for machine-readable checks on the current exported files.

The configuration/script and static-site extension is recorded separately in
[WORKFLOW_VALIDATION.json](WORKFLOW_VALIDATION.json): 20 CPU tests, 12 preserved
scientific implementation hashes, 41 installed-wheel module imports and a fresh
256/256 selected-ID agreement check. Historical checks below retain
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

This preparation does not rerun simulators or certify bitwise training
reproduction. Full upstream backbone rollouts in a fresh environment remain a
separate release-readiness check; the lightweight cached-data code does not
depend on those environments. No new experiment results are claimed.
