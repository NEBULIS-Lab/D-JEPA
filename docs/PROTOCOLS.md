# Evaluation populations and supervision schema

The released data contain complete candidate **inputs** for each included set.
Observed supervision is dense except Granular, where a Boolean supervision mask
is required. This is a decision-supervision release, not an archive of every
original video frame or every exploratory experiment.

Each split contains `inputs.npz`, optional `labels.npz`, and `metadata.json`.
NPZ arrays are numeric, Boolean or fixed-width strings; use `allow_pickle=False`.

- `candidate_ids`: original literal IDs, unique within a set; never infer identity
  from array position. Reference actions are excluded from deployable sets.
- `features`, `base_scores`: inference-ready relation inputs where exported.
- `candidate_actions`: task-specific actions; do not mix their units or horizons.
- `start_ids`, `episode_ids` or `source_identity_sha256`: original identity fields.
- `success`: observed candidate outcomes; never pass this into inference.
- `supervision_mask`: only `True` entries were executed/labeled. `False` means
  unknown, **not failure**.
- `base_ids` and `conditions`: identify repeated corruptions of the same start.

## Populations

| Task / split | Size | Interpretation |
|---|---:|---|
| PushT fitting pool | 512 | Relational fit/calibration = 448/64; plasticity = 384/128; exact module-specific ID lists in metadata |
| PushT development | 128 | Development/mechanism results, including multi-geometry |
| PushT independent confirmation | 256 | Frozen decision-boundary population; separate from fitting/development |
| Reacher train / calibration | 384 / 128 | Full predicted context/future/goal arrays plus dense candidate supervision |
| Reacher formal results | 128 | Recorded independent outcomes; not the 128 calibration starts |
| Granular train / calibration | 32 / 16 | Sparse candidate supervision with explicit masks |
| Granular formal inputs/results | 64 | Full inputs; only selected-action outcomes reported separately |
| PushObj train / calibration | 512 / 256 | Shapes T, L, Z and + |
| PushObj independent results | 300 | 100 each unseen I, small T and square; confirmation seeds 200 and 300 |
| PushT visual train / calibration | 126×7 / 63×7 | Seven known conditions on disjoint base starts |
| PushT visual confirmation results | 50×7 | 50 independent new base starts, not 350 independent starts |

The fixed PushT/PushObj action-selection evaluations execute 25 controls,
250 simulator integration steps, 2.5 seconds. They are not arbitrary-length
closed-loop episodes. Reacher and Granular keep their own task-specific horizons
and success metrics. The 63-candidate budget is a reported evaluation choice,
not a universal model requirement.

The main PushT independent archive retains both `local_candidate_positions` from
the execution input release and original `candidate_ids`. Its `start_ids` are
branch IDs and `episode_ids` are episode IDs. These distinctions are deliberate.

## Splits and reporting

Never train on `development`, `independent_256`, `formal`, or recorded result
files. For PushT's shared fitting pool, use the module-specific calibration ID
list before constructing your training loader. The fusion scalar has its own
recorded fitting protocol; do not reinterpret its provenance as independent test
performance. For visual shifts, split and bootstrap at the **base start** level.

Cached inference reproduces checkpoint decisions on the released features.
Rebuilding predictive features and executing new simulator trajectories additionally
requires the upstream environments/models and task preprocessing. The lightweight
CLI is not an end-to-end simulator benchmark runner.

Historical method keys in `results/` are immutable source identifiers. The
audience-facing method is D-JEPA; configurations are documented in CHECKPOINTS.md.
