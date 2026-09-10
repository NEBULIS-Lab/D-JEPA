# Robotics data and interfaces

All NPZ inputs use numeric or fixed-width string arrays and load with
`allow_pickle=False`. Keep measured outcomes separate from inference inputs.

## Candidate feature cache

Let B be the number of decision contexts, N the candidate count, G the number
of predictive geometries, and D the feature dimension.

| Field | Shape | Meaning |
|---|---|---|
| `terminal` | `[B,N,G,D]` float | Candidate terminal future descriptors |
| `goals` | `[B,G,D]` float | Goal descriptors in the same feature space |
| `candidate_ids` | `[B,N]` signed integer | Unique literal IDs per context |
| `native_costs` | `[B,N,G]` float | Frozen predictor costs, lower is better |
| `feature_spec` | scalar string | Exact descriptor/preprocessing identity |
| `source_model` | scalar string | Exact source-model identity |

The same fields suffice for `score_candidates`. The supplied native planning
pipeline constructs 2×2 spatially pooled features (`spatial_pool_2x2`); its
feature cache and trained head must use this matching descriptor.

Training additionally requires:

| Field | Shape | Meaning |
|---|---|---|
| `true_costs` | `[B,N]` | Measured physical costs; lower is better |
| `observed` | `[B,N]` boolean | All entries must be observed in this trainer |
| `provenance` | `[B,N]` string | `executed_real` or `executed_sim` |
| `evidence_ids` | `[B,N]` string | Nonempty references to executed candidate evidence |
| `start_ids` | `[B]` string | Context identity used to reject split overlap |
| `split` | `[B]` string | Both `train` and `val`; no formal/test rows |

Unexecuted candidates are unknown, not failures. Build complete observed
training pools; the present robotics trainer does not accept a sparse mask.
Split recordings by complete episode before generating overlapping windows,
and retain the episode-to-context mapping outside the cache. The loader checks
start-ID overlap, not an unstated episode hierarchy.

## Planning cache

| Field | Shape | Meaning |
|---|---|---|
| `context` | `[1,P,D]` | Current visual tokens |
| `goal` | `[1,P,D]` | Goal tokens |
| `pose` | `[1,7]` | xyz in metres, extrinsic xyz angles in radians, normalized gripper closedness |
| `dt` | scalar | Physical interval in seconds; must match configured model interval |
| `action_semantics` | scalar string | `eef_delta_xyz_extrinsic_xyz_gripper` |
| `source_model` | scalar string | Resolved path of the explicitly supplied AC checkpoint |
| `data_origin` | scalar string | `recorded_real` or `recorded_sim` |

Optional external proposals require **both** `actions` `[N,H,7]` and
`candidate_ids` `[N]`. Actions use Cartesian translation increments, composed
rotation increments and gripper-closedness increments; H must equal the
configuration horizon. Raw joint targets cannot be passed as Cartesian actions.

The planning command saves `plan.json` and `candidate_pool.npz` with the
literal pool and selected action sequence. These are prediction-space outputs,
not commands addressed to a physical robot. Source-model paths are runtime
identities written on the user's machine; no author machine paths are required.

## Recorded HDF5 adapter

The supplied reader accepts the following six-joint recording schema:

| Dataset | Shape / type |
|---|---|
| `left_arm/joint` | `[T,6]` float, recorded joint feedback |
| `left_arm/gripper` | `[T]` or `[T,1]` float |
| `left_arm/action` | `[T,7]` float, commanded joint targets and gripper |
| `left_arm/timestamp` | `[T]` integer nanoseconds |
| `cam_head/color`, `cam_wrist/color` | `[T,H,W,3]` uint8 RGB |
| `cam_head/timestamp`, `cam_wrist/timestamp` | `[T]` integer nanoseconds |

The full recording audit reads both cameras; selective RGB preparation reads
the requested camera. Preserve commands separately from feedback and retain
actual timestamps rather than replacing them with nominal frame indices.

`audit_data` expects `raw_hdf5/<task>/*.hdf5` plus
`lerobot/piper_d455_<task>/meta/info.json` under `--root`. This adapter's original
directory naming is a data-format convention, not a dependency on an actuator
SDK. For a different recording layout use `djepa_robot.data.read_episode` /
`audit_dataset` directly with converted inputs or supply a matching reader.

## Optional nominal motion diagnostic

```bash
python -m scripts.audit_motion \
  --config configs/robot.local.json \
  --raw-dir data/raw_hdf5/task --output outputs/motion
```

The user-supplied JSON defines `urdf`, `base`, `tip`, ordered `joint_names`,
`gripper_record_to_m`, `gripper_closed_m`, `gripper_open_m`,
`window_min_dt_s`, `window_max_dt_s` and `calibration_status`. Six recorded joint
columns must already use the units and ordering expected by that URDF.
No private robot description or measured calibration values are supplied.

The converter reports nominal flange motion, commanded targets, measured
transitions and timing-valid windows; it does not certify TCP calibration.
`evaluate_recorded_motion` can compare these arrays against visual caches with
matching timestamps and checkpoint identity using `--allow-nominal-flange`.
Observed next-state deltas are retrospective diagnostics, not deployable action
proposals or additional supervised physical trials.
