# Native reproduction

The native path rebuilds D-JEPA predictions from RGB observations and raw PushT
controls, then executes the selected literal candidate in the original PushT
physics. It is distinct from the lightweight cached-feature evaluator.

## Capability boundary

| Task | Predictor features from raw observations | Native physics rollout | Status |
|---|---:|---:|---|
| PushT fixed candidate protocol | Yes | Yes | Implemented here for 63 deployable candidates and exactly 25 controls |
| PushT visual shifts | No | No | Released features support cached evaluation only; the shift renderer/runtime is not packaged here |
| PushObj unseen geometry | No | No | Released supervision and heads do not include a portable native simulator adapter |
| DMC-Reacher | No | No | Full feature-level weights are released, but the recorded native observation/action preparation is not packaged here |
| Granular manipulation | No | No | Released inputs support sparse-supervision evaluation; native simulator setup is not packaged here |
| Real robot | Separate package | Not applicable | See `../real_robot/README.md` for offline RGB encoding, learning and planning; device control is user-supplied |

The tests in `tests/test_native.py` are CPU **contract tests**. Their small,
deterministic predictor and environment stand-ins test tensor alignment,
literal-candidate identity, float-before-half score ordering, and the complete
25-control execution loop. They are not evidence that the external upstream
packages, checkpoint templates, renderer, or physics have run successfully on a
new machine. An actual native run produces metadata with mode
`native_predictor_execution` and `native_physics_selected_rollouts`.

## Frozen PushT contract

`configs/native/pusht-independent.yaml` authenticates the full exact-realization
checkpoint and freezes the independent split, preprocessing, gate, and protocol.
Its checkpoint path is resolved **relative to the native YAML file**; with the
standard downloader destination, it resolves to
`checkpoints/pusht-exact-realization-full`.

The inputs use the original stable-worldmodel PushT action units: every two-vector
is a relative control in `[-1, 1]`; the environment targets
`agent_position + 100 * action`. A candidate contains 25 controls at 10 Hz. Each
control advances ten 0.01-second physics integrations, giving 250 integrations
and 2.5 seconds per fixed-horizon rollout. Termination signals are recorded but
do not shorten this fixed protocol.

Candidate zero (the source reference) is not deployable. Each start must contain
63 unique positive literal `candidate_ids`; selection and physics are joined by
literal ID and by a SHA-256 of each candidate's 25 raw controls, never by an
unverified array position.

## Required raw bundle

`prepare_features.py` reads a safe NPZ (`allow_pickle=False`) with:

| Array | Shape / dtype | Meaning |
|---|---|---|
| `start_ids` | `(B,) int64` | Unique literal branch/start IDs |
| `split` | `(B,)` fixed-width string | Every value is `independent_256` |
| `candidate_ids` | `(B,63) int64` | Positive, unique literal candidate IDs |
| `candidate_actions` | `(B,63,25,2)` floating | Original raw relative controls |
| `history_actions` | `(B,10,2)` floating | Ten preceding raw controls |
| `context_pixels` | `(B,3,H,W,3) uint8` | Frames at five-control spacing |
| `goal_pixels` | `(B,H,W,3) uint8` | Goal observation used by both encoders |

`evaluate_rollouts.py` additionally requires `initial_state` and `goal_state`,
both `(B,7)` floating arrays, in the recorded PushT state convention.

The released supervision archive does **not** itself contain one NPZ with this
complete native schema. In particular, the independent inputs contain context
pixels and history but not raw initial/goal states or goal pixels; development
contains states but not the raw predictor observation bundle. Joining the
runtime-action release does not fill those missing fields. Supply them from the
upstream source records or a newly collected, explicitly identified native
population. Do not relabel cached futures as rebuilt native features.

## Recorded upstream runtime

Install the upstream dependencies recorded in `docs/THIRD_PARTY.md`. The loader
is concrete: `djepa.native.load_recorded_pusht_model` imports
`stable_worldmodel.wm.utils.load_pretrained`, instantiates the TD-JEPA template,
loads the trusted serialized LeWM template, verifies both template hashes against
the full profile's lineage, and then strict-loads every tensor from
`pusht-exact-realization-full`. The exact runtime commonly needs explicit Python
roots for these source trees:

```text
TD-JEPA/
stable-worldmodel/
stable-pretraining/
le-wm/
```

After downloading the full profile and preparing `raw-independent.npz`, run:

```bash
python scripts/prepare_features.py \
  --config configs/native/pusht-independent.yaml \
  --inputs raw-independent.npz \
  --output outputs/native/native-features.npz \
  --tdjepa-template /path/to/recorded/weights_epoch_10.pt \
  --lewm-template /path/to/recorded/lewm_object.ckpt \
  --python-root /path/to/TD-JEPA \
  --python-root /path/to/stable-worldmodel \
  --python-root /path/to/stable-pretraining \
  --python-root /path/to/le-wm \
  --device cpu
```

The output retains both predicted five-block futures and goal latents, native
costs, 386-dimensional relational features, all gate scores, lifted futures,
literal selected IDs, and candidate-action hashes. It contains no outcome label.

Run the corresponding physical trajectories with the same raw bundle:

```bash
python scripts/evaluate_rollouts.py \
  --config configs/native/pusht-independent.yaml \
  --inputs raw-independent.npz \
  --features outputs/native/native-features.npz \
  --output outputs/native/native-rollouts.npz \
  --python-root /path/to/stable-worldmodel
```

`native-rollouts.npz` stores the initial state plus all 25 post-control states,
the exact chosen controls, rewards, termination/truncation flags, final physical
success, and distance. Add `--start-id ID` to select literal starts.

For 25-frame, 10-fps MP4s, install the optional dependencies with
`pip install -e '.[media]'`, then add `--video-dir outputs/native/videos`.
Video files and their SHA-256/frame-count receipts are
listed in the rollout JSON sidecar. Outputs are never overwritten.
