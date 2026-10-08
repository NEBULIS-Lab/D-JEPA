# Where candidate actions come from

D-JEPA evaluates a supplied set of action sequences. The proposal stage constructs
that set for the current context; predictive models evaluate the futures, and
decision alignment chooses the action. The proposal sources below implement the
paper's unified candidate-construction interface (Appendix A.1).

## Sources and entry points

| Task | Proposal source and selectable set | Public entry point |
|---|---|---|
| PushT confirmation | A recorded reference segment plus Gaussian perturbations, block permutations, smooth random and uniform random controls. Generate 256; native TD-JEPA cost retains 64 including the reference; exclude reference ID 0 to obtain 63. | `scripts/generate_candidates.py`, `pusht-confirmation` profile and `retain` command |
| Reacher; Cube task coverage | Reference-centred sampling with 24 Gaussian perturbations, seven block permutations, 16 smooth random and 16 uniform random sequences; exclude reference ID 0. | Same script, `reference-control` profile |
| PushObj unseen shapes; PushT appearance | The same 64-member family mixture, using official pixel-displacement segments, normalized by 100 and quantized as recorded; exclude reference ID 0. | Same script, `visual-shape` profile |
| Granular | A five-control replay window: 35 scale/rotation variants, eight fractional temporal shifts, 16 endpoint offsets and four identity-seeded mixed variants. All 63 are selectable, including replay anchor ID 1. | Same script, `granular` profile |
| RoboTwin | Native VLA actions plus scene-conditioned end-effector trajectory proposals; the included grasp builder produces 17 candidates. | [`scripts/robotics/build_candidates.py`](../scripts/robotics/build_candidates.py), [robotics guide](ROBOTICS.md) |
| Physical robot | Native-cost CEM searches bounded action sequences; the resulting shared pool is scored before sending the selected ID to the device controller. | [`real_robot/djepa_robot/planning.py`](../real_robot/djepa_robot/planning.py), [`real_robot/scripts/plan_latents.py`](../real_robot/scripts/plan_latents.py) |
| Autonomous driving | The 32 native Drive-JEPA trajectory proposals, each containing eight ego-frame poses over four seconds. | [`scripts/driving/export_candidates.sh`](../scripts/driving/export_candidates.sh), [driving guide](DRIVING.md) |

The robotics builder contains the released scene-conditioned grasp interface;
the driving exporter calls the upstream Drive-JEPA model. Follow the linked
guides for their model/environment dependencies. Device-specific robot drivers
remain separate from the offline planning package.

## Quick CPU-only example

From the repository root, after installing the package:

```bash
pip install -e '.[dev]'
python examples/generate_candidates.py --output outputs/candidate-example
python -m unittest discover -s tests -p test_candidate_generation.py -v
```

The example creates synthetic reference arrays and exercises all four profiles,
including PushT retention. Its illustrative squared-control costs demonstrate
the file interface only; experimental retention uses actual native predictive
costs. No GPU, model download or simulator is needed for candidate construction.

## PushT: generate, predict, retain

Supply one reference action segment as an NPY array of shape `[25,2]`, in the
original normalized relative-control units `[-1,1]`. In the reported protocol,
this is the recorded segment associated with the restored start and goal. It is
the centre of the perturbation family and is removed from the selectable set.

```bash
python scripts/generate_candidates.py generate \
  --config configs/candidates/pusht-confirmation.yaml \
  --reference-actions data/my-start/reference_actions.npy \
  --start-id 42 --seed 123 --output outputs/my-start/proposals
```

This produces `candidates.npz` and `metadata.json`. The NPZ holds:

| Field | Shape / meaning |
|---|---|
| `start_ids` | `[1]`, the supplied start identity |
| `candidate_ids` | `[1,K]`, persistent IDs, never reindexed after retention |
| `candidate_actions` | `[1,K,H,D]`, literal controls |
| `candidate_types` | `[1,K]`, proposal families |
| `candidate_action_sha256` | `[1,K]`, per-action hashes of canonical little-endian float32 controls |
| `generator_seed` | `[1]`, when the profile uses a numerical RNG seed |

Run the native TD-JEPA predictor on **all 256 candidates for this same context
and goal**, using the original preprocessing and action history. Save its
lower-is-better native goal costs as follows:

```python
import numpy as np

with np.load('outputs/my-start/proposals/candidates.npz', allow_pickle=False) as pool:
    identity = {key: pool[key] for key in
                ('start_ids', 'candidate_ids', 'candidate_action_sha256')}
    actions = pool['candidate_actions'].astype(np.float32)
    # Your upstream TD-JEPA inference supplies this array, in the original ID order:
    # native_costs = predictor_goal_cost(context, goal, history_actions, actions)
    # Expected shape: [1,256]. These are predictions, not executed outcome costs.
    np.savez_compressed('outputs/my-start/tdjepa-costs.npz',
                        native_costs=native_costs, **identity)
```

The block above is an integration pattern: connect your existing predictor call
where indicated. It does not initialize or replace that predictor. Then run:

```bash
python scripts/generate_candidates.py retain \
  --pool outputs/my-start/proposals/candidates.npz \
  --scores outputs/my-start/tdjepa-costs.npz \
  --output outputs/my-start/selectable
```

Retention keeps the 16 lowest-cost non-reference IDs, then adds 47 more by
round-robin coverage of lexicographically sorted families, ordered within each
family by `(native cost, candidate ID)`. Thus it reproduces the recorded
64-member retained pool with its reference removed. The generator and retention
API take no execution labels; the command also verifies start, candidate and
action-hash correspondence. Evaluation-population eligibility is a separate
protocol step described in the paper, not part of this action generator.

Attach the resulting 63 actions and literal IDs to the same start's context,
goal and history arrays for [`prepare_pusht_features`](../src/djepa/native/pusht.py).
That existing function evaluates the retained candidates with D-JEPA's predictive
paths. The paper-reproduction CLI retains its recorded split checks; use the
Python interface when supplying a new start.

## Other control tasks

For Reacher or Cube, provide the corresponding reference `[25,D]` array in
the task's native normalized control coordinates. `D` follows the environment;
the array's floating-point dtype is preserved. Use the recorded generator seed
when regenerating an existing pool, or an explicitly chosen seed for a new start:

```bash
python scripts/generate_candidates.py generate \
  --config configs/candidates/reference-control.yaml \
  --reference-actions data/my-start/reference_actions.npy \
  --start-id 42 --seed 123 --output outputs/my-start/selectable
```

For PushObj or PushT appearance, provide the official **pixel-displacement**
reference `[25,2]` array; the `visual-shape` profile divides it by 100 internally.
Do not normalize that input a second time. Without `--seed`, the initial seed is
`20260902 + start_id`; the builder records the first collision-free seed used.
For a recorded PushObj segment, supply the same fingerprint-derived integer
start identity used by its data preparation, or pass the recorded initial seed.

```bash
python scripts/generate_candidates.py generate \
  --config configs/candidates/visual-shape.yaml \
  --reference-actions data/my-start/pixel_actions.npy \
  --start-id 42 --output outputs/my-start/selectable
```

Every profile uses Gaussian standard deviations `0.05, 0.15, 0.30`, five
five-control blocks, uniform random controls in `[-1,1]`, and smooth controls
`a[0]=u[0]`, `a[t]=0.8*a[t-1]+0.2*u[t]`. The numerical contracts differ:

| Profile | Precision and uniqueness rule |
|---|---|
| `pusht-confirmation` | Generate 256-member float32 banks at successive seeds, quantize to float16, and refill missing family quotas with byte-unique actions. Preserve encounter-order IDs. |
| `reference-control` | Fill each family in place with byte-unique actions in the reference dtype using one RNG stream. |
| `visual-shape` | Regenerate the whole 64-member bank at successive seeds until all float16-quantized sequences are unique; expose those values as float32. |

Insufficiently distinct reference blocks fail explicitly rather than silently
changing the family mixture. The fixed configurations reproduce the published
construction rules; changing a seed or input constructs a new pool, not the
released evaluation population.

## Granular replay transformations

Supply the original `[20,4]` replay array. Each row encodes start and endpoint
coordinates; `--offset` selects a five-control window. Coordinates are bounded
by `[-4,4]`, not the normalized control range used above.

```bash
python scripts/generate_candidates.py generate \
  --config configs/candidates/granular.yaml \
  --reference-actions data/my-start/replay_actions.npy \
  --start-id scene-42 --goal-id target-42 --offset 2 \
  --output outputs/my-start/selectable
```

The config explicitly selects `standard` difficulty; use the recorded tier
(`conservative`, `standard` or `broad`) for an existing pool. Scale factors are
`1, .70, .85, 1.15, 1.30`; rotations are `0, ±10, ±20, ±30` degrees. Temporal
shifts use circular linear interpolation at `±.25, ±.75, ±1.25, ±1.75` controls.
Endpoint radii are `.15, .30`, each at eight compass directions. Four mixed
variants are deterministically derived from the immutable protocol hash
namespace, start ID, goal ID and variant name. No trial outcomes enter the grid.

## Using a new test set

Build one pool per new context with the appropriate proposal source, evaluate
the predicted futures, and pass the aligned actions/IDs to the selector. For
reference-centred evaluations, the new input must include its reference segment;
the generator does not synthesize that reference from an image or recover it
from outcome labels. Native VLA, CEM and driving interfaces instead construct
their proposals using their own action-producing model or search procedure.

All compared selectors should share the same pool. Candidate count is determined
by the proposal protocol; the released task-specific adapters retain their
recorded shape contracts. Model fitting and collecting execution supervision
remain separate from this CPU-only proposal step.

All output directories must be new. Metadata records the profile, source/config
identity, actual RNG seed, action precision and output hash. Tests cover exact
source-generator fixtures, reference exclusion, seeded reproducibility, family
coverage, malformed inputs, label-free interfaces and non-overwriting output.
Original implementation fingerprints and extraction notes are recorded in
[candidate source provenance](CANDIDATE_SOURCE_PROVENANCE.json).
