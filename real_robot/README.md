# D-JEPA · Offline Robotics

Recorded observations → action-conditioned futures → decision alignment →
selected prediction-space actions.

This independent package provides the robotics learning and offline planning
pipeline. It uses V-JEPA 2-AC for action-conditioned prediction and trains a
set-wise relational head on measured candidate outcomes. Install it separately
from the simulation package; no simulator, camera driver, robot SDK, or control
service is required for the core workflows.

Device-specific acquisition, calibration and execution adapters belong to the
user's robot integration. They are intentionally outside this release, rather
than pending public components. All supplied commands operate on files and
return offline outputs; they do not send robot commands.

## Installation

From the repository root, create a dedicated Python 3.10+ environment:

```bash
python -m venv .venv-robot
source .venv-robot/bin/activate
cd real_robot
python -m pip install -r requirements.txt
bash scripts/run_checks.sh
```

Run the commands below **from `real_robot/`**. The `djepa_robot` package and
this directory's `scripts` are independent of `src/djepa/` and the simulation
scripts at the repository root. Core dependencies are NumPy, SciPy, h5py and
PyTorch; pytest is used by the tests. Commands default to CPU.

## Workflows

| Workflow | Entry point | Output |
|---|---|---|
| Recording validation | `python -m scripts.audit_data` | Per-episode timing, field and frame-count checks |
| RGB preparation / encoding | `python -m scripts.cache_rgb` | Recorded RGB, optional visual latents and source receipt |
| Official predictor diagnostic | `python -m scripts.smoke_ac` | Action-response and recorded-transition diagnostics |
| Relational learning | `python -m scripts.train_alignment` | `alignment.pt` and training receipt |
| Label-free candidate scoring | `python -m scripts.score_candidates` | Native/aligned scores and literal selected IDs |
| Offline planning | `python -m scripts.plan_latents` | Selected actions, shared candidate pool and prediction costs |
| Recorded motion conversion | `python -m scripts.audit_motion` | Nominal FK and command-versus-observation arrays |
| Recorded-window prediction | `python -m scripts.evaluate_recorded_motion` | Prediction diagnostics on timing-valid recorded transitions |

Each entry point supports `--help`. Existing output directories are not silently
overwritten. See [data and interface contracts](DATA_FORMAT.md) before preparing
inputs.

### Train and score

Prepare measured candidate supervision using the documented NPZ schema, then:

```bash
python -m scripts.train_alignment \
  --cache data/measured-candidates.npz \
  --epochs 200 --batch-size 8 --learning-rate 0.001 --seed 0 \
  --device cpu --output outputs/alignment

python -m scripts.score_candidates \
  --cache data/inference-candidates.npz \
  --checkpoint outputs/alignment/alignment.pt \
  --output outputs/scored
```

The trainer uses measured lower-is-better costs, pairwise ordering and bounded
correction regularization. Train and validation start identities must be
disjoint. The final epoch is saved; the training receipt retains the complete
validation trace. Scoring does not open outcome labels. Synthetic fixtures need
explicit `--allow-synthetic` and retain their test-only identity in checkpoints.

### V-JEPA 2-AC dependency

For raw-image encoding and world-model planning, obtain the official
[V-JEPA 2 source and AC checkpoint](https://github.com/facebookresearch/vjepa2)
under their upstream terms. The recorded source revision is
`204698b45b3712590f06245fbfba32d3be539812`.

For the official RGB preprocessing, install `python -m pip install -e '.[preprocessing]'`
(torchvision, OpenCV and Pillow). Full model construction additionally requires
that checkout's compatible dependencies. Keep torchvision matched to your
PyTorch installation. Pass the checkout location
and the AC **encoder-plus-predictor** checkpoint explicitly:

```bash
python -m scripts.cache_rgb \
  --episode data/episode.hdf5 --camera cam_head --indices 0,1,2 \
  --checkout vendor/vjepa2 --checkpoint checkpoints/vjepa2-ac-vitg.pt \
  --device cpu --output outputs/visual

python -m scripts.plan_latents \
  --cache data/planning-latents.npz \
  --checkout vendor/vjepa2 --checkpoint checkpoints/vjepa2-ac-vitg.pt \
  --alignment-checkpoint outputs/alignment/alignment.pt \
  --config configs/offline_planning.json \
  --device cpu --output outputs/plan
```

The image cache preserves recorded joints and timestamps; it does not invent
Cartesian poses. The planning cache therefore requires a separately prepared
pose and verified action semantics. Its feature/model identity must match the
alignment checkpoint. Omit `--alignment-checkpoint` for native-only planning.
The loader uses explicit local sources and checked weights, with no automatic
checkpoint download or untrained inference fallback.

The example configuration specifies prediction-space search bounds, not
hardware limits. CEM first searches with comparable native costs; alignment
then operates once on the shared pool. Alternatively, supply already converted
candidate action chunks in the planning cache. Selected IDs always refer to
literal scored action sequences.

## Device integration boundary

Connect another robot by producing the inputs in [DATA_FORMAT.md](DATA_FORMAT.md)
and consuming the selected prediction-space sequence in your own controller.
Coordinate transforms, TCP calibration, joint/Cartesian conversion, actuator
limits and execution monitoring remain in that device-specific layer. The
reference recording reader targets a six-joint-plus-gripper HDF5 schema; other
recording schemas need a reader/converter, not a change to the alignment model.

Neither raw recordings nor private URDFs, calibration values, credentials, SDK
configuration or robot-trained weights are bundled here. Existing simulation
checkpoint profiles are not relabeled as robotics checkpoints.

## Tests

```bash
bash scripts/run_checks.sh
```

Tests cover action semantics, rotation composition, deterministic CEM, recorded
timestamps, model loading, learning and label-free scoring, output protection,
and the offline planning boundary. Two tests require the official preprocessing
checkout; run them explicitly after installing its dependencies:

```bash
DJEPA_VJEPA2_CHECKOUT=/path/to/vjepa2 bash scripts/run_checks.sh
```

Without that variable, those two integration tests are reported as skipped.
The ordinary suite does not load large pretrained weights or initialize hardware.
Its synthetic learning/planning runs test software behavior, not task success.

## Layout

```text
djepa_robot/    actions · data · vision · world_model · alignment · training
               planning · kinematics · motion_audit · recorded_windows
scripts/       offline workflow entry points
configs/       prediction-space planning example
tests/         CPU tests and optional upstream preprocessing integration
```

See [source provenance](SOURCE_PROVENANCE.json) for the exported offline modules
and [third-party notices](../docs/THIRD_PARTY.md) for upstream dependencies.
