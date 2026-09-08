# Install, configure and reproduce

## Installation

Use Python 3.10+ and a PyTorch build appropriate for your machine. The portable
training and cached evaluation commands run on CPU. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e '.[download,dev]'
bash scripts/smoke_test.sh
```

`pyproject.toml` is the dependency authority; `requirements.txt` delegates to it.
No site-specific runtime settings or accelerator indices are required.

## Download

```bash
python scripts/download_artifacts.py --profile pusht-relational --dataset
python -m zipfile -e data/D-JEPA-supervision-v1.zip data
```

The download helper selects only requested profiles, resolves Hub revisions to
immutable commits, and verifies weight/ZIP hashes against the release manifests.
It records revisions in local download receipts. Rerunning resumes Hub-cache
transfers and verifies existing outputs without replacing them. Destination
copies are published atomically only after checksum verification. An existing
file with the wrong checksum is never overwritten; move that named file aside
before retrying. No automatic archive extraction occurs.

Use `--revision COMMIT` and `--dataset-revision COMMIT` to replay the same release.
`--checkpoint-dir` and `--data-dir` change destinations. The helper uses the HF
library's normal authentication if needed; it never asks you to embed a token.

## YAML and command-line execution

Paths in YAML and CLI arguments are relative to the **working directory**, not
the YAML file. Run from the repository root unless using absolute paths.
Unknown/duplicate keys and invalid scalar types are rejected. CLI values override
YAML defaults. Historical `model.pt` / `config.json` files stay unchanged.

```bash
bash scripts/evaluate.sh --config configs/evaluation/pusht-independent.yaml
bash scripts/train.sh --config configs/training/pushobj.yaml --epochs 120
```

Training requires disjoint train/calibration base identities. The exposed
learning rate, batch size, weight decay, gradient clipping and evaluation interval
drive the actual trainer; defaults match the previous portable trainer.
Training writes `resolved_config.json`; evaluation writes a `.config.json`
sidecar. Existing output destinations cause an error, not an overwrite.

## Replay coverage

| YAML | Population | Replay output |
|---|---|---|
| `evaluation/pusht-independent.yaml` | 256 independent starts | Relational choices and labeled metrics |
| `evaluation/pushobj-calibration.yaml` | 256 calibration starts | Calibration choices and metrics; not formal unseen-300 |
| `evaluation/visual-shifts-calibration.yaml` | 63 calibration base IDs × 7 conditions | Condition-row metrics; not 441 independent starts |
| `evaluation/granular-formal.yaml` | 64 formal input sets | Selected IDs only; dense formal labels unavailable |

The full-model Reacher checkpoint is available, but its formal native scoring
is not an inference-ready cached-score release. No generic MSE substitute or
misleading Reacher replay YAML is provided. Other modules expose their original
implementations/objectives; the portable training CLI currently supports the
three-coordinate ordinal instance. Additional released-data recipes for PushT
relational alignment and Granular spatial/multiview alignment are available in
[MODULE_TRAINING.md](MODULE_TRAINING.md). Native predictor feature extraction
and fixed-horizon physics execution are described in
[NATIVE_REPRODUCING.md](NATIVE_REPRODUCING.md), with their required raw inputs.

```bash
python scripts/download_artifacts.py --profile pushobj-unseen-shapes --profile pusht-visual-shifts --profile granular-relational
bash scripts/reproduce.sh configs/evaluation/pusht-independent.yaml configs/evaluation/pushobj-calibration.yaml configs/evaluation/visual-shifts-calibration.yaml configs/evaluation/granular-formal.yaml
python scripts/summarize_results.py outputs/replay/pusht-independent.json outputs/replay/pushobj-calibration.json outputs/replay/visual-shifts-calibration.json outputs/replay/granular-formal.json --output outputs/replay-summary
```

Summaries produce one row per report, preserve observed/unobserved counts and do
not pool protocols. They consume evaluator outputs, not arbitrary historical
JSON schemas. These are **cached-decision replays**, not new simulator rollouts.
Calibrated composition remains a separate configuration from the relational
replay. See [protocols](PROTOCOLS.md) for the distinction. Numerical results are
presented on the [project website](https://nebulis-lab.com/D-JEPA#results); reference
outcomes remain in the HF supervision archive, not in the code repository.

## Preflight and resumable paper matrix

```bash
python scripts/doctor.py --checkpoint checkpoints/pusht-relational
python scripts/verify_dataset.py --data-root data/D-JEPA-supervision-v1 --checksums
bash scripts/reproduce_paper.sh --dry-run
bash scripts/reproduce_paper.sh
bash scripts/reproduce_paper.sh --resume
```

Download all four profiles in the replay coverage table before running the whole
matrix. Select individual runs with `--only pusht-independent` (repeatable).
`--data-root`, `--checkpoint-root`, `--matrix` and `--output` accept custom paths.
The doctor runs on CPU and does not initialize a GPU or simulator.
Dataset validation checks candidate identities, finite inputs, feature shapes
and boolean observation masks; `--checksums` additionally verifies every file
listed in the extracted dataset manifest. Use `--split PATH --checkpoint PROFILE`
for a single split's feature-dimension preflight.

The matrix preserves protocol names: PushT independent, Granular formal,
PushObj calibration and visual-shift calibration are separate runs. For PushT
and Granular, selected IDs must exactly match their recorded authorities from
the downloaded dataset. Other matrix rows report their calibration metrics
without claiming a fresh formal benchmark. The matrix never trains on these rows.

Each run creates `report.json`, `report.config.json`, `run.log` and `receipt.json`
under `outputs/paper/RUN_ID/`. The receipt records SHA-256 fingerprints for inputs,
checkpoint, settings, package source and outputs. `--resume` skips only a completed
run with matching fingerprints; it refuses changed or partial runs. To recover
from an interrupted run, inspect its log and select a new output directory (or
move that specific incomplete run aside). No output is silently overwritten.

```bash
python scripts/summarize_results.py outputs/paper/*/report.json --output outputs/paper-summary
```

Summary tables stay in ignored local outputs, not in the published source tree.

## Layout and compatibility

`src/djepa/models/` holds scientific implementations; `objectives/` groups losses,
`data/` loads supervision, `evaluation/` scores candidates, and `cli/` implements
configuration-aware commands. Root-level modules forward old imports to the
same implementations. Tensor names and arithmetic are preserved.

`SOURCE_PROVENANCE.json` records original exports; `PACKAGE_LAYOUT.json` maps
them to current implementation paths and records current hashes. An implementation
relocation is not a new model or experiment.

## Examples and site

```bash
python examples/predict_cached.py checkpoints/pusht-relational data/D-JEPA-supervision-v1/pusht/independent_256/inputs.npz
python examples/sparse_metrics.py
bash scripts/preview_site.sh 8000
```

The static project page is `docs/index.html`; existing technical documents remain
alongside it. Preview binds to localhost and does not publish anything. A custom
Python executable can be selected with `DJEPA_PYTHON` for shell wrappers.
