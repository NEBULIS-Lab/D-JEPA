# D-JEPA: A Decision-Aligned Latent World Model

**[Project website](https://nebulis-lab.com/D-JEPA)** ·
**[Model repository](https://huggingface.co/Shuaijun/D-JEPA)** ·
**[Decision-supervision dataset](https://huggingface.co/datasets/Shuaijun/D-JEPA-Dataset)** ·
**[Code](https://github.com/NEBULIS-Lab/D-JEPA)**

D-JEPA learns decision-relevant relations among predicted futures, combines
evidence from predictive geometries, and expresses decision structure in latent
future representations. It builds on pretrained predictive models.

## Release contents

- **[GitHub code repository](https://github.com/NEBULIS-Lab/D-JEPA):** scientific modules, checkpoint loading, cached-feature
  inference/training, tests, protocol descriptions, and recorded paper results.
- **[Hugging Face Model](https://huggingface.co/Shuaijun/D-JEPA):** task/module checkpoint profiles (`model.pt` + `config.json`).
- **[Hugging Face Dataset](https://huggingface.co/datasets/Shuaijun/D-JEPA-Dataset):** one `D-JEPA-supervision-v1.zip` containing supervision,
  candidate inputs, fixed identities, and result authorities.
- **Zenodo:** author-approved final paper archive and DOI, at final release.
- **[Project website](https://nebulis-lab.com/D-JEPA):** curated complete-horizon visual examples, separately hosted.

The authors designated the code repository `NEBULIS-Lab/D-JEPA`, model repository
`Shuaijun/D-JEPA` and dataset repository `Shuaijun/D-JEPA-Dataset`. The project
license, author metadata and final paper DOI remain author-supplied release information.

## Quick start: replay a released checkpoint

Python 3.10 or newer; CPU works for the lightweight package.

```bash
pip install -e .
unzip D-JEPA-supervision-v1.zip -d data
djepa-evaluate \
  --checkpoint checkpoints/pusht-relational \
  --inputs data/D-JEPA-supervision-v1/pusht/independent_256/inputs.npz \
  --labels data/D-JEPA-supervision-v1/pusht/independent_256/labels.npz \
  --output outputs/pusht-independent.json
```

The checkpoint argument is the downloaded **profile directory**, not its `.pt`
file. The command computes choices from frozen inputs first; labels are opened
only afterward for evaluation. Output files are never silently overwritten.
This reproduces cached candidate decisions, not a new simulator run.

## Train a task-local alignment module

```bash
djepa-train \
  --train data/D-JEPA-supervision-v1/pushobj/train \
  --calibration data/D-JEPA-supervision-v1/pushobj/calibration \
  --output outputs/pushobj-trained --epochs 120
```

This portable CPU trainer uses the released three-coordinate ordinal features,
full-set success-mass objective, local ordering, preservation and calibration
gate. It rejects overlapping train/calibration base identities. The historical
paper checkpoints are released directly; this command is not a promise of
bitwise-identical retraining across software/hardware versions.

## Method modules

| Module | Source | Purpose |
|---|---|---|
| Relational alignment | `src/djepa/relational.py` | Two normalized future–goal descriptors and ordinal coordinates; set-wise bounded correction and decision-tail objective |
| Predictive plasticity | `src/djepa/plasticity.py`, `predictor_boundary.py` | Restricted final-predictor adaptation and preservation losses |
| Multi-geometry alignment | `src/djepa/multi_geometry.py` | Two dense descriptors plus four ordinal geometries |
| Exact representation realization | `src/djepa/exact_realization.py` | Preserve earlier futures and encode the calibrated terminal ordering in goal-distance geometry |
| Temporal transport | `src/djepa/temporal_transport.py`, `transport_objective.py` | Learned, bounded, same-action five-step future updates |
| Task-local instances | `src/djepa/ordinal.py`, `granular.py` | Ordinal and spatial-feature input adapters, including sparse-supervision loss |

Use descriptive profiles as configurations of **D-JEPA**, not separate competing
methods. Legacy identifiers in raw result authorities and provenance identify
the original experiments and are intentionally retained.

## Validated result populations

| Population | D-JEPA configuration | Success |
|---|---|---:|
| PushT independent 256 starts | Calibrated composition | 225/256 |
| Same PushT population | Relational alignment / exact realization | 223/256 |
| Reacher independent 128 starts | Relational selector with learned transport module | 120/128 |
| Granular 64 starts | Relational alignment | 25/64 standard; 12/64 strict |
| PushObj unseen shapes, 300 starts | Task-local ordinal alignment | 161/300 |
| PushT known visual corruptions, 50 fresh base starts | Task-local ordinal alignment | Seven separate conditions; see protocol/result files |

The 128-start PushT multi-geometry result is a separate development/mechanism
population. It is not substituted for the independent 256-start result.

## Documentation and testing

- [Protocols and data schema](docs/PROTOCOLS.md)
- [Checkpoint profiles and loading](docs/CHECKPOINTS.md)
- [Dependencies and third-party notices](docs/THIRD_PARTY.md)
- [Local release validation scope](docs/VALIDATION.md)
- `docs/SOURCE_PROVENANCE.json`: source/export hashes for mechanically extracted modules.
- `results/`: recorded authorities, not newly run experiments.

```bash
python -m unittest discover -s tests -v
```

Ongoing expansion and real-robot experiments are not part of this version's
results. They can be added as subsequent versioned releases.
