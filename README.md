<p align="center">
  <a href="https://nebulis-lab.com/D-JEPA">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/branding/jepa-full-logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="assets/branding/jepa-full-logo-light.svg">
      <img src="assets/branding/jepa-full-logo-light.svg" alt="D-JEPA" width="600">
    </picture>
  </a>
</p>

<h1 align="center">D-JEPA: A Decision-Aligned Latent World Model</h1>

<p align="center">
  Learning decision-relevant structure from predicted futures.
</p>

<p align="center">
  <a href="https://nebulis-lab.com/D-JEPA"><img src="https://img.shields.io/badge/Project-Website-A64CA6?style=flat&amp;labelColor=494150" alt="Project website"></a>
  <a href="https://huggingface.co/Shuaijun/D-JEPA"><img src="https://img.shields.io/badge/Hugging_Face-Model-8464A5?style=flat&amp;logo=huggingface&amp;logoColor=white&amp;labelColor=494150" alt="Model repository"></a>
  <a href="https://huggingface.co/datasets/Shuaijun/D-JEPA-Dataset"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-8464A5?style=flat&amp;logo=huggingface&amp;logoColor=white&amp;labelColor=494150" alt="Decision-supervision dataset"></a>
  <a href="https://github.com/NEBULIS-Lab/D-JEPA"><img src="https://img.shields.io/badge/GitHub-Code-777083?style=flat&amp;logo=github&amp;logoColor=white&amp;labelColor=494150" alt="Code repository"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#documentation">Documentation</a>
</p>

## Overview

**D-JEPA aligns predictive geometry with action selection.** It learns
decision-relevant relations among predicted futures, combines evidence across
predictive geometries, and expresses decision structure in latent future
representations—all built on pretrained predictive models.

This repository includes scientific modules, checkpoint loading, cached-feature
inference and training, tests, and recorded results. Task/module checkpoints are
distributed through the model repository; decision supervision, candidate inputs,
fixed identities and result authorities are packaged in **one supervision ZIP**.

### Method at a glance

| Module | Role |
|---|---|
| [Relational alignment](src/djepa/relational.py) | Learn set-wise, bounded corrections from future–goal descriptors and ordinal coordinates. |
| [Predictive plasticity](src/djepa/plasticity.py) | Adapt the final predictor with preservation losses. |
| [Multi-geometry alignment](src/djepa/multi_geometry.py) | Combine two dense descriptors and four ordinal geometries. |
| [Exact representation realization](src/djepa/exact_realization.py) | Encode terminal ordering in goal-distance geometry while preserving earlier futures. |
| [Temporal transport](src/djepa/temporal_transport.py) | Learn bounded, same-action updates to five-step future representations. |
| [Ordinal](src/djepa/ordinal.py) / [spatial adapters](src/djepa/granular.py) | Support task-local inputs and sparse supervision. |

<details>
<summary>Module composition and checkpoint dependencies</summary>

These are configurations of **D-JEPA**, not separate competing methods. Legacy
identifiers in raw result authorities and provenance retain the identities of
the original experiments.

The [predictor boundary](src/djepa/predictor_boundary.py) defines restricted
adaptation; the [transport objective](src/djepa/transport_objective.py) accompanies
the temporal module. See [checkpoint profiles](docs/CHECKPOINTS.md) for full-model
dependencies and loading requirements.

</details>

## Quick Start

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-8464A5?style=flat&logo=python&logoColor=white&labelColor=494150)
![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-8464A5?style=flat&logo=pytorch&logoColor=white&labelColor=494150)

The lightweight package supports CPU execution. From the repository root:

```bash
pip install -e .
```

### Replay a released checkpoint

Download the **`pusht-relational/` profile** from the model repository into
`checkpoints/pusht-relational/`, and place `D-JEPA-supervision-v1.zip` in the
repository root. Then run:

```bash
unzip D-JEPA-supervision-v1.zip -d data
djepa-evaluate \
  --checkpoint checkpoints/pusht-relational \
  --inputs data/D-JEPA-supervision-v1/pusht/independent_256/inputs.npz \
  --labels data/D-JEPA-supervision-v1/pusht/independent_256/labels.npz \
  --output outputs/pusht-independent.json
```

Expected recorded result: **223/256** successful choices on the independent
PushT population using relational alignment.

<details>
<summary>Evaluation protocol and file layout</summary>

The checkpoint argument is the downloaded **profile directory**, containing
`model.pt` and `config.json`, not its `.pt` file alone. Choices are computed from
frozen inputs first; labels are opened only afterward for evaluation. Output
files are never silently overwritten. This replays cached candidate decisions,
not a new simulator run.

See [protocols and schema](docs/PROTOCOLS.md) for candidate identities, split
definitions and task-specific label semantics.

</details>

### Train a task-local alignment module

```bash
djepa-train \
  --train data/D-JEPA-supervision-v1/pushobj/train \
  --calibration data/D-JEPA-supervision-v1/pushobj/calibration \
  --output outputs/pushobj-trained --epochs 120
```

<details>
<summary>Training objective and reproducibility</summary>

This portable CPU trainer uses the released three-coordinate ordinal features,
full-set success-mass objective, local ordering, preservation and calibration
gate. It rejects overlapping train/calibration base identities. The historical
paper checkpoints are released directly; this command is not a promise of
bitwise-identical retraining across software/hardware versions.

</details>

## Results

| Task / protocol | D-JEPA configuration | Success |
|---|---|---:|
| [PushT · independent 256](results/pusht-independent-256.json) | Calibrated composition | **225/256** |
| Same PushT population | Relational alignment / exact realization | 223/256 |
| [Reacher · independent 128](results/reacher-128.json) | Relational selector + learned transport module | **120/128** |
| [Granular · 64 starts](results/granular-64.json) | Relational alignment | **25/64** standard; **12/64** strict |
| [PushObj · unseen shapes, 300 starts](results/pushobj-unseen-300.json) | Task-local ordinal alignment | **161/300** |
| [PushT · visual shifts, 50 fresh base starts](results/pusht-visual-shifts-50.json) | Task-local ordinal alignment | Seven separate conditions |

<details>
<summary>Result populations and release coverage</summary>

The 128-start PushT multi-geometry result is a separate development/mechanism
population, recorded [here](results/multi-geometry-development.json). It is not
substituted for the independent 256-start result. Visual-shift evaluation uses
50 fresh base starts across seven known corruption conditions, not 350
independent base starts.

Ongoing expansion and real-robot experiments are not part of this version's
results. They can be added as subsequent versioned releases.

</details>

## Documentation

| Guide | Contents |
|---|---|
| [Protocols and data schema](docs/PROTOCOLS.md) | Evaluation populations, identities and supervision semantics |
| [Checkpoint profiles](docs/CHECKPOINTS.md) | Loading, composition and upstream dependencies |
| [Validation](docs/VALIDATION.md) | Tested release scope and cached-decision replay |
| [Source provenance](docs/SOURCE_PROVENANCE.json) | Source/export hashes for extracted modules |
| [Recorded results](results/) | Experiment authorities, not newly run evaluations |

### Run tests

```bash
python -m unittest discover -s tests -v
```

## Acknowledgments

D-JEPA builds on pretrained predictive models and research infrastructure,
including [Temporal-Distance-JEPA](https://github.com/HKBU-KnowComp/TD-JEPA),
[stable-worldmodel / LeWM](https://github.com/galilai-group/stable-worldmodel),
and [stable-pretraining](https://github.com/galilai-group/stable-pretraining).
See [third-party notices](docs/THIRD_PARTY.md) for recorded revisions, dependency
attribution and release terms. Project and model/data licenses remain
author-specified release information.
