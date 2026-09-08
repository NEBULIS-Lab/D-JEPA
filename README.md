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
  <a href="https://huggingface.co/Shuaijun/D-JEPA"><img src="https://img.shields.io/badge/Hugging_Face-Model-FFDB67?style=flat&amp;logo=huggingface&amp;logoColor=white&amp;labelColor=494150" alt="Model repository"></a>
  <a href="https://huggingface.co/datasets/Shuaijun/D-JEPA-Dataset"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-8464A5?style=flat&amp;logo=huggingface&amp;logoColor=white&amp;labelColor=494150" alt="Decision-supervision dataset"></a>
  <a href="https://github.com/NEBULIS-Lab/D-JEPA"><img src="https://img.shields.io/badge/GitHub-Code-777083?style=flat&amp;logo=github&amp;logoColor=white&amp;labelColor=494150" alt="Code repository"></a>
  <a href="https://example.com" title="Paper link placeholder"><img src="https://img.shields.io/badge/Paper-arXiv-ECA896?style=flat&amp;logo=arxiv&amp;logoColor=white&amp;labelColor=494150" alt="Paper on arXiv (link forthcoming)"></a>
  <a href="https://example.com" title="Video link placeholder"><img src="https://img.shields.io/badge/Video-YouTube-F6C17F?style=flat&amp;logo=youtube&amp;logoColor=white&amp;labelColor=494150" alt="Video on YouTube (link forthcoming)"></a>
</p>

<p align="center"><sub>Paper and video links are placeholders; final links are forthcoming.</sub></p>

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
inference and training, tests, and reproducibility tools. Task/module checkpoints are
distributed through the model repository; decision supervision, candidate inputs,
fixed identities and result authorities are packaged in **one supervision ZIP**.

### Method at a glance

| Module | Role |
|---|---|
| [Relational alignment](src/djepa/models/relational.py) | Learn set-wise, bounded corrections from future–goal descriptors and ordinal coordinates. |
| [Predictive plasticity](src/djepa/models/plasticity.py) | Adapt the final predictor with preservation losses. |
| [Multi-geometry alignment](src/djepa/models/multi_geometry.py) | Combine two dense descriptors and four ordinal geometries. |
| [Exact representation realization](src/djepa/models/exact_realization.py) | Encode terminal ordering in goal-distance geometry while preserving earlier futures. |
| [Temporal transport](src/djepa/models/temporal_transport.py) | Learn bounded, same-action updates to five-step future representations. |
| [Ordinal](src/djepa/models/ordinal.py) / [spatial adapters](src/djepa/models/granular.py) | Support task-local inputs and sparse supervision. |

<details>
<summary>Module composition and checkpoint dependencies</summary>

These are configurations of **D-JEPA**, not separate competing methods. Legacy
identifiers in raw result authorities and provenance retain the identities of
the original experiments.

The [predictor boundary](src/djepa/models/predictor_boundary.py) defines restricted
adaptation; the [transport objective](src/djepa/objectives/transport_objective.py) accompanies
the temporal module. See [checkpoint profiles](docs/CHECKPOINTS.md) for full-model
dependencies and loading requirements.

</details>

## Quick Start

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-8464A5?style=flat&logo=python&logoColor=white&labelColor=494150)
![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-8464A5?style=flat&logo=pytorch&logoColor=white&labelColor=494150)

The lightweight package supports CPU execution. From the repository root:

```bash
pip install -r requirements.txt
```

For verified Hub downloads, install `pip install -e '.[download]'`. See the
[reproduction guide](docs/REPRODUCING.md) for all configurations and populations.

### Replay a released checkpoint

Download and verify the **`pusht-relational/` profile** plus supervision:

```bash
python scripts/download_artifacts.py --profile pusht-relational --dataset
python -m zipfile -e data/D-JEPA-supervision-v1.zip data
bash scripts/reproduce.sh
```

This replays the relational checkpoint on the independent PushT population.
See the [project website](https://nebulis-lab.com/D-JEPA#results) for reported results.

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
bash scripts/train.sh --config configs/training/pushobj.yaml --epochs 120
```

YAML configures the actual run; CLI options take precedence. Each run records
its resolved settings. Paths are relative to the working directory. Existing
`djepa-train` / `djepa-evaluate` commands and Python import paths remain supported.

<details>
<summary>Training objective and reproducibility</summary>

This portable CPU trainer uses the released three-coordinate ordinal features,
full-set success-mass objective, local ordering, preservation and calibration
gate. It rejects overlapping train/calibration base identities. The historical
paper checkpoints are released directly; this command is not a promise of
bitwise-identical retraining across software/hardware versions.

</details>

## Results

Visit the [project website](https://nebulis-lab.com/D-JEPA#results) for numerical
comparisons and [qualitative demonstrations](https://nebulis-lab.com/D-JEPA#demonstrations).
Machine-readable reference outcomes are distributed in the
[Hugging Face supervision archive](https://huggingface.co/datasets/Shuaijun/D-JEPA-Dataset),
not as a separate results directory in this code repository.

## Documentation

| Guide | Contents |
|---|---|
| [Reproduction guide](docs/REPRODUCING.md) | Installation, verified downloads, YAML runs and summary scripts |
| [Protocols and data schema](docs/PROTOCOLS.md) | Evaluation populations, identities and supervision semantics |
| [Checkpoint profiles](docs/CHECKPOINTS.md) | Loading, composition and upstream dependencies |
| [Validation](docs/VALIDATION.md) | Tested release scope and cached-decision replay |
| [Source provenance](docs/SOURCE_PROVENANCE.json) | Source/export hashes for extracted modules |
| [Website maintenance](docs/SITE.md) | Local preview, theme and author-artwork slots |

### Repository map

```text
src/djepa/      models · objectives · data · evaluation · cli
configs/       executable training and evaluation YAMLs
scripts/       download · train · evaluate · reproduce · summarize · preview
examples/      label-free inference and sparse metrics
tests/         scientific behavior and workflow tests
docs/          project website and technical documentation
assets/        original brand assets
```

Legacy top-level Python modules forward to the organized implementations;
scientific source hashes are mapped in [PACKAGE_LAYOUT.json](docs/PACKAGE_LAYOUT.json).

### Run tests

```bash
bash scripts/smoke_test.sh
```

## Acknowledgments

We thank the [LeWorldModel (LeWM)](https://github.com/Mengarr/lewm) and
[JEPA](https://github.com/facebookresearch/jepa) research teams for their
foundational work and open-source resources.

Additional implementation dependencies, recorded revisions and release terms
are documented in [third-party notices](docs/THIRD_PARTY.md).
