# Dependencies and release permissions

The lightweight package uses PyTorch, NumPy and PyYAML. Optional artifact downloads
use huggingface-hub; optional video export uses imageio/imageio-ffmpeg;
development packaging uses build. Its extracted scientific modules
are project-authored; source/export hashes are in SOURCE_PROVENANCE.json.

## Research foundations

We acknowledge [LeWorldModel (LeWM)](https://github.com/Mengarr/lewm) and
[JEPA](https://github.com/facebookresearch/jepa) as research foundations. These
project-level acknowledgments are distinct from the exact implementation
dependencies and revisions below.

## Implementation dependencies

Full pretrained-predictor reconstruction additionally uses these recorded sources:

| Dependency | Upstream source | Recorded source revision |
|---|---|---|
| Temporal-Distance-JEPA | https://github.com/HKBU-KnowComp/TD-JEPA | `b4c17ca4649c9bf47272fa66c38da7a684f2a020` |
| stable-worldmodel / LeWM | https://github.com/galilai-group/stable-worldmodel | `464f97f51a287f4d16e0caad02e8a9fe87c3847c` |
| stable-pretraining | https://github.com/galilai-group/stable-pretraining | `6a567ebd41bac4d11419bcb7bdfbd6506af576f5` |
| V-JEPA 2 / 2-AC (offline robotics) | https://github.com/facebookresearch/vjepa2 | `204698b45b3712590f06245fbfba32d3be539812` |

The independent `real_robot/` package additionally uses SciPy and h5py.
Official V-JEPA 2 preprocessing and model construction use an explicitly
provided local checkout with its own compatible dependencies (including
torchvision). Upstream code, weights, robot SDKs and robot-specific control or
calibration files are not vendored into this package. Their licenses remain
those supplied by their authors; recording/checkpoint distribution is separate
from releasing the offline integration code.

The recorded source trees declare MIT licensing (TD-JEPA's LICENSE and the two
stable packages' package metadata). Those declarations do not automatically
establish the terms for every pretrained weight or derived dataset. DINO-WM,
JEPA-WM and task-specific extension predictor dependencies are identified by
their checkpoint hashes in model/data provenance and need their own release
term checks. Upstream source revisions do not replace actual file hashes where
the experiment used local changes.

## Task interfaces

The project-authored robot interfaces use RoboTwin's RLDS/simulator conventions
and explicit pi0.5 action conversions. Full EE branch execution accepts an
external CoWAM-compatible simulator adapter; neither that adapter nor RoboTwin
simulator code, policy weights or assets is bundled. See [robotic manipulation](ROBOTICS.md).
RLDS decoding optionally uses TensorFlow CPU and TensorFlow Datasets.

The driving interfaces call an externally supplied Drive-JEPA checkout and its
NAVSIM v1 scorer. Install that checkout's requirements and retain its license;
Drive-JEPA, NAVSIM/nuPlan source, maps, sensor data and pretrained assets are not
copied into this package. See [autonomous driving](DRIVING.md). The extracted
D-JEPA modules and wrappers are project-authored; code-file provenance is in
`TASK_SOURCE_PROVENANCE.json`. No hardware-specific control interfaces or
cluster configuration are included.

## Project license

Project-authored D-JEPA code, including the independent offline robotics
package, is released under the [Apache License 2.0](../LICENSE).
This does not replace third-party licenses or grant additional rights to
upstream code, model weights, or datasets. Checkpoint and dataset terms are
specified by their respective releases.
