# Training released D-JEPA modules

The public trainer covers modules for which the supervision archive contains
all inputs needed by the native model and objective:

| Profile | Public training coverage | Split rule |
|---|---|---|
| `pusht-relational` | Complete | Read the exact 64 calibration start IDs from `relational_calibration_ids`; fit on the other 448 pool starts |
| `granular-relational` | Complete | Fit on `granular/train`; select the checkpoint and strict gate on `granular/calibration` |
| `granular-multiview` | Complete | Fit four-view convex weights on masked training labels only, then train and calibrate as above |

Run from the repository root after placing the extracted supervision archive
under `data/`:

```bash
bash scripts/train_modules.sh --config configs/training/pusht-relational.yaml
bash scripts/train_modules.sh --config configs/training/granular-relational.yaml
bash scripts/train_modules.sh --config configs/training/granular-multiview.yaml
```

Command-line arguments override YAML values. Each command refuses to overwrite
its output directory. It writes:

- `model.pt`: tensor-only state dictionary, strict-loadable through
  `djepa.inference.load_profile`;
- `config.json`: inference configuration, calibrated strict-gate threshold and
  SHA-256 of `model.pt`;
- `resolved_config.json`: effective command configuration;
- `training_receipt.json`: SHA-256 for every input, label and metadata file,
  plus the exact fit/calibration identities;
- `metrics.jsonl`: update and calibration trace.

All three routes use the released candidate features and the existing public
model/objective implementations. Granular labels are always intersected with
`supervision_mask`; an unobserved entry is never interpreted as a failure.
Training rejects `development`, `formal` and independent splits, and rejects
overlapping train/calibration identities. If calibration selects the baseline
for every row, positive infinity is exported as the largest finite float64
value. This JSON-safe sentinel preserves the disabled strict gate for every
valid finite advantage, including mixed float64/float32 rounding at the nominal
correction bound.

For a trained `granular-multiview` profile, construct its cached inference
inputs from the released four-view ranks (the archive's `features` field is the
2305-wide relational input):

```python
import json
import numpy as np
from djepa.inference import predict

profile = "outputs/granular-multiview-trained"
config = json.load(open(f"{profile}/config.json"))
with np.load("data/D-JEPA-supervision-v1/granular/formal/inputs.npz",
             allow_pickle=False) as arrays:
    ranks = arrays["multiview_ranks"]
    result = predict(profile, {
        "features": ranks.astype(np.float32),
        "base_scores": ranks @ np.asarray(config["fusion_weights"], dtype=np.float64),
        "candidate_ids": arrays["candidate_ids"],
    })
```

These commands reproduce the released training procedure and checkpoint
interface. Floating-point/library differences and fresh initialization mean
they are not a claim of bitwise reconstruction of the separately published
historical checkpoint; use that checkpoint for exact archived decision replay.

## Modules that cannot be faithfully retrained from this archive

No executable mode is exposed for the following incomplete routes:

- Reacher relational/temporal transport and the full Reacher world model need
  the exact frozen LeWM and TD-JEPA native criterion implementations and their
  compatible upstream weights/preprocessing. The released `true_cost` is
  physical supervision; it cannot replace the differentiable native criterion
  that scores original and transported futures.
- PushT predictive adaptation needs the compatible upstream predictor and its
  original adaptation inputs/objective. The tensor patch can be loaded as
  documented in `CHECKPOINTS.md`, but it cannot be reconstructed from cached
  decision features alone.
- PushT multi-geometry training needs the additional JEPA-WM and DINO-WM
  geometries on an authorized fitting population. Those fields are not present
  in the PushT fitting pool, and the development set is never a training source.

These are input-availability boundaries, not placeholder commands. The released
checkpoints remain usable for their documented inference interfaces.
