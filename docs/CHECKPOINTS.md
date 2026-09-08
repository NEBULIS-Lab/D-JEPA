# Checkpoint profiles

Every profile includes a tensor-only `model.pt` and JSON configuration. Load with
`torch.load(path, map_location="cpu", weights_only=True)`. The published file hash
differs from the original training container because metadata and optimizer state
are not embedded. Tensor values are preserved exactly; source and export hashes
are separate fields in the configuration.

| Profile | Contents | Inference interface |
|---|---|---|
| `pusht-relational` | 386-input alignment module | `djepa.inference.predict` |
| `pusht-predictive-adaptation` | Final predictor block and projection parameters | Load exact base checkpoint, then overwrite only these keys |
| `pusht-adapted-predictor-full` | Adapted predictor including frozen parameters | Instantiate the recorded Temporal-Distance-JEPA architecture, strict-load |
| `pusht-multi-geometry` | 388-input alignment module | Two normalized descriptors + four ordinal coordinates; JEPA-WM base rank |
| `pusht-exact-realization-full` | Two predictors + relation module | `ExactRealizationWorldModel`; architecture and preprocessing in configuration |
| `reacher-relational` | 386-input alignment module | Native task criteria must produce the ordinal coordinates |
| `reacher-temporal-transport` | 5→32→1 coefficient module | `ReacherFutureResidualHead(hidden_dim=32, radius=0.1)` |
| `reacher-world-model-full` | Two predictors + both learned modules | Tensor namespaces `tdjepa`, `lewm`, `adapters` |
| `granular-relational` | 2305-input spatial relational module | `GranularSetRanker`; cached features include the base ordinal coordinate |
| `granular-multiview` | Four-coordinate same-backbone variant | `GranularSetRanker`; use frozen four-view fusion weights |
| `pushobj-unseen-shapes` | 3-input ordinal alignment module | `djepa.inference.predict` |
| `pusht-visual-shifts` | 3-input ordinal alignment module | `djepa.inference.predict` |

## Full predictive models

The original upstream implementations remain separate dependencies, not vendored
copies of entire research repositories. The exact PushT architecture is included
under `model_architecture` in its full checkpoint configuration. Instantiate the
two `*_config` objects using the upstream packages listed in THIRD_PARTY.md; then
construct `djepa.exact_realization.ExactRealizationWorldModel` with both models,
the recorded fusion/gate values, and recorded action normalization. Strict-load
the exported tensor state. All predictor computation remains inside this model;
one checkpoint does not mean one backbone or a distilled student.

`djepa.checkpoints.load_exact_world_model(profile, tdjepa=..., lewm=...)` wraps
this strict-loading procedure. `load_reacher_world_model` loads the two supplied
native backbones and the trained Reacher feature-level model. Native backbone
architecture/preprocessing compatibility is required in both cases.

For a predictor patch:

```python
import torch
patch = torch.load("checkpoints/pusht-predictive-adaptation/model.pt",
                   map_location="cpu", weights_only=True)
state = exact_pretrained_model.state_dict()
if not patch or not set(patch).issubset(state):
    raise ValueError("patch and backbone do not match")
for key, value in patch.items():
    if state[key].shape != value.shape:
        raise ValueError(f"shape mismatch: {key}")
state.update(patch)
exact_pretrained_model.load_state_dict(state, strict=True)
```

Verify the base checkpoint hash in the patch configuration first. These are
replacement learned parameters, not additive deltas.

The Reacher adapter data retain full context/future/goal features. Its native
Temporal-Distance criterion is learned; do not substitute generic terminal MSE.
The formal physical choice uses the relational score. The transport module's
distance is evaluated as a separate representation diagnostic.

## Gates and composition

Core relation/Granular gate: compare `base_at_base_choice - refined_at_refined_choice`
against the calibrated threshold. Task-local ordinal gate: compare both choices
under refined scores. The strict comparison is `>`; ties do not activate the gate.
Exact PushT realization quantizes gate advantage to four decimals; the standalone
relation profile retains its original unquantized gate. Reacher uses eight.

Predictive composition is the calibrated relational decision with an adapted-
future proposal; it is not an arithmetic average or a success-label switch. The
paper's 225/256 is this composition, not the 223/256 relation checkpoint alone.

Full upstream-containing checkpoints are prepared locally. Authors must settle
the applicable upstream weight redistribution terms before uploading them.
