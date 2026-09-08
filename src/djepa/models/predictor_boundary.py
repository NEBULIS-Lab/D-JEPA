from __future__ import annotations
import torch

def configure_predictor_boundary(model: torch.nn.Module) -> list[str]:
    """Freeze TD-JEPA except its final conditional predictor block and projection."""
    if hasattr(model, "residual_transition") or any("residual" in key for key in model.state_dict()):
        raise ValueError("D-JEPA must not contain a residual transition")
    try:
        final_layer = model.predictor.transformer.layers[5]
        pred_proj = model.pred_proj
    except (AttributeError, IndexError) as error:
        raise ValueError("D-JEPA requires predictor.transformer.layers[5] and pred_proj") from error
    allowed = {id(parameter) for module in (final_layer, pred_proj) for parameter in module.parameters()}
    names: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = id(parameter) in allowed
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if not names or any(not (name.startswith("predictor.transformer.layers.5.") or name.startswith("pred_proj.")) for name in names):
        raise ValueError("D-JEPA trainable parameter boundary is invalid")
    set_predictor_mode(model)
    return names


def set_predictor_mode(model: torch.nn.Module) -> None:
    """All frozen modules stay eval; only the exact two allowed modules train."""
    model.eval()
    model.predictor.transformer.layers[5].train()
    model.pred_proj.train()
