"""Local V-JEPA2-AC loading and causal action-conditioned latent rollout.

Interface reference: facebookresearch/vjepa2 notebooks/utils/world_model_wrapper.py.
No hardware, remote hub entry, or randomly initialized inference fallback.
"""

from pathlib import Path
import math
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from .actions import CandidateBatch, advance_pose


def _finite(x, name):
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be a finite floating tensor")


def native_cost(terminal, goal):
    """Full patchwise L1, lower is better; no spatial pooling before this cost."""
    _finite(terminal, "terminal")
    _finite(goal, "goal")
    if terminal.ndim != 3 or goal.shape != (1, *terminal.shape[1:]):
        raise ValueError("terminal [N,P,D] and shared goal [1,P,D] required")
    return (terminal.float() - goal.float()).abs().mean(dim=(1, 2))


def spatial_features(representations):
    """Prototype 2x2 spatial pool for the small relational head, NOT native scoring.

    This robot-specific feature adapter is new and awaits measured-data validation.
    Patch order is the official row-major square image grid.
    """
    _finite(representations, "representations")
    if representations.ndim != 3:
        raise ValueError("representations must be [N,P,D]")
    n, patches, dim = representations.shape
    side = math.isqrt(patches)
    if side * side != patches or side < 2:
        raise ValueError("2x2 feature pooling requires a square patch grid >=2x2")
    images = representations.float().transpose(1, 2).reshape(n, dim, side, side)
    return F.adaptive_avg_pool2d(images, (2, 2)).flatten(1)


class ACPredictor:
    def __init__(self, predictor, *, expected_dt, max_history=32, normalize_reps=True):
        if not np.isfinite(expected_dt) or expected_dt <= 0 or max_history < 1:
            raise ValueError("explicit positive model transition dt and history required")
        self.predictor = predictor.eval()
        self.expected_dt = float(expected_dt)
        self.max_history = int(max_history)
        self.normalize_reps = bool(normalize_reps)

    @torch.inference_mode()
    def rollout(self, context, pose, candidates: CandidateBatch):
        _finite(context, "context")
        _finite(pose, "pose")
        if context.ndim != 3 or context.shape[0] != 1 or min(context.shape[1:]) < 1 or pose.shape != (1, 7):
            raise ValueError("one initial context [1,P,D] and pose [1,7] required")
        if pose.device != context.device or torch.any((pose[:, -1] < 0) | (pose[:, -1] > 1)):
            raise ValueError("pose device or normalized gripper contract mismatch")
        if not np.isclose(candidates.dt, self.expected_dt, rtol=1e-6, atol=1e-9):
            raise ValueError("candidate dt does not match explicit model transition dt")
        n, horizon, _ = candidates.actions.shape
        if horizon > self.max_history:
            raise ValueError("requested rollout exceeds predictor causal history")
        actions = torch.tensor(candidates.actions, device=context.device, dtype=context.dtype)
        patches, dim = context.shape[1:]
        frames = context.expand(n, -1, -1)[:, None]
        states = pose.to(dtype=context.dtype).expand(n, -1)[:, None]
        futures = []
        for step in range(horizon):
            result = self.predictor(frames.flatten(1, 2), actions[:, :step + 1], states)
            if result.shape != (n, (step + 1) * patches, dim):
                raise ValueError("AC predictor output shape differs from causal input token shape")
            _finite(result, "prediction")
            next_frame = result[:, -patches:]
            if self.normalize_reps:
                next_frame = F.layer_norm(next_frame, (dim,))
            futures.append(next_frame)
            next_pose = advance_pose(states[:, -1].float().cpu().numpy(), candidates.actions[:, step])
            states = torch.cat((states, torch.tensor(next_pose, device=context.device,
                                                     dtype=context.dtype)[:, None]), dim=1)
            frames = torch.cat((frames, next_frame[:, None]), dim=1)
        return torch.stack(futures, dim=1)


def load_checked_state(module, state, *, role, allowed_unexpected=()):
    """Validate all keys/shapes before mutating a module. Never silently drop weights."""
    cleaned = {}
    for name, tensor in state.items():
        key = name
        while key.startswith(("module.", "backbone.")):
            key = key.split(".", 1)[1]
        if key in cleaned:
            raise ValueError(f"{role}: duplicate normalized checkpoint key {key}")
        cleaned[key] = tensor
    expected = module.state_dict()
    missing = sorted(expected.keys() - cleaned.keys())
    unexpected = sorted(cleaned.keys() - expected.keys())
    bad_shapes = [k for k in expected.keys() & cleaned.keys()
                  if not isinstance(cleaned[k], torch.Tensor) or cleaned[k].shape != expected[k].shape]
    if missing or set(unexpected) - set(allowed_unexpected) or bad_shapes:
        raise ValueError(f"{role}: missing={missing}, unexpected={unexpected}, bad_shapes={bad_shapes}")
    module.load_state_dict({key: cleaned[key] for key in expected}, strict=True)
    return {"role": role, "missing": missing, "unexpected_allowed": unexpected,
            "loaded_keys": len(expected)}


def load_local_vjepa2_ac(checkout, checkpoint, *, expected_dt, device="cpu"):
    checkout = Path(checkout).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    if not (checkout / "hubconf.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("explicit official local checkout and checkpoint required")
    # weights_only avoids arbitrary pickle globals; no unsafe load fallback.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not {"encoder", "predictor"} <= state.keys():
        raise ValueError("not a V-JEPA2-AC encoder+predictor checkpoint")
    # source=local and pretrained=False prohibit torch.hub network weight fetching.
    encoder, predictor = torch.hub.load(str(checkout), "vjepa2_ac_vit_giant",
                                        source="local", pretrained=False)
    receipts = [load_checked_state(encoder, state["encoder"], role="encoder",
                                  allowed_unexpected=("pos_embed",)),
                load_checked_state(predictor, state["predictor"], role="predictor")]
    encoder, predictor = encoder.eval().to(device), predictor.eval().to(device)
    for model in (encoder, predictor):
        model.requires_grad_(False)
    git = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    receipt = {"checkout": str(checkout), "checkpoint": str(checkpoint),
               "checkpoint_bytes": checkpoint.stat().st_size,
               "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
               "git_commit": git.stdout.strip() if git.returncode == 0 else None,
               "weights": receipts, "expected_dt": expected_dt,
               "timing_status": "explicit configuration, not verified PiPER calibration",
               "robot_validation": "not established by loading"}
    return encoder, ACPredictor(predictor, expected_dt=expected_dt), receipt
