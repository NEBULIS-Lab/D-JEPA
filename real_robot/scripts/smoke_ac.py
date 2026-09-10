"""Small real-checkpoint check on the official recorded Franka example, offline."""

import argparse
from pathlib import Path

import numpy as np
import torch

from djepa_robot.actions import CandidateBatch, SEMANTICS, pose_delta
from djepa_robot.training import source_receipt
from djepa_robot.vision import encode_rgb_frames, official_ac_transform
from djepa_robot.world_model import load_local_vjepa2_ac, native_cost
from .common import new_output, write_json


def reference_diagnostics(model, context, goal, states):
    if states.shape != (2, 7):
        raise ValueError("official paired states [2,7] required")
    delta = pose_delta(states[0], states[1])
    candidate_actions = np.stack((delta, np.zeros(7), -delta))[:, None]
    batch = CandidateBatch(np.arange(3), candidate_actions, SEMANTICS, model.expected_dt)
    future = model.rollout(context, torch.tensor(states[:1], dtype=context.dtype, device=context.device), batch)
    costs = native_cost(future[:, -1], goal)
    response = (future[0] - future[1]).abs().mean()
    return {"artifact_scope": "official_reference_checkpoint_smoke_not_PiPER_evaluation",
            "hardware_execution": False, "robot_success_rate": None,
            "future_shape": list(future.shape), "all_finite": bool(torch.isfinite(future).all()),
            "candidate_definitions": ["delta_between_recorded_poses", "zero_delta", "negated_recorded_delta"],
            "pose_delta_is": "recorded pose difference, not an original actuator command log",
            "costs_to_recorded_goal": costs.cpu().tolist(),
            "recorded_transition_prediction_l1": float(costs[0]),
            "copy_current_latent_l1": float(native_cost(context, goal)[0]),
            "action_response_l1": float(response), "action_response_nonzero": bool(response > 0),
            "counterfactual_outcomes": "unknown",
            "timing": "reference NPZ has no timestamps; configured dt is an interface value, not measured timing"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkout", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if Path(args.output).exists():
        p.error("output exists; choose a new run directory")
    example = Path(args.checkout).resolve(strict=True) / "notebooks/franka_example_traj.npz"
    with np.load(example, allow_pickle=False) as f:
        frames, states = f["observations"], f["states"]
    if frames.shape != (1, 2, 256, 256, 3) or states.shape != (1, 2, 7):
        raise ValueError("unexpected official example schema")
    transform = official_ac_transform(args.checkout)
    print("Loading explicit local AC checkpoint", flush=True)
    encoder, model, receipt = load_local_vjepa2_ac(args.checkout, args.checkpoint, expected_dt=0.1,
                                                 device=args.device)
    features = encode_rgb_frames(encoder, frames[0], transform, batch_size=1, device=args.device)
    del encoder
    print("Encoded official reference pair; predicting three action-conditioned futures", flush=True)
    result = reference_diagnostics(model, features[:1].to(args.device), features[1:].to(args.device), states[0])
    result["world_model"], result["reference_data"] = receipt, source_receipt(example)
    output = new_output(args.output)
    write_json(output / "smoke.json", result)
    print(output / "smoke.json", flush=True)


if __name__ == "__main__":
    main()
