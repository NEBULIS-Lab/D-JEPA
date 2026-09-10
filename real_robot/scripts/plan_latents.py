"""Offline AC planning from explicitly prepared latents; NO hardware execution.

Either run policy-free CEM or provide already-converted, calibrated proposals in
the input NPZ. Raw π0.5 joint commands are deliberately not accepted here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from djepa_robot.actions import CandidateBatch, SEMANTICS
from djepa_robot.planning import cem
from djepa_robot.training import SYNTHETIC_SCOPE, load_alignment, source_receipt
from djepa_robot.world_model import load_local_vjepa2_ac, native_cost, spatial_features
from .common import new_output, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkout", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/offline_planning.json"))
    p.add_argument("--alignment-checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args()
    with np.load(args.cache, allow_pickle=False) as f:
        if "action_semantics" not in f or str(f["action_semantics"].item()) != SEMANTICS:
            raise ValueError("action semantics mismatch: raw joint commands cannot enter Cartesian prediction")
        required = {"context", "goal", "pose", "dt", "source_model", "data_origin"}
        if not required <= set(f.files):
            raise ValueError(f"missing latent cache fields: {sorted(required - set(f.files))}")
        cache = {k: f[k] for k in f.files}
    if Path(args.output).exists():
        raise ValueError("output exists; select a new directory")
    with open(args.config) as f:
        config = json.load(f)
    dt = float(cache["dt"].item())
    if not np.isfinite(dt) or dt <= 0 or not np.isclose(dt, config["model_dt"], rtol=1e-6, atol=1e-9):
        raise ValueError("cache dt differs from model transition dt")
    origin = str(cache["data_origin"].item())
    if origin not in ("recorded_real", "recorded_sim", "synthetic_test"):
        raise ValueError("data_origin must identify recorded real/simulation or synthetic test latents")
    if origin == "synthetic_test" and not args.allow_synthetic:
        raise ValueError("synthetic fixtures require explicit opt-in")
    checkpoint_path = str(Path(args.checkpoint).resolve(strict=True))
    if str(cache["source_model"].item()) != checkpoint_path:
        raise ValueError("cached latents must declare the exact source model checkpoint path")
    context = torch.as_tensor(cache["context"], dtype=torch.float32, device=args.device)
    goal = torch.as_tensor(cache["goal"], dtype=torch.float32, device=args.device)
    pose = torch.as_tensor(cache["pose"], dtype=torch.float32, device=args.device)
    if context.ndim != 3 or context.shape[0] != 1 or goal.shape != context.shape or pose.shape != (1, 7):
        raise ValueError("context/goal [1,P,D] and pose [1,7] required")
    if any(not torch.isfinite(x).all() for x in (context, goal, pose)):
        raise ValueError("nonfinite latent inputs")
    chunk_size = config.get("chunk_size", 4)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("positive integer chunk size required")
    proposal = None
    if "actions" in cache or "candidate_ids" in cache:
        if not {"actions", "candidate_ids"} <= cache.keys():
            raise ValueError("external proposals need both actions and candidate_ids")
        proposal = CandidateBatch(cache["candidate_ids"], cache["actions"], SEMANTICS, dt)
        proposal.validate_bounds(config["lower"], config["upper"])
        if proposal.actions.shape[1] != config["horizon"]:
            raise ValueError("external proposal horizon differs from configured horizon")
    encoder, model, model_receipt = load_local_vjepa2_ac(args.checkout, args.checkpoint,
                                                       expected_dt=config["model_dt"], device=args.device)
    del encoder  # Input latents already encoded; do not retain the giant encoder on GPU.

    def evaluate(batch, keep_terminal=False):
        costs, terminals = [], []
        for begin in range(0, len(batch.ids), chunk_size):
            small = CandidateBatch(batch.ids[begin:begin + chunk_size], batch.actions[begin:begin + chunk_size],
                                   batch.semantics, batch.dt)
            terminal = model.rollout(context, pose, small)[:, -1]
            costs.append(native_cost(terminal, goal).cpu().numpy())
            if keep_terminal:
                terminals.append(terminal.cpu())
        return np.concatenate(costs), torch.cat(terminals) if keep_terminal else None

    if proposal is None:
        result = cem(lambda b: evaluate(b)[0], lower=config["lower"], upper=config["upper"],
                     horizon=config["horizon"], samples=config["samples"], elites=config["elites"],
                     iterations=config["iterations"], seed=config["seed"], dt=dt)
        pool = result.final_batch
        # Retain the best actually evaluated native candidate, if from an earlier population.
        if result.candidate_id not in pool.ids:
            pool = CandidateBatch(np.append(pool.ids, result.candidate_id),
                                   np.concatenate((pool.actions, result.actions[None])), SEMANTICS, dt)
        route, history = "policy_free_native_CEM_then_shared_pool_scoring", result.history
    else:
        pool, route, history = proposal, "explicit_converted_external_proposals", []
    costs, terminal = evaluate(pool, keep_terminal=True)
    native_idx = int(np.lexsort((pool.ids, costs))[0])
    scores, aligned_idx, alignment_receipt = costs.copy(), None, None
    if args.alignment_checkpoint:
        head, metadata = load_alignment(args.alignment_checkpoint, allow_synthetic=args.allow_synthetic)
        if metadata["source_model"] != checkpoint_path or metadata["feature_spec"] != "spatial_pool_2x2":
            raise ValueError("alignment checkpoint feature/model provenance does not match this rollout")
        if metadata["artifact_scope"] == SYNTHETIC_SCOPE and origin != "synthetic_test":
            raise ValueError("synthetic alignment weights cannot be used on recorded robot/simulation inputs")
        with torch.inference_mode():
            aligned = head(spatial_features(terminal)[None, :, None, :],
                           spatial_features(goal.cpu())[:, None, :],
                           torch.tensor(pool.ids.copy())[None],
                           native_costs=torch.tensor(costs, dtype=torch.float32)[None, :, None])
        scores = aligned["scores"][0].numpy()
        aligned_idx = int(np.lexsort((pool.ids, scores))[0])
        alignment_receipt = source_receipt(args.alignment_checkpoint)
    chosen = native_idx if aligned_idx is None else aligned_idx
    output = new_output(args.output)
    with (output / "candidate_pool.npz").open("xb") as f:
        np.savez_compressed(f, candidate_ids=pool.ids, actions=pool.actions,
                            native_costs=costs, selection_scores=scores, terminal=terminal.numpy(),
                            dt=np.array(dt), action_semantics=np.array(SEMANTICS))
    write_json(output / "plan.json", {
        "artifact_scope": SYNTHETIC_SCOPE if origin == "synthetic_test" else "offline_prediction_not_robot_success",
        "hardware_execution": False, "route": route, "config": config, "data_origin": origin,
        "input_cache": source_receipt(args.cache), "world_model": model_receipt,
        "alignment_checkpoint": alignment_receipt, "candidate_count": len(pool.ids),
        "native_selected_id": int(pool.ids[native_idx]),
        "aligned_selected_id": None if aligned_idx is None else int(pool.ids[aligned_idx]),
        "selected_prediction_space_actions": pool.actions[chosen].tolist(),
        "action_warning": "NOT PiPER motor commands; do not send to hardware",
        "cem_history": history, "set_score_scope": "one fixed pool shared by both scorers"})
    print(output / "plan.json")


if __name__ == "__main__":
    main()
