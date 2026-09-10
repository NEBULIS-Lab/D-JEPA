"""Small retrospective native-AC diagnostic, not a deployed D-JEPA evaluation."""

import argparse
from pathlib import Path

import numpy as np
import torch

from djepa_robot.actions import CandidateBatch, SEMANTICS
from djepa_robot.recorded_windows import cached_transition_pairs
from djepa_robot.training import source_receipt
from djepa_robot.world_model import ACPredictor, load_local_vjepa2_ac, native_cost
from .common import new_output, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", required=True)
    p.add_argument("--visual", nargs="+", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkout", default="vendor/vjepa2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-nominal-flange", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if not args.allow_nominal_flange:
        p.error("this diagnostic requires explicit acknowledgement of uncalibrated nominal flange geometry")
    if Path(args.output).exists():
        p.error("output exists; choose a new run directory")
    with np.load(args.motion, allow_pickle=False) as f:
        motion = {k: f[k] for k in f.files}
    prepared = []
    for path in args.visual:
        with np.load(path, allow_pickle=False) as f:
            visual = {k: f[k] for k in f.files}
        if str(visual["source_model"].item()) != str(Path(args.checkpoint).resolve(strict=True)):
            raise ValueError("visual checkpoint provenance mismatch")
        ids = visual["frame_indices"]
        if not np.array_equal(visual["robot_timestamp_ns"], motion["timestamp_ns"][ids]):
            raise ValueError("visual/motion recorded timestamps do not match")
        if not np.allclose(visual["joint_state"], motion["joint_state"][ids], atol=0, rtol=0):
            raise ValueError("visual/motion recorded states do not match")
        pairs = cached_transition_pairs(ids, motion["timestamp_ns"], min_dt=.08, max_dt=.15)
        prepared.append((path, visual, pairs))
    if not any(pairs for _, _, pairs in prepared):
        raise ValueError("no timing-valid consecutive cached frames")
    print("Loading official checkpoint for recorded-window diagnostic", flush=True)
    encoder, model, receipt = load_local_vjepa2_ac(args.checkout, args.checkpoint, expected_dt=.1, device=args.device)
    del encoder
    rows = []
    for path, visual, pairs in prepared:
        for i, j, frame in pairs:
            dt = float(motion["timestamp_ns"][frame+1] - motion["timestamp_ns"][frame]) * 1e-9
            local_model = ACPredictor(model.predictor, expected_dt=dt)
            # Different analyses, not interchangeable action labels:
            # measured next-pose difference uses future feedback (retrospective);
            # target difference uses the recorded command, with tracking error.
            actions = np.stack((motion["observed_transition_delta"][frame],
                                motion["command_target_delta"][frame], np.zeros(7)))[:, None]
            batch = CandidateBatch(np.arange(3), actions, SEMANTICS, dt)
            context = torch.tensor(visual["latents"][i:i+1], device=args.device)
            goal = torch.tensor(visual["latents"][j:j+1], device=args.device)
            pose = torch.tensor(motion["nominal_flange_pose"][frame:frame+1], dtype=context.dtype, device=args.device)
            future = local_model.rollout(context, pose, batch)
            costs = native_cost(future[:, -1], goal).cpu().tolist()
            rows.append({"camera": str(visual["camera"].item()), "frame": frame, "next_frame": frame+1,
                         "actual_dt_s": dt, "realized_motion_conditioned_l1": costs[0],
                         "command_target_conditioned_l1": costs[1], "zero_action_conditioned_l1": costs[2],
                         "copy_current_latent_l1": float(native_cost(context, goal)[0]),
                         "state_within_nominal_limits": not bool(motion["state_joint_limit_violations"][frame:frame+2].any()),
                         "command_within_nominal_limits": not bool(motion["command_joint_limit_violations"][frame].any())})
            print(rows[-1], flush=True)
    output = new_output(args.output)
    write_json(output / "prediction_diagnostic.json", {
        "artifact_scope": "retrospective_nominal_PiPER_native_AC_diagnostic_not_D_JEPA_success",
        "hardware_execution": False, "hardware_calibrated": False, "robot_success_rate": None,
        "motion_source": source_receipt(args.motion), "visual_sources": [source_receipt(x) for x in args.visual],
        "world_model": receipt, "rows": rows,
        "interpretation": "realized-motion conditioning uses future feedback; command-conditioned branch does not",
        "timing": "measured dt retained; stock AC predictor has no explicit dt input and is not timing-adapted",
        "sample_identity": "camera views of the same transitions are not independent trials",
        "counterfactual_outcomes": "unknown; zero-action prediction compared to recorded target only as diagnostic"})
    print(output / "prediction_diagnostic.json", flush=True)


if __name__ == "__main__":
    main()
