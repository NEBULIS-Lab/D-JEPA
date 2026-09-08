"""Source-faithful fixed-25-control PushT prediction and physics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from djepa.evaluation.inference import ordinal_ranks
from djepa.evaluation.native_scoring import native_terminal_cost


CANDIDATE_COUNT = 63
CONTROL_HORIZON = 25
ACTION_BLOCK = 5
FUTURE_BLOCKS = 5
LATENT_DIM = 192


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    _require(result.dtype.kind != "O", f"{name} cannot use object dtype")
    return result


def _validate_common(inputs: Mapping[str, Any], *, pixels: bool, physics: bool) -> int:
    required = {"start_ids", "candidate_ids", "candidate_actions"}
    if pixels:
        required |= {"history_actions", "context_pixels", "goal_pixels"}
    if physics:
        required |= {"initial_state", "goal_state"}
    _require(isinstance(inputs, Mapping) and required <= set(inputs), f"PushT inputs require {sorted(required)}")
    starts = _array(inputs["start_ids"], "start_ids")
    ids = _array(inputs["candidate_ids"], "candidate_ids")
    actions = _array(inputs["candidate_actions"], "candidate_actions")
    _require(starts.ndim == 1 and starts.dtype == np.int64 and len(starts) > 0 and len(set(starts.tolist())) == len(starts), "start_ids must be nonempty unique int64")
    batch = len(starts)
    _require(ids.dtype == np.int64 and ids.shape == (batch, CANDIDATE_COUNT), "candidate_ids must be int64 with shape (B,63)")
    _require(bool(np.all(ids > 0)) and all(len(set(row.tolist())) == CANDIDATE_COUNT for row in ids), "candidate IDs must be positive and unique per start")
    _require(actions.shape == (batch, CANDIDATE_COUNT, CONTROL_HORIZON, 2) and actions.dtype.kind == "f", "candidate_actions must be floating point with shape (B,63,25,2)")
    _require(bool(np.isfinite(actions).all()) and bool(np.all(actions >= -1.0)) and bool(np.all(actions <= 1.0)), "candidate actions must be finite original PushT relative controls in [-1,1]")
    if pixels:
        history = _array(inputs["history_actions"], "history_actions")
        context = _array(inputs["context_pixels"], "context_pixels")
        goal = _array(inputs["goal_pixels"], "goal_pixels")
        _require(history.shape == (batch, 10, 2) and history.dtype.kind == "f" and np.isfinite(history).all(), "history_actions must be finite floating point with shape (B,10,2)")
        _require(context.ndim == 5 and context.shape[:2] == (batch, 3) and context.shape[-1] == 3 and context.dtype == np.uint8, "context_pixels must be uint8 RGB with shape (B,3,H,W,3)")
        _require(goal.ndim == 4 and goal.shape[0] == batch and goal.shape[-1] == 3 and goal.dtype == np.uint8 and goal.shape[1:3] == context.shape[2:4], "goal_pixels must align with context RGB resolution")
    if physics:
        initial = _array(inputs["initial_state"], "initial_state")
        goal_state = _array(inputs["goal_state"], "goal_state")
        _require(initial.shape == goal_state.shape == (batch, 7) and initial.dtype.kind == goal_state.dtype.kind == "f", "initial_state and goal_state must be floating point (B,7)")
        _require(np.isfinite(initial).all() and np.isfinite(goal_state).all(), "PushT physical states must be finite")
    return batch


def candidate_action_sha256(actions: Any) -> np.ndarray:
    """Bind every candidate to its 25 raw controls in canonical float32 bytes."""
    values = _array(actions, "candidate_actions")
    _require(values.ndim == 4 and values.shape[1:] == (CANDIDATE_COUNT, CONTROL_HORIZON, 2), "candidate action digest requires shape (B,63,25,2)")
    output = np.empty(values.shape[:2], dtype="S64")
    for row in range(values.shape[0]):
        for candidate in range(CANDIDATE_COUNT):
            canonical = np.ascontiguousarray(values[row, candidate], dtype="<f4")
            output[row, candidate] = hashlib.sha256(canonical.tobytes()).hexdigest().encode("ascii")
    return output


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _predictions(model: torch.nn.Module, context: torch.Tensor, goals: torch.Tensor, actions: torch.Tensor):
    per_start: list[tuple[torch.Tensor, ...]] = []
    for index in range(context.shape[0]):
        values: list[torch.Tensor] = []
        for backbone in (model.lewm, model.tdjepa):
            output = backbone.rollout({"pixels": context[index : index + 1].unsqueeze(1)}, actions[index : index + 1])
            _require(isinstance(output, Mapping) and "predicted_emb" in output, "embedded backbone rollout lacks predicted_emb")
            future = output["predicted_emb"]
            _require(isinstance(future, torch.Tensor) and future.ndim == 4 and future.shape[-2] >= FUTURE_BLOCKS and future.shape[-1] == LATENT_DIM, "embedded backbone future schema changed")
            future = future[..., -FUTURE_BLOCKS:, :]
            encoded = backbone.encode({"pixels": goals[index : index + 1, None]})
            _require(isinstance(encoded, Mapping) and "emb" in encoded, "embedded backbone goal encoder lacks emb")
            goal = encoded["emb"][:, 0]
            _require(future.shape == (1, CANDIDATE_COUNT, FUTURE_BLOCKS, LATENT_DIM) and goal.shape == (1, LATENT_DIM), "embedded backbone latent shapes changed")
            values.extend((future, goal))
        raw_lewm_future, raw_lewm_goal, raw_tdjepa_future, raw_tdjepa_goal = values
        # The reviewed path computes costs in float32 before descriptor storage
        # is rounded through float16.  Reversing these two operations can change
        # boundary ranks.
        per_start.append((
            raw_lewm_future.half().float(),
            raw_lewm_goal.half().float(),
            raw_tdjepa_future.half().float(),
            raw_tdjepa_goal.half().float(),
            native_terminal_cost(raw_lewm_future.float(), raw_lewm_goal.float()),
            native_terminal_cost(raw_tdjepa_future.float(), raw_tdjepa_goal.float()),
        ))
    return tuple(torch.cat([row[index] for row in per_start], dim=0) for index in range(6))


@torch.no_grad()
def prepare_pusht_features(model: torch.nn.Module, inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Run both embedded predictors from raw PushT observations and controls.

    ``model`` must be the strict-loaded ``ExactRealizationWorldModel`` (or an
    interface-compatible instance) with its pretrained TD-JEPA and LeWM
    predictors attached.  This path never consumes released cached futures.
    """
    batch = _validate_common(inputs, pixels=True, physics=False)
    _require(callable(getattr(model, "forward_from_futures", None)) and model.tdjepa is not None and model.lewm is not None, "feature preparation requires an instantiated full exact-realization model")
    device = _model_device(model)
    context_pixels = torch.as_tensor(inputs["context_pixels"], device=device)
    goal_pixels = torch.as_tensor(inputs["goal_pixels"], device=device)
    history_actions = torch.as_tensor(inputs["history_actions"], device=device)
    candidate_actions = torch.as_tensor(inputs["candidate_actions"], device=device)
    candidate_ids = torch.as_tensor(inputs["candidate_ids"], dtype=torch.int64, device=device)
    context = model._preprocess_pixels(context_pixels, context=True)
    goals = model._preprocess_pixels(goal_pixels, context=False)
    model_actions = model._assemble_actions(history_actions, candidate_actions)
    lewm_future, lewm_goal, td_future, td_goal, lewm_cost, td_cost = _predictions(model, context, goals, model_actions)
    exact = model.forward_from_futures(lewm_future, lewm_goal, td_future, td_goal, lewm_cost, td_cost, candidate_ids)

    ids_np = np.asarray(inputs["candidate_ids"], dtype=np.int64)
    lewm_rank = ordinal_ranks(lewm_cost.cpu().numpy().astype(np.float64), ids_np).astype(np.float32)
    td_rank = ordinal_ranks(td_cost.cpu().numpy().astype(np.float64), ids_np).astype(np.float32)
    features = torch.cat((
        F.layer_norm(lewm_future[:, :, -1] - lewm_goal[:, None], (LATENT_DIM,)),
        F.layer_norm(td_future[:, :, -1] - td_goal[:, None], (LATENT_DIM,)),
        torch.as_tensor(lewm_rank, device=device)[..., None],
        torch.as_tensor(td_rank, device=device)[..., None],
    ), dim=-1)
    final = exact.final_gated_score.detach().cpu().numpy()
    positions = np.asarray([np.lexsort((ids_np[row], final[row]))[0] for row in range(batch)], dtype=np.int64)
    rows = np.arange(batch)

    def host(value: torch.Tensor, dtype: Any | None = None) -> np.ndarray:
        result = value.detach().cpu().numpy()
        return result.astype(dtype, copy=False) if dtype is not None else result

    return {
        "start_ids": np.asarray(inputs["start_ids"], dtype=np.int64).copy(),
        "candidate_ids": ids_np.copy(),
        "candidate_action_sha256": candidate_action_sha256(inputs["candidate_actions"]),
        "lewm_future": host(lewm_future, np.float16),
        "lewm_goal": host(lewm_goal, np.float16),
        "lewm_cost": host(lewm_cost, np.float32),
        "tdjepa_future": host(td_future, np.float16),
        "tdjepa_goal": host(td_goal, np.float16),
        "tdjepa_cost": host(td_cost, np.float32),
        "features": host(features, np.float32),
        "base_scores": host(exact.base_score, np.float64),
        "refined_scores": host(exact.refined_score, np.float32),
        "final_gated_scores": final.astype(np.float64, copy=False),
        "gate_active": host(exact.gate_active, np.bool_),
        "lifted_future": host(exact.future, np.float32),
        "native_cost": host(exact.native_cost, np.float32),
        "selected_positions": positions,
        "selected_ids": ids_np[rows, positions].copy(),
    }


def _decode_digests(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    _require(raw.ndim == 2 and raw.shape[1] == CANDIDATE_COUNT and raw.dtype.kind in "SU", "candidate action identity matrix is invalid")
    return raw.astype("U64")


def _check_official_timing(env: Any) -> None:
    control_hz = getattr(env, "control_hz", getattr(env, "metadata", {}).get("render_fps"))
    _require(float(control_hz) == 10.0, "PushT environment control rate differs from 10 Hz")
    if hasattr(env, "dt"):
        _require(float(env.dt) == 0.01, "PushT physics dt differs from 0.01 seconds")
        _require(int(1 / (float(env.dt) * float(control_hz))) == 10, "PushT integration steps per control differ from 10")


def evaluate_pusht_rollouts(env: Any, inputs: Mapping[str, Any], decisions: Mapping[str, Any], *, capture_frames: bool = False) -> dict[str, np.ndarray]:
    """Execute selected literal candidates for all 25 controls in PushT physics."""
    batch = _validate_common(inputs, pixels=False, physics=True)
    _require(isinstance(decisions, Mapping), "decisions must be a mapping")
    required = {"start_ids", "candidate_ids", "candidate_action_sha256", "selected_ids"}
    _require(required <= set(decisions), f"decisions require {sorted(required)}")
    starts = np.asarray(inputs["start_ids"], dtype=np.int64)
    ids = np.asarray(inputs["candidate_ids"], dtype=np.int64)
    _require(np.array_equal(np.asarray(decisions["start_ids"]), starts), "decision start identity differs from rollout inputs")
    _require(np.array_equal(np.asarray(decisions["candidate_ids"]), ids), "decision candidate identity differs from rollout inputs")
    actual_digest = candidate_action_sha256(inputs["candidate_actions"]).astype("U64")
    _require(np.array_equal(_decode_digests(decisions["candidate_action_sha256"]), actual_digest), "candidate action identity differs from scored candidates")
    selected_ids = np.asarray(decisions["selected_ids"])
    _require(selected_ids.dtype.kind in "iu" and selected_ids.shape == (batch,), "selected_ids must be one integer literal per start")
    positions = np.empty(batch, dtype=np.int64)
    for row in range(batch):
        matches = np.flatnonzero(ids[row] == selected_ids[row])
        _require(len(matches) == 1, "selected literal candidate is unavailable")
        positions[row] = int(matches[0])
    _check_official_timing(env)

    actions = np.asarray(inputs["candidate_actions"], dtype=np.float32)
    initial = np.asarray(inputs["initial_state"], dtype=np.float64)
    goals = np.asarray(inputs["goal_state"], dtype=np.float64)
    trajectories = np.empty((batch, CONTROL_HORIZON + 1, 7), dtype=np.float32)
    rewards = np.empty((batch, CONTROL_HORIZON), dtype=np.float32)
    terminated = np.empty((batch, CONTROL_HORIZON), dtype=np.bool_)
    truncated = np.empty((batch, CONTROL_HORIZON), dtype=np.bool_)
    success = np.empty(batch, dtype=np.bool_)
    final_distance = np.empty(batch, dtype=np.float32)
    frame_rows: list[np.ndarray] = []
    selected_actions = np.empty((batch, CONTROL_HORIZON, 2), dtype=np.float32)
    for row in range(batch):
        env.reset(seed=0, options={"state": initial[row], "goal_state": goals[row]})
        trajectories[row, 0] = np.asarray(env._get_obs(), dtype=np.float32)
        selected_actions[row] = actions[row, positions[row]]
        frames: list[np.ndarray] = []
        for step, action in enumerate(selected_actions[row]):
            result = env.step(action)
            _require(isinstance(result, tuple) and len(result) == 5, "PushT step must follow the Gymnasium five-value API")
            _, reward, done, was_truncated, _ = result
            rewards[row, step] = float(reward)
            terminated[row, step] = bool(done)
            truncated[row, step] = bool(was_truncated)
            trajectories[row, step + 1] = np.asarray(env._get_obs(), dtype=np.float32)
            if capture_frames:
                frame = np.asarray(env.render())
                _require(frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8, "PushT render must return uint8 RGB")
                frames.append(frame.copy())
        success[row], final_distance[row] = env.eval_state(goals[row], np.asarray(env._get_obs(), dtype=np.float64))
        if capture_frames:
            frame_rows.append(np.stack(frames))
    result = {
        "start_ids": starts.copy(),
        "candidate_ids": ids.copy(),
        "candidate_action_sha256": candidate_action_sha256(inputs["candidate_actions"]),
        "selected_ids": selected_ids.astype(np.int64, copy=True),
        "selected_positions": positions,
        "selected_actions": selected_actions,
        "selected_action_sha256": actual_digest[np.arange(batch), positions].astype("S64"),
        "trajectory_states": trajectories,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "success": success,
        "final_distance": final_distance,
    }
    if capture_frames:
        try:
            result["frames"] = np.stack(frame_rows)
        except ValueError as error:
            raise ValueError("PushT render resolution changed between rollouts") from error
    return result


def make_official_pusht_environment() -> Any:
    """Instantiate the recorded stable-worldmodel PushT environment."""
    try:
        from stable_worldmodel.envs.pusht.env import PushT
    except ImportError as error:
        raise RuntimeError("stable-worldmodel with PushT dependencies is required for native physics") from error
    return PushT(resolution=224, render_mode="rgb_array", relative=True, with_target=True)


def load_recorded_pusht_model(
    checkpoint_dir: str | Path,
    *,
    tdjepa_template: str | Path,
    lewm_template: str | Path,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Instantiate recorded upstream classes, then strict-load the full export.

    The two template checkpoints supply the original Python architectures.  All
    learned tensors are subsequently replaced by ``pusht-exact-realization-full``.
    Their hashes are nevertheless checked against that profile's lineage so a
    different upstream realization cannot be substituted silently.
    """
    profile = Path(checkpoint_dir)
    config = json.loads((profile / "config.json").read_text(encoding="utf-8"))
    lineage = config.get("lineage", {}).get("source_sha256", {})
    from .config import sha256_file

    td_path, lewm_path = Path(tdjepa_template), Path(lewm_template)
    _require(sha256_file(td_path) == lineage.get("tdjepa_weights"), "TD-JEPA template SHA-256 differs from recorded upstream")
    _require(sha256_file(lewm_path) == lineage.get("lewm_object"), "LeWM template SHA-256 differs from recorded upstream")
    try:
        from stable_worldmodel.wm.utils import load_pretrained
    except ImportError as error:
        raise RuntimeError("recorded stable-worldmodel/TD-JEPA Python packages are not importable") from error
    tdjepa = load_pretrained(str(td_path)).eval().requires_grad_(False)
    # This is the original trusted LeWM object checkpoint named by a pinned hash,
    # not user-provided arbitrary pickle data.
    lewm = torch.load(lewm_path, map_location="cpu", weights_only=False).eval().requires_grad_(False)
    from djepa.checkpoints import load_exact_world_model

    return load_exact_world_model(profile, tdjepa=tdjepa, lewm=lewm).to(device)


__all__ = [
    "candidate_action_sha256",
    "evaluate_pusht_rollouts",
    "make_official_pusht_environment",
    "load_recorded_pusht_model",
    "prepare_pusht_features",
]
