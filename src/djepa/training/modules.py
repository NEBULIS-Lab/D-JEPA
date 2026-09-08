"""Faithful training over complete public D-JEPA supervision inputs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.supervision import require_disjoint
from ..models.granular import GranularSetRanker
from ..models.ordinal import DecisionAligner
from ..models.relational import (
    calibrate_confidence_threshold,
    gated_candidate_ids,
    relational_loss,
)
from ..objectives.granular import masked_refinement_loss


CANDIDATE_COUNT = 63
SUPPORTED_MODULES = (
    "pusht-relational",
    "granular-relational",
    "granular-multiview",
)
PROTECTED_SPLITS = frozenset({"development", "formal", "independent", "independent_256"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_hashes(directory: Path) -> dict[str, str]:
    return {
        name: _sha256(directory / name)
        for name in ("inputs.npz", "labels.npz", "metadata.json")
    }


def _metadata(directory: Path, *, task: str, split: str) -> dict[str, Any]:
    directory = Path(directory)
    for name in ("inputs.npz", "labels.npz", "metadata.json"):
        _require((directory / name).is_file(), f"missing released split member: {name}")
    try:
        value = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("released split metadata is invalid") from error
    _require(isinstance(value, dict), "released split metadata must be a JSON object")
    actual_split = value.get("split")
    _require(actual_split not in PROTECTED_SPLITS, f"protected split cannot be used for training: {actual_split}")
    _require(value.get("task") == task and actual_split == split, f"expected {task}/{split} released split")
    _require(type(value.get("count")) is int and value["count"] > 0, "metadata count is invalid")
    _require(value.get("candidate_count") == CANDIDATE_COUNT, "metadata candidate count is invalid")
    return value


def _candidate_contract(ids: np.ndarray, rows: int) -> None:
    _require(ids.dtype == np.int64 and ids.shape == (rows, CANDIDATE_COUNT), "candidate IDs must be int64 (N,63)")
    _require(bool((ids > 0).all()), "candidate IDs must be positive")
    _require(all(np.unique(row).size == CANDIDATE_COUNT for row in ids), "candidate IDs must be unique within each row")


@dataclass(frozen=True)
class SplitData:
    features: np.ndarray
    base_scores: np.ndarray
    candidate_ids: np.ndarray
    success: np.ndarray
    supervision_mask: np.ndarray
    identities: np.ndarray
    multiview_ranks: np.ndarray | None
    metadata: dict[str, Any]
    hashes: dict[str, str]


def _load_pusht_pool(directory: Path) -> tuple[SplitData, np.ndarray, np.ndarray]:
    root = Path(directory)
    metadata = _metadata(root, task="pusht", split="train")
    with np.load(root / "inputs.npz", allow_pickle=False) as inputs:
        required = {"features", "base_scores", "candidate_ids", "start_ids"}
        _require(required.issubset(inputs.files), "PushT pool input arrays are incomplete")
        features = np.asarray(inputs["features"])
        base = np.asarray(inputs["base_scores"])
        ids = np.asarray(inputs["candidate_ids"])
        identities = np.asarray(inputs["start_ids"])
    with np.load(root / "labels.npz", allow_pickle=False) as labels:
        _require("success" in labels.files, "PushT pool success labels are missing")
        success = np.asarray(labels["success"])
    rows = int(metadata["count"])
    _require(features.shape == (rows, CANDIDATE_COUNT, 386) and np.issubdtype(features.dtype, np.floating), "PushT relational features must be floating (N,63,386)")
    _require(base.shape == (rows, CANDIDATE_COUNT) and np.issubdtype(base.dtype, np.floating), "PushT base scores must be floating (N,63)")
    _candidate_contract(ids, rows)
    _require(identities.shape == (rows,) and identities.dtype.kind in "iu", "PushT start IDs must be integer (N,)")
    _require(np.unique(identities).size == rows, "PushT start IDs must be unique")
    _require(success.dtype == np.bool_ and success.shape == ids.shape, "PushT success labels must be Boolean (N,63)")
    _require(bool(np.isfinite(features).all()) and bool(np.isfinite(base).all()), "PushT relational inputs must be finite")
    raw_calibration = metadata.get("relational_calibration_ids")
    _require(isinstance(raw_calibration, list) and raw_calibration, "PushT metadata lacks relational calibration IDs")
    calibration_ids = np.asarray(raw_calibration)
    _require(calibration_ids.dtype.kind in "iu" and np.unique(calibration_ids).size == len(calibration_ids), "PushT relational calibration IDs are invalid")
    position = {int(value): index for index, value in enumerate(identities.tolist())}
    _require(all(int(value) in position for value in calibration_ids), "PushT relational calibration ID is absent from the fitting pool")
    calibration_rows = np.asarray([position[int(value)] for value in calibration_ids], dtype=np.int64)
    calibration_set = {int(value) for value in calibration_ids}
    fit_rows = np.asarray([index for index, value in enumerate(identities) if int(value) not in calibration_set], dtype=np.int64)
    _require(fit_rows.size > 0, "PushT relational fitting split is empty")
    require_disjoint(identities[fit_rows], identities[calibration_rows])
    data = SplitData(
        features=np.ascontiguousarray(features, dtype=np.float32),
        base_scores=np.ascontiguousarray(base, dtype=np.float64),
        candidate_ids=np.ascontiguousarray(ids),
        success=np.ascontiguousarray(success),
        supervision_mask=np.ones_like(success, dtype=np.bool_),
        identities=np.array(identities, copy=True),
        multiview_ranks=None,
        metadata=metadata,
        hashes=_source_hashes(root),
    )
    return data, fit_rows, calibration_rows


def _load_granular_split(directory: Path, *, split: str) -> SplitData:
    root = Path(directory)
    metadata = _metadata(root, task="granular", split=split)
    with np.load(root / "inputs.npz", allow_pickle=False) as inputs:
        required = {"features", "base_scores", "candidate_ids", "multiview_ranks", "start_ids"}
        _require(required.issubset(inputs.files), "Granular input arrays are incomplete")
        features = np.asarray(inputs["features"])
        base = np.asarray(inputs["base_scores"])
        ids = np.asarray(inputs["candidate_ids"])
        ranks = np.asarray(inputs["multiview_ranks"])
        identities = np.asarray(inputs["start_ids"])
    with np.load(root / "labels.npz", allow_pickle=False) as labels:
        _require({"success", "supervision_mask"}.issubset(labels.files), "Granular sparse labels are incomplete")
        success = np.asarray(labels["success"])
        mask = np.asarray(labels["supervision_mask"])
    rows = int(metadata["count"])
    _require(features.shape == (rows, CANDIDATE_COUNT, 2305) and np.issubdtype(features.dtype, np.floating), "Granular relational features must be floating (N,63,2305)")
    _require(base.shape == (rows, CANDIDATE_COUNT) and np.issubdtype(base.dtype, np.floating), "Granular base scores must be floating (N,63)")
    _candidate_contract(ids, rows)
    _require(ranks.dtype == np.float64 and ranks.shape == (rows, CANDIDATE_COUNT, 4), "Granular multiview ranks must be float64 (N,63,4)")
    _require(identities.shape == (rows,) and identities.dtype.kind in "US", "Granular start IDs must be fixed-width strings")
    _require(np.unique(identities).size == rows, "Granular start IDs must be unique")
    _require(success.dtype == mask.dtype == np.bool_ and success.shape == mask.shape == ids.shape, "Granular labels and supervision mask must be Boolean (N,63)")
    _require(bool((mask.sum(axis=1) == 16).all()), "Granular release requires exactly 16 supervised candidates per row")
    _require(bool(np.isfinite(features).all()) and bool(np.isfinite(base).all()) and bool(np.isfinite(ranks).all()), "Granular inputs must be finite")
    return SplitData(
        features=np.ascontiguousarray(features, dtype=np.float32),
        base_scores=np.ascontiguousarray(base, dtype=np.float64),
        candidate_ids=np.ascontiguousarray(ids),
        success=np.ascontiguousarray(success),
        supervision_mask=np.ascontiguousarray(mask),
        identities=np.array(identities, copy=True),
        multiview_ranks=np.ascontiguousarray(ranks),
        metadata=metadata,
        hashes=_source_hashes(root),
    )


def _selected_columns(scores: np.ndarray, ids: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(scores)
    visible = np.ones_like(values, dtype=np.bool_) if mask is None else np.asarray(mask)
    _require(values.shape == ids.shape == visible.shape and visible.dtype == np.bool_, "selection arrays must align")
    _require(bool(visible.any(axis=1).all()), "each selection row requires a visible candidate")
    masked = np.where(visible, values, np.inf)
    return np.asarray([np.lexsort((ids[row], masked[row]))[0] for row in range(len(ids))], dtype=np.int64)


def _calibrate_masked(data: SplitData, refined: np.ndarray) -> tuple[float, int]:
    base, ids, labels, mask = data.base_scores, data.candidate_ids, data.success, data.supervision_mask
    base_columns = _selected_columns(base, ids, mask)
    refined_columns = _selected_columns(refined, ids, mask)
    rows = np.arange(len(ids))
    advantage = base[rows, base_columns] - refined[rows, refined_columns]
    unique = np.unique(advantage)
    thresholds: list[float] = [math.inf]
    thresholds.extend(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:]))
    thresholds.append(float(np.nextafter(unique[0], -math.inf)))
    best: tuple[int, int, float] | None = None
    selected_threshold = math.inf
    for threshold in thresholds:
        use_refined = advantage > threshold
        columns = np.where(use_refined, refined_columns, base_columns)
        record = (int(labels[rows, columns].sum()), int(use_refined.sum()), -float(threshold))
        if best is None or record > best:
            best, selected_threshold = record, float(threshold)
    assert best is not None
    return selected_threshold, best[0]


def _simplex_weights(parts: int):
    for first in range(parts + 1):
        for second in range(parts - first + 1):
            for third in range(parts - first - second + 1):
                yield (
                    first / parts,
                    second / parts,
                    third / parts,
                    (parts - first - second - third) / parts,
                )


def _fit_masked_fusion(data: SplitData, *, grid_step: float) -> tuple[float, ...]:
    _require(data.multiview_ranks is not None, "Granular multiview ranks are unavailable")
    parts_float = 1.0 / float(grid_step)
    parts = int(round(parts_float))
    _require(0 < grid_step <= 1 and parts >= 1 and np.isclose(parts_float, parts, rtol=0, atol=1e-12), "MG grid step must divide one exactly")
    ranks, ids, labels, mask = data.multiview_ranks, data.candidate_ids, data.success, data.supervision_mask
    rows = np.arange(len(ids))
    best_key: tuple[int, float, float, tuple[float, ...]] | None = None
    best_weights: tuple[float, ...] | None = None
    for weights in _simplex_weights(parts):
        fused = np.tensordot(ranks, np.asarray(weights, dtype=np.float64), axes=([2], [0]))
        selected = _selected_columns(fused, ids, mask)
        correct = int(labels[rows, selected].sum())
        valid = (labels & mask).any(axis=1) & ((~labels) & mask).any(axis=1)
        positive_min = np.where(labels & mask, fused, np.inf).min(axis=1)
        negative_min = np.where((~labels) & mask, fused, np.inf).min(axis=1)
        margin = float(np.mean((negative_min - positive_min)[valid])) if bool(valid.any()) else 0.0
        key = (correct, round(margin, 12), float(np.square(weights).sum()), tuple(weights))
        if best_key is None or key > best_key:
            best_key, best_weights = key, tuple(float(value) for value in weights)
    assert best_weights is not None
    return best_weights


def _score(model: torch.nn.Module, features: np.ndarray, base: np.ndarray, *, batch_size: int) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            scores, _ = model(
                torch.from_numpy(features[start : start + batch_size]),
                torch.from_numpy(base[start : start + batch_size].astype(np.float32)),
            )
            values.append(scores.float().cpu().numpy().astype(np.float64))
    return np.concatenate(values)


def _training_config(args: Any) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "tail_k": int(args.tail_k),
        "evaluate_every": int(args.evaluate_every),
        "mg_grid_step": float(args.mg_grid_step),
    }


def _validate_args(args: Any) -> None:
    _require(args.module in SUPPORTED_MODULES, f"module must be one of {SUPPORTED_MODULES}")
    _require(min(args.updates, args.batch_size, args.tail_k, args.evaluate_every) > 0, "training schedule values must be positive")
    _require(args.tail_k <= CANDIDATE_COUNT, "tail_k cannot exceed 63")
    values = np.asarray([args.learning_rate, args.weight_decay, args.max_grad_norm, args.max_correction, args.mg_grid_step], dtype=np.float64)
    _require(bool(np.isfinite(values).all()), "training values must be finite")
    _require(args.learning_rate > 0 and args.weight_decay >= 0 and args.max_grad_norm > 0, "optimizer values are invalid")
    _require(args.hidden_dim > 0 and args.hidden_dim % 4 == 0 and args.low_rank > 0 and args.max_correction > 0, "model dimensions are invalid")
    _require(0 < args.mg_grid_step <= 1, "MG grid step must be in (0,1]")


def finite_gate_threshold(threshold: float, *, max_correction: float) -> float:
    """Represent the calibration no-refinement choice with a finite threshold.

    The largest finite float64 value is a JSON-safe positive-infinity sentinel.
    All valid gate advantages are finite, so the strict ``>`` comparison can
    never activate. This also avoids mixed float64/float32 rounding at the
    nominal correction bound.
    """
    value = float(threshold)
    bound = float(max_correction)
    _require(math.isfinite(bound) and bound > 0, "max correction must be finite and positive")
    if math.isinf(value) and value > 0:
        return float(np.finfo(np.float64).max)
    _require(math.isfinite(value), "calibrated gate threshold is invalid")
    return value


def train_released_module(args: Any) -> Path:
    """Train one supported profile and atomically refuse an existing output."""
    _validate_args(args)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")

    if args.module == "pusht-relational":
        _require(args.calibration is None, "PushT relational calibration is derived from pool metadata")
        pool, fit_rows, calibration_rows = _load_pusht_pool(Path(args.train))
        _require(args.batch_size <= len(fit_rows), "batch size exceeds PushT fitting split")
        train = pool
        calibration = pool
        train_features, train_base = pool.features, pool.base_scores
        calibration_features, calibration_base = pool.features, pool.base_scores
        fusion_weights = None
        model: torch.nn.Module = DecisionAligner(386, args.max_correction)
        model_config = {"input_dim": 386, "max_correction": float(args.max_correction)}
        sources = {"pool": pool.hashes}
        splits = {
            "identity_field": "start_ids",
            "fit_ids": [_json_value(value) for value in pool.identities[fit_rows]],
            "calibration_ids": [_json_value(value) for value in pool.identities[calibration_rows]],
        }
    else:
        _require(args.calibration is not None, "Granular training requires an explicit calibration split")
        train = _load_granular_split(Path(args.train), split="train")
        calibration = _load_granular_split(Path(args.calibration), split="calibration")
        require_disjoint(train.identities, calibration.identities)
        _require(args.batch_size <= len(train.identities), "batch size exceeds Granular training split")
        fit_rows = np.arange(len(train.identities), dtype=np.int64)
        calibration_rows = np.arange(len(calibration.identities), dtype=np.int64)
        if args.module == "granular-relational":
            train_features, calibration_features = train.features, calibration.features
            train_base, calibration_base = train.base_scores, calibration.base_scores
            fusion_weights = None
            input_dim = 2305
        else:
            fusion_weights = _fit_masked_fusion(train, grid_step=float(args.mg_grid_step))
            assert train.multiview_ranks is not None and calibration.multiview_ranks is not None
            train_features = np.ascontiguousarray(train.multiview_ranks, dtype=np.float32)
            calibration_features = np.ascontiguousarray(calibration.multiview_ranks, dtype=np.float32)
            train_base = np.tensordot(train.multiview_ranks, np.asarray(fusion_weights), axes=([2], [0]))
            calibration_base = np.tensordot(calibration.multiview_ranks, np.asarray(fusion_weights), axes=([2], [0]))
            input_dim = 4
        model = GranularSetRanker(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            low_rank=args.low_rank,
            max_correction=args.max_correction,
        )
        model_config = {
            "input_dim": input_dim,
            "hidden_dim": int(args.hidden_dim),
            "low_rank": int(args.low_rank),
            "max_correction": float(args.max_correction),
        }
        sources = {"train": train.hashes, "calibration": calibration.hashes}
        splits = {
            "identity_field": "start_ids",
            "fit_ids": [_json_value(value) for value in train.identities],
            "calibration_ids": [_json_value(value) for value in calibration.identities],
        }

    torch.set_num_threads(2)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    # Recreate after seeding so initialization itself is reproducible.
    if args.module == "pusht-relational":
        model = DecisionAligner(386, args.max_correction)
    else:
        model = GranularSetRanker(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output.mkdir(parents=True, exist_ok=False)
    metrics_path = output / "metrics.jsonl"
    best_key: tuple[int, float, int] | None = None
    best_threshold = math.inf
    best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
    with metrics_path.open("x", encoding="utf-8") as metrics_stream:
        for update in range(1, args.updates + 1):
            prefix = "pusht-relational" if args.module == "pusht-relational" else f"granular-{args.module}"
            step_seed = int.from_bytes(hashlib.sha256(f"{prefix}:{args.seed}:{update}".encode()).digest()[:8], "little")
            chosen = np.random.default_rng(step_seed).choice(fit_rows, size=args.batch_size, replace=False)
            index = torch.as_tensor(chosen, dtype=torch.long)
            model.train()
            scores, correction = model(
                torch.from_numpy(train_features)[index],
                torch.from_numpy(train_base.astype(np.float32))[index],
            )
            if args.module == "pusht-relational":
                loss, components = relational_loss(
                    scores,
                    torch.from_numpy(train.success)[index],
                    correction,
                    tail_k=args.tail_k,
                )
            else:
                loss, components = masked_refinement_loss(
                    scores,
                    torch.from_numpy(train.success)[index],
                    correction,
                    torch.from_numpy(train.supervision_mask)[index],
                    tail_k=args.tail_k,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            _require(bool(torch.isfinite(grad_norm)), "training gradient norm became nonfinite")
            optimizer.step()
            row: dict[str, Any] = {
                "update": update,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm.detach()),
                **components,
            }
            if update == 1 or update % args.evaluate_every == 0 or update == args.updates:
                refined = _score(
                    model,
                    calibration_features[calibration_rows],
                    calibration_base[calibration_rows],
                    batch_size=args.batch_size,
                )
                if args.module == "pusht-relational":
                    base = calibration.base_scores[calibration_rows]
                    ids = calibration.candidate_ids[calibration_rows]
                    labels = calibration.success[calibration_rows]
                    threshold = calibrate_confidence_threshold(base, refined, ids, labels)
                    selected = gated_candidate_ids(base, refined, ids, threshold=threshold)
                    positions = np.argmax(ids == selected[:, None], axis=1)
                    correct = int(labels[np.arange(len(labels)), positions].sum())
                else:
                    calibration_view = SplitData(
                        features=calibration_features[calibration_rows],
                        base_scores=np.asarray(calibration_base[calibration_rows]),
                        candidate_ids=calibration.candidate_ids[calibration_rows],
                        success=calibration.success[calibration_rows],
                        supervision_mask=calibration.supervision_mask[calibration_rows],
                        identities=calibration.identities[calibration_rows],
                        multiview_ranks=None,
                        metadata=calibration.metadata,
                        hashes=calibration.hashes,
                    )
                    threshold, correct = _calibrate_masked(calibration_view, refined)
                threshold = finite_gate_threshold(
                    threshold, max_correction=float(args.max_correction)
                )
                trust = float(components["trust_loss"])
                row.update(
                    {
                        "calibration_correct": correct,
                        "calibration_count": int(len(calibration_rows)),
                        "gate_threshold": float(threshold),
                    }
                )
                key = (correct, -trust, -update)
                if best_key is None or key > best_key:
                    best_key = key
                    best_threshold = float(threshold)
                    best_state = copy.deepcopy(
                        {name: value.detach().cpu() for name, value in model.state_dict().items()}
                    )
            metrics_stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            metrics_stream.flush()
    _require(best_key is not None, "training produced no calibration checkpoint")
    model_path = output / "model.pt"
    torch.save(best_state, model_path)
    weights_sha256 = _sha256(model_path)
    training = _training_config(args)
    if args.module == "pusht-relational":
        config = {
            "project": "D-JEPA",
            "profile": args.module,
            "architecture": "set_aligner",
            "input_dim": 386,
            "max_correction": float(args.max_correction),
            "fusion_alpha": float(train.metadata["fusion_alpha"]),
            "gate_threshold": best_threshold,
            "gate_mode": "base_minus_refined",
            "gate_decimals": None,
            "weights_sha256": weights_sha256,
            "training": training,
        }
    else:
        config = {
            "project": "D-JEPA",
            "profile": args.module,
            "architecture": "granular",
            "model": model_config,
            "input_dim": model_config["input_dim"],
            "gate_threshold": best_threshold,
            "gate_mode": "base_minus_refined",
            "gate_decimals": None,
            "fusion_weights": list(fusion_weights) if fusion_weights is not None else None,
            "weights_sha256": weights_sha256,
            "training": training,
        }
    receipt = {
        "schema": "djepa_public_module_training_v1",
        "module": args.module,
        "sources": sources,
        "splits": splits,
        "weights_sha256": weights_sha256,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


__all__ = ["SUPPORTED_MODULES", "finite_gate_threshold", "train_released_module"]
