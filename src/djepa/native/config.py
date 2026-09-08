"""Frozen public contract for the native PushT reproduction path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


_TOP_LEVEL = {"schema", "task", "evaluation_split", "checkpoint", "protocol", "decision", "preprocessing"}
_PROTOCOL = {
    "candidate_count": 63,
    "context_frames": 3,
    "history_controls": 10,
    "control_horizon": 25,
    "action_block": 5,
    "future_blocks": 5,
    "action_units": "relative_position_delta",
    "action_low": [-1.0, -1.0],
    "action_high": [1.0, 1.0],
    "control_hz": 10,
    "physics_dt": 0.01,
    "integrator_steps_per_control": 10,
}
_DECISION = {
    "fusion_alpha": 0.42,
    "gate_threshold": -0.03162526342176623,
    "gate_comparison": "strict_gt",
    "gate_advantage_decimals": 4,
}
_PREPROCESSING = {
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "action_mean": [-0.007812565192580223, 0.006860686466097832],
    "action_std": [0.2084674835205078, 0.20674866437911987],
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_numbers(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_numbers(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(expected, float):
        return type(actual) in (int, float) and float(actual) == expected
    return actual == expected


def _require_exact(mapping: Any, expected: Mapping[str, Any], label: str) -> None:
    _require(isinstance(mapping, Mapping) and set(mapping) == set(expected), f"native {label} keys changed")
    for key, value in expected.items():
        _require(_same_numbers(mapping[key], value), f"native {label}.{key} changed")


def _validate_profile_config(value: Any) -> None:
    _require(isinstance(value, Mapping), "checkpoint config is not a mapping")
    _require(value.get("architecture") == "exact_realization_world_model", "checkpoint is not the full exact PushT profile")
    decision = value.get("decision")
    _require(isinstance(decision, Mapping), "checkpoint decision metadata is missing")
    expected_decision = {
        "fusion_alpha": _DECISION["fusion_alpha"],
        "gate_threshold": _DECISION["gate_threshold"],
        "gate_comparison": "strict_gt",
        "advantage_quantization": "decimal_half_even_4",
    }
    for key, expected in expected_decision.items():
        _require(_same_numbers(decision.get(key), expected), f"checkpoint decision.{key} changed")
    preprocessing = value.get("preprocessing")
    _require(isinstance(preprocessing, Mapping), "checkpoint preprocessing metadata is missing")
    expected_preprocessing = dict(_PREPROCESSING)
    expected_preprocessing.update(context_frames=3, action_block=5, future_blocks=5)
    for key, expected in expected_preprocessing.items():
        _require(_same_numbers(preprocessing.get(key), expected), f"checkpoint preprocessing.{key} changed")


def load_native_config(path: str | Path) -> dict[str, Any]:
    """Load a fail-closed config and authenticate the full checkpoint files."""
    config_path = Path(path).resolve()
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"native configuration is unreadable: {config_path}") from error
    _require(isinstance(value, Mapping) and set(value) == _TOP_LEVEL, "native configuration keys changed")
    _require(value["schema"] == "djepa_native_pusht_v1" and value["task"] == "pusht", "unsupported native configuration")
    _require(value["evaluation_split"] == "independent_256", "native evaluation split must remain independent_256")
    _require_exact(value["protocol"], _PROTOCOL, "protocol")
    _require_exact(value["decision"], _DECISION, "decision")
    _require_exact(value["preprocessing"], _PREPROCESSING, "preprocessing")

    checkpoint = value["checkpoint"]
    _require(isinstance(checkpoint, Mapping) and set(checkpoint) == {"path", "weights_sha256", "config_sha256"}, "native checkpoint descriptor changed")
    checkpoint_dir = (config_path.parent / str(checkpoint["path"])).resolve()
    weights_path, profile_config_path = checkpoint_dir / "model.pt", checkpoint_dir / "config.json"
    _require(weights_path.is_file() and profile_config_path.is_file(), f"native checkpoint is incomplete: {checkpoint_dir}")
    _require(sha256_file(weights_path) == checkpoint["weights_sha256"], "native checkpoint weight SHA-256 mismatch")
    _require(sha256_file(profile_config_path) == checkpoint["config_sha256"], "native checkpoint config SHA-256 mismatch")
    try:
        profile_config = json.loads(profile_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint config is invalid JSON") from error
    _validate_profile_config(profile_config)
    resolved = dict(value)
    resolved["checkpoint"] = dict(checkpoint)
    resolved["checkpoint_dir"] = checkpoint_dir
    return resolved


__all__ = ["load_native_config", "sha256_file"]
