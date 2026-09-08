"""Strict, label-free Reacher input and planning contracts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import numpy as np
import torch
from torch import Tensor


class ReacherContractError(ValueError):
    """Raised when a Reacher release crosses the model boundary incorrectly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReacherContractError(message)


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True, order="C")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class ReacherPlanningProfile:
    mode: str
    temperature: float
    terminal_weight: float
    mse_blend: float

    @classmethod
    def reviewed(cls) -> "ReacherPlanningProfile":
        return cls(
            mode="td_jepa",
            temperature=0.1,
            terminal_weight=0.3,
            mse_blend=0.0,
        )

    def validate_reviewed(self) -> "ReacherPlanningProfile":
        expected = self.reviewed()
        _require(self == expected, "reviewed Reacher planning profile differs")
        return self


@dataclass(frozen=True)
class ReacherRawBatch:
    context_pixels: np.ndarray
    goal_pixels: np.ndarray
    history_actions: np.ndarray
    candidate_actions: np.ndarray
    candidate_ids: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    _FIELDS = frozenset(
        {
            "context_pixels",
            "goal_pixels",
            "history_actions",
            "candidate_actions",
            "candidate_ids",
            "action_mean",
            "action_std",
        }
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ReacherRawBatch may not be subclassed")

    def _validate(self) -> None:
        context = np.asarray(self.context_pixels)
        goal = np.asarray(self.goal_pixels)
        history = np.asarray(self.history_actions)
        candidates = np.asarray(self.candidate_actions)
        ids = np.asarray(self.candidate_ids)
        mean = np.asarray(self.action_mean)
        std = np.asarray(self.action_std)
        _require(context.dtype == np.uint8 and context.ndim == 5 and context.shape[1:] == (3, 224, 224, 3), "context_pixels must be uint8 (B,3,224,224,3)")
        batch = int(context.shape[0])
        _require(batch > 0, "Reacher batch must be nonempty")
        _require(goal.dtype == np.uint8 and goal.shape == (batch, 1, 224, 224, 3), "goal_pixels must be uint8 (B,1,224,224,3)")
        _require(history.dtype == np.float64 and history.shape == (batch, 10, 2), "history_actions must be float64 (B,10,2)")
        _require(candidates.dtype == np.float64 and candidates.shape == (batch, 63, 25, 2), "candidate_actions must be float64 (B,63,25,2)")
        _require(bool(np.isfinite(history).all()), "history_actions must be finite")
        _require(bool(np.isfinite(candidates).all()), "candidate_actions must be finite")
        literal_ids = np.arange(1, 64, dtype=np.int64)
        _require(ids.dtype == np.int64 and ids.shape in {(63,), (batch, 63)}, "candidate_ids must be int64 (63,) or (B,63)")
        expected_ids = literal_ids if ids.ndim == 1 else np.broadcast_to(literal_ids, (batch, 63))
        _require(bool(np.array_equal(ids, expected_ids)), "candidate_ids must be exact ordered literals 1..63")
        _require(mean.dtype == np.float32 and mean.shape == (2,) and bool(np.isfinite(mean).all()), "action_mean must be finite float32 (2,)")
        _require(std.dtype == np.float32 and std.shape == (2,) and bool(np.isfinite(std).all()) and bool((std > 0).all()), "action_std must be positive finite float32 (2,)")

    def __post_init__(self) -> None:
        self._validate()
        for name in self._FIELDS:
            object.__setattr__(self, name, _readonly_copy(np.asarray(getattr(self, name))))

    @classmethod
    def from_arrays(
        cls,
        *,
        context_pixels: np.ndarray,
        goal_pixels: np.ndarray,
        history_actions: np.ndarray,
        candidate_actions: np.ndarray,
        candidate_ids: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ) -> "ReacherRawBatch":
        return cls.from_mapping(
            {
                "context_pixels": context_pixels,
                "goal_pixels": goal_pixels,
                "history_actions": history_actions,
                "candidate_actions": candidate_actions,
                "candidate_ids": candidate_ids,
                "action_mean": action_mean,
                "action_std": action_std,
            }
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, np.ndarray]) -> "ReacherRawBatch":
        _require(isinstance(value, Mapping) and set(value) == cls._FIELDS, "Reacher batch requires exact fields")
        arrays = {name: np.asarray(value[name]) for name in cls._FIELDS}
        context = arrays["context_pixels"]
        goal = arrays["goal_pixels"]
        history = arrays["history_actions"]
        candidates = arrays["candidate_actions"]
        ids = arrays["candidate_ids"]
        mean = arrays["action_mean"]
        std = arrays["action_std"]

        _require(context.dtype == np.uint8 and context.ndim == 5 and context.shape[1:] == (3, 224, 224, 3), "context_pixels must be uint8 (B,3,224,224,3)")
        batch = int(context.shape[0])
        _require(batch > 0, "Reacher batch must be nonempty")
        _require(goal.dtype == np.uint8 and goal.shape == (batch, 1, 224, 224, 3), "goal_pixels must be uint8 (B,1,224,224,3)")
        _require(history.dtype == np.float64 and history.shape == (batch, 10, 2), "history_actions must be float64 (B,10,2)")
        _require(candidates.dtype == np.float64 and candidates.shape == (batch, 63, 25, 2), "candidate_actions must be float64 (B,63,25,2)")
        _require(bool(np.isfinite(history).all()), "history_actions must be finite")
        _require(bool(np.isfinite(candidates).all()), "candidate_actions must be finite")

        literal_ids = np.arange(1, 64, dtype=np.int64)
        _require(ids.dtype == np.int64 and ids.shape in {(63,), (batch, 63)}, "candidate_ids must be int64 (63,) or (B,63)")
        expected_ids = literal_ids if ids.ndim == 1 else np.broadcast_to(literal_ids, (batch, 63))
        _require(bool(np.array_equal(ids, expected_ids)), "candidate_ids must be exact ordered literals 1..63")
        _require(mean.dtype == np.float32 and mean.shape == (2,) and bool(np.isfinite(mean).all()), "action_mean must be finite float32 (2,)")
        _require(std.dtype == np.float32 and std.shape == (2,) and bool(np.isfinite(std).all()) and bool((std > 0).all()), "action_std must be positive finite float32 (2,)")

        return cls(
            context_pixels=_readonly_copy(context),
            goal_pixels=_readonly_copy(goal),
            history_actions=_readonly_copy(history),
            candidate_actions=_readonly_copy(candidates),
            candidate_ids=_readonly_copy(ids),
            action_mean=_readonly_copy(mean),
            action_std=_readonly_copy(std),
        )

    def as_model_tensors(
        self, device: str | torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._validate()
        target = torch.device(device)
        context = torch.as_tensor(np.array(self.context_pixels, copy=True), device=target)
        goal = torch.as_tensor(np.array(self.goal_pixels[:, 0], copy=True), device=target)
        mean = torch.as_tensor(np.array(self.action_mean, copy=True), dtype=torch.float32, device=target)
        std = torch.as_tensor(np.array(self.action_std, copy=True), dtype=torch.float32, device=target)
        history = torch.as_tensor(np.array(self.history_actions, copy=True), dtype=torch.float32, device=target)
        candidates = torch.as_tensor(np.array(self.candidate_actions, copy=True), dtype=torch.float32, device=target)
        history = (history - mean) / std
        candidates = (candidates - mean) / std
        ids = np.array(self.candidate_ids, copy=True)
        if ids.ndim == 1:
            ids = np.broadcast_to(ids, (self.context_pixels.shape[0], 63)).copy()
        ids_tensor = torch.as_tensor(ids, dtype=torch.int64, device=target)
        return context, goal, history, candidates, ids_tensor


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ReacherPublicationAuthority:
    task_name: str
    capacity_decision: str
    capacity_schema: str
    capacity_completion_sha256: str
    train_authority_schema: str
    train_authority_completion_sha256: str
    teacher_schema: str
    teacher_completion_sha256: str
    teacher_checkpoint_sha256: str
    teacher_config_sha256: str

    _FIELDS = frozenset(
        {
            "task_name",
            "capacity_decision",
            "capacity_schema",
            "capacity_completion_sha256",
            "train_authority_schema",
            "train_authority_completion_sha256",
            "teacher_schema",
            "teacher_completion_sha256",
            "teacher_checkpoint_sha256",
            "teacher_config_sha256",
        }
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReacherPublicationAuthority":
        _require(isinstance(value, Mapping) and set(value) == cls._FIELDS, "Reacher publication authority requires exact fields")
        _require(all(isinstance(value[name], str) for name in cls._FIELDS), "Reacher publication authority fields must be strings")
        result = cls(**{name: str(value[name]) for name in cls._FIELDS})
        _require(result.task_name == "DMC-Reacher", "publication task must be DMC-Reacher")
        _require(result.capacity_decision == "PASS", "Reacher capacity decision must be PASS")
        _require(result.capacity_schema == "dtail_b2_capacity_gate_release_v1", "Reacher capacity schema is invalid")
        _require(result.train_authority_schema == "dtail_b2_full_pool_release_v1", "Reacher train authority schema is invalid")
        _require(result.teacher_schema == "dtail_unified_reacher_teacher_release_v1", "Reacher teacher schema is invalid")
        hash_fields = (
            result.capacity_completion_sha256,
            result.train_authority_completion_sha256,
            result.teacher_completion_sha256,
            result.teacher_checkpoint_sha256,
            result.teacher_config_sha256,
        )
        _require(all(_SHA256.fullmatch(item) for item in hash_fields), "Reacher publication authority SHA-256 is invalid")
        return result

    def as_mapping(self) -> Mapping[str, str]:
        return {name: str(getattr(self, name)) for name in sorted(self._FIELDS)}
