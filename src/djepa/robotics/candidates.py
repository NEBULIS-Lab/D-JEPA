"""Candidate identity, trajectory diversity, and Oracle-coverage gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np


_CANDIDATE_COUNT = 17
_PREFIX_LENGTH = 16
_ACTION_DIM = 14


@dataclass(frozen=True)
class Candidate:
    role: str
    noise_seed: int
    standardized_actions: np.ndarray
    success: bool
    progress: float

    def __post_init__(self) -> None:
        if self.role not in {"native", "proposal"}:
            raise ValueError("candidate role must be native or proposal")
        if not isinstance(self.noise_seed, int) or self.noise_seed < 0:
            raise ValueError("candidate noise_seed must be a nonnegative integer")
        actions = np.asarray(self.standardized_actions)
        if actions.shape != (_PREFIX_LENGTH, _ACTION_DIM):
            raise ValueError(
                f"candidate actions must have shape ({_PREFIX_LENGTH},{_ACTION_DIM}), got {actions.shape}"
            )
        if actions.dtype != np.float32 or not np.isfinite(actions).all():
            raise ValueError("candidate actions must be finite float32")
        if not isinstance(self.success, (bool, np.bool_)):
            raise ValueError("candidate success must be boolean")
        if not math.isfinite(float(self.progress)):
            raise ValueError("candidate progress must be finite")
        detached = actions.copy()
        detached.flags.writeable = False
        object.__setattr__(self, "standardized_actions", detached)
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "progress", float(self.progress))


@dataclass(frozen=True)
class CandidateSet:
    task: str
    context_id: str
    candidates: tuple[Candidate, ...]
    post_outcome_selected: bool = False

    def __post_init__(self) -> None:
        if not self.task or not self.context_id:
            raise ValueError("candidate task and context_id must be nonempty")
        if self.post_outcome_selected:
            raise ValueError("post-outcome candidate replacement is forbidden")
        if len(self.candidates) != _CANDIDATE_COUNT:
            raise ValueError(f"candidate set must contain exactly {_CANDIDATE_COUNT} candidates")
        if self.candidates[0].role != "native":
            raise ValueError("candidate index zero must be native")
        if any(candidate.role != "proposal" for candidate in self.candidates[1:]):
            raise ValueError("all nonzero candidates must have proposal role")
        seeds = [candidate.noise_seed for candidate in self.candidates]
        if len(set(seeds)) != len(seeds):
            raise ValueError("candidate noise seeds must be unique")


@dataclass(frozen=True)
class CoverageThresholds:
    minimum_mixed_fraction: float = 0.30
    minimum_oracle_gain: float = 0.15
    minimum_diverse_fraction: float = 0.80
    minimum_clusters: int = 4

    def __post_init__(self) -> None:
        values = (
            self.minimum_mixed_fraction,
            self.minimum_oracle_gain,
            self.minimum_diverse_fraction,
        )
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError("coverage fractions must lie in [0,1]")
        if self.minimum_clusters < 1 or self.minimum_clusters > _CANDIDATE_COUNT:
            raise ValueError("minimum_clusters is outside candidate count")


@dataclass(frozen=True)
class CoverageSummary:
    contexts: int
    mixed_contexts: int
    native_successes: int
    oracle_successes: int
    diverse_contexts: int

    def __post_init__(self) -> None:
        counts = (
            self.contexts,
            self.mixed_contexts,
            self.native_successes,
            self.oracle_successes,
            self.diverse_contexts,
        )
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("coverage counts must be nonnegative integers")
        if any(value > self.contexts for value in counts[1:]):
            raise ValueError("coverage event counts cannot exceed contexts")
        if self.oracle_successes < self.native_successes:
            raise ValueError("Oracle successes cannot be below native successes")


@dataclass(frozen=True)
class CoverageDecision:
    admitted: bool
    mixed_fraction: float
    oracle_gain: float
    diverse_fraction: float
    gates: Mapping[str, bool]


def _trajectory_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def trajectory_cluster_count(actions: Iterable[np.ndarray], threshold: float) -> int:
    """Return deterministic complete-link clusters at a fixed distance cut."""

    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("cluster distance threshold must be positive and finite")
    rows = tuple(np.asarray(value) for value in actions)
    if not rows:
        return 0
    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(rows))]

    def complete_link(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        return max(_trajectory_distance(rows[i], rows[j]) for i in left for j in right)

    while True:
        choices = [
            (complete_link(clusters[i], clusters[j]), i, j)
            for i in range(len(clusters))
            for j in range(i + 1, len(clusters))
        ]
        if not choices:
            break
        distance, left_index, right_index = min(choices)
        if distance > threshold:
            break
        merged = tuple(sorted((*clusters[left_index], *clusters[right_index])))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (left_index, right_index)
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: cluster[0])
    return len(clusters)


def summarize_coverage(
    candidate_sets: Iterable[CandidateSet],
    *,
    cluster_distance: float,
    minimum_clusters: int,
) -> CoverageSummary:
    """Summarize every fixed candidate set without filtering failed contexts."""

    rows = tuple(candidate_sets)
    if len({(row.task, row.context_id) for row in rows}) != len(rows):
        raise ValueError("candidate context identities must be unique")
    mixed = 0
    native_successes = 0
    oracle_successes = 0
    diverse = 0
    for row in rows:
        utilities = [(int(candidate.success), candidate.progress) for candidate in row.candidates]
        mixed += int(min(utilities) != max(utilities))
        native_successes += int(row.candidates[0].success)
        oracle_successes += int(any(candidate.success for candidate in row.candidates))
        clusters = trajectory_cluster_count(
            (candidate.standardized_actions for candidate in row.candidates),
            cluster_distance,
        )
        diverse += int(clusters >= minimum_clusters)
    return CoverageSummary(
        contexts=len(rows),
        mixed_contexts=mixed,
        native_successes=native_successes,
        oracle_successes=oracle_successes,
        diverse_contexts=diverse,
    )


def admit_coverage(
    summary: CoverageSummary,
    thresholds: CoverageThresholds,
) -> CoverageDecision:
    """Apply the three predeclared proposal-capability gates."""

    if summary.contexts == 0:
        fractions = (0.0, 0.0, 0.0)
    else:
        fractions = (
            summary.mixed_contexts / summary.contexts,
            (summary.oracle_successes - summary.native_successes) / summary.contexts,
            summary.diverse_contexts / summary.contexts,
        )
    mixed_fraction, oracle_gain, diverse_fraction = fractions
    gates = {
        "mixed_outcomes": mixed_fraction >= thresholds.minimum_mixed_fraction,
        "oracle_headroom": oracle_gain >= thresholds.minimum_oracle_gain,
        "trajectory_diversity": diverse_fraction >= thresholds.minimum_diverse_fraction,
    }
    return CoverageDecision(
        admitted=all(gates.values()),
        mixed_fraction=mixed_fraction,
        oracle_gain=oracle_gain,
        diverse_fraction=diverse_fraction,
        gates=gates,
    )
