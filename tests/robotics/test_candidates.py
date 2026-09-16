import numpy as np
import pytest

from djepa.robotics.candidates import (
    Candidate,
    CandidateSet,
    CoverageSummary,
    CoverageThresholds,
    admit_coverage,
    summarize_coverage,
)


def make_candidate_set(
    *,
    seeds=range(17),
    native_success: bool = False,
    rescue_index: int | None = 1,
    context: str = "context-0",
    cluster_count: int = 5,
) -> CandidateSet:
    candidates = []
    for index, seed in enumerate(seeds):
        cluster = index % cluster_count
        actions = np.full((16, 14), cluster * 0.5, dtype=np.float32)
        candidates.append(
            Candidate(
                role="native" if index == 0 else "proposal",
                noise_seed=seed,
                standardized_actions=actions,
                success=native_success if index == 0 else index == rescue_index,
                progress=1.0 if index == rescue_index else float(native_success),
            )
        )
    return CandidateSet(task="grab_roller", context_id=context, candidates=tuple(candidates))


def test_candidate_set_requires_native_plus_sixteen_unique_seeds() -> None:
    candidate_set = make_candidate_set(seeds=range(17))

    assert candidate_set.candidates[0].role == "native"
    assert len(candidate_set.candidates) == 17
    assert len({row.noise_seed for row in candidate_set.candidates}) == 17
    assert not candidate_set.candidates[0].standardized_actions.flags.writeable


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        (list(range(16)), "17"),
        ([0, *range(16)], "unique"),
    ],
)
def test_candidate_set_rejects_invalid_seed_contract(seeds, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_candidate_set(seeds=seeds)


def test_candidate_set_rejects_missing_native() -> None:
    candidates = list(make_candidate_set().candidates)
    candidates[0] = Candidate(
        role="proposal",
        noise_seed=0,
        standardized_actions=np.zeros((16, 14), dtype=np.float32),
        success=False,
        progress=0.0,
    )

    with pytest.raises(ValueError, match="native"):
        CandidateSet("grab_roller", "context", tuple(candidates))


def test_coverage_admission_enforces_all_three_gates() -> None:
    summary = CoverageSummary(
        contexts=20,
        mixed_contexts=8,
        native_successes=8,
        oracle_successes=12,
        diverse_contexts=18,
    )

    decision = admit_coverage(summary, CoverageThresholds())

    assert decision.admitted
    assert decision.mixed_fraction == 0.4
    assert decision.oracle_gain == pytest.approx(0.2)
    assert decision.diverse_fraction == 0.9


@pytest.mark.parametrize(
    "summary",
    [
        CoverageSummary(20, 5, 8, 12, 18),
        CoverageSummary(20, 8, 8, 10, 18),
        CoverageSummary(20, 8, 8, 12, 15),
    ],
)
def test_coverage_admission_rejects_each_failed_gate(summary: CoverageSummary) -> None:
    decision = admit_coverage(summary, CoverageThresholds())

    assert not decision.admitted
    assert not all(decision.gates.values())


def test_summarize_coverage_counts_native_oracle_mixed_and_diverse() -> None:
    sets = [
        make_candidate_set(context="rescue", native_success=False, rescue_index=1, cluster_count=5),
        make_candidate_set(context="native", native_success=True, rescue_index=None, cluster_count=5),
        make_candidate_set(context="flat", native_success=False, rescue_index=None, cluster_count=1),
    ]

    summary = summarize_coverage(sets, cluster_distance=0.25, minimum_clusters=4)

    assert summary == CoverageSummary(
        contexts=3,
        mixed_contexts=2,
        native_successes=1,
        oracle_successes=2,
        diverse_contexts=2,
    )
