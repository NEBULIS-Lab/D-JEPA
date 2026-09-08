"""Complete-set decision-tail objective for constrained TD-JEPA plasticity."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from djepa.predictor_boundary import configure_predictor_boundary


CANDIDATE_COUNT = 63


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def configure_plastic_tdjepa(model: nn.Module) -> tuple[str, ...]:
    """Enable only the exact final TD-JEPA block used by the plastic arm."""
    names = tuple(configure_predictor_boundary(model))
    _require(
        bool(names)
        and all(
            name.startswith("predictor.transformer.layers.5.") or name.startswith("pred_proj.")
            for name in names
        ),
        "D-JEPA predictive plasticity trainable boundary is invalid",
    )
    # Plasticity changes parameters, not stochastic execution. Keeping every
    # module in eval mode preserves exact Frozen predictions at update zero;
    # eval mode does not disable autograd for the selected parameters.
    model.eval()
    return names


def _validate_relations(
    physical_success: torch.Tensor,
    reference_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    tail_k: int,
) -> tuple[int, torch.device]:
    _require(
        physical_success.dtype == torch.bool
        and physical_success.ndim == 2
        and physical_success.shape[1] == CANDIDATE_COUNT,
        "D-JEPA predictive plasticity requires Boolean success labels over 63 candidates",
    )
    _require(
        reference_scores.shape == physical_success.shape
        and torch.is_floating_point(reference_scores)
        and bool(torch.isfinite(reference_scores).all()),
        "D-JEPA predictive plasticity reference scores must be finite and aligned",
    )
    _require(
        candidate_ids.shape == physical_success.shape
        and candidate_ids.dtype in (torch.int32, torch.int64)
        and bool((candidate_ids > 0).all())
        and all(torch.unique(row).numel() == CANDIDATE_COUNT for row in candidate_ids),
        "D-JEPA predictive plasticity requires 63 unique deployable candidate IDs without candidate zero",
    )
    _require(1 <= int(tail_k) <= CANDIDATE_COUNT, "D-JEPA predictive plasticity tail size is invalid")
    positives = physical_success.sum(dim=1)
    _require(bool(((positives > 0) & (positives < CANDIDATE_COUNT)).all()), "each D-JEPA predictive plasticity row requires both success classes")
    return physical_success.shape[0], physical_success.device


def _tail_positions(
    reference_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    tail_k: int,
    physical_success: torch.Tensor | None = None,
) -> list[list[int]]:
    rows: list[list[int]] = []
    labels_cpu = None if physical_success is None else physical_success.detach().cpu()
    for row, (scores, ids) in enumerate(zip(reference_scores.detach().cpu(), candidate_ids.detach().cpu(), strict=True)):
        ordered = sorted(range(CANDIDATE_COUNT), key=lambda index: (float(scores[index]), int(ids[index])))
        end = int(tail_k)
        if labels_cpu is not None:
            while end < CANDIDATE_COUNT:
                values = {bool(labels_cpu[row, position]) for position in ordered[:end]}
                if len(values) == 2:
                    break
                end += 1
        rows.append(ordered[:end])
    return rows


def all_tail_boundary_pairs(
    physical_success: torch.Tensor,
    reference_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    tail_k: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Enumerate every success/failure relation in the fixed reference tail."""
    batch, device = _validate_relations(physical_success, reference_scores, candidate_ids, tail_k=tail_k)
    tails = _tail_positions(reference_scores, candidate_ids, tail_k, physical_success)
    output: list[tuple[torch.Tensor, torch.Tensor]] = []
    for row in range(batch):
        success = [position for position in tails[row] if bool(physical_success[row, position])]
        failure = [position for position in tails[row] if not bool(physical_success[row, position])]
        _require(bool(success) and bool(failure), "reference decision tail must contain both success classes")
        success_positions = torch.tensor(
            [position for position in success for _ in failure], dtype=torch.long, device=device
        )
        failure_positions = torch.tensor(
            [position for _ in success for position in failure], dtype=torch.long, device=device
        )
        output.append((success_positions, failure_positions))
    return tuple(output)


def plasticity_loss(
    student_future: torch.Tensor,
    teacher_future: torch.Tensor,
    goal_z: torch.Tensor,
    physical_success: torch.Tensor,
    candidate_ids: torch.Tensor,
    reference_scores: torch.Tensor,
    *,
    tail_k: int,
    temperature: float = 0.05,
    margin: float = 0.02,
    boundary_weight: float = 0.5,
    rank_preserve_weight: float = 0.25,
    teacher_latent_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Optimize deployed 63-way success ordering under a Frozen trust region."""
    _require(
        student_future.ndim == 4
        and student_future.shape[1] == CANDIDATE_COUNT
        and student_future.shape == teacher_future.shape
        and torch.is_floating_point(student_future)
        and torch.is_floating_point(teacher_future)
        and bool(torch.isfinite(student_future).all())
        and bool(torch.isfinite(teacher_future).all()),
        "D-JEPA predictive plasticity future tensors must have matching finite (B,63,T,D) shape",
    )
    batch, _, _, latent = student_future.shape
    _require(
        goal_z.shape == (batch, latent)
        and torch.is_floating_point(goal_z)
        and bool(torch.isfinite(goal_z).all()),
        "D-JEPA predictive plasticity goal latent shape is invalid",
    )
    _validate_relations(physical_success, reference_scores, candidate_ids, tail_k=tail_k)
    _require(
        temperature > 0.0
        and margin >= 0.0
        and boundary_weight >= 0.0
        and rank_preserve_weight >= 0.0
        and teacher_latent_weight >= 0.0,
        "D-JEPA predictive plasticity loss constants are invalid",
    )

    student_terminal = student_future[:, :, -1].float()
    teacher_terminal = teacher_future[:, :, -1].float().detach()
    goal = goal_z[:, None].float()
    student_scores = (student_terminal - goal).square().mean(dim=-1)
    teacher_scores = (teacher_terminal - goal).square().mean(dim=-1)

    scaled = -student_scores / float(temperature)
    positive_logits = scaled.masked_fill(~physical_success, -torch.inf)
    listwise = (torch.logsumexp(scaled, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()

    boundary_terms: list[torch.Tensor] = []
    preserve_terms: list[torch.Tensor] = []
    tails = _tail_positions(reference_scores, candidate_ids, tail_k, physical_success)
    pairs = all_tail_boundary_pairs(physical_success, reference_scores, candidate_ids, tail_k=tail_k)
    for row, (success_positions, failure_positions) in enumerate(pairs):
        boundary_terms.append(
            F.softplus(
                (float(margin) + student_scores[row, success_positions] - student_scores[row, failure_positions])
                / float(temperature)
            ).mean()
        )
        tail = set(tails[row])
        non_tail = torch.tensor([index for index in range(CANDIDATE_COUNT) if index not in tail], device=student_scores.device)
        if non_tail.numel() > 1:
            teacher_prob = F.softmax(-teacher_scores[row, non_tail] / float(temperature), dim=0)
            student_log_prob = F.log_softmax(-student_scores[row, non_tail] / float(temperature), dim=0)
            preserve_terms.append(F.kl_div(student_log_prob, teacher_prob, reduction="sum"))
        for label in (False, True):
            same_class = [index for index in tails[row] if bool(physical_success[row, index]) is label]
            if len(same_class) > 1:
                index = torch.tensor(same_class, device=student_scores.device)
                student_delta = student_scores[row, index, None] - student_scores[row, None, index]
                teacher_delta = teacher_scores[row, index, None] - teacher_scores[row, None, index]
                preserve_terms.append(F.smooth_l1_loss(student_delta / float(temperature), teacher_delta / float(temperature)))

    boundary = torch.stack(boundary_terms).mean()
    rank_preserve = torch.stack(preserve_terms).mean() if preserve_terms else student_scores.sum() * 0.0
    teacher_latent = F.mse_loss(
        F.layer_norm(student_terminal, (latent,)),
        F.layer_norm(teacher_terminal, (latent,)),
    )
    total = (
        listwise
        + float(boundary_weight) * boundary
        + float(rank_preserve_weight) * rank_preserve
        + float(teacher_latent_weight) * teacher_latent
    )
    return total, {
        "candidate_count": CANDIDATE_COUNT,
        "decision_tail_candidate_count": float(sum(map(len, tails)) / len(tails)),
        "boundary_pair_count": int(sum(success.numel() for success, _ in pairs)),
        "listwise_loss": float(listwise.detach()),
        "boundary_loss": float(boundary.detach()),
        "rank_preserve_loss": float(rank_preserve.detach()),
        "teacher_latent_loss": float(teacher_latent.detach()),
        "student_teacher_score_mse": float(F.mse_loss(student_scores, teacher_scores).detach()),
    }
