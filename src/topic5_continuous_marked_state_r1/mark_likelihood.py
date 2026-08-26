"""Exact tied-group, unordered without-replacement mark likelihood."""
from __future__ import annotations

from dataclasses import dataclass

import torch


NEG_INF = float("-inf")


def log_elementary_symmetric_all(logits: torch.Tensor,
                                 candidate: torch.Tensor) -> torch.Tensor:
    """Return log e_k(exp(logits[candidate])) for every k=0..N.

    The leading dimensions are arbitrary and the final dimension indexes
    contacts.  Dynamic programming is exact, differentiable and performed in
    log space.
    """
    if logits.shape != candidate.shape or logits.ndim < 1:
        raise ValueError("logits/candidate shapes disagree")
    candidate = candidate.to(torch.bool)
    n = logits.shape[-1]
    shape = logits.shape[:-1]
    # ``logaddexp(-inf, -inf)`` has an undefined derivative (0/0) in PyTorch
    # even though its forward value is mathematically harmless.  A finite
    # dtype-scaled sentinel keeps unreachable DP cells outside any realistic
    # logit range while preserving finite gradients in reachable cells.
    unreachable = torch.finfo(logits.dtype).min / 4.0
    dp = logits.new_full((*shape, n + 1), unreachable)
    dp[..., 0] = 0.0
    for index in range(n):
        value = logits[..., index]
        include = candidate[..., index]
        shifted = torch.cat(
            [logits.new_full((*shape, 1), unreachable), dp[..., :-1] + value.unsqueeze(-1)],
            dim=-1,
        )
        updated = torch.logaddexp(dp, shifted)
        dp = torch.where(include.unsqueeze(-1), updated, dp)
    return dp


def log_elementary_symmetric(logits: torch.Tensor, candidate: torch.Tensor,
                             cardinality: torch.Tensor | int) -> torch.Tensor:
    all_values = log_elementary_symmetric_all(logits, candidate)
    k = torch.as_tensor(cardinality, dtype=torch.long, device=logits.device)
    target_shape = logits.shape[:-1]
    if k.ndim == 0:
        k = k.expand(target_shape)
    if k.shape != target_shape:
        raise ValueError("cardinality shape does not match batch shape")
    if bool((k < 0).any()) or bool((k > candidate.sum(-1)).any()):
        raise ValueError("cardinality exceeds eligible support")
    return all_values.gather(-1, k.unsqueeze(-1)).squeeze(-1)


def conditional_k_subset_log_prob(logits: torch.Tensor, target: torch.Tensor,
                                  candidate: torch.Tensor) -> torch.Tensor:
    """Exact conditional log probability of an unordered size-k target subset."""
    if logits.shape != target.shape or logits.shape != candidate.shape:
        raise ValueError("subset tensors have different shapes")
    target = target.to(torch.bool)
    candidate = candidate.to(torch.bool)
    if bool((target & ~candidate).any()):
        raise ValueError("target contains an ineligible contact")
    k = target.sum(-1)
    numerator = torch.where(target, logits, torch.zeros_like(logits)).sum(-1)
    return numerator - log_elementary_symmetric(logits, candidate, k)


@dataclass(frozen=True)
class TiedMarkTerms:
    group_size_log_prob: torch.Tensor
    subset_log_prob: torch.Tensor
    event_log_prob: torch.Tensor
    group_size_step_log_prob: torch.Tensor
    subset_step_log_prob: torch.Tensor
    active_step: torch.Tensor
    select_step: torch.Tensor


def tied_group_mark_log_prob(group_ids: torch.Tensor,
                             group_count: torch.Tensor,
                             size_logits: torch.Tensor,
                             contact_logits: torch.Tensor,
                             node_mask: torch.Tensor | None = None) -> TiedMarkTerms:
    """Score complete events including their terminal STOP step.

    Parameters
    ----------
    group_ids:
        ``(B,N)`` dense 0..K-1 on participants and -1 otherwise.
    group_count:
        ``(B,)`` number of tied groups K.
    size_logits:
        ``(B,S,N+1)`` logits for group size 0..N, where S >= max(K)+1.
    contact_logits:
        ``(B,S,N)`` contact logits before each group.
    node_mask:
        optional ``(B,N)`` real-contact mask for padded batches.
    """
    if group_ids.ndim != 2:
        raise ValueError("group_ids must have shape (B,N)")
    batch, n_contacts = group_ids.shape
    if group_count.shape != (batch,):
        raise ValueError("group_count shape disagrees")
    if contact_logits.ndim != 3 or contact_logits.shape[:1] != (batch,):
        raise ValueError("contact_logits must have shape (B,S,N)")
    n_steps = contact_logits.shape[1]
    if contact_logits.shape[2] != n_contacts:
        raise ValueError("contact logits have wrong contact count")
    if size_logits.shape != (batch, n_steps, n_contacts + 1):
        raise ValueError("size logits must cover sizes 0..N at every step")
    if bool((group_count < 1).any()) or bool((group_count + 1 > n_steps).any()):
        raise ValueError("missing select or terminal step")
    if node_mask is None:
        node_mask = torch.ones_like(group_ids, dtype=torch.bool)
    else:
        node_mask = node_mask.to(torch.bool)
        if node_mask.shape != group_ids.shape:
            raise ValueError("node mask shape disagrees")
    participating = group_ids >= 0
    if bool((participating & ~node_mask).any()):
        raise ValueError("padded/ineligible contact participates")
    if bool((group_ids[~participating] != -1).any()):
        raise ValueError("non-participant group id is not -1")
    # Validate dense 0..K-1 labels in one batched device operation.  The
    # previous per-row ``torch.unique`` loop caused one host/device
    # synchronization per event and made high-event patients need hours.
    max_groups = int(group_count.max().detach().cpu())
    steps = torch.arange(max_groups, device=group_ids.device)
    present = (group_ids.unsqueeze(-1) == steps.view(1, 1, -1)).any(dim=1)
    expected = steps.view(1, -1) < group_count.unsqueeze(-1)
    if not bool(torch.equal(present, expected)):
        raise ValueError("group ids are not dense 0..K-1")

    recruited = torch.zeros_like(node_mask)
    size_step = size_logits.new_zeros((batch, n_steps))
    subset_step = contact_logits.new_zeros((batch, n_steps))
    active_step = torch.zeros((batch, n_steps), dtype=torch.bool, device=group_ids.device)
    select_step = torch.zeros_like(active_step)
    size_index = torch.arange(n_contacts + 1, device=group_ids.device).view(1, -1)

    for step in range(n_steps):
        active = step <= group_count
        selecting = step < group_count
        if not bool(active.any()):
            break
        eligible = node_mask & ~recruited
        target = participating & (group_ids == step)
        target_size = target.sum(-1)
        if bool((selecting & (target_size < 1)).any()):
            raise ValueError("select step has an empty tied group")
        if bool((~selecting & active & (target_size != 0)).any()):
            raise ValueError("terminal STOP step selects contacts")
        eligible_count = eligible.sum(-1)
        valid_size = size_index <= eligible_count.unsqueeze(-1)
        masked_size = torch.where(
            valid_size, size_logits[:, step],
            torch.full_like(size_logits[:, step], NEG_INF),
        )
        size_logp = torch.log_softmax(masked_size, dim=-1).gather(
            -1, target_size.unsqueeze(-1)
        ).squeeze(-1)
        size_step[:, step] = torch.where(active, size_logp, torch.zeros_like(size_logp))
        if bool(selecting.any()):
            subset_logp = conditional_k_subset_log_prob(
                contact_logits[:, step], target, eligible
            )
            subset_step[:, step] = torch.where(
                selecting, subset_logp, torch.zeros_like(subset_logp)
            )
        active_step[:, step] = active
        select_step[:, step] = selecting
        recruited = recruited | target

    group_size_total = size_step.sum(-1)
    subset_total = subset_step.sum(-1)
    return TiedMarkTerms(
        group_size_log_prob=group_size_total,
        subset_log_prob=subset_total,
        event_log_prob=group_size_total + subset_total,
        group_size_step_log_prob=size_step,
        subset_step_log_prob=subset_step,
        active_step=active_step,
        select_step=select_step,
    )
