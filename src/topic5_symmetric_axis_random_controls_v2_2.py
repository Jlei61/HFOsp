"""Vectorized fixed-axis controls for the frozen Topic-5 v2.2 model.

This module only batches independent fixed-axis fits.  It does not add a new
state, decoder, contact-mixing path, or target input.  The equations mirror the
scalar implementation in ``topic5_symmetric_axis_propagation_state_v2_2`` and
exist to make the 256-direction Claim-3 null computationally tractable.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def fixed_axis_operator_batch(
    *,
    coords: Tensor,
    axes: Tensor,
    anisotropy_ratio: Tensor | float,
    gamma_raw: Tensor,
    gain_raw: Tensor,
    local_scale: Tensor | float,
    eps: float = 1.0e-8,
) -> Tensor:
    """Return one symmetric operator per fixed physical axis.

    Parameters with a leading ``direction`` dimension are independent.  Only
    ``gamma_raw`` and ``gain_raw`` are intended to be optimized.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [contact, 3]")
    if axes.ndim != 2 or axes.shape[1] != 3:
        raise ValueError("axes must have shape [direction, 3]")
    n_directions = axes.shape[0]
    if gamma_raw.shape != (n_directions,) or gain_raw.shape != (n_directions,):
        raise ValueError("gamma_raw and gain_raw must match directions")

    centered = coords - coords.mean(dim=0, keepdim=True)
    axes_unit = axes / torch.clamp(
        torch.linalg.vector_norm(axes, dim=1, keepdim=True), min=eps
    )
    delta = centered[:, None, :] - centered[None, :, :]
    distance_sq = torch.sum(delta.square(), dim=-1)
    scale = torch.as_tensor(local_scale, device=coords.device, dtype=coords.dtype)
    ratio = torch.as_tensor(
        anisotropy_ratio, device=coords.device, dtype=coords.dtype
    )
    if not torch.isfinite(scale) or bool(scale <= eps):
        raise ValueError("local_scale must be finite and positive")
    if not torch.isfinite(ratio) or bool(ratio < 1.0):
        raise ValueError("anisotropy_ratio must be finite and at least one")

    diagonal = torch.eye(
        coords.shape[0], device=coords.device, dtype=torch.bool
    )
    local = torch.exp(-distance_sq / (2.0 * scale.square())).masked_fill(
        diagonal, 0.0
    )
    local = local / torch.clamp(torch.linalg.vector_norm(local), min=eps)

    parallel = torch.abs(torch.einsum("ijd,ad->aij", delta, axes_unit))
    perpendicular_sq = torch.clamp(
        distance_sq[None, :, :] - parallel.square(), min=0.0
    )
    axial = torch.exp(
        -parallel.square() / (2.0 * (ratio * scale).square())
        - perpendicular_sq / (2.0 * scale.square())
    ).masked_fill(diagonal[None, :, :], 0.0)
    axial = axial / torch.clamp(
        torch.linalg.vector_norm(axial, dim=(1, 2), keepdim=True), min=eps
    )

    gamma = torch.sigmoid(gamma_raw)
    gain = F.softplus(gain_raw) + eps
    adjacency = (
        (1.0 - gamma)[:, None, None] * local[None, :, :]
        + gamma[:, None, None] * axial
    )
    degree = torch.sum(adjacency, dim=2)
    inv_sqrt_degree = torch.rsqrt(torch.clamp(degree, min=eps))
    return (
        gain[:, None, None]
        * inv_sqrt_degree[:, :, None]
        * adjacency
        * inv_sqrt_degree[:, None, :]
    )


def _future_first_arrival_nll_batch(
    *,
    state: Tensor,
    eligible: Tensor,
    groups: Tensor,
    current_step: int,
    horizon: int,
    operator: Tensor,
    node_bias: Tensor,
    rho_p: Tensor,
    c0: Tensor,
    c_p: Tensor,
    c_n: Tensor,
    eps: float,
) -> Tensor:
    """Direction-by-event H-step first-arrival NLL."""
    n_directions, batch_size, n_contacts = state.shape
    dtype = state.dtype
    eligible_float = eligible.to(dtype)
    not_arrived = eligible_float[None, :, :].expand(
        n_directions, -1, -1
    ).clone()
    survival = torch.ones(
        (n_directions, batch_size), device=state.device, dtype=dtype
    )
    rollout_state = state
    base_seen = n_contacts - eligible_float.sum(dim=1)
    q_steps: list[Tensor] = []

    for _ in range(horizon):
        remaining_weight = not_arrived.sum(dim=2)
        weighted_mean = (
            not_arrived * rollout_state
        ).sum(dim=2) / torch.clamp(remaining_weight, min=eps)
        expected_seen = (
            base_seen[None, :]
            + (
                eligible_float[None, :, :] - not_arrived
            ).sum(dim=2)
        ) / float(n_contacts)
        stop_logit = c0 + c_p * weighted_mean + c_n * expected_seen
        p_stop = torch.sigmoid(stop_logit)

        hazard = torch.sigmoid(
            node_bias[None, None, :] + rollout_state
        ) * eligible_float[None, :, :]
        log_empty = torch.where(
            eligible[None, :, :],
            torch.log1p(-torch.clamp(hazard, max=1.0 - eps)),
            torch.zeros_like(hazard),
        ).sum(dim=2)
        z = -torch.expm1(log_empty)
        forced_stop = (remaining_weight <= eps) | (z <= eps)
        conditional_hazard = torch.where(
            forced_stop[:, :, None],
            torch.zeros_like(hazard),
            hazard / torch.clamp(z[:, :, None], min=eps),
        )
        p_stop = torch.where(forced_stop, torch.ones_like(p_stop), p_stop)
        activation = not_arrived * conditional_hazard
        q_steps.append(
            survival[:, :, None] * (1.0 - p_stop[:, :, None]) * activation
        )
        survival = survival * (1.0 - p_stop)
        not_arrived = not_arrived * (1.0 - conditional_hazard)
        rollout_state = rho_p * rollout_state + torch.einsum(
            "dbc,dic->dbi", activation, operator
        )

    q_stack = torch.stack(q_steps, dim=2)
    q_sum = q_stack.sum(dim=2)
    offset = groups - int(current_step)
    target_arrives = eligible & (offset >= 1) & (offset <= horizon)
    gather_index = torch.clamp(offset - 1, min=0, max=horizon - 1)
    arrival_probability = torch.gather(
        q_stack.permute(0, 1, 3, 2),
        3,
        gather_index[None, :, :, None].expand(n_directions, -1, -1, -1),
    ).squeeze(-1)
    class_probability = torch.where(
        target_arrives[None, :, :],
        arrival_probability,
        1.0 - q_sum,
    )
    contact_nll = -torch.log(torch.clamp(class_probability, min=eps))
    eligible_count = torch.clamp(eligible.sum(dim=1), min=1)
    return (
        contact_nll * eligible_float[None, :, :]
    ).sum(dim=2) / eligible_count[None, :]


def fixed_axis_event_losses_batch(
    *,
    operator: Tensor,
    groups: Tensor,
    counts: Tensor,
    node_bias: Tensor,
    rho_p: Tensor,
    c0: Tensor,
    c_p: Tensor,
    c_n: Tensor,
    training_horizon: int,
    eps: float = 1.0e-8,
) -> dict[str, Tensor]:
    """Mirror scalar event-first losses for many independent fixed axes."""
    if operator.ndim != 3 or operator.shape[1] != operator.shape[2]:
        raise ValueError("operator must have shape [direction, contact, contact]")
    if groups.ndim != 2 or counts.shape != (groups.shape[0],):
        raise ValueError("groups/counts are not aligned")
    n_directions, n_contacts, _ = operator.shape
    if groups.shape[1] != n_contacts or node_bias.shape != (n_contacts,):
        raise ValueError("contact dimensions are not aligned")
    if training_horizon < 0:
        raise ValueError("training_horizon must be non-negative")

    batch_size = groups.shape[0]
    dtype = operator.dtype
    state = torch.zeros(
        (n_directions, batch_size, n_contacts),
        device=groups.device,
        dtype=dtype,
    )
    next_sum = torch.zeros(
        (n_directions, batch_size), device=groups.device, dtype=dtype
    )
    future_sum = torch.zeros_like(next_sum)
    decision_count = torch.zeros(
        batch_size, device=groups.device, dtype=dtype
    )
    max_steps = int(torch.max(counts).item())

    for step in range(max_steps):
        active = counts > step
        current = (groups == step).to(dtype)
        state = rho_p * state + torch.einsum(
            "bc,dic->dbi", current, operator
        )
        seen = (groups >= 0) & (groups <= step)
        eligible = ~seen
        eligible_count = eligible.sum(dim=1)
        eligible_float = eligible.to(dtype)
        mean_drive = (
            state * eligible_float[None, :, :]
        ).sum(dim=2) / torch.clamp(eligible_count[None, :], min=1)
        seen_fraction = seen.to(dtype).mean(dim=1)
        stop_logit = c0 + c_p * mean_drive + c_n * seen_fraction[None, :]
        stop_logit = torch.where(
            eligible_count[None, :] > 0,
            stop_logit,
            torch.full_like(stop_logit, torch.inf),
        )

        logits = node_bias[None, None, :] + state
        target_float = (groups == (step + 1)).to(dtype)
        terminal = counts == (step + 1)
        log_hazard = F.logsigmoid(logits)
        log_one_minus = F.logsigmoid(-logits)
        bernoulli = (
            target_float[None, :, :] * log_hazard
            + (
                eligible_float[None, :, :] - target_float[None, :, :]
            )
            * log_one_minus
        ).sum(dim=2)
        log_empty = (
            eligible_float[None, :, :] * log_one_minus
        ).sum(dim=2)
        log_empty = torch.clamp(
            log_empty, max=-torch.finfo(log_empty.dtype).eps
        )
        log_z = torch.log(-torch.expm1(log_empty))
        log_probability = torch.where(
            terminal[None, :],
            F.logsigmoid(stop_logit),
            F.logsigmoid(-stop_logit) + bernoulli - log_z,
        )
        normalized_nll = -log_probability / torch.clamp(
            eligible_count[None, :], min=1
        )
        next_sum = next_sum + torch.where(
            active[None, :], normalized_nll, torch.zeros_like(normalized_nll)
        )

        if training_horizon > 0:
            future_nll = _future_first_arrival_nll_batch(
                state=state,
                eligible=eligible,
                groups=groups,
                current_step=step,
                horizon=training_horizon,
                operator=operator,
                node_bias=node_bias,
                rho_p=rho_p,
                c0=c0,
                c_p=c_p,
                c_n=c_n,
                eps=eps,
            )
            future_sum = future_sum + torch.where(
                active[None, :], future_nll, torch.zeros_like(future_nll)
            )
        decision_count = decision_count + active.to(dtype)

    event_next = next_sum / torch.clamp(decision_count[None, :], min=1.0)
    if training_horizon > 0:
        event_future = future_sum / torch.clamp(
            decision_count[None, :], min=1.0
        )
        event_objective = event_next + event_future
    else:
        event_future = torch.full_like(event_next, torch.nan)
        event_objective = event_next
    return {
        "event_next_nll": event_next,
        "event_future_nll": event_future,
        "event_objective": event_objective,
    }
