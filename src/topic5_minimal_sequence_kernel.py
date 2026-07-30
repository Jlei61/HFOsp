"""Minimal within-event sequence-kernel tools for Topic 5.

This module deliberately separates:

* the joint STOP/contact likelihood used by the frozen models;
* finite recent-rank prediction without a recurrent hidden state; and
* input-output lag kernels of the selected diagonal linear state.

All time indices are within-event rank steps.  Nothing here represents
inter-event or continuous biological time.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object
    nn = None

from src.topic5_rank_distribution import StaticSequenceContactQuery


if nn is not None:

    def decomposed_next_set_stop_loss(
        outputs: Mapping[str, Tensor],
        group_ids: Tensor,
        group_count: Tensor,
    ) -> dict[str, Tensor]:
        """Exactly decompose the frozen joint categorical likelihood.

        For nonterminal decisions the original loss is the sum of a binary
        continue decision and contact-set identity conditional on continuing.
        For terminal decisions it is the STOP loss.  Event-balanced additive
        terms use zero contact contribution at the terminal decision, while
        ``event_contact_choice_nll`` averages contact identity only across
        nonterminal decisions.
        """

        contact_logits = outputs["contact_logits"]
        stop_logits = outputs["stop_logits"]
        candidate_mask = outputs.get("candidate_mask")
        if contact_logits.ndim != 3 or stop_logits.shape != contact_logits.shape[:2]:
            raise ValueError("STOP/contact logits must align on batch and step")
        if candidate_mask is not None and candidate_mask.shape != contact_logits.shape:
            raise ValueError("candidate mask must align with contact logits")
        if group_ids.ndim != 2 or group_count.shape != (group_ids.shape[0],):
            raise ValueError("group ids/counts are not aligned")

        per_total = []
        per_stop = []
        per_contact = []
        per_stop_probability = []
        per_terminal = []
        per_active = []
        for step in range(contact_logits.shape[1]):
            active = group_count >= step
            terminal = group_count == step
            logits = contact_logits[:, step]
            contact_log_z = torch.logsumexp(logits, dim=1)
            total_log_z = torch.logaddexp(stop_logits[:, step], contact_log_z)
            log_p_stop = stop_logits[:, step] - total_log_z
            log_p_continue = contact_log_z - total_log_z

            target_set = group_ids == step
            if candidate_mask is not None:
                invalid_target = target_set & ~candidate_mask[:, step]
                if bool(torch.any(invalid_target & active[:, None])):
                    raise ValueError("target contact is absent from candidate mask")
            target_log_mass = torch.logsumexp(
                logits.masked_fill(~target_set, -1.0e9), dim=1
            )
            contact_choice = -(target_log_mass - contact_log_z)
            contact_choice = torch.where(
                terminal, torch.zeros_like(contact_choice), contact_choice
            )
            stop_decision = torch.where(
                terminal, -log_p_stop, -log_p_continue
            )
            total = stop_decision + contact_choice

            per_total.append(total)
            per_stop.append(stop_decision)
            per_contact.append(contact_choice)
            per_stop_probability.append(torch.exp(log_p_stop))
            per_terminal.append(terminal)
            per_active.append(active)

        decision_total = torch.stack(per_total, dim=1)
        decision_stop = torch.stack(per_stop, dim=1)
        decision_contact = torch.stack(per_contact, dim=1)
        stop_probability = torch.stack(per_stop_probability, dim=1)
        terminal_mask = torch.stack(per_terminal, dim=1)
        decision_mask = torch.stack(per_active, dim=1)
        nonterminal_mask = decision_mask & ~terminal_mask
        decision_count = decision_mask.sum(1).clamp_min(1)
        nonterminal_count = nonterminal_mask.sum(1).clamp_min(1)

        event_total = (decision_total * decision_mask).sum(1) / decision_count
        event_stop = (decision_stop * decision_mask).sum(1) / decision_count
        event_contact_contribution = (
            decision_contact * decision_mask
        ).sum(1) / decision_count
        event_contact_choice = (
            decision_contact * nonterminal_mask
        ).sum(1) / nonterminal_count
        event_continue = (
            decision_stop * nonterminal_mask
        ).sum(1) / nonterminal_count
        event_terminal_stop = (
            decision_stop * terminal_mask
        ).sum(1) / terminal_mask.sum(1).clamp_min(1)

        return {
            "total": event_total.mean(),
            "event_total_nll": event_total,
            "event_stop_contribution_nll": event_stop,
            "event_contact_contribution_nll": event_contact_contribution,
            "event_contact_choice_nll": event_contact_choice,
            "event_continue_nll": event_continue,
            "event_terminal_stop_nll": event_terminal_stop,
            "decision_total_nll": decision_total,
            "decision_stop_nll": decision_stop,
            "decision_contact_choice_nll": decision_contact,
            "decision_mask": decision_mask,
            "nonterminal_mask": nonterminal_mask,
            "terminal_mask": terminal_mask,
            "stop_probability": stop_probability,
        }


    class ResidualFIRH3SequenceModel(StaticSequenceContactQuery):
        """Three-lag ordered residual over an unordered-prefix baseline.

        The model has no temporally persistent hidden state.  At each decision
        it recomputes the unordered baseline and adds three lag-specific
        projections of the most recent rank-set tokens.
        """

        def __init__(
            self,
            contact_feature_dim: int,
            *,
            history_lags: int = 3,
            **kwargs,
        ):
            if int(history_lags) != 3:
                raise ValueError("the frozen closeout model requires exactly 3 lags")
            super().__init__(contact_feature_dim, mode="unordered", **kwargs)
            self.history_lags = int(history_lags)
            self.lag_projections = nn.ModuleList(
                nn.Linear(
                    self.contact_embedding_dim,
                    self.hidden_size,
                    bias=False,
                )
                for _ in range(self.history_lags)
            )
            for projection in self.lag_projections:
                nn.init.zeros_(projection.weight)

        def freeze_unordered_baseline(self) -> None:
            """Freeze every parameter except the three ordered projections."""

            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for projection in self.lag_projections:
                for parameter in projection.parameters():
                    parameter.requires_grad_(True)

        def ordered_parameters(self) -> list[Tensor]:
            return [
                parameter
                for projection in self.lag_projections
                for parameter in projection.parameters()
            ]

        def forward(
            self,
            contact_features: Tensor,
            contact_mask: Tensor,
            group_ids: Tensor,
            group_count: Tensor,
            local_offset: Tensor,
        ) -> dict[str, Tensor]:
            embedding, encoder_input = self._encode(contact_features, local_offset)
            recruited = torch.zeros_like(contact_mask)
            last_set = torch.zeros_like(contact_mask)
            maximum = int(group_count.max().item())
            contact_logits = []
            stop_logits = []
            candidate_masks = []
            for prediction_step in range(maximum + 1):
                hidden = self._static_hidden(
                    embedding, contact_mask, recruited, last_set
                )
                residual = torch.zeros_like(hidden)
                for lag, projection in enumerate(self.lag_projections):
                    source_step = prediction_step - 1 - lag
                    if source_step < 0:
                        continue
                    source_set = (group_ids == source_step) & contact_mask
                    weight = source_set.to(embedding.dtype).unsqueeze(-1)
                    token = (
                        (embedding * weight).sum(1)
                        / weight.sum(1).clamp_min(1.0)
                    )
                    active = (group_count > source_step).unsqueeze(1)
                    residual = residual + torch.where(
                        active, projection(token), torch.zeros_like(residual)
                    )
                hidden = hidden + residual
                candidate = contact_mask & ~recruited
                action, stop = self._decode(
                    embedding, encoder_input, hidden, candidate
                )
                contact_logits.append(action)
                stop_logits.append(stop)
                candidate_masks.append(candidate)
                if prediction_step == maximum:
                    break
                current = (group_ids == prediction_step) & contact_mask
                active = (group_count > prediction_step).unsqueeze(1)
                recruited = torch.where(active, recruited | current, recruited)
                last_set = torch.where(active, current, last_set)
            return {
                "contact_logits": torch.stack(contact_logits, dim=1),
                "stop_logits": torch.stack(stop_logits, dim=1),
                "candidate_mask": torch.stack(candidate_masks, dim=1),
            }


    @torch.no_grad()
    def linear_state_contact_lag_kernels(
        model: nn.Module,
        contact_features: Tensor,
        local_offset: Tensor,
        *,
        max_lag: int = 5,
    ) -> dict[str, Tensor]:
        """Return contact-space invariant kernels of a fitted linear state."""

        if not hasattr(model, "persistence") or not hasattr(
            model, "input_projection"
        ):
            raise TypeError("model is not a compatible linear-state recurrence")
        if int(max_lag) < 0:
            raise ValueError("max_lag must be nonnegative")
        features = contact_features
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("contact features must contain one patient")
        embedding, _ = model._encode(features, local_offset)
        embedding = embedding[0]
        token_weight = model.input_projection.weight[
            :, : model.contact_embedding_dim
        ]
        source_drive = token_weight @ embedding.T
        contact_readout = (
            embedding @ model.action_query.weight
        ) / np.sqrt(float(model.contact_embedding_dim))
        stop_readout = model.stop_head.weight
        persistence = model.persistence
        contact_kernels = []
        stop_kernels = []
        power = torch.ones_like(persistence)
        for _ in range(int(max_lag) + 1):
            propagated = power[:, None] * source_drive
            contact_kernels.append(contact_readout @ propagated)
            stop_kernels.append(stop_readout @ propagated)
            power = power * persistence
        return {
            "contact": torch.stack(contact_kernels, dim=0),
            "stop": torch.stack(stop_kernels, dim=0),
            "persistence": persistence.detach().clone(),
            "contact_embedding": embedding.detach().clone(),
        }


    @torch.no_grad()
    def linear_state_lag_ablation_outputs(
        model: nn.Module,
        contact_features: Tensor,
        contact_mask: Tensor,
        group_ids: Tensor,
        group_count: Tensor,
        local_offset: Tensor,
        *,
        ablate_lags: Optional[Iterable[int]] = None,
        ablate_from_lag: Optional[int] = None,
    ) -> dict[str, Tensor]:
        """Replay a linear state while removing contact identity at chosen lags.

        Candidate masks, input-projection bias, prefix progress and set-size
        covariates remain unchanged.  Only the identity-bearing contact token
        is zeroed.
        """

        if not hasattr(model, "persistence") or not hasattr(
            model, "input_projection"
        ):
            raise TypeError("model is not a compatible linear-state recurrence")
        exact = {int(value) for value in (ablate_lags or [])}
        if any(value < 0 for value in exact):
            raise ValueError("lag indices must be nonnegative")
        if ablate_from_lag is not None and int(ablate_from_lag) < 0:
            raise ValueError("ablate_from_lag must be nonnegative")

        groups = group_ids.to(dtype=torch.long)
        counts = group_count.to(dtype=torch.long)
        embedding, encoder_input = model._encode(contact_features, local_offset)
        maximum = int(counts.max().item())
        contact_logits = []
        stop_logits = []
        candidate_masks = []
        for prediction_step in range(maximum + 1):
            hidden = model._initial_hidden(embedding, contact_mask)
            recruited = torch.zeros_like(contact_mask)
            for source_step in range(prediction_step):
                current = (groups == source_step) & contact_mask
                updated_recruited = recruited | current
                weight = current.to(embedding.dtype).unsqueeze(-1)
                token = (
                    (embedding * weight).sum(1)
                    / weight.sum(1).clamp_min(1.0)
                )
                lag = prediction_step - 1 - source_step
                remove = lag in exact or (
                    ablate_from_lag is not None
                    and lag >= int(ablate_from_lag)
                )
                if remove:
                    token = torch.zeros_like(token)
                denominator = (
                    contact_mask.sum(1).clamp_min(1).to(embedding.dtype)
                )
                progress = (
                    updated_recruited.sum(1).to(embedding.dtype) / denominator
                )
                new_fraction = (
                    current.sum(1).to(embedding.dtype) / denominator
                )
                external = model.input_projection(
                    torch.cat(
                        [token, progress[:, None], new_fraction[:, None]], dim=1
                    )
                )
                updated_hidden = model.persistence * hidden + external
                active = (counts > source_step).unsqueeze(1)
                hidden = torch.where(active, updated_hidden, hidden)
                recruited = torch.where(
                    active, updated_recruited, recruited
                )
            recruited = (
                (groups >= 0)
                & (groups < prediction_step)
                & contact_mask
            )
            candidate = contact_mask & ~recruited
            action, stop = model._decode(
                embedding, encoder_input, hidden, candidate
            )
            contact_logits.append(action)
            stop_logits.append(stop)
            candidate_masks.append(candidate)
        return {
            "contact_logits": torch.stack(contact_logits, dim=1),
            "stop_logits": torch.stack(stop_logits, dim=1),
            "candidate_mask": torch.stack(candidate_masks, dim=1),
        }


def block_hankel_from_lag_kernels(
    kernels: np.ndarray,
    *,
    block_rows: int = 3,
    block_columns: int = 3,
) -> np.ndarray:
    """Construct a finite block Hankel matrix from ``K_0, K_1, ...``."""

    values = np.asarray(kernels, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("kernels must have shape [lag, output, input]")
    rows = int(block_rows)
    columns = int(block_columns)
    if rows < 1 or columns < 1:
        raise ValueError("block dimensions must be positive")
    required = rows + columns - 1
    if values.shape[0] < required:
        raise ValueError(f"at least {required} lag kernels are required")
    return np.row_stack(
        [
            np.column_stack([values[row + column] for column in range(columns)])
            for row in range(rows)
        ]
    )


def hankel_singular_summary(hankel: np.ndarray) -> dict[str, object]:
    """Summarize finite-horizon input-output order without a state claim."""

    matrix = np.asarray(hankel, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("hankel must be a nonempty matrix")
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular**2
    total = float(np.sum(energy))
    if total <= 0:
        cumulative = np.zeros_like(energy)
        rank90 = rank95 = 0
        effective = 0.0
    else:
        cumulative = np.cumsum(energy) / total
        rank90 = int(np.searchsorted(cumulative, 0.90) + 1)
        rank95 = int(np.searchsorted(cumulative, 0.95) + 1)
        effective = float(total**2 / np.sum(energy**2))
    return {
        "singular_values": singular,
        "cumulative_energy": cumulative,
        "rank90": rank90,
        "rank95": rank95,
        "effective_order": effective,
    }


def merge_frozen_groups_by_lag_tolerance(
    group_ids: np.ndarray,
    group_count: np.ndarray,
    lag_raw: np.ndarray,
    *,
    tolerance_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge adjacent frozen rank sets without reconstructing their base order.

    The canonical zero-tolerance encoding was produced from pre-save float64
    lag values, while ``event_lag_raw`` is stored as float32. Reconstructing
    exact ties from the saved lags can therefore invent equalities through
    rounding. The frozen groups remain the zero-tolerance source of truth;
    positive tolerances may only merge adjacent frozen groups.
    """

    groups = np.asarray(group_ids, np.int16)
    counts = np.asarray(group_count, np.int16)
    lags = np.asarray(lag_raw, float)
    if groups.ndim != 2 or lags.shape != groups.shape:
        raise ValueError("group_ids and lag_raw must be aligned event x contact arrays")
    if counts.shape != (groups.shape[0],):
        raise ValueError("group_count must contain one value per event")
    tolerance = float(tolerance_seconds)
    if tolerance < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    if tolerance == 0:
        return groups.copy(), counts.copy()

    merged = np.full_like(groups, -1)
    merged_counts = np.zeros_like(counts)
    for event_index, (event, count, event_lag) in enumerate(
        zip(groups, counts, lags)
    ):
        count_int = int(count)
        if count_int <= 0:
            continue
        representatives = np.full(count_int, np.nan, dtype=float)
        for group_index in range(count_int):
            values = event_lag[(event == group_index) & np.isfinite(event_lag)]
            if values.size:
                representatives[group_index] = float(np.median(values))
        new_group = 0
        merged[event_index, event == 0] = new_group
        for group_index in range(1, count_int):
            left = representatives[group_index - 1]
            right = representatives[group_index]
            should_merge = (
                np.isfinite(left)
                and np.isfinite(right)
                and float(right - left) <= tolerance + 1e-8
            )
            if not should_merge:
                new_group += 1
            merged[event_index, event == group_index] = new_group
        merged_counts[event_index] = new_group + 1
    return merged, merged_counts
