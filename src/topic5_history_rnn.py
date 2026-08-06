"""Cross-event history-state utilities for the Topic 5 early-ictal bridge.

This module deliberately separates three objects:

``h_event``
    A within-one-event rank-prefix state that resets at every event.
``u_event``
    The final within-event embedding.
``z_history``
    A state that persists across chronologically ordered events and decays in
    real time.

The timeline helpers do not read ictal target values.  PyTorch components are
optional so the metadata-only G0 audit can run in the base environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - G0 is intentionally torch-free
    torch = None
    Tensor = object
    nn = None


@dataclass(frozen=True)
class CausalPrefix:
    """Indices and provenance for one pre-seizure causal event history."""

    event_indices: np.ndarray
    segment_id: int
    last_event_index: int
    cutoff_epoch: float
    previous_postictal_end_epoch: float
    exclusion_reason: str

    @property
    def available(self) -> bool:
        return bool(self.event_indices.size)


def normalize_contact_name(value: object) -> str:
    """Normalize only cosmetic contact-name differences.

    No shaft/contact inference or fuzzy matching is allowed because that can
    silently join different contacts.
    """

    return str(value).strip().upper().replace(" ", "")


def exact_contact_join(
    interictal_contacts: Sequence[object], target_contacts: Sequence[object]
) -> np.ndarray:
    """Return target indices for interictal contacts under an exact 1:1 join."""

    left = [normalize_contact_name(value) for value in interictal_contacts]
    right = [normalize_contact_name(value) for value in target_contacts]
    if len(set(left)) != len(left):
        raise ValueError("duplicate normalized interictal contact name")
    if len(set(right)) != len(right):
        raise ValueError("duplicate normalized target contact name")
    lookup = {name: index for index, name in enumerate(right)}
    missing = [name for name in left if name not in lookup]
    if missing:
        raise ValueError(f"contacts absent from target inventory: {missing}")
    return np.asarray([lookup[name] for name in left], dtype=np.int64)


def build_continuous_segment_ids(
    block_stems: Sequence[object],
    block_metadata: Mapping[str, Mapping[str, object]],
    *,
    max_block_gap_seconds: float = 2.0,
    allow_cross_recording_contiguous: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign event-level segment IDs from recording-aware block continuity.

    Events in the same block are continuous.  Crossing blocks is allowed only
    when both blocks belong to the same recording, their block numbers are
    consecutive, and the inventory reports no material gap.  A long interval
    between events inside otherwise continuous blocks is *not* a reset.
    """

    stems = np.asarray([str(value) for value in block_stems])
    if stems.ndim != 1:
        raise ValueError("block_stems must be one-dimensional")
    segment = np.zeros(stems.size, dtype=np.int32)
    reset = np.zeros(stems.size, dtype=bool)
    if not stems.size:
        return segment, reset
    reset[0] = True
    current_segment = 0
    for index in range(1, stems.size):
        previous_stem = stems[index - 1]
        current_stem = stems[index]
        if current_stem == previous_stem:
            segment[index] = current_segment
            continue
        if previous_stem not in block_metadata or current_stem not in block_metadata:
            raise KeyError(f"block inventory missing {previous_stem} or {current_stem}")
        previous = block_metadata[previous_stem]
        current = block_metadata[current_stem]
        same_recording = str(previous["recording_id"]) == str(current["recording_id"])
        index_key = (
            "sequence_index"
            if "sequence_index" in previous and "sequence_index" in current
            else "block_no"
        )
        consecutive = int(current[index_key]) == int(previous[index_key]) + 1
        gap = float(current["block_start_epoch"]) - float(previous["block_end_epoch"])
        continuous = (
            (same_recording or bool(allow_cross_recording_contiguous))
            and consecutive
            and np.isfinite(gap)
            and gap <= float(max_block_gap_seconds)
            and gap >= -float(max_block_gap_seconds)
        )
        if not continuous:
            current_segment += 1
            reset[index] = True
        segment[index] = current_segment
    return segment, reset


def select_causal_prefix(
    event_times: Sequence[float],
    segment_ids: Sequence[int],
    recording_ids: Sequence[object],
    *,
    seizure_recording_id: object,
    clinical_onset_epoch: float,
    guard_seconds: float,
    previous_postictal_end_epoch: float = -np.inf,
) -> CausalPrefix:
    """Select the final continuous, post-reset history before onset minus guard."""

    times = np.asarray(event_times, dtype=np.float64)
    segments = np.asarray(segment_ids, dtype=np.int64)
    recordings = np.asarray([str(value) for value in recording_ids])
    if not (times.ndim == segments.ndim == recordings.ndim == 1):
        raise ValueError("timeline arrays must be one-dimensional")
    if not (times.size == segments.size == recordings.size):
        raise ValueError("timeline arrays must align")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("event times must be finite and chronological")
    cutoff = float(clinical_onset_epoch) - float(guard_seconds)
    same_recording = recordings == str(seizure_recording_id)
    candidate = np.flatnonzero(
        same_recording
        & (times < cutoff)
        & (times >= float(previous_postictal_end_epoch))
    )
    if not candidate.size:
        return CausalPrefix(
            event_indices=np.asarray([], dtype=np.int64),
            segment_id=-1,
            last_event_index=-1,
            cutoff_epoch=cutoff,
            previous_postictal_end_epoch=float(previous_postictal_end_epoch),
            exclusion_reason="no_event_after_last_reset_before_guard",
        )
    last = int(candidate[-1])
    segment_id = int(segments[last])
    selected = candidate[segments[candidate] == segment_id]
    return CausalPrefix(
        event_indices=selected.astype(np.int64, copy=False),
        segment_id=segment_id,
        last_event_index=last,
        cutoff_epoch=cutoff,
        previous_postictal_end_epoch=float(previous_postictal_end_epoch),
        exclusion_reason="",
    )


def prefix_matched_order_indices(
    target_index: int,
    *,
    window: int,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    """Permute exactly the observed events in a recent causal prefix.

    The returned absolute indices contain the same event multiset as
    ``[start, target_index)`` and can never include the target or future data.
    """

    target = int(target_index)
    width = int(window)
    if target <= 0:
        raise ValueError("target_index must have at least one preceding event")
    if width <= 0:
        raise ValueError("window must be positive")
    start = max(0, target - width)
    absolute = np.arange(start, target, dtype=np.int64)
    # The most recent event is an explicit shared M1 covariate and must remain
    # fixed.  Only the preceding event identities are permuted across their
    # original time slots.
    if len(absolute) > 2:
        order = rng.permutation(len(absolute) - 1)
        if np.array_equal(order, np.arange(len(absolute) - 1)):
            order = np.roll(order, 1)
        absolute = np.concatenate([absolute[:-1][order], absolute[-1:]])
    return start, absolute


def center_contact_field(value):
    """Center a contact field while preserving NumPy or Torch semantics."""

    if torch is not None and isinstance(value, Tensor):
        return value - value.mean(dim=-1, keepdim=True)
    array = np.asarray(value)
    return array - np.mean(array, axis=-1, keepdims=True)


if nn is not None:

    @torch.no_grad()
    def encode_within_event(
        event_model: nn.Module,
        contact_features: Tensor,
        group_ids: Tensor,
        group_count: Tensor,
        *,
        local_offset: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Return final EventRNN states and contact embeddings.

        The EventRNN hidden state is explicitly reinitialized for every row in
        the batch.  This function never carries ``h_event`` across events;
        cross-event persistence is the sole responsibility of
        :class:`TimeDecayHistoryGRU`.
        """

        if contact_features.ndim != 3:
            raise ValueError("contact_features must be [batch,contact,feature]")
        batch, n_contacts, _ = contact_features.shape
        if group_ids.shape != (batch, n_contacts):
            raise ValueError("group_ids must align with event/contact axes")
        if group_count.shape != (batch,):
            raise ValueError("group_count must align with event batch")
        if local_offset is None:
            local_offset = contact_features.new_zeros(
                (n_contacts, int(event_model.local_offset_dim))
            )
        contact_mask = torch.ones(
            (batch, n_contacts), dtype=torch.bool, device=contact_features.device
        )
        embedding, _ = event_model._encode(contact_features, local_offset)
        hidden = event_model._initial_hidden(embedding, contact_mask)
        recruited = torch.zeros_like(contact_mask)
        max_groups = int(group_count.max().item()) if batch else 0
        for step in range(max_groups):
            current = (group_ids == step) & contact_mask
            active = (group_count > step).unsqueeze(1)
            updated_recruited = recruited | current
            updated_hidden = event_model._advance(
                embedding,
                current,
                updated_recruited,
                hidden,
                contact_mask,
            )
            hidden = torch.where(active, updated_hidden, hidden)
            recruited = torch.where(active, updated_recruited, recruited)
        return hidden, embedding

    class TimeDecayHistoryGRU(nn.Module):
        """GRU history state with positive, dimension-wise real-time decay."""

        def __init__(
            self,
            event_embedding_dim: int,
            history_dim: int = 32,
            *,
            initial_half_life_hours: float = 2.0,
        ):
            super().__init__()
            self.event_embedding_dim = int(event_embedding_dim)
            self.history_dim = int(history_dim)
            if float(initial_half_life_hours) <= 0:
                raise ValueError("initial_half_life_hours must be positive")
            rate_per_second = np.log(2.0) / (float(initial_half_life_hours) * 3600.0)
            raw = np.log(np.expm1(rate_per_second))
            self.raw_decay_rate = nn.Parameter(
                torch.full((self.history_dim,), float(raw))
            )
            self.cell = nn.GRUCell(self.event_embedding_dim, self.history_dim)

        @property
        def decay_rate_per_second(self) -> Tensor:
            return torch.nn.functional.softplus(self.raw_decay_rate)

        def decay(self, state: Tensor, delta_t_seconds: Tensor) -> Tensor:
            if delta_t_seconds.ndim == state.ndim - 1:
                delta_t_seconds = delta_t_seconds.unsqueeze(-1)
            return state * torch.exp(
                -self.decay_rate_per_second * delta_t_seconds.clamp_min(0.0)
            )

        def step(
            self,
            event_embedding: Tensor,
            state: Tensor,
            delta_t_seconds: Tensor,
            reset: Optional[Tensor] = None,
        ) -> Tensor:
            if reset is not None:
                if reset.ndim == state.ndim - 1:
                    reset = reset.unsqueeze(-1)
                state = torch.where(reset.bool(), torch.zeros_like(state), state)
            return self.cell(
                event_embedding,
                self.decay(state, delta_t_seconds),
            )

        def forward(
            self,
            event_embeddings: Tensor,
            delta_t_seconds: Tensor,
            reset_mask: Tensor,
            initial_state: Optional[Tensor] = None,
        ) -> Tensor:
            if event_embeddings.ndim != 3:
                raise ValueError("event_embeddings must be [batch,event,feature]")
            batch, n_events, _ = event_embeddings.shape
            if delta_t_seconds.shape != (batch, n_events):
                raise ValueError("delta_t_seconds must align with batch/event")
            if reset_mask.shape != (batch, n_events):
                raise ValueError("reset_mask must align with batch/event")
            state = (
                event_embeddings.new_zeros((batch, self.history_dim))
                if initial_state is None
                else initial_state
            )
            states = []
            for event_index in range(n_events):
                state = self.step(
                    event_embeddings[:, event_index],
                    state,
                    delta_t_seconds[:, event_index],
                    reset_mask[:, event_index],
                )
                states.append(state)
            return torch.stack(states, dim=1)

        def forward_masked(
            self,
            event_embeddings: Tensor,
            delta_t_seconds: Tensor,
            reset_mask: Tensor,
            event_mask: Tensor,
            initial_state: Optional[Tensor] = None,
        ) -> tuple[Tensor, Tensor]:
            """Run padded sequences without changing state on padded steps."""

            if event_mask.shape != event_embeddings.shape[:2]:
                raise ValueError("event_mask must align with batch/event")
            batch, n_events, _ = event_embeddings.shape
            state = (
                event_embeddings.new_zeros((batch, self.history_dim))
                if initial_state is None
                else initial_state
            )
            states = []
            for event_index in range(n_events):
                proposal = self.step(
                    event_embeddings[:, event_index],
                    state,
                    delta_t_seconds[:, event_index],
                    reset_mask[:, event_index],
                )
                active = event_mask[:, event_index].unsqueeze(-1)
                state = torch.where(active, proposal, state)
                states.append(state)
            return torch.stack(states, dim=1), state


    class MatchedUnorderedHistory(nn.Module):
        """Permutation-invariant pooling of the same event embeddings.

        The separately supplied ``last_event`` term is frozen by contract and
        is intentionally shared with the chronological model's M1 covariates.
        """

        def __init__(self, event_embedding_dim: int, output_dim: int = 32):
            super().__init__()
            input_dim = 4 * int(event_embedding_dim) + 3
            self.output = nn.Sequential(
                nn.Linear(input_dim, int(output_dim)),
                nn.SiLU(),
                nn.Linear(int(output_dim), int(output_dim)),
            )

        def forward(
            self,
            event_embeddings: Tensor,
            event_mask: Tensor,
            scalar_context: Tensor,
        ) -> Tensor:
            if event_embeddings.ndim != 3 or event_mask.ndim != 2:
                raise ValueError("unordered history inputs have invalid rank")
            if event_embeddings.shape[:2] != event_mask.shape:
                raise ValueError("event mask must align with event embeddings")
            weight = event_mask.to(event_embeddings.dtype).unsqueeze(-1)
            count = weight.sum(1).clamp_min(1.0)
            mean = (event_embeddings * weight).sum(1) / count
            centered = (event_embeddings - mean[:, None]) * weight
            variance = centered.square().sum(1) / count
            floor = torch.finfo(event_embeddings.dtype).min
            maximum = event_embeddings.masked_fill(
                ~event_mask.unsqueeze(-1), floor
            ).max(1).values
            maximum = torch.where(
                event_mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum)
            )
            last_index = event_mask.sum(1).clamp_min(1) - 1
            last = event_embeddings[
                torch.arange(event_embeddings.shape[0], device=event_embeddings.device),
                last_index,
            ]
            features = torch.cat([mean, variance, maximum, last, scalar_context], dim=-1)
            return self.output(features)


    class MatchedUnorderedSummary(nn.Module):
        """Read precomputed causal mean/max/last summaries.

        The summary at event ``e`` uses events from the true segment start
        through ``e``.  It is therefore orderless without introducing an
        artificial fixed-window reset.
        """

        def __init__(self, event_embedding_dim: int, output_dim: int = 32):
            super().__init__()
            input_dim = 3 * int(event_embedding_dim) + 3
            self.output = nn.Sequential(
                nn.Linear(input_dim, int(output_dim)),
                nn.SiLU(),
                nn.Linear(int(output_dim), int(output_dim)),
            )

        def forward(self, summary: Tensor) -> Tensor:
            if summary.shape[-1] != self.output[0].in_features:
                raise ValueError("unordered summary feature dimension drifted")
            return self.output(summary)


    class ContactFieldReadout(nn.Module):
        """Shared latent-to-contact query used for next-event and ictal fields."""

        def __init__(
            self,
            state_dim: int,
            contact_embedding_dim: int,
            *,
            centered: bool,
        ):
            super().__init__()
            self.centered = bool(centered)
            self.state_to_query = nn.Linear(
                int(state_dim), int(contact_embedding_dim), bias=False
            )

        def forward(self, state: Tensor, contact_embedding: Tensor) -> Tensor:
            query = self.state_to_query(state)
            if query.ndim == 2:
                score = torch.einsum("bce,be->bc", contact_embedding, query)
            elif query.ndim == 3:
                score = torch.einsum("bce,bte->btc", contact_embedding, query)
            else:
                raise ValueError("state must be [batch,state] or [batch,time,state]")
            score = score / np.sqrt(float(contact_embedding.shape[-1]))
            return center_contact_field(score) if self.centered else score


    class NextEventFieldHeads(nn.Module):
        """Participation and relative-rank heads from one shared state."""

        def __init__(self, state_dim: int, contact_embedding_dim: int):
            super().__init__()
            self.participation = ContactFieldReadout(
                state_dim, contact_embedding_dim, centered=False
            )
            self.relative_rank = ContactFieldReadout(
                state_dim, contact_embedding_dim, centered=False
            )

        def forward(self, state: Tensor, contact_embedding: Tensor) -> dict[str, Tensor]:
            return {
                "participation_logits": self.participation(state, contact_embedding),
                "relative_rank": self.relative_rank(state, contact_embedding),
            }


    def next_event_field_loss(
        prediction: Mapping[str, Tensor],
        participation: Tensor,
        relative_rank: Tensor,
        *,
        rank_weight: float = 0.2,
        event_weight: Optional[Tensor] = None,
        contact_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Patient-balanced next-event participation and rank-field loss."""

        target = participation.to(prediction["participation_logits"].dtype)
        contact_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction["participation_logits"], target, reduction="none"
        )
        if contact_mask is None:
            expanded_contact_mask = torch.ones_like(contact_bce, dtype=torch.bool)
        else:
            expanded_contact_mask = contact_mask.bool()
            while expanded_contact_mask.ndim < contact_bce.ndim:
                expanded_contact_mask = expanded_contact_mask.unsqueeze(-2)
            expanded_contact_mask = expanded_contact_mask.expand_as(contact_bce)
        event_bce = (contact_bce * expanded_contact_mask).sum(-1) / (
            expanded_contact_mask.sum(-1).clamp_min(1)
        )
        rank_mask = (
            participation.bool()
            & torch.isfinite(relative_rank)
            & expanded_contact_mask
        )
        rank_error = torch.nn.functional.smooth_l1_loss(
            prediction["relative_rank"],
            torch.nan_to_num(relative_rank),
            reduction="none",
        )
        event_rank = (rank_error * rank_mask).sum(-1) / rank_mask.sum(-1).clamp_min(1)
        if event_weight is None:
            event_weight = torch.ones_like(event_bce)
        denominator = event_weight.sum().clamp_min(1.0)
        participation_loss = (event_bce * event_weight).sum() / denominator
        rank_loss = (event_rank * event_weight).sum() / denominator
        return {
            "total": participation_loss + float(rank_weight) * rank_loss,
            "participation_bce": participation_loss,
            "relative_rank_huber": rank_loss,
            "event_participation_bce": event_bce,
            "event_relative_rank_huber": event_rank,
        }
