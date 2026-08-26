"""Minimal exact-likelihood T1 pilot with one causally persistent state."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .baseline import ExactHistoryMarkDecoder, HistoryIntensity
from .bridge_e1 import (
    BridgeE1Design, RawAnchorReader, observation_batch,
)
from .mark_likelihood import tied_group_mark_log_prob
from .observer import ObservationTransformer
from .state import ControlledPersistentState


T1_PILOT_REVISION = "r1_t1_exact_filtered_state_swap_v2"


class PersistentEventModel(nn.Module):
    """Frozen history baseline plus state-only zero-effect residual heads."""

    def __init__(self, baseline_checkpoint: dict, history_dim: int,
                 n_mark_contacts: int, adjacency: np.ndarray,
                 explicit_dim: int, *, raw_enabled: bool = False,
                 state_dim: int = 8, d_model: int = 64):
        super().__init__()
        self.observer = ObservationTransformer(
            explicit_dim, d_model=d_model, patch_samples=128,
            n_heads=4, temporal_layers=2, spatial_layers=1,
            raw_enabled=raw_enabled,
        )
        self.state = ControlledPersistentState(d_model, state_dim)
        self.timing_baseline = HistoryIntensity(history_dim, history_visible=True)
        self.timing_baseline.load_state_dict(baseline_checkpoint["timing"]["history"])
        self.mark_baseline = ExactHistoryMarkDecoder(
            history_dim, n_mark_contacts, adjacency, history_visible=True
        )
        self.mark_baseline.load_state_dict(baseline_checkpoint["mark"]["history"])
        for frozen in (self.timing_baseline, self.mark_baseline):
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
        self.state_timing = nn.Linear(state_dim, 1, bias=False)
        self.state_contact = nn.Linear(state_dim, n_mark_contacts, bias=False)
        self.state_size = nn.Linear(state_dim, n_mark_contacts + 1, bias=False)
        # Exact parity with the frozen baseline is checked before optimisation.
        # The non-zero candidate path in ControlledPersistentState avoids a
        # permanently dead bilinear state once these heads take their first step.
        for head in (self.state_timing, self.state_contact, self.state_size):
            nn.init.zeros_(head.weight)

    def timing_log_rate(self, history: torch.Tensor,
                        state: torch.Tensor) -> torch.Tensor:
        return self.timing_baseline(history) + self.state_timing(state).squeeze(-1)

    def mark_terms(self, history: torch.Tensor, state: torch.Tensor,
                   group_ids: torch.Tensor, group_count: torch.Tensor):
        size, contact = self.mark_baseline.logits(history, group_ids, group_count)
        size = size + self.state_size(state).unsqueeze(1)
        contact = contact + self.state_contact(state).unsqueeze(1)
        return tied_group_mark_log_prob(group_ids, group_count, size, contact)


@dataclass(frozen=True)
class StateTrace:
    anchor_state: torch.Tensor
    event_state: torch.Tensor
    quadrature_state: torch.Tensor


def _encode_anchors(model: PersistentEventModel, design: BridgeE1Design,
                    reader: RawAnchorReader, *, device: torch.device | str,
                    anchor_batch_size: int,
                    embedding_permutation: np.ndarray | None = None) -> torch.Tensor:
    rows = []
    for lo in range(0, len(design.anchor_time), int(anchor_batch_size)):
        ids = np.arange(lo, min(lo + int(anchor_batch_size), len(design.anchor_time)))
        batch = observation_batch(
            reader, design, ids, device,
            read_raw=model.observer.raw is not None,
        )
        rows.append(model.observer(**batch))
    value = torch.cat(rows, dim=0)
    if embedding_permutation is not None:
        permutation = torch.as_tensor(
            embedding_permutation, dtype=torch.long, device=value.device
        )
        if permutation.shape != (len(design.anchor_time),):
            raise ValueError("anchor embedding permutation has wrong shape")
        value = value[permutation]
    return value


def filtered_state_trace(model: PersistentEventModel, design: BridgeE1Design,
                         reader: RawAnchorReader, *, device: torch.device | str,
                         anchor_batch_size: int = 8,
                         correction_enabled: bool = True,
                         validation_correction_off: bool = False,
                         embedding_permutation: np.ndarray | None = None) -> StateTrace:
    """Causal filtered states; validation naturally warm-starts from TRAIN."""
    embedding = _encode_anchors(
        model, design, reader, device=device,
        anchor_batch_size=anchor_batch_size,
        embedding_permutation=embedding_permutation,
    )
    anchor_state: list[torch.Tensor | None] = [None] * len(design.anchor_time)
    event_state: list[torch.Tensor | None] = [None] * len(design.event_anchor)
    q_state: list[torch.Tensor | None] = [None] * len(design.quadrature_anchor)
    for session in np.unique(design.anchor_session):
        anchors = np.flatnonzero(design.anchor_session == session)
        anchors = anchors[np.argsort(design.anchor_time[anchors], kind="stable")]
        state = embedding.new_zeros(model.state.dim)
        cursor = float(design.anchor_time[anchors[0]])
        for position, anchor in enumerate(anchors):
            time = float(design.anchor_time[anchor])
            delta = 0.0 if position == 0 else max((time - cursor) / 60.0, 0.0)
            state = model.state.assimilate(
                state, delta, embedding[anchor],
                enabled=(
                    correction_enabled
                    and not (
                        validation_correction_off
                        and int(design.anchor_split[anchor]) == 1
                    )
                ),
            )
            anchor_state[int(anchor)] = state
            event_rows = np.flatnonzero(design.event_anchor == anchor)
            if len(event_rows):
                delta = torch.as_tensor(
                    (design.event_time[event_rows] - time) / 60.0,
                    dtype=state.dtype, device=state.device,
                ).clamp(min=0.0)
                values = model.state.generator.from_anchor(state, delta)
                for row, value in zip(event_rows.tolist(), values):
                    event_state[int(row)] = value
            q_rows = np.flatnonzero(design.quadrature_anchor == anchor)
            if len(q_rows):
                delta = torch.as_tensor(
                    (design.quadrature_time[q_rows] - time) / 60.0,
                    dtype=state.dtype, device=state.device,
                ).clamp(min=0.0)
                values = model.state.generator.from_anchor(state, delta)
                for row, value in zip(q_rows.tolist(), values):
                    q_state[int(row)] = value
            cursor = time
    if any(value is None for value in anchor_state + event_state + q_state):
        raise RuntimeError("T1 state scan did not cover every design row")
    return StateTrace(
        anchor_state=torch.stack(anchor_state),
        event_state=torch.stack(event_state),
        quadrature_state=torch.stack(q_state),
    )


@dataclass(frozen=True)
class T1Metrics:
    joint_nll_per_event: float
    timing_nll_per_event: float
    mark_nll_per_event: float
    group_size_nll_per_event: float
    subset_nll_per_event: float
    n_events: int
    n_anchors: int
    recorded_seconds: float


def t1_loss_terms(model: PersistentEventModel, design: BridgeE1Design,
                  reader: RawAnchorReader, anchor_ids: np.ndarray, *,
                  device: torch.device | str,
                  anchor_batch_size: int = 8,
                  correction_enabled: bool = True,
                  validation_correction_off: bool = False,
                  embedding_permutation: np.ndarray | None = None,
                  state_permutation: np.ndarray | None = None,
                  ) -> dict[str, torch.Tensor | int | float]:
    trace = filtered_state_trace(
        model, design, reader, device=device,
        anchor_batch_size=anchor_batch_size,
        correction_enabled=correction_enabled,
        validation_correction_off=validation_correction_off,
        embedding_permutation=embedding_permutation,
    )
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    event_row = np.flatnonzero(np.isin(design.event_anchor, anchor_ids))
    q_row = np.flatnonzero(np.isin(design.quadrature_anchor, anchor_ids))
    selected_event_state = trace.event_state[event_row]
    selected_q_state = trace.quadrature_state[q_row]
    if state_permutation is not None:
        permutation = np.asarray(state_permutation, dtype=np.int64)
        if permutation.shape != (len(design.anchor_time),):
            raise ValueError("anchor state permutation has wrong shape")
        event_values: list[torch.Tensor] = []
        for row in event_row:
            target_anchor = int(design.event_anchor[row])
            donor_anchor = int(permutation[target_anchor])
            delta = torch.as_tensor(
                [(design.event_time[row] - design.anchor_time[target_anchor]) / 60.0],
                dtype=trace.anchor_state.dtype, device=trace.anchor_state.device,
            ).clamp(min=0.0)
            event_values.append(model.state.generator.from_anchor(
                trace.anchor_state[donor_anchor], delta
            )[0])
        q_values: list[torch.Tensor] = []
        for row in q_row:
            target_anchor = int(design.quadrature_anchor[row])
            donor_anchor = int(permutation[target_anchor])
            delta = torch.as_tensor(
                [(design.quadrature_time[row] - design.anchor_time[target_anchor]) / 60.0],
                dtype=trace.anchor_state.dtype, device=trace.anchor_state.device,
            ).clamp(min=0.0)
            q_values.append(model.state.generator.from_anchor(
                trace.anchor_state[donor_anchor], delta
            )[0])
        selected_event_state = torch.stack(event_values)
        selected_q_state = torch.stack(q_values)
    event_history = torch.as_tensor(design.event_history[event_row], device=device)
    q_history = torch.as_tensor(design.quadrature_history[q_row], device=device)
    event_log = model.timing_log_rate(event_history, selected_event_state).sum()
    q_log = model.timing_log_rate(q_history, selected_q_state)
    weight = torch.as_tensor(
        design.quadrature_weight_seconds[q_row], dtype=q_log.dtype, device=device
    )
    survival = torch.sum(weight * torch.exp(torch.clamp(q_log, max=20.0)))
    if len(event_row):
        mark = model.mark_terms(
            event_history, selected_event_state,
            torch.as_tensor(design.event_group_ids[event_row], dtype=torch.long, device=device),
            torch.as_tensor(design.event_group_count[event_row], dtype=torch.long, device=device),
        )
        mark_log = mark.event_log_prob.sum()
        size_log = mark.group_size_log_prob.sum()
        subset_log = mark.subset_log_prob.sum()
    else:
        mark_log = event_log.new_zeros(())
        size_log = event_log.new_zeros(())
        subset_log = event_log.new_zeros(())
    return {
        "event_log": event_log, "survival": survival,
        "mark_log": mark_log, "size_log": size_log,
        "subset_log": subset_log, "n_events": int(len(event_row)),
        "recorded_seconds": float(weight.detach().sum().cpu()),
    }


@torch.no_grad()
def evaluate_t1(model: PersistentEventModel, design: BridgeE1Design,
                reader: RawAnchorReader, split: str, *,
                device: torch.device | str,
                anchor_batch_size: int = 8,
                correction_enabled: bool = True,
                validation_correction_off: bool = False,
                embedding_permutation: np.ndarray | None = None,
                state_permutation: np.ndarray | None = None) -> T1Metrics:
    model.eval()
    anchors = design.anchor_ids(split)
    terms = t1_loss_terms(
        model, design, reader, anchors, device=device,
        anchor_batch_size=anchor_batch_size,
        correction_enabled=correction_enabled,
        validation_correction_off=validation_correction_off,
        embedding_permutation=embedding_permutation,
        state_permutation=state_permutation,
    )
    denom = max(int(terms["n_events"]), 1)
    timing = (float(terms["survival"]) - float(terms["event_log"])) / denom
    mark = -float(terms["mark_log"]) / denom
    return T1Metrics(
        joint_nll_per_event=timing + mark,
        timing_nll_per_event=timing,
        mark_nll_per_event=mark,
        group_size_nll_per_event=-float(terms["size_log"]) / denom,
        subset_nll_per_event=-float(terms["subset_log"]) / denom,
        n_events=int(terms["n_events"]), n_anchors=int(len(anchors)),
        recorded_seconds=float(terms["recorded_seconds"]),
    )


def fit_t1(model: PersistentEventModel, design: BridgeE1Design,
           reader: RawAnchorReader, *, seed: int,
           device: torch.device | str, epochs: int = 8,
           anchor_batch_size: int = 8,
           learning_rate: float = 3e-4) -> PersistentEventModel:
    """Inner-TRAIN epoch selection including epoch zero, then full refit."""
    train = design.anchor_ids("train")
    if len(train) < 10:
        raise ValueError("T1 pilot needs at least ten TRAIN anchors")
    cut = int(np.clip(math.floor(0.8 * len(train)), 1, len(train) - 1))
    inner_train, inner_validation = train[:cut], train[cut:]
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def objective(anchor_ids: np.ndarray) -> torch.Tensor:
        terms = t1_loss_terms(
            model, design, reader, anchor_ids, device=device,
            anchor_batch_size=anchor_batch_size,
        )
        return (
            terms["survival"] - terms["event_log"] - terms["mark_log"]
        ) / max(int(terms["n_events"]), 1)

    best_epoch = 0
    with torch.no_grad():
        best_value = float(objective(inner_validation))
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=float(learning_rate), weight_decay=1e-3,
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss = objective(inner_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            value = float(objective(inner_validation))
        if value < best_value:
            best_value = value
            best_epoch = epoch

    model.load_state_dict(initial)
    if best_epoch:
        optimizer = torch.optim.AdamW(
            [value for value in model.parameters() if value.requires_grad],
            lr=float(learning_rate), weight_decay=1e-3,
        )
        for _ in range(best_epoch):
            model.train()
            loss = objective(train)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.selected_epochs = int(best_epoch)
    model.inner_validation_joint_nll = float(best_value)
    return model.eval()


def matched_wrong_time_permutation(design: BridgeE1Design, *, split: str,
                                   min_separation_seconds: float = 300.0
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest history/time-of-day match within session, excluding nearby anchors.

    The permutation is identity where the sampled pilot has no admissible donor;
    callers must report the returned matched mask as the denominator.
    """
    target = design.anchor_ids(split)
    permutation = np.arange(len(design.anchor_time), dtype=np.int64)
    matched = np.zeros(len(design.anchor_time), dtype=bool)
    # count traces, load summaries, time of day, and session elapsed.  These are
    # standardized TRAIN-only deterministic features in anchor_history.
    feature_index = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    for row in target:
        candidate = target[
            (design.anchor_session[target] == design.anchor_session[row])
            & (np.abs(design.anchor_time[target] - design.anchor_time[row])
               >= float(min_separation_seconds))
        ]
        if not len(candidate):
            continue
        delta = design.anchor_history[candidate][:, feature_index] - design.anchor_history[
            row, feature_index
        ]
        distance = np.sum(delta.astype(np.float64) ** 2, axis=1)
        donor = int(candidate[int(np.argmin(distance))])
        permutation[row] = donor
        matched[row] = True
    return permutation, matched
