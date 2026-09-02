"""The minimal perturbation set, replayed on a frozen checkpoint.

Three arms, and only three, because the plan fixes them:

``real_sequence``                    the record as it happened
``no_event_feedback``                the same model with the event edge switched
                                     off *inside the exposure window only*
``state_matched_mark_replacement``   the same event count at the same instants,
                                     each event's content swapped for a
                                     state-matched donor's

Which separates the two estimands the plan refuses to let merge:

*Burden*  ``real - no_event_feedback``.  Nothing matches or regresses out the
          exposure window's event count; that count **is** the exposure.
*Content* ``real - state_matched_mark_replacement``.  Count and instants are
          preserved bit-for-bit; only the marks move.

Every arm starts from the identical pre-state at the identical exposure start and
is scored against the identical future block.  The future block is decoded in one
shot from the anchor state, so there is no teacher forcing inside it to switch off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .models import H3Model
from .runtime import SubjectContext
from .train import SubjectTensors, _segment_inputs, resolve_window, TrainConfig

PRIMARY_ARMS = ("real_sequence", "no_event_feedback", "state_matched_mark_replacement")
SECONDARY_ARMS = ("rate_preserving_mark_shuffle", "burst_thinning")

# Pre-registered before any perturbation score existed.
BURST_QUANTILE = 0.25          # an interval below this patient quantile is "in a burst"
BURST_THINNING_FRACTION = 0.5  # half of those events are dropped
DONOR_POOL_SIZE = 4096


@dataclass
class PerturbationBlock:
    subject: str
    horizon: int
    segment: int
    anchor_index: int
    anchor_time: float
    exposure_index: int
    n_exposure_events: int


def collect_anchor_states(
    model: H3Model, data: SubjectTensors, cfg: TrainConfig, device: torch.device
) -> list[Tensor]:
    """The causal state arriving at every anchor of every segment.

    One pass, carried across windows with ``detach`` only.  Perturbations then
    start from a row of this table, which is what makes "same pre-state" literal
    rather than approximate.
    """

    cfg = resolve_window(data, cfg)
    out: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for seg, tl in enumerate(data.timelines):
            carry = None
            pieces: list[Tensor] = []
            for lo in range(0, tl.n_steps, cfg.window_steps):
                hi = min(lo + cfg.window_steps, tl.n_steps)
                local = np.flatnonzero((tl.anchor_step >= lo) & (tl.anchor_step < hi))
                want = torch.from_numpy(
                    (tl.anchor_step[local] - lo).astype(np.int64)
                ).to(device)
                dt, drive, impulse = _segment_inputs(model, data, seg, lo, hi)
                states, carry = model.rollout(dt, drive, impulse, want, state_init=carry)
                if local.size:
                    pieces.append(states)
            out.append(
                torch.cat(pieces, dim=0)
                if pieces
                else torch.zeros(0, model.cfg.d_state, device=device)
            )
    return out


def build_donor_pool(
    ctx: SubjectContext,
    data: SubjectTensors,
    anchor_states: Sequence[Tensor],
    seed: int,
) -> tuple[Tensor, Tensor]:
    """TRAIN-split events and the state that was current when each arrived.

    Donors come only from TRAIN, so a replacement mark can never carry information
    about a held-out future block.  The "state that was current" is the state at
    the last anchor at or before the event -- an exact per-event state would need
    every row of the rollout materialised for 235,000 events, and the anchor grid
    is 5 minutes, which is finer than any timescale this state can express.
    """

    rows: list[int] = []
    states: list[Tensor] = []
    rng = np.random.default_rng(seed)
    for seg, tl in enumerate(data.timelines):
        split = data.anchor_split[seg]
        is_train = np.asarray([s == "train" for s in split], dtype=bool)
        if not is_train.any() or anchor_states[seg].shape[0] == 0:
            continue
        event_mask = tl.event_row >= 0
        ev_times = tl.step_time[event_mask]
        ev_rows = tl.event_row[event_mask]
        if ev_rows.size == 0:
            continue
        pos = np.searchsorted(tl.anchor_time, ev_times, side="right") - 1
        keep = (pos >= 0) & is_train[np.clip(pos, 0, is_train.size - 1)]
        if not keep.any():
            continue
        rows.append(ev_rows[keep])
        states.append(anchor_states[seg][torch.from_numpy(pos[keep]).to(anchor_states[seg].device)])
    if not rows:
        raise ValueError(f"{ctx.subject}: no TRAIN events available as donors")
    all_rows = np.concatenate(rows)
    all_states = torch.cat(states, dim=0)
    if all_rows.size > DONOR_POOL_SIZE:
        pick = rng.choice(all_rows.size, size=DONOR_POOL_SIZE, replace=False)
        all_rows, all_states = all_rows[pick], all_states[torch.from_numpy(pick).to(all_states.device)]
    return torch.from_numpy(all_rows.astype(np.int64)).to(all_states.device), all_states


def state_matched_marks(
    data: SubjectTensors,
    recipient_rows: Tensor,
    recipient_state: Tensor,
    donor_rows: Tensor,
    donor_states: Tensor,
) -> Tensor:
    """Each exposure event's mark replaced by its nearest state-matched donor's.

    Nearest in standardised state space.  Count features are untouched by
    construction -- this function returns marks only -- so the replacement leaves
    the event count and every event instant exactly as recorded.
    """

    if donor_rows.numel() == 0:
        return data.mark_features[recipient_rows]
    mu = donor_states.mean(0, keepdim=True)
    sd = donor_states.std(0, keepdim=True).clamp_min(1e-6)
    a = (recipient_state - mu) / sd
    b = (donor_states - mu) / sd
    d2 = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    nearest = donor_rows[d2.argmin(dim=1)]
    return data.mark_features[nearest]


def perturb_exposure(
    model: H3Model,
    data: SubjectTensors,
    seg: int,
    lo: int,
    hi: int,
    pre_state: Tensor,
    arm: str,
    *,
    donor: tuple[Tensor, Tensor] | None = None,
    pre_state_for_match: Tensor | None = None,
    rng: np.random.Generator | None = None,
    burst_mask: np.ndarray | None = None,
) -> Tensor:
    """Roll the exposure window under one perturbation and return the anchor state.

    ``[lo, hi]`` is inclusive of the anchor step, so the returned state is exactly
    the one the decoder would read there -- before the anchor's own event, if any.
    """

    tl = data.timelines[seg]
    rows_np = tl.event_row[lo : hi + 1]
    is_event = rows_np >= 0
    device = pre_state.device

    if arm == "real_sequence":
        dt, drive, impulse = _segment_inputs(model, data, seg, lo, hi + 1)
    elif arm == "no_event_feedback":
        dt, drive, impulse = _segment_inputs(
            model, data, seg, lo, hi + 1, enable_count=False, enable_mark=False
        )
    elif arm == "state_matched_mark_replacement":
        override = None
        if is_event.any() and donor is not None:
            rows = torch.from_numpy(rows_np[is_event].astype(np.int64)).to(device)
            match_state = (
                pre_state.unsqueeze(0).expand(int(is_event.sum()), -1)
                if pre_state_for_match is None
                else pre_state_for_match
            )
            override = state_matched_marks(data, rows, match_state, donor[0], donor[1])
        dt, drive, impulse = _segment_inputs(
            model, data, seg, lo, hi + 1, mark_override=override
        )
    elif arm == "rate_preserving_mark_shuffle":
        override = None
        if is_event.any():
            rows = rows_np[is_event]
            perm = (rng or np.random.default_rng(0)).permutation(rows.size)
            override = data.mark_features[
                torch.from_numpy(rows[perm].astype(np.int64)).to(device)
            ]
        dt, drive, impulse = _segment_inputs(
            model, data, seg, lo, hi + 1, mark_override=override
        )
    elif arm == "burst_thinning":
        dt, drive, impulse = _segment_inputs(model, data, seg, lo, hi + 1)
        if burst_mask is not None:
            drop = torch.from_numpy(burst_mask[lo : hi + 1]).to(device)
            impulse = impulse * (~drop).float().unsqueeze(-1)
    else:
        raise ValueError(f"unknown perturbation arm {arm!r}")

    want = torch.tensor([hi - lo], dtype=torch.long, device=device)
    states, _final = model.rollout(dt, drive, impulse, want, state_init=pre_state)
    return states[0]


def burst_event_mask(
    data: SubjectTensors, seg: int, seed: int
) -> np.ndarray:
    """Which timeline steps are burst events chosen for thinning.

    A burst event is one whose preceding interval is below the patient's own lower
    quartile.  Half of them are dropped, chosen deterministically from the seed so
    the arm is reproducible rather than a fresh draw each run.
    """

    tl = data.timelines[seg]
    mask = np.zeros(tl.n_steps, dtype=bool)
    is_event = tl.event_row >= 0
    idx = np.flatnonzero(is_event)
    if idx.size < 4:
        return mask
    times = tl.step_time[idx]
    gaps = np.diff(times, prepend=times[0])
    threshold = np.quantile(gaps[1:], BURST_QUANTILE) if gaps.size > 1 else 0.0
    in_burst = idx[gaps <= threshold]
    if in_burst.size == 0:
        return mask
    rng = np.random.default_rng(seed + 991)
    chosen = rng.choice(
        in_burst, size=int(round(BURST_THINNING_FRACTION * in_burst.size)), replace=False
    )
    mask[chosen] = True
    return mask
