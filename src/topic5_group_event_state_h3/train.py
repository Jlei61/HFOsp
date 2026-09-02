"""One training protocol, three arms.

Every knob that is not the feedback edge is held identical across ``M0``, ``M1``
and ``M2``: the same optimizer, the same learning rate, the same number of
truncated-BPTT windows per epoch in the same order, the same early-stopping rule,
and the same checkpoint criterion -- the interictal inner-validation future-block
objective, which never sees a seizure, a development-test block, or any H3
contrast.  If any of those differed, "M1 beat M0" would be a statement about
training budget.

Endpoints are kept apart end to end.  ``count`` and ``conditional mark`` are
optimised with equal weight and *reported* separately, because an arm that only
sharpened the event rate is a rate model, and the numbers have to be able to say
so rather than hiding inside one summed score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .features import EventFeatures, build_block_targets, train_standardiser
from .models import H3Config, H3Model, KIND_EVENT, parameter_report
from .timeline import SegmentTimeline, label_anchors
from .support import Interval


@dataclass
class TrainConfig:
    max_epochs: int = 30
    min_epochs: int = 8
    patience: int = 8
    lr: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    window_steps: int = 0             # 0 = derive from target_windows (see below)
    target_windows: int = 48          # optimizer steps per epoch, held equal across arms
    window_steps_range: tuple[int, int] = (512, 16384)
    max_train_seconds: float = 2400.0
    amp: bool = False                 # the recurrence is exact in fp32 by design

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubjectTensors:
    """Everything one patient's rollout needs, already on the device."""

    timelines: list[SegmentTimeline]
    step_dt: list[Tensor]             # (S,) per segment
    step_drive_in: list[Tensor]       # (S, D) per segment
    step_drive_valid: list[Tensor]    # (S,) per segment
    step_event_row: list[Tensor]      # (S,) long, -1 for non-event steps
    anchor_step: list[Tensor]
    anchor_split: list[np.ndarray]
    anchor_valid: list[dict[int, np.ndarray]]
    targets: list[dict[int, dict[str, Tensor]]]
    count_features: Tensor            # (n_events, 4)
    mark_features: Tensor             # (n_events, F)
    mark_loc: np.ndarray
    mark_scale: np.ndarray
    count_log_mu_init: dict[int, float]
    n_drive_features: int
    mark_groups: tuple[tuple[str, tuple[int, int]], ...]
    train_event_rate_hz: float


def _standardise(x: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - loc) / scale).astype(np.float32)


def prepare_subject(
    features: EventFeatures,
    timelines: Sequence[SegmentTimeline],
    intervals: Sequence[Interval],
    horizons_minutes: Sequence[int],
    device: torch.device,
) -> SubjectTensors:
    """Assemble tensors once; every epoch and every arm then reuses them.

    All standardisation constants come from TRAIN anchors only.  Fitting them on
    the whole stream would leak the held-out distribution into the model's own
    normalisation, which is quiet and real.
    """

    horizons = [int(h) for h in horizons_minutes]
    splits, valids, targets = [], [], []
    for tl in timelines:
        split, valid = label_anchors(tl, intervals, horizons)
        splits.append(split)
        valids.append(valid)
        targets.append(
            {
                h: build_block_targets(features, tl.anchor_time, float(h) * 60.0)
                for h in horizons
            }
        )

    # TRAIN-only statistics -------------------------------------------------
    train_mark_rows, train_counts = [], {h: [] for h in horizons}
    for tl_i, tl in enumerate(timelines):
        is_train = np.asarray([s == "train" for s in splits[tl_i]], dtype=bool)
        for h in horizons:
            keep = is_train & valids[tl_i][h]
            tgt = targets[tl_i][h]
            if keep.any():
                train_counts[h].append(tgt.count[keep])
                if h == horizons[0]:
                    train_mark_rows.append(tgt.mark_mean[keep & tgt.has_events])
    mark_rows = (
        np.concatenate(train_mark_rows, axis=0)
        if train_mark_rows
        else np.zeros((0, features.mark_features.shape[1]), np.float32)
    )
    mark_loc, mark_scale = train_standardiser(
        mark_rows, np.ones(mark_rows.shape[0], dtype=bool)
    )
    count_log_mu_init = {
        h: float(np.log1p(np.median(np.concatenate(train_counts[h])))) if train_counts[h] else 0.0
        for h in horizons
    }

    # TRAIN-only event rate: events per recorded second inside the TRAIN pieces.
    train_seconds = sum(i.duration for i in intervals if i.split == "train")
    train_events = 0
    for interval in intervals:
        if interval.split != "train":
            continue
        lo_i = int(np.searchsorted(features.t_abs, interval.start, side="left"))
        hi_i = int(np.searchsorted(features.t_abs, interval.stop, side="left"))
        train_events += hi_i - lo_i
    train_rate = float(train_events) / max(train_seconds, 1.0)

    drive_loc_rows = np.concatenate([tl.cell_features for tl in timelines], axis=0)
    drive_loc, drive_scale = train_standardiser(
        drive_loc_rows, np.ones(drive_loc_rows.shape[0], dtype=bool)
    )

    count_loc, count_scale = train_standardiser(
        features.count_features, np.ones(features.count_features.shape[0], dtype=bool)
    )
    # The occurrence channel is a constant 1.0 by construction; standardising it
    # would divide by a zero scale and erase the impulse the M1 edge is made of.
    count_scale = np.where(count_scale > 1e-6, count_scale, 1.0)
    count_loc[0] = 0.0
    count_scale[0] = 1.0

    step_dt, drive_in, drive_valid, event_row, anchor_step = [], [], [], [], []
    tgt_tensors: list[dict[int, dict[str, Tensor]]] = []
    for tl_i, tl in enumerate(timelines):
        dt = np.zeros(tl.n_steps, dtype=np.float64)
        if tl.n_steps > 1:
            dt[:-1] = np.diff(tl.step_time)
        step_dt.append(torch.from_numpy(dt.astype(np.float32)).to(device))
        cell_std = _standardise(tl.cell_features, drive_loc, drive_scale)
        drive_in.append(torch.from_numpy(cell_std[tl.step_cell]).to(device))
        drive_valid.append(torch.from_numpy(tl.cell_valid[tl.step_cell]).to(device))
        event_row.append(torch.from_numpy(tl.event_row).to(device))
        anchor_step.append(torch.from_numpy(tl.anchor_step).to(device))
        per_h: dict[int, dict[str, Tensor]] = {}
        for h in horizons:
            tgt = targets[tl_i][h]
            per_h[h] = {
                "count": torch.from_numpy(tgt.count).to(device),
                "has_events": torch.from_numpy(tgt.has_events).to(device),
                "mark_mean": torch.from_numpy(
                    _standardise(tgt.mark_mean, mark_loc, mark_scale)
                ).to(device),
            }
        tgt_tensors.append(per_h)

    return SubjectTensors(
        timelines=list(timelines),
        step_dt=step_dt,
        step_drive_in=drive_in,
        step_drive_valid=drive_valid,
        step_event_row=event_row,
        anchor_step=anchor_step,
        anchor_split=splits,
        anchor_valid=valids,
        targets=tgt_tensors,
        count_features=torch.from_numpy(
            _standardise(features.count_features, count_loc, count_scale)
        ).to(device),
        mark_features=torch.from_numpy(
            _standardise(features.mark_features, mark_loc, mark_scale)
        ).to(device),
        mark_loc=mark_loc,
        mark_scale=mark_scale,
        count_log_mu_init=count_log_mu_init,
        n_drive_features=int(timelines[0].cell_features.shape[1]) if timelines else 0,
        mark_groups=tuple(sorted(features.mark_group_slices.items())),
        train_event_rate_hz=train_rate,
    )


def _segment_inputs(
    model: H3Model,
    data: SubjectTensors,
    seg: int,
    lo: int,
    hi: int,
    *,
    enable_count: bool = True,
    enable_mark: bool = True,
    mark_override: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(dt, drive, impulse)`` for one truncated-BPTT window of one segment.

    ``mark_override`` carries the content perturbation: the event times and count
    are untouched, only the mark vector each event contributes is swapped, which
    is exactly what separates the content estimand from the burden one.
    """

    dt = data.step_dt[seg][lo:hi]
    drive = model.drive(data.step_drive_in[seg][lo:hi], data.step_drive_valid[seg][lo:hi])
    rows = data.step_event_row[seg][lo:hi]
    is_event = rows >= 0
    impulse = torch.zeros(hi - lo, model.cfg.d_state, device=dt.device)
    if bool(is_event.any()) and (model.count_adapter is not None or model.mark_adapter is not None):
        idx = rows[is_event]
        # ``mark_override`` is already one row per *event* in this window, aligned to
        # ``idx``; indexing it with the step-level mask would mix the two frames.
        marks = data.mark_features[idx] if mark_override is None else mark_override
        kick = model.event_impulse(
            data.count_features[idx],
            marks,
            enable_count=enable_count,
            enable_mark=enable_mark,
        )
        impulse = impulse.index_put((torch.nonzero(is_event).flatten(),), kick)
    return dt, drive, impulse


def _score_window(
    model: H3Model,
    data: SubjectTensors,
    seg: int,
    states: Tensor,
    anchor_ids: np.ndarray,
    horizons: Sequence[int],
    keep_mask: np.ndarray,
) -> tuple[Tensor, dict[str, list[np.ndarray]], int]:
    """Loss over the anchors in one window, its scores, and its term count.

    The term count is returned rather than inferred: a window can hold anchors
    that are valid for no horizon at all -- near the end of a short segment, where
    every block would overrun -- and its loss is then a constant zero with no
    graph behind it.  Stepping on that raises; silently skipping it without
    counting would hide how much of an epoch was empty.
    """

    device = states.device
    loss = torch.zeros((), device=device)
    n_terms = 0
    collected: dict[str, list[np.ndarray]] = {}
    for horizon in horizons:
        valid = keep_mask & data.anchor_valid[seg][horizon][anchor_ids]
        if not valid.any():
            continue
        sel = torch.from_numpy(np.flatnonzero(valid)).to(device)
        ids = anchor_ids[valid]
        tgt = data.targets[seg][horizon]
        scores = model.score_blocks(
            states.index_select(0, sel),
            horizon,
            tgt["count"][ids],
            tgt["has_events"][ids],
            tgt["mark_mean"][ids],
        )
        count_ll = scores["count"]
        has = scores["has_events"]
        mark_ll = scores["mark"]
        # Equal weight, never summed into one headline: the mark term is averaged
        # over its own dimensions so a 42-dimensional mark cannot swamp a scalar
        # count simply by being wider.
        loss = loss - count_ll.mean()
        n_terms += 1
        denom = has.float().sum().clamp_min(1.0)
        loss = loss - mark_ll.sum() / (denom * mark_ll.shape[1])
        n_terms += 1
        collected.setdefault(f"count_{horizon}", []).append(
            count_ll.detach().float().cpu().numpy()
        )
        collected.setdefault(f"mark_{horizon}", []).append(
            mark_ll.detach().float().mean(dim=1).cpu().numpy()
        )
        # The named parts of the conditional-mark endpoint.  Reporting only the
        # total would hide an arm that sharpened participation while blurring
        # multiband, which is a different scientific statement.
        group_scores = np.stack(
            [
                mark_ll[:, a:b].detach().float().mean(dim=1).cpu().numpy()
                for _name, (a, b) in data.mark_groups
            ],
            axis=1,
        )
        collected.setdefault(f"mark_groups_{horizon}", []).append(group_scores)
        collected.setdefault(f"has_{horizon}", []).append(
            has.detach().cpu().numpy().astype(bool)
        )
        # Segment id travels with every row: a block is identified by
        # (subject, split, horizon, segment, anchor), never by array position.
        collected.setdefault(f"anchor_ids_{horizon}", []).append(ids)
        collected.setdefault(f"segment_{horizon}", []).append(
            np.full(ids.size, seg, dtype=np.int64)
        )
        collected.setdefault(f"count_true_{horizon}", []).append(
            data.targets[seg][horizon]["count"][ids].detach().cpu().numpy()
        )
    return (loss / max(n_terms, 1)), collected, n_terms


def run_epoch(
    model: H3Model,
    data: SubjectTensors,
    horizons: Sequence[int],
    device: torch.device,
    cfg: TrainConfig,
    *,
    train_split: str,
    optimizer: torch.optim.Optimizer | None = None,
    collect_splits: Sequence[str] = (),
) -> tuple[float, dict[str, dict[str, np.ndarray]]]:
    """One causal pass over every segment, in recording order.

    Windows inside a segment are strictly ordered and the state is carried across
    them with ``detach`` only -- never re-initialised.  Shuffling windows, or
    resetting between them, silently converts a slow-state model into a
    short-memory one while every score still looks plausible.
    """

    is_train = optimizer is not None
    cfg = resolve_window(data, cfg)
    total, n_batches, n_empty_windows = 0.0, 0, 0
    out: dict[str, dict[str, list[np.ndarray]]] = {s: {} for s in collect_splits}

    for seg, tl in enumerate(data.timelines):
        carry = None
        n_steps = tl.n_steps
        for lo in range(0, n_steps, cfg.window_steps):
            hi = min(lo + cfg.window_steps, n_steps)
            want_local = np.flatnonzero(
                (tl.anchor_step >= lo) & (tl.anchor_step < hi)
            )
            if want_local.size == 0:
                with torch.no_grad():
                    dt, drive, impulse = _segment_inputs(model, data, seg, lo, hi)
                    _states, carry = model.rollout(
                        dt, drive, impulse,
                        torch.zeros(0, dtype=torch.long, device=device),
                        state_init=carry,
                    )
                    carry = carry.detach()
                continue

            want = torch.from_numpy(
                (tl.anchor_step[want_local] - lo).astype(np.int64)
            ).to(device)
            context = torch.enable_grad() if is_train else torch.no_grad()
            with context:
                dt, drive, impulse = _segment_inputs(model, data, seg, lo, hi)
                states, new_carry = model.rollout(
                    dt, drive, impulse, want, state_init=carry
                )

                if is_train:
                    keep = np.asarray(
                        [data.anchor_split[seg][i] == train_split for i in want_local]
                    )
                    if keep.any():
                        loss, _scores, n_terms = _score_window(
                            model, data, seg, states, want_local, horizons, keep
                        )
                        if n_terms and loss.requires_grad and torch.isfinite(loss):
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), cfg.grad_clip
                            )
                            if torch.isfinite(norm):
                                optimizer.step()
                            total += float(loss.detach())
                            n_batches += 1
                    states = states.detach()
                    new_carry = new_carry.detach()

                for split_name in collect_splits:
                    keep = np.asarray(
                        [data.anchor_split[seg][i] == split_name for i in want_local]
                    )
                    if not keep.any():
                        continue
                    with torch.no_grad():
                        loss_s, collected, n_terms = _score_window(
                            model, data, seg, states.detach(), want_local, horizons, keep
                        )
                    if not n_terms:
                        continue
                    if split_name not in out:
                        out[split_name] = {}
                    for key, values in collected.items():
                        out[split_name].setdefault(key, []).extend(values)
                    out[split_name].setdefault("_loss", []).append(
                        np.asarray([float(loss_s)])
                    )
                    out[split_name].setdefault("_segment", []).append(
                        np.full(1, seg, dtype=np.int64)
                    )
            carry = new_carry.detach() if is_train else new_carry

    merged: dict[str, dict[str, np.ndarray]] = {}
    for split_name, blocks in out.items():
        merged[split_name] = {
            key: (np.concatenate(vals, axis=0) if vals else np.zeros(0))
            for key, vals in blocks.items()
        }
    return (total / max(n_batches, 1)), merged


def validation_objective(collected: Mapping[str, np.ndarray], horizons: Sequence[int]) -> float:
    """Checkpoint criterion: interictal future-block negative log score only.

    Deliberately blind to seizures, to development-test blocks and to any
    M0/M1/M2 contrast, so the selected checkpoint cannot have been chosen for
    producing the answer this line is testing.
    """

    total, n = 0.0, 0
    for horizon in horizons:
        count = collected.get(f"count_{horizon}")
        mark = collected.get(f"mark_{horizon}")
        has = collected.get(f"has_{horizon}")
        if count is None or count.size == 0:
            continue
        # Weighted by how many blocks each horizon actually has, not by an
        # unweighted mean of per-horizon means.  A 120-minute term resting on five
        # validation blocks otherwise contributes a third of the criterion and
        # swings it by more than a whole epoch of real improvement.
        total -= float(np.sum(count))
        n += int(count.size)
        if mark is not None and mark.size and has is not None and bool(has.any()):
            total -= float(np.sum(mark[has]))
            n += int(has.sum())
    return total / n if n else float("nan")


def resolve_window(data: SubjectTensors, cfg: TrainConfig) -> TrainConfig:
    """Fix the number of optimizer steps per epoch, not the window length.

    A fixed window length gives a 24-hour patient four updates an epoch and a
    400-hour patient sixty, so "same number of epochs" would mean two different
    training budgets.  Deriving the window from the timeline makes the budget the
    same quantity for everyone, and -- because it depends only on the timeline --
    identical across the three arms.
    """

    if cfg.window_steps:
        return cfg
    total = int(sum(tl.n_steps for tl in data.timelines))
    lo, hi = cfg.window_steps_range
    window = int(min(max(total // max(cfg.target_windows, 1), lo), hi))
    return replace(cfg, window_steps=window)


def train_arm(
    model: H3Model,
    data: SubjectTensors,
    horizons: Sequence[int],
    device: torch.device,
    cfg: TrainConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit one arm under the shared protocol and return its selection history."""

    cfg = resolve_window(data, cfg)
    torch.manual_seed(int(seed))
    model.decoder.initialise(data.count_log_mu_init, np.zeros(data.mark_features.shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[dict[str, Any]] = []
    best = {"objective": float("inf"), "epoch": -1}
    best_state: dict[str, Tensor] | None = None
    started = time.time()
    stop_reason = "max_epochs"

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss, _ = run_epoch(
            model, data, horizons, device, cfg,
            train_split="train", optimizer=optimizer,
        )
        model.eval()
        with torch.no_grad():
            _loss, collected = run_epoch(
                model, data, horizons, device, cfg,
                train_split="train", collect_splits=("inner_validation",),
            )
        objective = validation_objective(collected.get("inner_validation", {}), horizons)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "inner_validation_objective": float(objective),
                "seconds": round(time.time() - started, 1),
            }
        )
        if math.isfinite(objective) and objective < best["objective"] - 1e-6:
            best = {"objective": float(objective), "epoch": epoch}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best["epoch"] >= cfg.patience and epoch + 1 >= cfg.min_epochs:
            stop_reason = "early_stopping"
            break
        if time.time() - started > cfg.max_train_seconds and epoch + 1 >= cfg.min_epochs:
            stop_reason = "time_budget"
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "window_steps": int(cfg.window_steps),
        "selected_epoch": best["epoch"],
        "inner_validation_objective": best["objective"],
        "n_epochs_run": len(history),
        "stop_reason": stop_reason,
        "train_seconds": round(time.time() - started, 1),
        "history": history,
        "parameters": parameter_report(model),
    }
