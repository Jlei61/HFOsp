"""Small-memory index contracts for Topic 5 event-innovation analyses.

One position in every sequence is one complete interictal event.  This module
contains no within-event recurrence, features, model, or result access.  Dense
anchors are stored as columnar scalar bounds; event-index windows are resolved
only for one requested row, so an 80-event history is never copied per anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SourceSegment:
    """Metadata required to decide whether state may cross a source boundary."""

    source_id: str
    start_time: float
    stop_time: float
    continuity_group: str | None
    montage_hash: str | None
    continuity_verified: bool


@dataclass(frozen=True)
class ContinuityDecision:
    source_id: str
    continuity_unit_id: str
    previous_source_id: str | None
    gap_seconds: float | None
    montage_compatible: bool | None
    decision: str
    reason: str


@dataclass(frozen=True)
class ContinuitySequence:
    """Chronological event indices in one verified continuity unit."""

    continuity_unit_id: str
    event_indices: np.ndarray
    event_times: np.ndarray
    source_ids: np.ndarray


@dataclass(frozen=True)
class SingleEventAnchors:
    """Columnar bounds for ``pre -> one innovation event -> post`` anchors.

    Positions are half-open bounds into the declared ``ContinuitySequence``.
    ``sequence_index`` refers to that split's sequence tuple.  All arrays have
    one scalar per anchor and use int32, avoiding copied history windows.
    """

    sequence_index: np.ndarray
    pre_start: np.ndarray
    pre_stop: np.ndarray
    innovation_position: np.ndarray
    post_start: np.ndarray
    post_stop: np.ndarray
    pre_events: int
    horizon: int

    def __len__(self) -> int:
        return int(len(self.sequence_index))


@dataclass(frozen=True)
class CumulativeAnchors:
    """Columnar bounds for ``pre -> repeated exposure -> post`` anchors."""

    sequence_index: np.ndarray
    pre_start: np.ndarray
    pre_stop: np.ndarray
    exposure_start: np.ndarray
    exposure_stop: np.ndarray
    post_start: np.ndarray
    post_stop: np.ndarray
    pre_events: int
    exposure_events: int
    horizon: int

    def __len__(self) -> int:
        return int(len(self.sequence_index))


@dataclass(frozen=True)
class AnchorSplits:
    """Dense train anchors and non-overlapping validation/test anchors."""

    train: SingleEventAnchors | CumulativeAnchors
    validation: SingleEventAnchors | CumulativeAnchors
    test: SingleEventAnchors | CumulativeAnchors


@dataclass(frozen=True)
class BlockedChronologicalFold:
    """Scalar position bounds for one expanding, future-blind cross-fit fold."""

    sequence_index: int
    fold_index: int
    train_start: int
    train_stop: int
    embargo_start: int
    embargo_stop: int
    validation_start: int
    validation_stop: int
    embargo_events: int


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _int32(values: np.ndarray | Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(parts).astype(np.int32, copy=False)


def assign_continuity_units(
    segments: Sequence[SourceSegment],
    *,
    maximum_gap_seconds: float,
    maximum_overlap_seconds: float = 0.0,
) -> tuple[ContinuityDecision, ...]:
    """Join only adjacent files with verified metadata continuity.

    Event density never enters this decision.  Missing group or montage
    metadata, an excessive gap/overlap, or an unverified relation forces reset.
    """

    max_gap = float(maximum_gap_seconds)
    max_overlap = float(maximum_overlap_seconds)
    if max_gap < 0 or max_overlap < 0:
        raise ValueError("gap and overlap tolerances must be non-negative")
    if not segments:
        raise ValueError("at least one source segment is required")

    checked: list[SourceSegment] = []
    seen: set[str] = set()
    for segment in segments:
        source_id = str(segment.source_id)
        if source_id in seen:
            raise ValueError(f"duplicate source_id: {source_id}")
        seen.add(source_id)
        start = _finite(segment.start_time, f"{source_id}.start_time")
        stop = _finite(segment.stop_time, f"{source_id}.stop_time")
        if stop <= start:
            raise ValueError(f"{source_id}: stop_time must exceed start_time")
        checked.append(
            SourceSegment(
                source_id=source_id,
                start_time=start,
                stop_time=stop,
                continuity_group=(
                    None
                    if segment.continuity_group is None
                    else str(segment.continuity_group)
                ),
                montage_hash=(
                    None if segment.montage_hash is None else str(segment.montage_hash)
                ),
                continuity_verified=bool(segment.continuity_verified),
            )
        )
    ordered = sorted(checked, key=lambda item: (item.start_time, item.source_id))

    decisions: list[ContinuityDecision] = []
    unit_number = -1
    previous: SourceSegment | None = None
    for current in ordered:
        gap: float | None = None
        compatible: bool | None = None
        join = False
        if previous is None:
            reason = "first_source"
        else:
            gap = float(current.start_time - previous.stop_time)
            if previous.montage_hash is not None and current.montage_hash is not None:
                compatible = previous.montage_hash == current.montage_hash
            if not (previous.continuity_verified and current.continuity_verified):
                reason = "continuity_relationship_unverified"
            elif previous.continuity_group is None or current.continuity_group is None:
                reason = "continuity_group_missing"
            elif previous.continuity_group != current.continuity_group:
                reason = "independent_continuity_group"
            elif compatible is None:
                reason = "montage_hash_missing"
            elif not compatible:
                reason = "montage_mismatch"
            elif gap < -max_overlap:
                reason = "overlap_exceeds_tolerance"
            elif gap > max_gap:
                reason = "gap_exceeds_tolerance"
            else:
                join = True
                reason = "verified_continuous_artificial_split"
        if not join:
            unit_number += 1
        decisions.append(
            ContinuityDecision(
                source_id=current.source_id,
                continuity_unit_id=f"continuity_{unit_number:04d}",
                previous_source_id=None if previous is None else previous.source_id,
                gap_seconds=gap,
                montage_compatible=compatible,
                decision="join_previous" if join else "reset",
                reason=reason,
            )
        )
        previous = current
    return tuple(decisions)


def build_continuity_sequences(
    event_time: np.ndarray,
    event_source_ids: np.ndarray,
    decisions: Sequence[ContinuityDecision],
    *,
    eligible_indices: Sequence[int] | None = None,
) -> tuple[ContinuitySequence, ...]:
    """Build chronological index-only sequences from continuity decisions."""

    times = np.asarray(event_time, dtype=float)
    sources = np.asarray(event_source_ids).astype(str)
    if times.ndim != 1 or sources.shape != times.shape:
        raise ValueError("event_time and event_source_ids must be aligned 1D arrays")
    if not np.all(np.isfinite(times)):
        raise ValueError("event_time contains non-finite values")
    eligible = (
        np.arange(len(times), dtype=np.int64)
        if eligible_indices is None
        else np.asarray(eligible_indices, dtype=np.int64)
    )
    if eligible.ndim != 1 or len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible_indices must be unique and one-dimensional")
    if np.any(eligible < 0) or np.any(eligible >= len(times)):
        raise ValueError("eligible_indices exceed the event arrays")
    if np.any(np.diff(eligible) <= 0):
        raise ValueError("eligible_indices must preserve canonical event order")

    source_to_unit: dict[str, str] = {}
    for decision in decisions:
        source_id = str(decision.source_id)
        if source_id in source_to_unit:
            raise ValueError(f"duplicate continuity decision for {source_id}")
        source_to_unit[source_id] = str(decision.continuity_unit_id)
    missing = sorted(set(sources[eligible]) - set(source_to_unit))
    if missing:
        raise ValueError(f"events lack continuity decisions: {missing}")

    unit_members: dict[str, list[int]] = {}
    for index in eligible:
        unit_members.setdefault(source_to_unit[str(sources[index])], []).append(int(index))
    output: list[ContinuitySequence] = []
    ordered_units = sorted(
        unit_members.items(),
        key=lambda item: float(np.min(times[np.asarray(item[1], dtype=np.int64)])),
    )
    for unit, members in ordered_units:
        indices = np.asarray(members, dtype=np.int64)
        if np.any(np.diff(times[indices]) < 0):
            raise ValueError(f"{unit}: canonical event order is not chronological")
        output.append(
            ContinuitySequence(
                continuity_unit_id=unit,
                event_indices=indices,
                event_times=times[indices].copy(),
                source_ids=sources[indices].copy(),
            )
        )
    if not output:
        raise ValueError("no eligible continuity sequence")
    return tuple(output)


def build_single_event_anchors(
    sequences: Sequence[ContinuitySequence],
    *,
    pre_events: int,
    horizon: int,
    stride: int,
) -> SingleEventAnchors:
    """Build columnar scalar bounds for ``past -> event -> future``."""

    pre, future, step = int(pre_events), int(horizon), int(stride)
    if min(pre, future, step) < 1:
        raise ValueError("pre_events, horizon, and stride must be positive")
    sequence_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    for sequence_index, sequence in enumerate(sequences):
        positions = np.arange(pre, len(sequence.event_indices) - future, step, dtype=np.int32)
        if len(positions):
            sequence_parts.append(np.full(len(positions), sequence_index, dtype=np.int32))
            position_parts.append(positions)
    sequence_column = _concat(sequence_parts)
    position = _concat(position_parts)
    return SingleEventAnchors(
        sequence_index=sequence_column,
        pre_start=position - pre,
        pre_stop=position.copy(),
        innovation_position=position.copy(),
        post_start=position + 1,
        post_stop=position + 1 + future,
        pre_events=pre,
        horizon=future,
    )


def build_cumulative_anchors(
    sequences: Sequence[ContinuitySequence],
    *,
    pre_events: int,
    exposure_events: int,
    horizon: int,
    stride: int,
) -> CumulativeAnchors:
    """Build columnar scalar bounds for ``past -> exposure -> future``."""

    pre = int(pre_events)
    exposure = int(exposure_events)
    future = int(horizon)
    step = int(stride)
    if min(pre, exposure, future, step) < 1:
        raise ValueError("all cumulative-anchor sizes must be positive")
    sequence_parts: list[np.ndarray] = []
    start_parts: list[np.ndarray] = []
    for sequence_index, sequence in enumerate(sequences):
        final_start = len(sequence.event_indices) - exposure - future
        starts = np.arange(pre, final_start + 1, step, dtype=np.int32)
        if len(starts):
            sequence_parts.append(np.full(len(starts), sequence_index, dtype=np.int32))
            start_parts.append(starts)
    sequence_column = _concat(sequence_parts)
    start = _concat(start_parts)
    exposure_stop = start + exposure
    return CumulativeAnchors(
        sequence_index=sequence_column,
        pre_start=start - pre,
        pre_stop=start.copy(),
        exposure_start=start.copy(),
        exposure_stop=exposure_stop,
        post_start=exposure_stop.copy(),
        post_stop=exposure_stop + future,
        pre_events=pre,
        exposure_events=exposure,
        horizon=future,
    )


def build_single_event_anchor_splits(
    split_sequences: Mapping[str, Sequence[ContinuitySequence]],
    *,
    pre_events: int,
    horizon: int,
) -> AnchorSplits:
    """Use stride one for train and horizon spacing for validation/test."""

    _require_splits(split_sequences)
    return AnchorSplits(
        train=build_single_event_anchors(
            split_sequences["train"], pre_events=pre_events, horizon=horizon, stride=1
        ),
        validation=build_single_event_anchors(
            split_sequences["validation"],
            pre_events=pre_events,
            horizon=horizon,
            stride=horizon,
        ),
        test=build_single_event_anchors(
            split_sequences["test"],
            pre_events=pre_events,
            horizon=horizon,
            stride=horizon,
        ),
    )


def build_cumulative_anchor_splits(
    split_sequences: Mapping[str, Sequence[ContinuitySequence]],
    *,
    pre_events: int,
    exposure_events: int,
    horizon: int,
) -> AnchorSplits:
    """Use dense train and max(exposure, horizon) formal spacing."""

    _require_splits(split_sequences)
    formal_stride = max(int(exposure_events), int(horizon))
    kwargs = {
        "pre_events": int(pre_events),
        "exposure_events": int(exposure_events),
        "horizon": int(horizon),
    }
    return AnchorSplits(
        train=build_cumulative_anchors(split_sequences["train"], stride=1, **kwargs),
        validation=build_cumulative_anchors(
            split_sequences["validation"], stride=formal_stride, **kwargs
        ),
        test=build_cumulative_anchors(
            split_sequences["test"], stride=formal_stride, **kwargs
        ),
    )


def _require_splits(
    split_sequences: Mapping[str, Sequence[ContinuitySequence]],
) -> None:
    missing = {"train", "validation", "test"} - set(split_sequences)
    if missing:
        raise ValueError(f"missing sequence splits: {sorted(missing)}")


def resolve_single_event_anchor(
    anchors: SingleEventAnchors,
    row: int,
    sequences: Sequence[ContinuitySequence],
) -> tuple[np.ndarray, int, np.ndarray]:
    """Resolve one anchor on demand without materializing the anchor table."""

    position = int(row)
    sequence = sequences[int(anchors.sequence_index[position])]
    indices = np.asarray(sequence.event_indices, dtype=np.int64)
    return (
        indices[int(anchors.pre_start[position]) : int(anchors.pre_stop[position])],
        int(indices[int(anchors.innovation_position[position])]),
        indices[int(anchors.post_start[position]) : int(anchors.post_stop[position])],
    )


def resolve_cumulative_anchor(
    anchors: CumulativeAnchors,
    row: int,
    sequences: Sequence[ContinuitySequence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one cumulative anchor on demand."""

    position = int(row)
    sequence = sequences[int(anchors.sequence_index[position])]
    indices = np.asarray(sequence.event_indices, dtype=np.int64)
    return (
        indices[int(anchors.pre_start[position]) : int(anchors.pre_stop[position])],
        indices[
            int(anchors.exposure_start[position]) : int(anchors.exposure_stop[position])
        ],
        indices[int(anchors.post_start[position]) : int(anchors.post_stop[position])],
    )


def build_blocked_chronological_crossfit_folds(
    sequences: Sequence[ContinuitySequence],
    *,
    n_splits: int,
    embargo_events: int,
    minimum_train_events: int,
    minimum_validation_events: int = 1,
) -> tuple[BlockedChronologicalFold, ...]:
    """Create expanding future-blind folds using scalar position bounds."""

    splits = int(n_splits)
    embargo = int(embargo_events)
    min_train = int(minimum_train_events)
    min_validation = int(minimum_validation_events)
    if splits < 1 or embargo < 0 or min_train < 1 or min_validation < 1:
        raise ValueError("invalid cross-fit configuration")
    folds: list[BlockedChronologicalFold] = []
    for sequence_index, sequence in enumerate(sequences):
        length = len(sequence.event_indices)
        if length < min_train + embargo + min_validation:
            continue
        quotient, remainder = divmod(length, splits + 1)
        sizes = [quotient + int(index < remainder) for index in range(splits + 1)]
        boundaries = np.cumsum([0, *sizes], dtype=np.int64)
        for chunk_index in range(1, splits + 1):
            validation_start = int(boundaries[chunk_index])
            validation_stop = int(boundaries[chunk_index + 1])
            train_stop = validation_start - embargo
            if train_stop < min_train or validation_stop - validation_start < min_validation:
                continue
            folds.append(
                BlockedChronologicalFold(
                    sequence_index=sequence_index,
                    fold_index=chunk_index - 1,
                    train_start=0,
                    train_stop=train_stop,
                    embargo_start=train_stop,
                    embargo_stop=validation_start,
                    validation_start=validation_start,
                    validation_stop=validation_stop,
                    embargo_events=embargo,
                )
            )
    if not folds:
        raise ValueError("no continuity sequence supports the cross-fit contract")
    return tuple(folds)


def resolve_crossfit_fold(
    fold: BlockedChronologicalFold,
    sequences: Sequence[ContinuitySequence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one cross-fit fold on demand."""

    indices = np.asarray(sequences[int(fold.sequence_index)].event_indices, dtype=np.int64)
    return (
        indices[fold.train_start : fold.train_stop],
        indices[fold.embargo_start : fold.embargo_stop],
        indices[fold.validation_start : fold.validation_stop],
    )


def _aligned_columns(anchors: SingleEventAnchors | CumulativeAnchors) -> bool:
    columns = [
        value
        for value in anchors.__dict__.values()
        if isinstance(value, np.ndarray)
    ]
    return bool(columns) and all(
        column.ndim == 1 and len(column) == len(anchors) for column in columns
    )


def _positions_belong_to_sequences(
    sequence_index: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    sequences: Sequence[ContinuitySequence],
) -> bool:
    if not len(sequence_index):
        return False
    valid_sequence = (sequence_index >= 0) & (sequence_index < len(sequences))
    if not np.all(valid_sequence):
        return False
    lengths = np.asarray(
        [len(sequence.event_indices) for sequence in sequences], dtype=np.int64
    )
    selected_lengths = lengths[sequence_index.astype(np.int64)]
    return bool(
        np.all(starts >= 0)
        and np.all(starts <= stops)
        and np.all(stops <= selected_lengths)
    )


def _formal_targets_nonoverlap(
    sequence_index: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> bool:
    for current in np.unique(sequence_index):
        mask = sequence_index == current
        order = np.argsort(starts[mask], kind="stable")
        selected_starts = starts[mask][order]
        selected_stops = stops[mask][order]
        if len(selected_starts) > 1 and np.any(selected_starts[1:] < selected_stops[:-1]):
            return False
    return True


def audit_single_event_anchors(
    anchors: SingleEventAnchors,
    sequences: Sequence[ContinuitySequence],
    *,
    require_nonoverlap_post: bool,
) -> dict[str, bool | int]:
    """Audit scalar bounds and declared-sequence provenance."""

    aligned = _aligned_columns(anchors)
    exact = bool(
        aligned
        and np.all(anchors.pre_stop - anchors.pre_start == anchors.pre_events)
        and np.all(anchors.pre_stop == anchors.innovation_position)
        and np.all(anchors.post_start == anchors.innovation_position + 1)
        and np.all(anchors.post_stop - anchors.post_start == anchors.horizon)
    )
    belongs = bool(
        aligned
        and _positions_belong_to_sequences(
            anchors.sequence_index, anchors.pre_start, anchors.post_stop, sequences
        )
    )
    nonoverlap = bool(
        not require_nonoverlap_post
        or _formal_targets_nonoverlap(
            anchors.sequence_index, anchors.post_start, anchors.post_stop
        )
    )
    return {
        "n_anchors": len(anchors),
        "nonempty": bool(len(anchors)),
        "columns_aligned": bool(aligned),
        "internally_disjoint": exact,
        "strict_event_order": exact,
        "positions_belong_to_declared_sequences": belongs,
        "formal_post_windows_nonoverlap": nonoverlap,
    }


def audit_cumulative_anchors(
    anchors: CumulativeAnchors,
    sequences: Sequence[ContinuitySequence],
    *,
    require_nonoverlap_post: bool,
) -> dict[str, bool | int]:
    """Audit cumulative scalar bounds and provenance."""

    aligned = _aligned_columns(anchors)
    exact = bool(
        aligned
        and np.all(anchors.pre_stop - anchors.pre_start == anchors.pre_events)
        and np.all(anchors.pre_stop == anchors.exposure_start)
        and np.all(
            anchors.exposure_stop - anchors.exposure_start == anchors.exposure_events
        )
        and np.all(anchors.exposure_stop == anchors.post_start)
        and np.all(anchors.post_stop - anchors.post_start == anchors.horizon)
    )
    belongs = bool(
        aligned
        and _positions_belong_to_sequences(
            anchors.sequence_index, anchors.pre_start, anchors.post_stop, sequences
        )
    )
    nonoverlap = bool(
        not require_nonoverlap_post
        or _formal_targets_nonoverlap(
            anchors.sequence_index, anchors.post_start, anchors.post_stop
        )
    )
    return {
        "n_anchors": len(anchors),
        "nonempty": bool(len(anchors)),
        "columns_aligned": bool(aligned),
        "internally_disjoint": exact,
        "strict_event_order": exact,
        "positions_belong_to_declared_sequences": belongs,
        "formal_post_windows_nonoverlap": nonoverlap,
    }


def audit_crossfit_folds(
    folds: Sequence[BlockedChronologicalFold],
    sequences: Sequence[ContinuitySequence],
) -> dict[str, bool | int]:
    """Audit future blindness, exact embargo, and sequence provenance."""

    belongs = all(
        0 <= fold.sequence_index < len(sequences)
        and 0 <= fold.train_start <= fold.train_stop
        <= fold.embargo_start <= fold.embargo_stop
        <= fold.validation_start <= fold.validation_stop
        <= len(sequences[fold.sequence_index].event_indices)
        for fold in folds
    )
    disjoint = all(
        fold.train_stop <= fold.embargo_start
        and fold.embargo_stop <= fold.validation_start
        for fold in folds
    )
    future_blind = all(
        fold.train_stop > fold.train_start
        and fold.validation_stop > fold.validation_start
        and fold.train_stop <= fold.validation_start
        for fold in folds
    )
    embargo_exact = all(
        fold.embargo_start == fold.train_stop
        and fold.embargo_stop == fold.validation_start
        and fold.embargo_stop - fold.embargo_start == fold.embargo_events
        for fold in folds
    )
    return {
        "n_folds": len(folds),
        "nonempty": bool(len(folds)),
        "bounds_belong_to_declared_sequences": bool(belongs),
        "train_embargo_validation_disjoint": bool(disjoint),
        "training_strictly_precedes_validation": bool(future_blind),
        "embargo_exact_and_ordered": bool(embargo_exact),
    }


def audit_phase0_contract(
    split_sequences: Mapping[str, Sequence[ContinuitySequence]],
    single_anchors: AnchorSplits,
    cumulative_anchors: AnchorSplits,
    crossfit_folds: Sequence[BlockedChronologicalFold],
) -> dict[str, object]:
    """Return one fail-closed report for the Phase-0 index contract."""

    _require_splits(split_sequences)
    split_names = ("train", "validation", "test")
    all_sequences = tuple(
        sequence for split in split_names for sequence in split_sequences[split]
    )
    sequences_valid = all(
        len(sequence.event_indices) == len(sequence.event_times)
        == len(sequence.source_ids)
        and len(sequence.event_indices) == len(np.unique(sequence.event_indices))
        and np.all(np.diff(sequence.event_times) >= 0)
        for sequence in all_sequences
    )
    unit_ids_unique = all(
        len({sequence.continuity_unit_id for sequence in split_sequences[split]})
        == len(split_sequences[split])
        for split in split_names
    )
    split_sets = {
        split: set(
            map(
                int,
                np.concatenate(
                    [sequence.event_indices for sequence in split_sequences[split]]
                )
                if split_sequences[split]
                else np.empty(0, dtype=np.int64),
            )
        )
        for split in split_names
    }
    split_disjoint = (
        split_sets["train"].isdisjoint(split_sets["validation"])
        and split_sets["train"].isdisjoint(split_sets["test"])
        and split_sets["validation"].isdisjoint(split_sets["test"])
    )

    single = {
        split: audit_single_event_anchors(
            getattr(single_anchors, split),
            split_sequences[split],
            require_nonoverlap_post=split != "train",
        )
        for split in split_names
    }
    cumulative = {
        split: audit_cumulative_anchors(
            getattr(cumulative_anchors, split),
            split_sequences[split],
            require_nonoverlap_post=split != "train",
        )
        for split in split_names
    }
    crossfit = audit_crossfit_folds(crossfit_folds, split_sequences["train"])
    checks: list[bool] = [sequences_valid, unit_ids_unique, split_disjoint]
    for family in (single, cumulative):
        for report in family.values():
            checks.extend(value for value in report.values() if isinstance(value, bool))
    checks.extend(value for value in crossfit.values() if isinstance(value, bool))
    anchors_provenance = all(
        report["positions_belong_to_declared_sequences"]
        for family in (single, cumulative)
        for report in family.values()
    )
    return {
        "status": "PASS" if all(checks) else "FAIL",
        "one_step_is_one_complete_event": True,
        "feature_windows_materialized": False,
        "anchor_storage": "columnar_scalar_position_bounds",
        "n_split_sequences": len(all_sequences),
        "sequence_event_indices_unique_and_times_monotonic": bool(sequences_valid),
        "continuity_unit_ids_unique_within_split": bool(unit_ids_unique),
        "split_event_indices_disjoint": bool(split_disjoint),
        "anchors_belong_to_declared_split_and_unit": bool(anchors_provenance),
        "crossfit_indices_belong_to_train_unit": bool(
            crossfit["bounds_belong_to_declared_sequences"]
        ),
        "single_event": single,
        "cumulative": cumulative,
        "crossfit": crossfit,
    }


__all__ = [
    "AnchorSplits",
    "BlockedChronologicalFold",
    "ContinuityDecision",
    "ContinuitySequence",
    "CumulativeAnchors",
    "SingleEventAnchors",
    "SourceSegment",
    "assign_continuity_units",
    "audit_crossfit_folds",
    "audit_cumulative_anchors",
    "audit_phase0_contract",
    "audit_single_event_anchors",
    "build_blocked_chronological_crossfit_folds",
    "build_continuity_sequences",
    "build_cumulative_anchor_splits",
    "build_cumulative_anchors",
    "build_single_event_anchor_splits",
    "build_single_event_anchors",
    "resolve_crossfit_fold",
    "resolve_cumulative_anchor",
    "resolve_single_event_anchor",
]
