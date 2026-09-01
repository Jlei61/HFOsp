"""patient -> recording -> seizure crosswalk with a per-onset containment audit.

Contract clauses pinned by ``tests/test_topic5_h2b_crosswalk.py``:

* **C1** the join key is the *recording code*, never the subject string alone.
  ``seizure_index`` in ``src/topic5_group_event_state/source_audit.py`` keys by
  ``(dataset, subject)`` only -- it answers the interictal-*exclusion* question
  ("is this event ictal?"), which is a different question from "which covered
  recording does this seizure live in?" (CLAUDE.md §6.1). It is deliberately
  **not** reused here.
* **C2** every seizure is checked onset-by-onset against the block spans of its
  own recording; the residual is reported so the audit is falsifiable.
* **C3** every input row receives exactly one :class:`Disposition`; the counts
  reconcile against the input length.
* **C4** ambiguity (duplicate id, onset inside two recordings) is surfaced, never
  resolved by picking one.
* **C5** dataset subjects with no seizure rows at all stay visible, because for
  Yuquan "not detected" must not read as "no seizures" (v0.1 data contract §11).
* **C6** the two subject collections are compared by explicit symmetric
  difference; key-set equality is never assumed.
* **C7** degenerate intervals are flagged, neither dropped nor silently kept.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Sequence


class Disposition(Enum):
    """Exactly one of these is assigned to every input seizure row (C3)."""

    MATCHED = "matched"
    SUBJECT_NOT_IN_DATASET = "subject_not_in_dataset"
    RECORDING_ABSENT = "recording_absent"
    ONSET_OUTSIDE_RECORDING = "onset_outside_recording"
    AMBIGUOUS_MULTIPLE_RECORDINGS = "ambiguous_multiple_recordings"
    DUPLICATE_SEIZURE_ID = "duplicate_seizure_id"
    INCOMPLETE_INTERVAL = "incomplete_interval"


@dataclass(frozen=True)
class BlockSpan:
    record_name: str
    start_epoch: float
    end_epoch: float

    def contains(self, t: float) -> bool:
        # Half-open so that back-to-back blocks cannot both claim the seam.
        return self.start_epoch <= t < self.end_epoch


@dataclass(frozen=True)
class RecordingSpan:
    subject: str
    dataset: str
    recording_code: str
    blocks: tuple[BlockSpan, ...]

    @property
    def start_epoch(self) -> float:
        return min(b.start_epoch for b in self.blocks)

    @property
    def end_epoch(self) -> float:
        return max(b.end_epoch for b in self.blocks)

    def block_containing(self, t: float) -> BlockSpan | None:
        for block in self.blocks:
            if block.contains(t):
                return block
        return None

    def gap_to(self, t: float) -> float:
        """Seconds from ``t`` to the nearest covered instant (0.0 if inside)."""
        if self.block_containing(t) is not None:
            return 0.0
        return min(
            min(abs(t - b.start_epoch), abs(t - b.end_epoch)) for b in self.blocks
        )


@dataclass(frozen=True)
class CrosswalkEntry:
    subject: str
    dataset: str
    seizure_id: str
    recording_code: str
    onset_epoch: float
    offset_epoch: float
    duration_sec: float
    disposition: Disposition
    flags: tuple[str, ...]
    block_record_name: str | None
    onset_offset_into_block_sec: float | None
    onset_gap_to_recording_sec: float | None
    containing_recording_codes: tuple[str, ...]


@dataclass(frozen=True)
class CrosswalkResult:
    dataset: str
    entries: tuple[CrosswalkEntry, ...]
    disposition_counts: Mapping[str, int]
    dataset_subjects_without_seizure_rows: tuple[str, ...]
    inventory_subjects_not_in_dataset: tuple[str, ...]
    n_input_rows: int
    per_subject: Mapping[str, Mapping[str, int]] = field(default_factory=dict)


def recording_code_of_record_name(dataset: str, record_name: str) -> str:
    """C1: derive the recording code from a block's ``record_name``.

    Epilepsiae blocks are ``<recording_id>_<block:04d>``; Yuquan blocks *are*
    recordings, so the record name is already the code.
    """

    name = str(record_name).strip()
    if dataset == "epilepsiae":
        head, sep, tail = name.rpartition("_")
        if sep and tail.isdigit():
            return head
        return name
    if dataset == "yuquan":
        return name
    raise ValueError(f"unsupported dataset {dataset!r}")


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def build_recording_index(
    block_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], RecordingSpan]:
    """Group covered blocks into ``(subject, recording_code) -> RecordingSpan``.

    Callers pass the blocks that actually carry group-event artifacts; a
    recording missing from this index is *uncovered*, which is a reportable
    disposition rather than a silent drop.
    """

    grouped: dict[tuple[str, str], list[BlockSpan]] = {}
    owners: dict[tuple[str, str], set[str]] = {}
    for row in block_rows:
        dataset = str(row["dataset"])
        subject = str(row["subject"])
        record_name = str(row["record_name"])
        code = recording_code_of_record_name(dataset, record_name)
        owners.setdefault((dataset, code), set()).add(subject)
        grouped.setdefault((subject, code), []).append(
            BlockSpan(
                record_name=record_name,
                start_epoch=_f(row["block_start_epoch"]),
                end_epoch=_f(row["block_end_epoch"]),
            )
        )

    for (dataset, code), subjects in owners.items():
        if len(subjects) > 1:
            raise ValueError(
                f"recording {code!r} ({dataset}) is shared by {sorted(subjects)}; "
                "a recording code must identify exactly one patient"
            )

    index: dict[tuple[str, str], RecordingSpan] = {}
    for (subject, code), blocks in grouped.items():
        dataset = subject.split("_", 1)[0]
        index[(subject, code)] = RecordingSpan(
            subject=subject,
            dataset=dataset,
            recording_code=code,
            blocks=tuple(sorted(blocks, key=lambda b: b.start_epoch)),
        )
    return index


def _interval_flags(onset: float, offset: float) -> tuple[str, ...]:
    """C7: degenerate intervals are described, not discarded."""

    flags: list[str] = []
    if not math.isfinite(onset):
        flags.append("onset_not_finite")
    if not math.isfinite(offset):
        flags.append("offset_not_finite")
    if math.isfinite(onset) and math.isfinite(offset) and offset <= onset:
        flags.append("zero_duration" if offset == onset else "negative_duration")
    return tuple(flags)


def crosswalk_seizures(
    seizure_rows: Sequence[Mapping[str, Any]],
    recording_index: Mapping[tuple[str, str], RecordingSpan],
    dataset: str,
    dataset_subjects: Iterable[str],
) -> CrosswalkResult:
    """Assign every seizure row exactly one disposition (C3).

    ``dataset_subjects`` are the dataset-qualified subjects under analysis
    (e.g. ``"yuquan_gaolan"``); ``seizure_rows`` carry the *bare* subject as it
    appears in the inventory CSV.
    """

    subjects = set(dataset_subjects)
    subject_recordings: dict[str, list[RecordingSpan]] = {}
    for (subj, _code), span in recording_index.items():
        subject_recordings.setdefault(subj, []).append(span)

    qualified = [f"{dataset}_{str(r['subject']).strip()}" for r in seizure_rows]
    id_counts = Counter(
        (subj, str(row.get("seizure_id", "")).strip())
        for subj, row in zip(qualified, seizure_rows)
    )

    entries: list[CrosswalkEntry] = []
    for subject, row in zip(qualified, seizure_rows):
        seizure_id = str(row.get("seizure_id", "")).strip()
        code = str(row.get("recording_id") or row.get("record") or "").strip()
        onset = _f(row.get("eeg_onset_epoch"))
        offset = _f(row.get("eeg_offset_epoch"))
        flags = _interval_flags(onset, offset)
        duration = offset - onset if math.isfinite(onset) and math.isfinite(offset) else math.nan

        span = recording_index.get((subject, code))
        containing = tuple(
            sorted(
                s.recording_code
                for s in subject_recordings.get(subject, [])
                if math.isfinite(onset) and s.block_containing(onset) is not None
            )
        )

        # Disposition ladder -- order is part of the contract.
        if subject not in subjects:
            disposition = Disposition.SUBJECT_NOT_IN_DATASET
        elif id_counts[(subject, seizure_id)] > 1:
            disposition = Disposition.DUPLICATE_SEIZURE_ID  # C4
        elif "onset_not_finite" in flags or "offset_not_finite" in flags:
            disposition = Disposition.INCOMPLETE_INTERVAL  # C7
        elif span is None:
            disposition = Disposition.RECORDING_ABSENT  # C1 negative
        elif len(containing) > 1:
            disposition = Disposition.AMBIGUOUS_MULTIPLE_RECORDINGS  # C4
        elif span.block_containing(onset) is None:
            disposition = Disposition.ONSET_OUTSIDE_RECORDING  # C2
        else:
            disposition = Disposition.MATCHED

        block = span.block_containing(onset) if span is not None else None
        entries.append(
            CrosswalkEntry(
                subject=subject,
                dataset=dataset,
                seizure_id=seizure_id,
                recording_code=code,
                onset_epoch=onset,
                offset_epoch=offset,
                duration_sec=duration,
                disposition=disposition,
                flags=flags,
                block_record_name=block.record_name if block is not None else None,
                onset_offset_into_block_sec=(
                    onset - block.start_epoch if block is not None else None
                ),
                onset_gap_to_recording_sec=(
                    span.gap_to(onset)
                    if span is not None and math.isfinite(onset)
                    else None
                ),
                containing_recording_codes=containing,
            )
        )

    counts = Counter(e.disposition.value for e in entries)
    with_rows = set(qualified)
    per_subject: dict[str, dict[str, int]] = {}
    for entry in entries:
        bucket = per_subject.setdefault(entry.subject, {})
        bucket[entry.disposition.value] = bucket.get(entry.disposition.value, 0) + 1

    return CrosswalkResult(
        dataset=dataset,
        entries=tuple(entries),
        disposition_counts=dict(counts),
        # C5 / C6: both directions of the symmetric difference are reported.
        dataset_subjects_without_seizure_rows=tuple(sorted(subjects - with_rows)),
        inventory_subjects_not_in_dataset=tuple(sorted(with_rows - subjects)),
        n_input_rows=len(seizure_rows),
        per_subject=per_subject,
    )
