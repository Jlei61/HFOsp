"""
Interictal synchrony aggregation above event-level PR4 outputs.

Mainline contract:
- consumes seizure inventory + event-level synchrony rows
- annotates each event by strict full-containment rules
- keeps temporal continuity explicit via per-subject event segments
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .epilepsiae_dataset import NONTRIVIAL_GAP_SEC
from .preprocessing import (
    build_recording_timeline,
    epoch_to_local_hour,
    parse_seizure_annotation_events,
    read_edf_record_info,
)


@dataclass(frozen=True)
class EpilepsiaeSyncAggregationConfig:
    post_ictal_minutes: float = 60.0
    offset_buffer_minutes: float = 10.0
    nontrivial_gap_sec: float = NONTRIVIAL_GAP_SEC
    day_start_hour: int = 8
    night_start_hour: int = 20

    @property
    def post_ictal_window_sec(self) -> float:
        return 60.0 * (float(self.post_ictal_minutes) + float(self.offset_buffer_minutes))


YUQUAN_SEIZURE_LABELS = (
    "EEG SZ",
    "SZ",
    "SZ1",
    "SZ2",
    "SZ3",
    "SZ4",
    "SZ5",
    "SZ6",
    "SZ7",
    "SZ8",
    "SZ9",
    "SZ10",
    "EEG onset",
    "seizure",
    "Seizure",
    "SEIZURE",
    "onset",
    "Onset",
    "ictal",
    "Ictal",
    "sz onset",
    "seizure onset",
    "clinical seizure",
    "subclinical seizure",
    "electrographic seizure",
)


@dataclass(frozen=True)
class YuquanSyncAggregationConfig:
    post_ictal_minutes: float = 60.0
    offset_buffer_minutes: float = 10.0
    nontrivial_gap_sec: float = 120.0
    timezone_name: str = "Asia/Shanghai"
    day_start_hour: int = 8
    night_start_hour: int = 20
    seizure_labels: Tuple[str, ...] = YUQUAN_SEIZURE_LABELS

    @property
    def post_ictal_window_sec(self) -> float:
        return 60.0 * (float(self.post_ictal_minutes) + float(self.offset_buffer_minutes))


def _read_csv_rows(csv_path: Path | str) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_float(value: object) -> Optional[float]:
    if value in (None, "", "nan", "NaN"):
        return None
    return float(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(median(values))


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _classify_day_night_from_hour(
    local_hour: int,
    *,
    day_start_hour: int,
    night_start_hour: int,
) -> str:
    if int(day_start_hour) <= int(local_hour) < int(night_start_hour):
        return "day"
    return "night"


def load_sync_event_rows(csv_path: Path | str) -> List[Dict[str, object]]:
    rows = _read_csv_rows(csv_path)
    float_cols = {
        "event_index",
        "event_start_sec_rel",
        "event_end_sec_rel",
        "event_center_sec_rel",
        "event_duration_sec",
        "event_start_epoch",
        "event_end_epoch",
        "event_center_epoch",
        "n_participating",
        "n_core",
        "n_penumbra",
        "sync_phase_global",
        "sync_phase_core",
        "sync_phase_penumbra",
        "sync_legacy_global",
        "sync_legacy_core",
        "sync_legacy_penumbra",
        "sync_span_global",
        "sync_span_core",
        "sync_span_penumbra",
        "jaccard_global_with_next",
        "jaccard_core_with_next",
        "jaccard_penumbra_with_next",
        "block_start_epoch",
        "block_end_epoch",
        "n_channels",
        "n_core_channels",
        "n_penumbra_channels",
    }
    out: List[Dict[str, object]] = []
    for row in rows:
        parsed: Dict[str, object] = dict(row)
        for col in float_cols:
            parsed[col] = _as_float(parsed.get(col))
        out.append(parsed)
    return out


def load_epilepsiae_sync_summary_rows(csv_path: Path | str) -> List[Dict[str, object]]:
    """Backward-compatible alias; now loads event rows."""
    return load_sync_event_rows(csv_path)


def load_epilepsiae_block_inventory_rows(csv_path: Path | str) -> List[Dict[str, object]]:
    rows = _read_csv_rows(csv_path)
    float_cols = {
        "block_start_epoch",
        "block_end_epoch",
        "gap_to_prev_sec",
        "sample_rate_sql",
        "head_start_epoch",
        "head_duration_sec",
        "head_sample_rate",
        "head_sql_start_delta_sec",
    }
    int_cols = {
        "block_no",
        "n_channels_sql",
        "head_n_channels",
        "intracranial_channels",
        "scalp_or_aux_channels",
        "block_start_local_hour",
        "block_end_local_hour",
    }
    bool_cols = {
        "head_exists",
        "data_exists",
        "raw_gpu_exists",
        "artifact_gpu_exists",
        "has_lagpat",
        "has_lagpat_freq",
        "has_packed_times",
        "has_packed_times_freq",
        "has_eeg",
    }
    out: List[Dict[str, object]] = []
    for row in rows:
        parsed: Dict[str, object] = dict(row)
        for col in float_cols:
            parsed[col] = _as_float(parsed.get(col))
        for col in int_cols:
            parsed[col] = None if parsed.get(col) in (None, "") else int(str(parsed[col]))
        for col in bool_cols:
            parsed[col] = _as_bool(parsed.get(col))
        out.append(parsed)
    return out


def load_epilepsiae_seizure_inventory_rows(csv_path: Path | str) -> List[Dict[str, object]]:
    rows = _read_csv_rows(csv_path)
    float_cols = {
        "eeg_onset_epoch",
        "eeg_offset_epoch",
        "clin_onset_epoch",
        "clin_offset_epoch",
        "eeg_duration_sec",
        "clin_duration_sec",
        "seizure_interval_from_prev_sec",
    }
    int_cols = {"eeg_onset_local_hour", "clin_onset_local_hour"}
    bool_cols = {"has_complete_eeg_interval", "has_complete_clin_interval"}
    out: List[Dict[str, object]] = []
    for row in rows:
        parsed: Dict[str, object] = dict(row)
        for col in float_cols:
            parsed[col] = _as_float(parsed.get(col))
        for col in int_cols:
            parsed[col] = None if parsed.get(col) in (None, "") else int(str(parsed[col]))
        for col in bool_cols:
            parsed[col] = _as_bool(parsed.get(col))
        out.append(parsed)
    return out


def _event_key(row: Mapping[str, object]) -> Tuple[str, str, int]:
    subject = str(row.get("subject", ""))
    block_stem = str(row.get("block_stem", ""))
    event_index = int(_as_float(row.get("event_index")) or -1)
    return (subject, block_stem, event_index)


def _event_bounds(row: Mapping[str, object]) -> Tuple[Optional[float], Optional[float]]:
    start = _as_float(row.get("event_start_epoch"))
    end = _as_float(row.get("event_end_epoch"))
    if start is None:
        start = _as_float(row.get("event_center_epoch"))
    if end is None:
        end = _as_float(row.get("event_center_epoch"))
    if start is None:
        start = _as_float(row.get("block_start_epoch"))
    if end is None:
        end = _as_float(row.get("block_end_epoch"))
    if start is not None and end is not None and end < start:
        end = start
    return start, end


def _build_event_continuous_segments(
    event_rows: Sequence[Mapping[str, object]],
    *,
    nontrivial_gap_sec: float,
) -> Dict[Tuple[str, str, int], Dict[str, object]]:
    by_subject: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in event_rows:
        by_subject[str(row["subject"])].append(row)

    out: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for subject, rows in by_subject.items():
        prev_end: Optional[float] = None
        segment_id = 0
        def _start_key(x: Mapping[str, object]) -> float:
            start, _ = _event_bounds(x)
            return float(start or 0.0)
        for row in sorted(rows, key=_start_key):
            start, end = _event_bounds(row)
            observed_gap = None if prev_end is None or start is None else float(start - prev_end)
            starts_new = bool(
                prev_end is None
                or observed_gap is None
                or observed_gap > float(nontrivial_gap_sec)
            )
            if starts_new:
                segment_id += 1
            out[_event_key(row)] = {
                "continuous_segment_id": segment_id,
                "gap_from_prev_observed_sec": observed_gap,
                "starts_new_continuous_segment": starts_new,
            }
            prev_end = end if end is not None else start
    return out


def _build_interval_rows_from_complete_seizures(
    complete_rows_by_subject: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    post_ictal_window_sec: float,
) -> List[Dict[str, object]]:
    interval_rows: List[Dict[str, object]] = []
    for subject, rows in complete_rows_by_subject.items():
        ordered = sorted(rows, key=lambda x: float(x["eeg_onset_epoch"] or 0.0))
        for idx in range(1, len(ordered)):
            prev = ordered[idx - 1]
            curr = ordered[idx]
            prev_onset = _as_float(prev["eeg_onset_epoch"])
            prev_offset = _as_float(prev["eeg_offset_epoch"])
            next_onset = _as_float(curr["eeg_onset_epoch"])
            next_offset = _as_float(curr.get("eeg_offset_epoch"))
            if prev_onset is None or prev_offset is None or next_onset is None:
                continue
            clean_start = prev_offset
            clean_end = next_onset
            post_end = min(clean_end, clean_start + float(post_ictal_window_sec))
            interval_rows.append(
                {
                    "subject": subject,
                    "patient_code": prev.get("patient_code", ""),
                    "seizure_interval_id": f"{subject}_szint_{idx:03d}",
                    "interval_index": idx,
                    "prev_seizure_id": prev["seizure_id"],
                    "next_seizure_id": curr["seizure_id"],
                    "prev_recording_id": prev.get("recording_id", ""),
                    "next_recording_id": curr.get("recording_id", ""),
                    "prev_eeg_onset_epoch": prev_onset,
                    "prev_eeg_offset_epoch": prev_offset,
                    "next_eeg_onset_epoch": next_onset,
                    "next_eeg_offset_epoch": next_offset,
                    "seizure_onset_to_onset_sec": next_onset - prev_onset,
                    "clean_between_seizures_start_epoch": clean_start,
                    "clean_between_seizures_end_epoch": clean_end,
                    "clean_between_seizures_sec": max(0.0, clean_end - clean_start),
                    "post_ictal_start_epoch": clean_start,
                    "post_ictal_end_epoch": post_end,
                    "post_ictal_available_sec": max(0.0, post_end - clean_start),
                    "interictal_start_epoch": post_end,
                    "interictal_end_epoch": clean_end,
                    "interictal_available_sec": max(0.0, clean_end - post_end),
                    "timezone_name": prev.get("timezone_name", "") or curr.get("timezone_name", ""),
                }
            )
    return interval_rows


def build_epilepsiae_seizure_intervals(
    seizure_rows: Sequence[Mapping[str, object]],
    *,
    config: Optional[EpilepsiaeSyncAggregationConfig] = None,
) -> List[Dict[str, object]]:
    cfg = config or EpilepsiaeSyncAggregationConfig()
    by_subject: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in seizure_rows:
        if _as_bool(row.get("has_complete_eeg_interval")):
            by_subject[str(row["subject"])].append(row)
    return _build_interval_rows_from_complete_seizures(
        by_subject,
        post_ictal_window_sec=cfg.post_ictal_window_sec,
    )


def _interval_overlap(start: float, end: float, left: float, right: float) -> bool:
    return max(start, left) < min(end, right)


def _classify_event_day_night(
    *,
    event_start: float,
    event_end: float,
    timezone_name: str,
    day_start_hour: int,
    night_start_hour: int,
) -> Tuple[str, str, bool]:
    end_probe = event_end if event_end <= event_start else max(event_start, event_end - 1e-6)
    start_hour = epoch_to_local_hour(event_start, timezone_name)
    end_hour = epoch_to_local_hour(end_probe, timezone_name)
    start_label = _classify_day_night_from_hour(
        start_hour,
        day_start_hour=day_start_hour,
        night_start_hour=night_start_hour,
    )
    end_label = _classify_day_night_from_hour(
        end_hour,
        day_start_hour=day_start_hour,
        night_start_hour=night_start_hour,
    )
    return start_label, end_label, start_label == end_label


def build_yuquan_seizure_inventory(
    data_root: Path | str,
    *,
    subjects: Optional[Sequence[str]] = None,
    config: Optional[YuquanSyncAggregationConfig] = None,
) -> List[Dict[str, object]]:
    cfg = config or YuquanSyncAggregationConfig()
    root = Path(data_root)
    subject_names = (
        sorted(str(x) for x in subjects)
        if subjects is not None
        else sorted(x.name for x in root.iterdir() if x.is_dir())
    )
    seizure_rows: List[Dict[str, object]] = []
    for subject in subject_names:
        subject_dir = root / subject
        edf_paths = sorted(subject_dir.glob("*.edf"))
        if not edf_paths:
            continue
        timeline = build_recording_timeline([str(x) for x in edf_paths])
        by_record = {
            str(row["record"]): dict(row) for row in timeline["records"]
        }
        seizure_idx = 0
        for edf_path in edf_paths:
            info = read_edf_record_info(edf_path)
            parsed = parse_seizure_annotation_events(
                edf_path,
                list(cfg.seizure_labels),
                float(info["start_epoch"]),
            )
            for onset_epoch, offset_epoch in parsed["intervals"]:
                seizure_idx += 1
                local_hour = epoch_to_local_hour(float(onset_epoch), cfg.timezone_name)
                seizure_rows.append(
                    {
                        "subject": subject,
                        "patient_code": subject,
                        "recording_id": str(info["record"]),
                        "record": str(info["record"]),
                        "seizure_id": f"{subject}_sz_{seizure_idx:03d}",
                        "eeg_onset_epoch": float(onset_epoch),
                        "eeg_offset_epoch": float(offset_epoch),
                        "eeg_duration_sec": float(offset_epoch - onset_epoch),
                        "has_complete_eeg_interval": True,
                        "timezone_name": cfg.timezone_name,
                        "eeg_onset_local_hour": local_hour,
                        "eeg_onset_day_night": _classify_day_night_from_hour(
                            local_hour,
                            day_start_hour=cfg.day_start_hour,
                            night_start_hour=cfg.night_start_hour,
                        ),
                        "record_start_epoch": by_record[str(info["record"])]["start_epoch"],
                        "record_end_epoch": by_record[str(info["record"])]["end_epoch"],
                    }
                )
    return seizure_rows


def build_yuquan_seizure_intervals(
    seizure_rows: Sequence[Mapping[str, object]],
    *,
    config: Optional[YuquanSyncAggregationConfig] = None,
) -> List[Dict[str, object]]:
    cfg = config or YuquanSyncAggregationConfig()
    by_subject: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in seizure_rows:
        if _as_bool(row.get("has_complete_eeg_interval")):
            by_subject[str(row["subject"])].append(row)
    return _build_interval_rows_from_complete_seizures(
        by_subject,
        post_ictal_window_sec=cfg.post_ictal_window_sec,
    )


def _annotate_sync_events_against_intervals(
    event_rows: Sequence[Mapping[str, object]],
    seizure_rows: Sequence[Mapping[str, object]],
    interval_rows: Sequence[Mapping[str, object]],
    *,
    nontrivial_gap_sec: float,
    day_start_hour: int,
    night_start_hour: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    segment_meta = _build_event_continuous_segments(
        event_rows, nontrivial_gap_sec=float(nontrivial_gap_sec)
    )
    seizures_by_subject: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in seizure_rows:
        if _as_bool(row.get("has_complete_eeg_interval")):
            seizures_by_subject[str(row["subject"])].append(row)
    intervals_by_subject: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in interval_rows:
        intervals_by_subject[str(row["subject"])].append(row)

    annotated_rows: List[Dict[str, object]] = []
    exclusion_counts: Counter[str] = Counter()
    for event in event_rows:
        subject = str(event["subject"])
        event_start, event_end = _event_bounds(event)
        if event_start is None or event_end is None:
            exclusion_counts["missing_event_epoch"] += 1
            continue

        segment = segment_meta[_event_key(event)]
        overlap_seizure = False
        for seizure in seizures_by_subject.get(subject, []):
            onset = _as_float(seizure["eeg_onset_epoch"])
            offset = _as_float(seizure["eeg_offset_epoch"])
            if onset is None or offset is None:
                continue
            if _interval_overlap(event_start, event_end, onset, offset):
                overlap_seizure = True
                break

        containing_intervals = [
            interval
            for interval in intervals_by_subject.get(subject, [])
            if event_start >= float(interval["clean_between_seizures_start_epoch"])
            and event_end <= float(interval["clean_between_seizures_end_epoch"])
        ]
        overlaps_any_interval = any(
            _interval_overlap(
                event_start,
                event_end,
                float(interval["clean_between_seizures_start_epoch"]),
                float(interval["clean_between_seizures_end_epoch"]),
            )
            for interval in intervals_by_subject.get(subject, [])
        )

        interval = containing_intervals[0] if len(containing_intervals) == 1 else None
        interval_status = "assigned"
        if overlap_seizure:
            interval_status = "overlaps_seizure"
        elif len(containing_intervals) > 1:
            interval_status = "multiple_interval_match"
        elif interval is None and overlaps_any_interval:
            interval_status = "crosses_interval_boundary"
        elif interval is None:
            interval_status = "outside_complete_intervals"

        phase_window_type = ""
        phase_eligible = False
        if interval is not None and not overlap_seizure:
            post_start = float(interval["post_ictal_start_epoch"])
            post_end = float(interval["post_ictal_end_epoch"])
            inter_start = float(interval["interictal_start_epoch"])
            inter_end = float(interval["interictal_end_epoch"])
            if post_end > post_start and event_start >= post_start and event_end <= post_end:
                phase_window_type = "post_ictal"
                phase_eligible = True
            elif inter_end > inter_start and event_start >= inter_start and event_end <= inter_end:
                phase_window_type = "interictal"
                phase_eligible = True
            else:
                phase_window_type = "phase_boundary_crossing"

        timezone_name = str(event.get("timezone_name") or interval.get("timezone_name") or "")
        diurnal_window_type = ""
        diurnal_eligible = False
        diurnal_start = ""
        diurnal_end = ""
        if timezone_name:
            diurnal_start, diurnal_end, diurnal_eligible = _classify_event_day_night(
                event_start=event_start,
                event_end=event_end,
                timezone_name=timezone_name,
                day_start_hour=day_start_hour,
                night_start_hour=night_start_hour,
            )
            if diurnal_eligible:
                diurnal_window_type = diurnal_start
            else:
                diurnal_window_type = "day_night_transition"
        elif event.get("timezone_name") or interval.get("timezone_name"):
            diurnal_window_type = "day_night_transition"

        reasons: List[str] = []
        if interval_status != "assigned":
            reasons.append(interval_status)
        if not phase_eligible:
            reasons.append(phase_window_type or "phase_not_eligible")
        if not diurnal_eligible:
            reasons.append(diurnal_window_type or "day_night_not_eligible")
        if float(segment["gap_from_prev_observed_sec"] or 0.0) > float(nontrivial_gap_sec):
            reasons.append("nontrivial_gap_before_event")

        row = {
            "subject": subject,
            "patient_code": event.get("patient_code", ""),
            "block_stem": event.get("block_stem", ""),
            "recording_id": event.get("recording_id", ""),
            "event_start_epoch": event_start,
            "event_end_epoch": event_end,
            "event_duration_sec": float(event_end - event_start),
            "event_start_day_night": diurnal_start,
            "event_end_day_night": diurnal_end,
            "timezone_name": timezone_name,
            "day_night_rule": event.get("day_night_rule", ""),
            "continuous_segment_id": int(segment["continuous_segment_id"]),
            "gap_from_prev_observed_sec": segment["gap_from_prev_observed_sec"],
            "starts_new_continuous_segment": bool(segment["starts_new_continuous_segment"]),
            "has_gap_before_event": bool(
                segment["gap_from_prev_observed_sec"] is not None
                and float(segment["gap_from_prev_observed_sec"]) > float(nontrivial_gap_sec)
            ),
            "interval_assignment_status": interval_status,
            "overlaps_complete_eeg_seizure": overlap_seizure,
            "phase_window_type": phase_window_type,
            "phase_eligible": phase_eligible,
            "diurnal_window_type": diurnal_window_type,
            "diurnal_eligible": diurnal_eligible,
            "exclusion_reasons": "|".join(dict.fromkeys(reasons)),
            "seizure_interval_id": "" if interval is None else interval["seizure_interval_id"],
            "prev_seizure_id": "" if interval is None else interval["prev_seizure_id"],
            "next_seizure_id": "" if interval is None else interval["next_seizure_id"],
            "prev_eeg_onset_epoch": None if interval is None else interval["prev_eeg_onset_epoch"],
            "prev_eeg_offset_epoch": None if interval is None else interval["prev_eeg_offset_epoch"],
            "next_eeg_onset_epoch": None if interval is None else interval["next_eeg_onset_epoch"],
            "next_eeg_offset_epoch": None if interval is None else interval["next_eeg_offset_epoch"],
            "clean_between_seizures_sec": None
            if interval is None
            else interval["clean_between_seizures_sec"],
            "post_ictal_available_sec": None
            if interval is None
            else interval["post_ictal_available_sec"],
            "interictal_available_sec": None
            if interval is None
            else interval["interictal_available_sec"],
        }
        for key, value in event.items():
            if key not in row:
                row[key] = value
        annotated_rows.append(row)
        for reason in reasons:
            exclusion_counts[reason] += 1

    summary = {
        "n_event_rows_input": len(event_rows),
        "n_annotated_rows": len(annotated_rows),
        "n_assigned_interval_rows": sum(
            1 for row in annotated_rows if row["interval_assignment_status"] == "assigned"
        ),
        "n_phase_eligible_rows": sum(1 for row in annotated_rows if row["phase_eligible"]),
        "n_diurnal_eligible_rows": sum(1 for row in annotated_rows if row["diurnal_eligible"]),
        "exclusion_reason_counts": dict(exclusion_counts),
    }
    return annotated_rows, list(interval_rows), summary


def annotate_epilepsiae_sync_events(
    event_rows: Sequence[Mapping[str, object]],
    seizure_rows: Sequence[Mapping[str, object]],
    *,
    config: Optional[EpilepsiaeSyncAggregationConfig] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    cfg = config or EpilepsiaeSyncAggregationConfig()
    intervals = build_epilepsiae_seizure_intervals(seizure_rows, config=cfg)
    return _annotate_sync_events_against_intervals(
        event_rows,
        seizure_rows,
        intervals,
        nontrivial_gap_sec=cfg.nontrivial_gap_sec,
        day_start_hour=cfg.day_start_hour,
        night_start_hour=cfg.night_start_hour,
    )


def annotate_yuquan_sync_events(
    event_rows: Sequence[Mapping[str, object]],
    seizure_rows: Sequence[Mapping[str, object]],
    *,
    config: Optional[YuquanSyncAggregationConfig] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    cfg = config or YuquanSyncAggregationConfig()
    intervals = build_yuquan_seizure_intervals(seizure_rows, config=cfg)
    return _annotate_sync_events_against_intervals(
        event_rows,
        seizure_rows,
        intervals,
        nontrivial_gap_sec=cfg.nontrivial_gap_sec,
        day_start_hour=cfg.day_start_hour,
        night_start_hour=cfg.night_start_hour,
    )


def _metric_values(rows: Sequence[Mapping[str, object]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        value = _as_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def _aggregate_sync_rows(
    annotated_rows: Sequence[Mapping[str, object]],
    seizure_interval_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    interval_by_id = {
        str(row["seizure_interval_id"]): dict(row) for row in seizure_interval_rows
    }
    grouped: Dict[Tuple[str, str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in annotated_rows:
        interval_id = str(row.get("seizure_interval_id") or "")
        if not interval_id:
            continue
        if _as_bool(row.get("phase_eligible")):
            grouped[(str(row["subject"]), interval_id, "phase", str(row["phase_window_type"]))].append(
                row
            )
        if _as_bool(row.get("diurnal_eligible")):
            grouped[
                (str(row["subject"]), interval_id, "diurnal", str(row["diurnal_window_type"]))
            ].append(row)

    metric_cols = [
        "sync_phase_global",
        "sync_legacy_global",
        "sync_span_global",
        "jaccard_global_with_next",
        "n_participating",
        "n_core",
        "n_penumbra",
    ]
    interval_window_rows: List[Dict[str, object]] = []
    subject_grouped: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for (subject, interval_id, family, window_type), rows in sorted(grouped.items()):
        interval = interval_by_id[interval_id]
        segment_ids = sorted({int(row["continuous_segment_id"]) for row in rows})
        out = {
            "subject": subject,
            "patient_code": rows[0].get("patient_code", ""),
            "seizure_interval_id": interval_id,
            "interval_index": interval["interval_index"],
            "window_family": family,
            "window_type": window_type,
            "n_events": len(rows),
            "n_continuous_segments": len(segment_ids),
            "continuous_segment_ids": "|".join(str(x) for x in segment_ids),
            "has_gap_split": len(segment_ids) > 1,
            "eligible_for_primary_stats": len(segment_ids) == 1,
            "event_coverage_sec": sum(float(row["event_duration_sec"]) for row in rows),
            "event_start_min_epoch": min(float(row["event_start_epoch"]) for row in rows),
            "event_end_max_epoch": max(float(row["event_end_epoch"]) for row in rows),
            "prev_seizure_id": interval["prev_seizure_id"],
            "next_seizure_id": interval["next_seizure_id"],
            "prev_eeg_onset_epoch": interval["prev_eeg_onset_epoch"],
            "prev_eeg_offset_epoch": interval["prev_eeg_offset_epoch"],
            "next_eeg_onset_epoch": interval["next_eeg_onset_epoch"],
            "next_eeg_offset_epoch": interval["next_eeg_offset_epoch"],
            "seizure_onset_to_onset_sec": interval["seizure_onset_to_onset_sec"],
            "clean_between_seizures_sec": interval["clean_between_seizures_sec"],
            "post_ictal_available_sec": interval["post_ictal_available_sec"],
            "interictal_available_sec": interval["interictal_available_sec"],
            "timezone_name": rows[0].get("timezone_name", ""),
            "day_night_rule": rows[0].get("day_night_rule", ""),
        }
        for metric in metric_cols:
            values = _metric_values(rows, metric)
            out[f"{metric}_mean_across_events"] = _mean_or_none(values)
            out[f"{metric}_median_across_events"] = _median_or_none(values)
        strata = [str(row.get("event_stratum", "")) for row in rows]
        out["frac_core_only_events"] = float(np.mean([x == "core_only" for x in strata]))
        out["frac_penumbra_only_events"] = float(np.mean([x == "penumbra_only" for x in strata]))
        out["frac_mixed_events"] = float(np.mean([x == "mixed" for x in strata]))
        interval_window_rows.append(out)
        subject_grouped[(subject, family, window_type)].extend(rows)

    subject_window_rows: List[Dict[str, object]] = []
    for (subject, family, window_type), rows in sorted(subject_grouped.items()):
        interval_ids = sorted({str(row["seizure_interval_id"]) for row in rows if row.get("seizure_interval_id")})
        segment_ids = sorted({int(row["continuous_segment_id"]) for row in rows})
        out = {
            "subject": subject,
            "patient_code": rows[0].get("patient_code", ""),
            "window_family": family,
            "window_type": window_type,
            "n_events": len(rows),
            "n_seizure_intervals": len(interval_ids),
            "seizure_interval_ids": "|".join(interval_ids),
            "n_continuous_segments": len(segment_ids),
            "has_gap_split": len(segment_ids) > 1,
            "event_coverage_sec": sum(float(row["event_duration_sec"]) for row in rows),
            "timezone_name": rows[0].get("timezone_name", ""),
            "day_night_rule": rows[0].get("day_night_rule", ""),
        }
        for metric in metric_cols:
            values = _metric_values(rows, metric)
            out[f"{metric}_mean_across_events"] = _mean_or_none(values)
            out[f"{metric}_median_across_events"] = _median_or_none(values)
        strata = [str(row.get("event_stratum", "")) for row in rows]
        out["frac_core_only_events"] = float(np.mean([x == "core_only" for x in strata]))
        out["frac_penumbra_only_events"] = float(np.mean([x == "penumbra_only" for x in strata]))
        out["frac_mixed_events"] = float(np.mean([x == "mixed" for x in strata]))
        subject_window_rows.append(out)

    summary = {
        "n_interval_window_rows": len(interval_window_rows),
        "n_subject_window_rows": len(subject_window_rows),
        "window_family_counts": {
            f"{family}:{window_type}": count
            for (family, window_type), count in Counter(
                (row["window_family"], row["window_type"]) for row in interval_window_rows
            ).items()
        },
    }
    return interval_window_rows, subject_window_rows, summary


def aggregate_epilepsiae_sync_rows(
    annotated_rows: Sequence[Mapping[str, object]],
    seizure_interval_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    return _aggregate_sync_rows(annotated_rows, seizure_interval_rows)


def aggregate_yuquan_sync_rows(
    annotated_rows: Sequence[Mapping[str, object]],
    seizure_interval_rows: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    return _aggregate_sync_rows(annotated_rows, seizure_interval_rows)


def _write_csv(path: Path | str, rows: Sequence[Mapping[str, object]]) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [dict(row) for row in rows]
    if not rows_list:
        out_path.write_text("", encoding="utf-8")
        return str(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        writer.writerows(rows_list)
    return str(out_path)


def run_epilepsiae_sync_aggregation(
    *,
    seizure_inventory_csv: Path | str,
    sync_event_csv: Path | str,
    output_dir: Path | str,
    config: Optional[EpilepsiaeSyncAggregationConfig] = None,
) -> Dict[str, object]:
    cfg = config or EpilepsiaeSyncAggregationConfig()
    sync_rows = load_sync_event_rows(sync_event_csv)
    seizure_rows = load_epilepsiae_seizure_inventory_rows(seizure_inventory_csv)

    annotated_rows, interval_rows, annotation_summary = annotate_epilepsiae_sync_events(
        sync_rows,
        seizure_rows,
        config=cfg,
    )
    interval_window_rows, subject_window_rows, aggregate_summary = aggregate_epilepsiae_sync_rows(
        annotated_rows,
        interval_rows,
    )

    out_dir = Path(output_dir)
    outputs = {
        "event_annotations_csv": _write_csv(
            out_dir / "epilepsiae_sync_event_annotations.csv", annotated_rows
        ),
        "interval_window_csv": _write_csv(
            out_dir / "epilepsiae_sync_interval_window_table.csv", interval_window_rows
        ),
        "subject_window_csv": _write_csv(
            out_dir / "epilepsiae_sync_subject_window_summary.csv", subject_window_rows
        ),
        "summary_json": str(out_dir / "epilepsiae_sync_aggregation_summary.json"),
    }
    summary = {
        "config": {
            "post_ictal_minutes": cfg.post_ictal_minutes,
            "offset_buffer_minutes": cfg.offset_buffer_minutes,
            "post_ictal_window_sec": cfg.post_ictal_window_sec,
            "nontrivial_gap_sec": cfg.nontrivial_gap_sec,
        },
        "annotation": annotation_summary,
        "aggregation": aggregate_summary,
        "outputs": outputs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs["summary_json"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


def run_yuquan_sync_aggregation(
    *,
    sync_event_csv: Path | str,
    output_dir: Path | str,
    data_root: Path | str = "/mnt/yuquan_data/yuquan_24h_edf",
    subjects: Optional[Sequence[str]] = None,
    config: Optional[YuquanSyncAggregationConfig] = None,
) -> Dict[str, object]:
    cfg = config or YuquanSyncAggregationConfig()
    sync_rows = load_sync_event_rows(sync_event_csv)
    for row in sync_rows:
        row.setdefault("timezone_name", cfg.timezone_name)
        row.setdefault(
            "day_night_rule",
            f"day={int(cfg.day_start_hour):02d}:00-{int(cfg.night_start_hour):02d}:00 local",
        )
    if subjects is None:
        subjects = sorted({str(row["subject"]) for row in sync_rows})
    seizure_rows = build_yuquan_seizure_inventory(
        data_root,
        subjects=subjects,
        config=cfg,
    )
    annotated_rows, interval_rows, annotation_summary = annotate_yuquan_sync_events(
        sync_rows,
        seizure_rows,
        config=cfg,
    )
    interval_window_rows, subject_window_rows, aggregate_summary = aggregate_yuquan_sync_rows(
        annotated_rows,
        interval_rows,
    )

    out_dir = Path(output_dir)
    outputs = {
        "event_annotations_csv": _write_csv(
            out_dir / "yuquan_sync_event_annotations.csv", annotated_rows
        ),
        "interval_window_csv": _write_csv(
            out_dir / "yuquan_sync_interval_window_table.csv", interval_window_rows
        ),
        "subject_window_csv": _write_csv(
            out_dir / "yuquan_sync_subject_window_summary.csv", subject_window_rows
        ),
        "summary_json": str(out_dir / "yuquan_sync_aggregation_summary.json"),
    }
    summary = {
        "config": {
            "post_ictal_minutes": cfg.post_ictal_minutes,
            "offset_buffer_minutes": cfg.offset_buffer_minutes,
            "post_ictal_window_sec": cfg.post_ictal_window_sec,
            "nontrivial_gap_sec": cfg.nontrivial_gap_sec,
            "timezone_name": cfg.timezone_name,
            "day_start_hour": cfg.day_start_hour,
            "night_start_hour": cfg.night_start_hour,
        },
        "annotation": annotation_summary,
        "aggregation": aggregate_summary,
        "outputs": outputs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs["summary_json"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary
