"""
Epilepsiae synchrony aggregation above block-level PR4 outputs.

This module stays deliberately conservative:
- it consumes existing inventory + manifest + block-level synchrony outputs
- it does not invent sub-block labels from 1h lagPat blocks
- blocks that cross seizure / post-ictal / day-night boundaries are excluded
  from the corresponding aggregate instead of being force-assigned
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .epilepsiae_dataset import NONTRIVIAL_GAP_SEC


@dataclass(frozen=True)
class EpilepsiaeSyncAggregationConfig:
    post_ictal_minutes: float = 60.0
    offset_buffer_minutes: float = 10.0
    nontrivial_gap_sec: float = NONTRIVIAL_GAP_SEC

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


def load_epilepsiae_sync_summary_rows(csv_path: Path | str) -> List[Dict[str, object]]:
    rows = _read_csv_rows(csv_path)
    float_cols = {
        "n_events",
        "n_channels",
        "n_core_channels",
        "n_penumbra_channels",
        "mean_sync_phase_global",
        "mean_sync_phase_core",
        "mean_sync_phase_penumbra",
        "mean_sync_legacy_global",
        "mean_sync_legacy_core",
        "mean_sync_legacy_penumbra",
        "mean_sync_span_global",
        "mean_sync_span_core",
        "mean_sync_span_penumbra",
        "mean_jaccard_global",
        "mean_jaccard_core",
        "mean_jaccard_penumbra",
        "frac_core_only_events",
        "frac_penumbra_only_events",
        "frac_mixed_events",
    }
    out: List[Dict[str, object]] = []
    for row in rows:
        parsed: Dict[str, object] = dict(row)
        for col in float_cols:
            parsed[col] = _as_float(parsed.get(col))
        out.append(parsed)
    return out


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


def _build_continuous_segments(
    block_rows: Sequence[Mapping[str, object]],
    *,
    nontrivial_gap_sec: float,
) -> Dict[Tuple[str, str], Dict[str, object]]:
    by_subject: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in block_rows:
        by_subject[str(row["subject"])].append(row)

    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    for subject, rows in by_subject.items():
        prev_end: Optional[float] = None
        segment_id = 0
        for row in sorted(rows, key=lambda x: float(x["block_start_epoch"] or 0.0)):
            start = _as_float(row["block_start_epoch"])
            end = _as_float(row["block_end_epoch"])
            observed_gap = None if prev_end is None or start is None else float(start - prev_end)
            starts_new = bool(
                prev_end is None
                or observed_gap is None
                or observed_gap > float(nontrivial_gap_sec)
            )
            if starts_new:
                segment_id += 1
            out[(subject, str(row["block_stem"]))] = {
                "continuous_segment_id": segment_id,
                "gap_from_prev_observed_sec": observed_gap,
                "starts_new_continuous_segment": starts_new,
            }
            prev_end = end
    return out


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

    interval_rows: List[Dict[str, object]] = []
    for subject, rows in by_subject.items():
        ordered = sorted(rows, key=lambda x: float(x["eeg_onset_epoch"] or 0.0))
        for idx in range(1, len(ordered)):
            prev = ordered[idx - 1]
            curr = ordered[idx]
            prev_onset = _as_float(prev["eeg_onset_epoch"])
            prev_offset = _as_float(prev["eeg_offset_epoch"])
            next_onset = _as_float(curr["eeg_onset_epoch"])
            next_offset = _as_float(curr["eeg_offset_epoch"])
            if prev_onset is None or prev_offset is None or next_onset is None:
                continue
            clean_start = prev_offset
            clean_end = next_onset
            post_end = min(clean_end, clean_start + cfg.post_ictal_window_sec)
            interval_rows.append(
                {
                    "subject": subject,
                    "patient_code": prev.get("patient_code", ""),
                    "seizure_interval_id": f"{subject}_szint_{idx:03d}",
                    "interval_index": idx,
                    "prev_seizure_id": prev["seizure_id"],
                    "next_seizure_id": curr["seizure_id"],
                    "prev_recording_id": prev["recording_id"],
                    "next_recording_id": curr["recording_id"],
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


def _interval_overlap(start: float, end: float, left: float, right: float) -> bool:
    return max(start, left) < min(end, right)


def annotate_epilepsiae_sync_blocks(
    sync_rows: Sequence[Mapping[str, object]],
    block_rows: Sequence[Mapping[str, object]],
    seizure_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    config: Optional[EpilepsiaeSyncAggregationConfig] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    cfg = config or EpilepsiaeSyncAggregationConfig()
    intervals = build_epilepsiae_seizure_intervals(seizure_rows, config=cfg)
    manifest_by_subject = {str(x["subject"]): dict(x) for x in manifest_rows}
    block_by_key = {
        (str(row["subject"]), str(row["block_stem"])): dict(row) for row in block_rows
    }
    segment_meta = _build_continuous_segments(
        block_rows, nontrivial_gap_sec=float(cfg.nontrivial_gap_sec)
    )
    seizures_by_subject: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in seizure_rows:
        if _as_bool(row.get("has_complete_eeg_interval")):
            seizures_by_subject[str(row["subject"])].append(row)
    intervals_by_subject: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in intervals:
        intervals_by_subject[str(row["subject"])].append(row)

    annotated_rows: List[Dict[str, object]] = []
    exclusion_counts: Counter[str] = Counter()
    join_miss_count = 0
    for sync in sync_rows:
        subject = str(sync["subject"])
        block_stem = str(sync["block_stem"])
        block = block_by_key.get((subject, block_stem))
        if block is None:
            join_miss_count += 1
            continue

        block_start = _as_float(block["block_start_epoch"])
        block_end = _as_float(block["block_end_epoch"])
        if block_start is None or block_end is None:
            exclusion_counts["missing_block_epoch"] += 1
            continue

        segment = segment_meta[(subject, block_stem)]
        overlap_seizure = False
        for seizure in seizures_by_subject.get(subject, []):
            onset = _as_float(seizure["eeg_onset_epoch"])
            offset = _as_float(seizure["eeg_offset_epoch"])
            if onset is None or offset is None:
                continue
            if _interval_overlap(block_start, block_end, onset, offset):
                overlap_seizure = True
                break

        containing_intervals = [
            interval
            for interval in intervals_by_subject.get(subject, [])
            if block_start >= float(interval["clean_between_seizures_start_epoch"])
            and block_end <= float(interval["clean_between_seizures_end_epoch"])
        ]
        overlaps_any_interval = any(
            _interval_overlap(
                block_start,
                block_end,
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
            if post_end > post_start and block_start >= post_start and block_end <= post_end:
                phase_window_type = "post_ictal"
                phase_eligible = True
            elif inter_end > inter_start and block_start >= inter_start and block_end <= inter_end:
                phase_window_type = "interictal"
                phase_eligible = True
            else:
                phase_window_type = "phase_boundary_crossing"

        diurnal_start = str(block.get("block_start_day_night") or "")
        diurnal_end = str(block.get("block_end_day_night") or "")
        diurnal_window_type = ""
        diurnal_eligible = False
        if diurnal_start in {"day", "night"} and diurnal_start == diurnal_end:
            diurnal_window_type = diurnal_start
            diurnal_eligible = True
        elif diurnal_start or diurnal_end:
            diurnal_window_type = "day_night_transition"

        reasons: List[str] = []
        if interval_status != "assigned":
            reasons.append(interval_status)
        if not phase_eligible:
            reasons.append(phase_window_type or "phase_not_eligible")
        if not diurnal_eligible:
            reasons.append(diurnal_window_type or "day_night_not_eligible")
        if float(segment["gap_from_prev_observed_sec"] or 0.0) > float(cfg.nontrivial_gap_sec):
            reasons.append("nontrivial_gap_before_block")

        manifest = manifest_by_subject.get(subject, {})
        row = {
            "subject": subject,
            "patient_code": sync.get("patient_code", block.get("patient_code", "")),
            "block_stem": block_stem,
            "recording_id": block.get("recording_id", ""),
            "block_start_epoch": block_start,
            "block_end_epoch": block_end,
            "block_duration_sec": float(block_end - block_start),
            "block_start_day_night": diurnal_start,
            "block_end_day_night": diurnal_end,
            "timezone_name": block.get("timezone_name", manifest.get("timezone_name", "")),
            "day_night_rule": manifest.get("day_night_rule", ""),
            "continuous_segment_id": int(segment["continuous_segment_id"]),
            "gap_from_prev_observed_sec": segment["gap_from_prev_observed_sec"],
            "starts_new_continuous_segment": bool(segment["starts_new_continuous_segment"]),
            "has_gap_before_block": bool(
                segment["gap_from_prev_observed_sec"] is not None
                and float(segment["gap_from_prev_observed_sec"]) > float(cfg.nontrivial_gap_sec)
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
        for key, value in sync.items():
            if key not in row:
                row[key] = value
        annotated_rows.append(row)
        for reason in reasons:
            exclusion_counts[reason] += 1

    summary = {
        "n_sync_rows_input": len(sync_rows),
        "n_join_miss": join_miss_count,
        "n_annotated_rows": len(annotated_rows),
        "n_assigned_interval_rows": sum(
            1 for row in annotated_rows if row["interval_assignment_status"] == "assigned"
        ),
        "n_phase_eligible_rows": sum(1 for row in annotated_rows if row["phase_eligible"]),
        "n_diurnal_eligible_rows": sum(1 for row in annotated_rows if row["diurnal_eligible"]),
        "exclusion_reason_counts": dict(exclusion_counts),
    }
    return annotated_rows, intervals, summary


def _metric_values(rows: Sequence[Mapping[str, object]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        value = _as_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def aggregate_epilepsiae_sync_rows(
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
        "n_events",
        "mean_sync_phase_global",
        "mean_sync_legacy_global",
        "mean_sync_span_global",
        "mean_jaccard_global",
        "frac_core_only_events",
        "frac_penumbra_only_events",
        "frac_mixed_events",
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
            "n_blocks": len(rows),
            "n_continuous_segments": len(segment_ids),
            "continuous_segment_ids": "|".join(str(x) for x in segment_ids),
            "has_gap_split": len(segment_ids) > 1,
            "eligible_for_primary_stats": len(segment_ids) == 1,
            "block_coverage_sec": sum(float(row["block_duration_sec"]) for row in rows),
            "block_start_min_epoch": min(float(row["block_start_epoch"]) for row in rows),
            "block_end_max_epoch": max(float(row["block_end_epoch"]) for row in rows),
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
            out[f"{metric}_mean_across_blocks"] = _mean_or_none(values)
            out[f"{metric}_median_across_blocks"] = _median_or_none(values)
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
            "n_blocks": len(rows),
            "n_seizure_intervals": len(interval_ids),
            "seizure_interval_ids": "|".join(interval_ids),
            "n_continuous_segments": len(segment_ids),
            "has_gap_split": len(segment_ids) > 1,
            "block_coverage_sec": sum(float(row["block_duration_sec"]) for row in rows),
            "timezone_name": rows[0].get("timezone_name", ""),
            "day_night_rule": rows[0].get("day_night_rule", ""),
        }
        for metric in metric_cols:
            values = _metric_values(rows, metric)
            out[f"{metric}_mean_across_blocks"] = _mean_or_none(values)
            out[f"{metric}_median_across_blocks"] = _median_or_none(values)
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
    block_inventory_csv: Path | str,
    seizure_inventory_csv: Path | str,
    sync_summary_csv: Path | str,
    manifest_csv: Path | str,
    output_dir: Path | str,
    config: Optional[EpilepsiaeSyncAggregationConfig] = None,
) -> Dict[str, object]:
    cfg = config or EpilepsiaeSyncAggregationConfig()
    sync_rows = load_epilepsiae_sync_summary_rows(sync_summary_csv)
    block_rows = load_epilepsiae_block_inventory_rows(block_inventory_csv)
    seizure_rows = load_epilepsiae_seizure_inventory_rows(seizure_inventory_csv)
    manifest_rows = _read_csv_rows(manifest_csv)

    annotated_rows, interval_rows, annotation_summary = annotate_epilepsiae_sync_blocks(
        sync_rows,
        block_rows,
        seizure_rows,
        manifest_rows,
        config=cfg,
    )
    interval_window_rows, subject_window_rows, aggregate_summary = aggregate_epilepsiae_sync_rows(
        annotated_rows,
        interval_rows,
    )

    out_dir = Path(output_dir)
    outputs = {
        "block_annotations_csv": _write_csv(
            out_dir / "epilepsiae_sync_block_annotations.csv", annotated_rows
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
