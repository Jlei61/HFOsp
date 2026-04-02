from __future__ import annotations

import csv
import json
from pathlib import Path

from src.interictal_synchrony_aggregation import (
    EpilepsiaeSyncAggregationConfig,
    annotate_epilepsiae_sync_blocks,
    aggregate_epilepsiae_sync_rows,
    build_epilepsiae_seizure_intervals,
    run_epilepsiae_sync_aggregation,
)


def test_build_epilepsiae_seizure_intervals_applies_postictal_buffer() -> None:
    rows = [
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "seizure_id": "sz1",
            "eeg_onset_epoch": 0.0,
            "eeg_offset_epoch": 100.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Europe/Berlin",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "seizure_id": "sz2",
            "eeg_onset_epoch": 10000.0,
            "eeg_offset_epoch": 10100.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Europe/Berlin",
        },
    ]
    intervals = build_epilepsiae_seizure_intervals(rows)
    assert len(intervals) == 1
    interval = intervals[0]
    assert interval["post_ictal_start_epoch"] == 100.0
    assert interval["post_ictal_end_epoch"] == 4300.0
    assert interval["interictal_start_epoch"] == 4300.0
    assert interval["interictal_end_epoch"] == 10000.0


def test_annotate_and_aggregate_epilepsiae_sync_blocks() -> None:
    seizure_rows = [
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_id": "b0",
            "seizure_id": "sz1",
            "eeg_onset_epoch": 0.0,
            "eeg_offset_epoch": 100.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Europe/Berlin",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_id": "b9",
            "seizure_id": "sz2",
            "eeg_onset_epoch": 10000.0,
            "eeg_offset_epoch": 10100.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Europe/Berlin",
        },
    ]
    block_rows = [
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0001",
            "block_start_epoch": 200.0,
            "block_end_epoch": 400.0,
            "gap_to_prev_sec": 0.0,
            "block_start_day_night": "day",
            "block_end_day_night": "day",
            "timezone_name": "Europe/Berlin",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0002",
            "block_start_epoch": 5000.0,
            "block_end_epoch": 5200.0,
            "gap_to_prev_sec": 0.0,
            "block_start_day_night": "day",
            "block_end_day_night": "day",
            "timezone_name": "Europe/Berlin",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0003",
            "block_start_epoch": 5200.0,
            "block_end_epoch": 5400.0,
            "gap_to_prev_sec": 0.0,
            "block_start_day_night": "day",
            "block_end_day_night": "day",
            "timezone_name": "Europe/Berlin",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0004",
            "block_start_epoch": 8000.0,
            "block_end_epoch": 8200.0,
            "gap_to_prev_sec": 2600.0,
            "block_start_day_night": "night",
            "block_end_day_night": "night",
            "timezone_name": "Europe/Berlin",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0005",
            "block_start_epoch": 7500.0,
            "block_end_epoch": 7800.0,
            "gap_to_prev_sec": 0.0,
            "block_start_day_night": "day",
            "block_end_day_night": "night",
            "timezone_name": "Europe/Berlin",
        },
    ]
    manifest_rows = [
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "day_night_rule": "day=08:00-20:00 local",
            "timezone_name": "Europe/Berlin",
        }
    ]
    sync_rows = [
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "block_stem": "r1_0001",
            "n_events": 10.0,
            "mean_sync_phase_global": 0.10,
            "mean_sync_legacy_global": 0.20,
            "mean_sync_span_global": 0.30,
            "mean_jaccard_global": 0.40,
            "frac_core_only_events": 1.0,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.0,
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "block_stem": "r1_0002",
            "n_events": 20.0,
            "mean_sync_phase_global": 0.20,
            "mean_sync_legacy_global": 0.30,
            "mean_sync_span_global": 0.40,
            "mean_jaccard_global": 0.50,
            "frac_core_only_events": 0.9,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.1,
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "block_stem": "r1_0003",
            "n_events": 30.0,
            "mean_sync_phase_global": 0.30,
            "mean_sync_legacy_global": 0.40,
            "mean_sync_span_global": 0.50,
            "mean_jaccard_global": 0.60,
            "frac_core_only_events": 0.8,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.2,
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "block_stem": "r1_0004",
            "n_events": 40.0,
            "mean_sync_phase_global": 0.40,
            "mean_sync_legacy_global": 0.50,
            "mean_sync_span_global": 0.60,
            "mean_jaccard_global": 0.70,
            "frac_core_only_events": 0.7,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.3,
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "block_stem": "r1_0005",
            "n_events": 50.0,
            "mean_sync_phase_global": 0.50,
            "mean_sync_legacy_global": 0.60,
            "mean_sync_span_global": 0.70,
            "mean_jaccard_global": 0.80,
            "frac_core_only_events": 0.6,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.4,
        },
    ]

    annotated, intervals, summary = annotate_epilepsiae_sync_blocks(
        sync_rows,
        block_rows,
        seizure_rows,
        manifest_rows,
    )
    assert summary["n_assigned_interval_rows"] == 5
    by_block = {row["block_stem"]: row for row in annotated}
    assert by_block["r1_0001"]["phase_window_type"] == "post_ictal"
    assert by_block["r1_0001"]["phase_eligible"] is True
    assert by_block["r1_0002"]["phase_window_type"] == "interictal"
    assert by_block["r1_0002"]["diurnal_window_type"] == "day"
    assert by_block["r1_0004"]["has_gap_before_block"] is True
    assert by_block["r1_0005"]["diurnal_eligible"] is False

    interval_rows, subject_rows, agg_summary = aggregate_epilepsiae_sync_rows(annotated, intervals)
    assert agg_summary["n_interval_window_rows"] >= 3
    by_key = {
        (row["window_family"], row["window_type"]): row for row in interval_rows
    }
    assert by_key[("phase", "post_ictal")]["n_blocks"] == 1
    assert by_key[("phase", "interictal")]["n_blocks"] == 4
    assert by_key[("phase", "interictal")]["has_gap_split"] is True
    assert by_key[("diurnal", "day")]["n_blocks"] == 3
    assert by_key[("diurnal", "day")]["eligible_for_primary_stats"] is False
    assert by_key[("diurnal", "night")]["n_blocks"] == 1
    assert len(subject_rows) >= 3


def test_run_epilepsiae_sync_aggregation_writes_outputs(tmp_path: Path) -> None:
    block_csv = tmp_path / "block.csv"
    seizure_csv = tmp_path / "seizure.csv"
    sync_csv = tmp_path / "sync.csv"
    manifest_csv = tmp_path / "manifest.csv"

    with open(block_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "subject",
                "patient_code",
                "recording_id",
                "block_stem",
                "block_start_epoch",
                "block_end_epoch",
                "gap_to_prev_sec",
                "block_start_day_night",
                "block_end_day_night",
                "timezone_name",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "1073",
                "patient_code": "FR_1073",
                "recording_id": "r1",
                "block_stem": "r1_0001",
                "block_start_epoch": "200",
                "block_end_epoch": "400",
                "gap_to_prev_sec": "0",
                "block_start_day_night": "day",
                "block_end_day_night": "day",
                "timezone_name": "Europe/Berlin",
            }
        )
    with open(seizure_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "subject",
                "patient_code",
                "recording_id",
                "block_id",
                "seizure_id",
                "eeg_onset_epoch",
                "eeg_offset_epoch",
                "has_complete_eeg_interval",
                "timezone_name",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "1073",
                "patient_code": "FR_1073",
                "recording_id": "r1",
                "block_id": "b0",
                "seizure_id": "sz1",
                "eeg_onset_epoch": "0",
                "eeg_offset_epoch": "100",
                "has_complete_eeg_interval": "True",
                "timezone_name": "Europe/Berlin",
            }
        )
        writer.writerow(
            {
                "subject": "1073",
                "patient_code": "FR_1073",
                "recording_id": "r1",
                "block_id": "b1",
                "seizure_id": "sz2",
                "eeg_onset_epoch": "10000",
                "eeg_offset_epoch": "10100",
                "has_complete_eeg_interval": "True",
                "timezone_name": "Europe/Berlin",
            }
        )
    with open(sync_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "subject",
                "patient_code",
                "block_stem",
                "n_events",
                "mean_sync_phase_global",
                "mean_sync_legacy_global",
                "mean_sync_span_global",
                "mean_jaccard_global",
                "frac_core_only_events",
                "frac_penumbra_only_events",
                "frac_mixed_events",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "1073",
                "patient_code": "FR_1073",
                "block_stem": "r1_0001",
                "n_events": "10",
                "mean_sync_phase_global": "0.1",
                "mean_sync_legacy_global": "0.2",
                "mean_sync_span_global": "0.3",
                "mean_jaccard_global": "0.4",
                "frac_core_only_events": "1.0",
                "frac_penumbra_only_events": "0.0",
                "frac_mixed_events": "0.0",
            }
        )
    with open(manifest_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["subject", "patient_code", "timezone_name", "day_night_rule"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "1073",
                "patient_code": "FR_1073",
                "timezone_name": "Europe/Berlin",
                "day_night_rule": "day=08:00-20:00 local",
            }
        )

    summary = run_epilepsiae_sync_aggregation(
        block_inventory_csv=block_csv,
        seizure_inventory_csv=seizure_csv,
        sync_summary_csv=sync_csv,
        manifest_csv=manifest_csv,
        output_dir=tmp_path / "out",
        config=EpilepsiaeSyncAggregationConfig(),
    )

    assert (tmp_path / "out" / "epilepsiae_sync_block_annotations.csv").exists()
    assert (tmp_path / "out" / "epilepsiae_sync_interval_window_table.csv").exists()
    assert (tmp_path / "out" / "epilepsiae_sync_subject_window_summary.csv").exists()
    summary_json = tmp_path / "out" / "epilepsiae_sync_aggregation_summary.json"
    assert summary_json.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["annotation"]["n_annotated_rows"] == 1
    assert summary["aggregation"]["n_interval_window_rows"] >= 1
