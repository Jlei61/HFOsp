from __future__ import annotations

import csv
import json
from pathlib import Path

from src.interictal_synchrony_aggregation import (
    EpilepsiaeSyncAggregationConfig,
    YuquanSyncAggregationConfig,
    annotate_epilepsiae_sync_events,
    annotate_yuquan_sync_events,
    aggregate_epilepsiae_sync_rows,
    aggregate_yuquan_sync_rows,
    build_epilepsiae_seizure_intervals,
    build_yuquan_seizure_intervals,
    run_epilepsiae_sync_aggregation,
    run_yuquan_sync_aggregation,
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


def test_annotate_and_aggregate_epilepsiae_sync_events() -> None:
    seizure_rows = [
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_id": "b0",
            "seizure_id": "sz1",
            "eeg_onset_epoch": 28800.0,
            "eeg_offset_epoch": 28900.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Europe/Berlin",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_id": "b9",
            "seizure_id": "sz2",
            "eeg_onset_epoch": 90000.0,
            "eeg_offset_epoch": 90100.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Europe/Berlin",
        },
    ]
    sync_rows = [
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0001",
            "block_start_epoch": 29000.0,
            "block_end_epoch": 29200.0,
            "block_start_day_night": "day",
            "block_end_day_night": "day",
            "timezone_name": "Europe/Berlin",
            "day_night_rule": "day=08:00-20:00 local",
            "n_events": 10.0,
            "sync_phase_global": 0.10,
            "sync_legacy_global": 0.20,
            "sync_span_global": 0.30,
            "jaccard_global_with_next": 0.40,
            "frac_core_only_events": 1.0,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.0,
            "event_index": 0.0,
            "event_start_epoch": 29020.0,
            "event_end_epoch": 29040.0,
            "event_center_epoch": 29030.0,
            "event_duration_sec": 20.0,
            "n_participating": 4.0,
            "n_core": 4.0,
            "n_penumbra": 0.0,
            "event_stratum": "core_only",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0002",
            "block_start_epoch": 50000.0,
            "block_end_epoch": 50200.0,
            "block_start_day_night": "day",
            "block_end_day_night": "day",
            "timezone_name": "Europe/Berlin",
            "day_night_rule": "day=08:00-20:00 local",
            "n_events": 20.0,
            "sync_phase_global": 0.20,
            "sync_legacy_global": 0.30,
            "sync_span_global": 0.40,
            "jaccard_global_with_next": 0.50,
            "frac_core_only_events": 0.9,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.1,
            "event_index": 1.0,
            "event_start_epoch": 50030.0,
            "event_end_epoch": 50050.0,
            "event_center_epoch": 50040.0,
            "event_duration_sec": 20.0,
            "n_participating": 5.0,
            "n_core": 5.0,
            "n_penumbra": 0.0,
            "event_stratum": "core_only",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0003",
            "block_start_epoch": 50200.0,
            "block_end_epoch": 50400.0,
            "block_start_day_night": "day",
            "block_end_day_night": "day",
            "timezone_name": "Europe/Berlin",
            "day_night_rule": "day=08:00-20:00 local",
            "n_events": 30.0,
            "sync_phase_global": 0.30,
            "sync_legacy_global": 0.40,
            "sync_span_global": 0.50,
            "jaccard_global_with_next": 0.60,
            "frac_core_only_events": 0.8,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.2,
            "event_index": 2.0,
            "event_start_epoch": 50230.0,
            "event_end_epoch": 50250.0,
            "event_center_epoch": 50240.0,
            "event_duration_sec": 20.0,
            "n_participating": 6.0,
            "n_core": 5.0,
            "n_penumbra": 1.0,
            "event_stratum": "mixed",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0004",
            "block_start_epoch": 80000.0,
            "block_end_epoch": 80200.0,
            "block_start_day_night": "night",
            "block_end_day_night": "night",
            "timezone_name": "Europe/Berlin",
            "day_night_rule": "day=08:00-20:00 local",
            "n_events": 40.0,
            "sync_phase_global": 0.40,
            "sync_legacy_global": 0.50,
            "sync_span_global": 0.60,
            "jaccard_global_with_next": 0.70,
            "frac_core_only_events": 0.7,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.3,
            "event_index": 3.0,
            "event_start_epoch": 80020.0,
            "event_end_epoch": 80040.0,
            "event_center_epoch": 80030.0,
            "event_duration_sec": 20.0,
            "n_participating": 5.0,
            "n_core": 4.0,
            "n_penumbra": 1.0,
            "event_stratum": "mixed",
        },
        {
            "subject": "1073",
            "patient_code": "FR_1073",
            "recording_id": "r1",
            "block_stem": "r1_0005",
            "block_start_epoch": 68350.0,
            "block_end_epoch": 68450.0,
            "block_start_day_night": "day",
            "block_end_day_night": "night",
            "timezone_name": "Europe/Berlin",
            "day_night_rule": "day=08:00-20:00 local",
            "n_events": 50.0,
            "sync_phase_global": 0.50,
            "sync_legacy_global": 0.60,
            "sync_span_global": 0.70,
            "jaccard_global_with_next": 0.80,
            "frac_core_only_events": 0.6,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.4,
            "event_index": 4.0,
            "event_start_epoch": 68390.0,
            "event_end_epoch": 68410.0,
            "event_center_epoch": 68400.0,
            "event_duration_sec": 20.0,
            "n_participating": 7.0,
            "n_core": 5.0,
            "n_penumbra": 2.0,
            "event_stratum": "mixed",
        },
    ]

    annotated, intervals, summary = annotate_epilepsiae_sync_events(
        sync_rows,
        seizure_rows,
    )
    assert summary["n_assigned_interval_rows"] == 5
    by_block = {row["block_stem"]: row for row in annotated}
    assert by_block["r1_0001"]["phase_window_type"] == "post_ictal"
    assert by_block["r1_0001"]["phase_eligible"] is True
    assert by_block["r1_0002"]["phase_window_type"] == "interictal"
    assert by_block["r1_0002"]["diurnal_window_type"] == "day"
    assert by_block["r1_0002"]["event_start_day_night"] == "day"
    assert by_block["r1_0002"]["event_end_day_night"] == "day"
    assert by_block["r1_0004"]["has_gap_before_event"] is True
    assert by_block["r1_0005"]["diurnal_eligible"] is False

    interval_rows, subject_rows, agg_summary = aggregate_epilepsiae_sync_rows(annotated, intervals)
    assert agg_summary["n_interval_window_rows"] >= 3
    by_key = {
        (row["window_family"], row["window_type"]): row for row in interval_rows
    }
    assert by_key[("phase", "post_ictal")]["n_events"] == 1
    assert by_key[("phase", "interictal")]["n_events"] == 4
    assert by_key[("phase", "interictal")]["has_gap_split"] is True
    assert by_key[("diurnal", "day")]["n_events"] == 3
    assert by_key[("diurnal", "day")]["eligible_for_primary_stats"] is False
    assert by_key[("diurnal", "night")]["n_events"] == 1
    assert len(subject_rows) >= 3


def test_run_epilepsiae_sync_aggregation_writes_outputs(tmp_path: Path) -> None:
    seizure_csv = tmp_path / "seizure.csv"
    sync_csv = tmp_path / "sync.csv"
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
                "recording_id",
                "block_stem",
                "block_start_epoch",
                "block_end_epoch",
                "block_start_day_night",
                "block_end_day_night",
                "timezone_name",
                "day_night_rule",
                "n_events",
                "sync_phase_global",
                "sync_legacy_global",
                "sync_span_global",
                "jaccard_global_with_next",
                "frac_core_only_events",
                "frac_penumbra_only_events",
                "frac_mixed_events",
                "event_index",
                "event_start_epoch",
                "event_end_epoch",
                "event_center_epoch",
                "event_duration_sec",
                "n_participating",
                "n_core",
                "n_penumbra",
                "event_stratum",
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
                "block_start_day_night": "day",
                "block_end_day_night": "day",
                "timezone_name": "Europe/Berlin",
                "day_night_rule": "day=08:00-20:00 local",
                "n_events": "10",
                "sync_phase_global": "0.1",
                "sync_legacy_global": "0.2",
                "sync_span_global": "0.3",
                "jaccard_global_with_next": "0.4",
                "frac_core_only_events": "1.0",
                "frac_penumbra_only_events": "0.0",
                "frac_mixed_events": "0.0",
                "event_index": "0",
                "event_start_epoch": "220",
                "event_end_epoch": "240",
                "event_center_epoch": "230",
                "event_duration_sec": "20",
                "n_participating": "4",
                "n_core": "4",
                "n_penumbra": "0",
                "event_stratum": "core_only",
            }
        )

    summary = run_epilepsiae_sync_aggregation(
        seizure_inventory_csv=seizure_csv,
        sync_event_csv=sync_csv,
        output_dir=tmp_path / "out",
        config=EpilepsiaeSyncAggregationConfig(),
    )

    assert (tmp_path / "out" / "epilepsiae_sync_event_annotations.csv").exists()
    assert (tmp_path / "out" / "epilepsiae_sync_interval_window_table.csv").exists()
    assert (tmp_path / "out" / "epilepsiae_sync_subject_window_summary.csv").exists()
    summary_json = tmp_path / "out" / "epilepsiae_sync_aggregation_summary.json"
    assert summary_json.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["annotation"]["n_annotated_rows"] == 1
    assert summary["aggregation"]["n_interval_window_rows"] >= 1


def test_build_and_annotate_yuquan_sync_events() -> None:
    seizure_rows = [
        {
            "subject": "litengsheng",
            "patient_code": "litengsheng",
            "recording_id": "rec01",
            "seizure_id": "litengsheng_sz_001",
            "eeg_onset_epoch": 100.0,
            "eeg_offset_epoch": 180.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Asia/Shanghai",
        },
        {
            "subject": "litengsheng",
            "patient_code": "litengsheng",
            "recording_id": "rec03",
            "seizure_id": "litengsheng_sz_002",
            "eeg_onset_epoch": 50000.0,
            "eeg_offset_epoch": 50080.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Asia/Shanghai",
        },
    ]
    intervals = build_yuquan_seizure_intervals(seizure_rows)
    assert len(intervals) == 1
    assert intervals[0]["prev_recording_id"] == "rec01"
    assert intervals[0]["next_recording_id"] == "rec03"

    sync_rows = [
        {
            "subject": "litengsheng",
            "patient_code": "litengsheng",
            "recording_id": "rec02",
            "block_stem": "rec02",
                "block_start_epoch": 190.0,
                "block_end_epoch": 230.0,
            "block_start_day_night": "day",
            "block_end_day_night": "day",
            "timezone_name": "Asia/Shanghai",
            "day_night_rule": "day=08:00-20:00 local",
            "n_events": 10.0,
            "sync_phase_global": 0.1,
            "sync_legacy_global": 0.2,
            "sync_span_global": 0.3,
            "jaccard_global_with_next": 0.4,
            "frac_core_only_events": 1.0,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.0,
            "sync_phase_global": 0.1,
            "event_index": 0.0,
            "event_start_epoch": 195.0,
            "event_end_epoch": 210.0,
            "event_center_epoch": 202.5,
            "event_duration_sec": 15.0,
            "n_participating": 4.0,
            "n_core": 4.0,
            "n_penumbra": 0.0,
            "event_stratum": "core_only",
        },
        {
            "subject": "litengsheng",
            "patient_code": "litengsheng",
            "recording_id": "rec02b",
            "block_stem": "rec02b",
                "block_start_epoch": 300.0,
                "block_end_epoch": 400.0,
            "block_start_day_night": "day",
            "block_end_day_night": "day",
            "timezone_name": "Asia/Shanghai",
            "day_night_rule": "day=08:00-20:00 local",
            "n_events": 20.0,
            "sync_phase_global": 0.2,
            "sync_legacy_global": 0.3,
            "sync_span_global": 0.4,
            "jaccard_global_with_next": 0.5,
            "frac_core_only_events": 0.9,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.1,
            "event_index": 1.0,
            "event_start_epoch": 320.0,
            "event_end_epoch": 340.0,
            "event_center_epoch": 330.0,
            "event_duration_sec": 20.0,
            "n_participating": 5.0,
            "n_core": 5.0,
            "n_penumbra": 0.0,
            "event_stratum": "core_only",
        },
        {
            "subject": "litengsheng",
            "patient_code": "litengsheng",
            "recording_id": "rec02c",
            "block_stem": "rec02c",
            "block_start_epoch": 45050.0,
            "block_end_epoch": 45150.0,
            "block_start_day_night": "night",
            "block_end_day_night": "night",
            "timezone_name": "Asia/Shanghai",
            "day_night_rule": "day=08:00-20:00 local",
            "n_events": 30.0,
            "sync_phase_global": 0.3,
            "sync_legacy_global": 0.4,
            "sync_span_global": 0.5,
            "jaccard_global_with_next": 0.6,
            "frac_core_only_events": 0.8,
            "frac_penumbra_only_events": 0.0,
            "frac_mixed_events": 0.2,
            "event_index": 2.0,
            "event_start_epoch": 45070.0,
            "event_end_epoch": 45090.0,
            "event_center_epoch": 45080.0,
            "event_duration_sec": 20.0,
            "n_participating": 5.0,
            "n_core": 4.0,
            "n_penumbra": 1.0,
            "event_stratum": "mixed",
        },
    ]
    annotated, interval_rows, summary = annotate_yuquan_sync_events(
        sync_rows,
        seizure_rows,
        config=YuquanSyncAggregationConfig(post_ictal_minutes=1.0, offset_buffer_minutes=0.0),
    )
    assert summary["n_assigned_interval_rows"] == 3
    by_block = {row["block_stem"]: row for row in annotated}
    assert by_block["rec02"]["phase_window_type"] == "post_ictal"
    assert by_block["rec02b"]["phase_window_type"] == "interictal"
    assert by_block["rec02c"]["diurnal_window_type"] == "night"

    interval_window_rows, subject_window_rows, agg_summary = aggregate_yuquan_sync_rows(
        annotated,
        interval_rows,
    )
    assert agg_summary["n_interval_window_rows"] >= 2
    assert any(row["window_type"] == "interictal" for row in interval_window_rows)
    assert len(subject_window_rows) >= 2


def test_run_yuquan_sync_aggregation_writes_outputs(tmp_path: Path) -> None:
    sync_csv = tmp_path / "yuquan_sync.csv"
    with open(sync_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "subject",
                "patient_code",
                "recording_id",
                "block_stem",
                "block_start_epoch",
                "block_end_epoch",
                "n_events",
                "sync_phase_global",
                "sync_legacy_global",
                "sync_span_global",
                "jaccard_global_with_next",
                "frac_core_only_events",
                "frac_penumbra_only_events",
                "frac_mixed_events",
                "event_index",
                "event_start_epoch",
                "event_end_epoch",
                "event_center_epoch",
                "event_duration_sec",
                "n_participating",
                "n_core",
                "n_penumbra",
                "event_stratum",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "litengsheng",
                "patient_code": "litengsheng",
                "recording_id": "rec02",
                "block_stem": "rec02",
                "block_start_epoch": "3600",
                "block_end_epoch": "4200",
                "n_events": "10",
                "sync_phase_global": "0.1",
                "sync_legacy_global": "0.2",
                "sync_span_global": "0.3",
                "jaccard_global_with_next": "0.4",
                "frac_core_only_events": "1.0",
                "frac_penumbra_only_events": "0.0",
                "frac_mixed_events": "0.0",
                "event_index": "0",
                "event_start_epoch": "3650",
                "event_end_epoch": "3700",
                "event_center_epoch": "3675",
                "event_duration_sec": "50",
                "n_participating": "4",
                "n_core": "4",
                "n_penumbra": "0",
                "event_stratum": "core_only",
            }
        )

    fake_seizure_rows = [
        {
            "subject": "litengsheng",
            "patient_code": "litengsheng",
            "recording_id": "rec01",
            "seizure_id": "litengsheng_sz_001",
            "eeg_onset_epoch": 100.0,
            "eeg_offset_epoch": 200.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Asia/Shanghai",
        },
        {
            "subject": "litengsheng",
            "patient_code": "litengsheng",
            "recording_id": "rec03",
            "seizure_id": "litengsheng_sz_002",
            "eeg_onset_epoch": 10000.0,
            "eeg_offset_epoch": 10100.0,
            "has_complete_eeg_interval": True,
            "timezone_name": "Asia/Shanghai",
        },
    ]

    import src.interictal_synchrony_aggregation as agg_mod

    original_builder = agg_mod.build_yuquan_seizure_inventory
    agg_mod.build_yuquan_seizure_inventory = lambda *args, **kwargs: fake_seizure_rows
    try:
        summary = run_yuquan_sync_aggregation(
            sync_event_csv=sync_csv,
            output_dir=tmp_path / "out",
            config=YuquanSyncAggregationConfig(),
        )
    finally:
        agg_mod.build_yuquan_seizure_inventory = original_builder

    assert (tmp_path / "out" / "yuquan_sync_event_annotations.csv").exists()
    assert (tmp_path / "out" / "yuquan_sync_interval_window_table.csv").exists()
    assert (tmp_path / "out" / "yuquan_sync_subject_window_summary.csv").exists()
    payload = json.loads(
        (tmp_path / "out" / "yuquan_sync_aggregation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["annotation"]["n_annotated_rows"] == 1
    assert summary["aggregation"]["n_interval_window_rows"] >= 1
