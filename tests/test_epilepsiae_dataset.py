from __future__ import annotations

from src.epilepsiae_dataset import (
    EpilepsiaeInventory,
    EpilepsiaeTimeConfig,
    build_epilepsiae_sync_subject_manifest,
    resolve_epilepsiae_timezone,
)


def test_resolve_epilepsiae_timezone_uses_hospital_map() -> None:
    out = resolve_epilepsiae_timezone(
        subject="1073",
        patient_code="FR_1073",
        hospital="UKLFR",
    )
    assert out["timezone_name"] == "Europe/Berlin"
    assert out["timezone_source"] == "hospital"
    assert out["reliable_without_override"] is True


def test_resolve_epilepsiae_timezone_prefers_recording_override() -> None:
    cfg = EpilepsiaeTimeConfig(
        recording_timezone_overrides={"1073/107300102": "Europe/Paris"},
    )
    out = resolve_epilepsiae_timezone(
        subject="1073",
        patient_code="FR_1073",
        hospital="UKLFR",
        recording_id="107300102",
        time_config=cfg,
    )
    assert out["timezone_name"] == "Europe/Paris"
    assert out["timezone_source"] == "recording_override"
    assert out["reliable_without_override"] is False


def test_manifest_keeps_short_interval_subjects_in_primary_ready_tier() -> None:
    inventory = EpilepsiaeInventory(
        subject_rows=[
            {
                "subject": "1073",
                "patient_code": "FR_1073",
                "artifact_subject_present": True,
                "has_refine_gpu": True,
                "n_lagpat_blocks": 10,
                "n_packed_times_blocks": 10,
                "n_gpu_artifact_blocks": 10,
                "n_complete_eeg_intervals": 2,
                "n_inter_record_gaps_gt2s": 0,
                "max_inter_record_gap_sec": 0.0,
                "max_block_gap_sec": 0.0,
                "timezone_name": "Europe/Berlin",
                "timezone_source": "hospital",
                "day_night_rule": "day=08:00-20:00 local",
                "day_night_reliable_without_override": True,
            }
        ],
        recording_rows=[],
        block_rows=[],
        seizure_rows=[
            {
                "subject": "1073",
                "has_complete_eeg_interval": True,
                "eeg_onset_epoch": 1000.0,
            },
            {
                "subject": "1073",
                "has_complete_eeg_interval": True,
                "eeg_onset_epoch": 1000.0 + 1800.0,
            },
        ],
        summary={},
    )

    rows = build_epilepsiae_sync_subject_manifest(
        inventory,
        min_complete_eeg_intervals=2,
        min_sync_interval_sec=3.0 * 3600.0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["ready_for_sync_analysis"] is True
    assert row["tier"] == "ready_full_artifacts"
    assert row["n_all_eligible_intervals"] == 1
    assert row["n_legacy_comparable_intervals"] == 0
    assert row["n_short_intervals"] == 1
    assert row["has_legacy_comparable_intervals"] is False
