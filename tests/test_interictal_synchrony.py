from __future__ import annotations

from pathlib import Path

import numpy as np

from src.interictal_synchrony import (
    EVENT_SYNC_SCHEMA_VERSION,
    BLOCK_SYNC_COMPAT_SCHEMA_VERSION,
    build_interictal_synchrony,
    build_block_summary_from_event_rows,
    build_event_rows_from_result,
    build_interictal_synchrony_from_legacy_lagpat,
    compute_adjacent_jaccard,
    compute_event_synchrony_metrics,
    compute_interictal_synchrony,
    load_interictal_synchrony_result,
    run_epilepsiae_interictal_synchrony_from_manifest,
    run_yuquan_interictal_synchrony,
    save_interictal_synchrony_result,
    select_core_penumbra_mask,
)


def test_compute_event_synchrony_metrics_known_values() -> None:
    lag_raw = np.array(
        [
            [0.10, 0.20],
            [0.10, 0.30],
            [0.10, np.nan],
        ],
        dtype=np.float64,
    )
    events_bool = np.array(
        [
            [True, True],
            [True, True],
            [True, False],
        ],
        dtype=bool,
    )
    event_windows = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64)

    metrics = compute_event_synchrony_metrics(lag_raw, events_bool, event_windows)
    assert np.isclose(metrics["phase"][0], 1.0)
    assert np.isclose(metrics["legacy"][0], 1.0)
    assert np.isclose(metrics["span"][0], 1.0)

    assert np.isclose(metrics["phase"][1], 0.0, atol=1e-8)
    assert np.isclose(metrics["legacy"][1], 0.5, atol=1e-8)
    assert np.isclose(metrics["span"][1], 0.8, atol=1e-8)
    assert np.array_equal(metrics["n_participating"], np.array([3, 2], dtype=np.int64))


def test_compute_interictal_synchrony_core_penumbra_and_jaccard() -> None:
    group_analysis = {
        "ch_names": ["A1", "B1", "C1"],
        "event_windows": np.array([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]], dtype=np.float64),
        "lag_raw": np.array(
            [
                [0.02, 0.10, np.nan],
                [0.03, 0.10, np.nan],
                [0.05, np.nan, 0.20],
            ],
            dtype=np.float64,
        ),
        "events_bool": np.array(
            [
                [True, True, False],
                [True, True, False],
                [True, False, True],
            ],
            dtype=bool,
        ),
    }

    result = compute_interictal_synchrony(group_analysis, core_channels=["A1", "B1"])
    assert np.array_equal(result.core_mask, np.array([True, True, False], dtype=bool))
    assert result.event_stratum.tolist() == ["mixed", "core_only", "penumbra_only"]
    assert np.array_equal(result.event_n_core, np.array([2, 2, 0], dtype=np.int64))
    assert np.array_equal(result.event_n_penumbra, np.array([1, 0, 1], dtype=np.int64))
    assert np.isclose(result.sync_legacy_core[1], 1.0)
    assert np.isclose(result.sync_span_global[0], 0.94, atol=1e-8)
    assert np.allclose(result.jaccard_global, np.array([2.0 / 3.0, 0.0], dtype=np.float64))
    assert np.allclose(result.jaccard_core, np.array([1.0, 0.0], dtype=np.float64))
    assert np.allclose(result.jaccard_penumbra, np.array([0.0, 0.0], dtype=np.float64))


def test_build_interictal_synchrony_from_group_analysis_npz(
    synthetic_group_analysis_npz: str,
) -> None:
    result = build_interictal_synchrony(synthetic_group_analysis_npz)
    assert result.ch_names == ["A1", "B1", "C1"]
    assert np.all(result.core_mask)
    assert np.all(result.event_stratum == "core_only")
    assert np.allclose(result.jaccard_global, np.ones((11,), dtype=np.float64))
    assert 0.0 <= result.summary["mean_sync_phase_global"] <= 1.0


def test_save_load_interictal_synchrony_roundtrip(tmp_path: Path) -> None:
    group_analysis = {
        "ch_names": ["A1", "B1"],
        "event_windows": np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64),
        "lag_raw": np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64),
        "events_bool": np.ones((2, 2), dtype=bool),
    }
    result = compute_interictal_synchrony(
        group_analysis,
        core_channels=["A1"],
        metadata={"subject": "1073", "block_stem": "107300102_0000"},
    )
    out_npz = tmp_path / "interictal_sync.npz"
    save_interictal_synchrony_result(result, str(out_npz))
    loaded = load_interictal_synchrony_result(str(out_npz))

    assert loaded.ch_names == result.ch_names
    assert np.array_equal(loaded.core_mask, result.core_mask)
    assert np.allclose(loaded.sync_phase_global, result.sync_phase_global, equal_nan=True)
    assert np.allclose(loaded.jaccard_global, result.jaccard_global, equal_nan=True)
    assert loaded.metadata == result.metadata
    assert loaded.summary["n_events"] == result.summary["n_events"]


def test_build_interictal_synchrony_from_legacy_lagpat(tmp_path: Path) -> None:
    lagpat_path = tmp_path / "block_lagPat.npz"
    packed_times_path = tmp_path / "block_packedTimes.npy"
    np.savez_compressed(
        lagpat_path,
        lagPatRaw=np.array([[0.01, 0.02], [0.03, np.nan]], dtype=np.float64),
        eventsBool=np.array([[1, 1], [1, 0]], dtype=np.float64),
        chnNames=np.array(["A1", "B1"], dtype=object),
        start_t=np.array([123.0], dtype=np.float64),
    )
    np.save(packed_times_path, np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64))

    result = build_interictal_synchrony_from_legacy_lagpat(
        str(lagpat_path),
        str(packed_times_path),
        core_channels=["A1"],
    )

    assert result.ch_names == ["A1", "B1"]
    assert np.array_equal(result.core_mask, np.array([True, False], dtype=bool))
    assert result.summary["n_events"] == 2.0


def test_build_event_rows_and_compat_summary() -> None:
    group_analysis = {
        "ch_names": ["A1", "B1"],
        "event_windows": np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
        "lag_raw": np.array([[0.0, 0.2], [0.0, 0.3]], dtype=np.float64),
        "events_bool": np.ones((2, 2), dtype=bool),
    }
    result = compute_interictal_synchrony(
        group_analysis,
        metadata={"subject": "S1", "block_start_epoch": 100.0, "block_stem": "S1_blk"},
    )
    rows = build_event_rows_from_result(result)
    assert len(rows) == 2
    assert rows[0]["schema_version"] == EVENT_SYNC_SCHEMA_VERSION
    assert rows[0]["event_start_epoch"] == 100.0
    assert rows[1]["event_center_epoch"] == 101.5
    compat = build_block_summary_from_event_rows(rows)
    assert compat["schema_version"] == BLOCK_SYNC_COMPAT_SCHEMA_VERSION
    assert compat["n_events"] == 2.0
    assert np.isclose(
        compat["mean_sync_legacy_global"],
        result.summary["mean_sync_legacy_global"],
        equal_nan=True,
    )


def test_run_epilepsiae_interictal_synchrony_from_manifest(tmp_path: Path) -> None:
    manifest_csv = tmp_path / "manifest.csv"
    artifact_root = tmp_path / "artifacts"
    subject_dir = artifact_root / "1073" / "all_recs"
    subject_dir.mkdir(parents=True)

    lagpat_path = subject_dir / "107300102_0000_lagPat.npz"
    packed_times_path = subject_dir / "107300102_0000_packedTimes.npy"
    np.savez_compressed(
        lagpat_path,
        lagPatRaw=np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64),
        eventsBool=np.ones((2, 2), dtype=np.float64),
        chnNames=np.array(["A1", "B1"], dtype=object),
    )
    np.save(packed_times_path, np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64))

    manifest_csv.write_text(
        "\n".join(
            [
                "subject,patient_code,tier,timezone_name,day_night_rule",
                "1073,FR_1073,ready_full_artifacts,Europe/Berlin,day=08:00-20:00 local",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fake_inventory = type(
        "Inventory",
        (),
        {
            "block_rows": [
                {
                    "subject": "1073",
                    "patient_code": "FR_1073",
                    "recording_id": "107300102",
                    "block_stem": "107300102_0000",
                    "block_start_epoch": 123.0,
                    "block_end_epoch": 483.0,
                    "block_start_day_night": "day",
                    "block_end_day_night": "day",
                    "timezone_name": "Europe/Berlin",
                }
            ]
        },
    )()

    import src.interictal_synchrony as sync_mod

    original_survey = sync_mod.survey_epilepsiae_dataset
    sync_mod.survey_epilepsiae_dataset = lambda: fake_inventory
    try:
        rows = run_epilepsiae_interictal_synchrony_from_manifest(
            str(manifest_csv),
            str(tmp_path / "outputs"),
            artifact_root=str(artifact_root),
        )
    finally:
        sync_mod.survey_epilepsiae_dataset = original_survey

    assert len(rows) == 1
    assert rows[0]["subject"] == "1073"
    assert rows[0]["recording_id"] == "107300102"
    assert rows[0]["block_start_epoch"] == 123.0
    assert rows[0]["block_end_epoch"] == 483.0
    assert Path(str(rows[0]["output_npz_path"])).exists()
    assert rows[0]["n_events"] == 2.0


def test_run_yuquan_interictal_synchrony(tmp_path: Path) -> None:
    artifact_root = tmp_path / "yuquan"
    subject_dir = artifact_root / "litengsheng"
    subject_dir.mkdir(parents=True)

    lagpat_path = subject_dir / "rec01_lagPat.npz"
    packed_times_path = subject_dir / "rec01_packedTimes.npy"
    edf_path = subject_dir / "rec01.edf"
    np.savez_compressed(
        lagpat_path,
        lagPatRaw=np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64),
        eventsBool=np.ones((2, 2), dtype=np.float64),
        chnNames=np.array(["A1", "B1"], dtype=object),
    )
    np.save(packed_times_path, np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64))
    edf_path.write_bytes(b"fake")

    import src.interictal_synchrony as sync_mod

    original_read = sync_mod.read_edf_record_info
    sync_mod.read_edf_record_info = lambda path: {
        "record": "rec01",
        "start_epoch": 0.0,
        "end_epoch": 7200.0,
    }
    try:
        rows = run_yuquan_interictal_synchrony(
            str(tmp_path / "outputs"),
            artifact_root=str(artifact_root),
            subjects=["litengsheng"],
        )
    finally:
        sync_mod.read_edf_record_info = original_read

    assert len(rows) == 1
    assert rows[0]["subject"] == "litengsheng"
    assert rows[0]["recording_id"] == "rec01"
    assert rows[0]["block_start_day_night"] == "day"
    assert rows[0]["block_end_day_night"] == "day"
    assert Path(str(rows[0]["output_npz_path"])).exists()
    assert rows[0]["n_events"] == 2.0


def test_select_core_penumbra_mask_rejects_empty_overlap() -> None:
    try:
        select_core_penumbra_mask(["A1", "B1"], ["C1"])
    except ValueError as exc:
        assert "No overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError for disjoint core_channels.")


def test_compute_adjacent_jaccard_handles_short_input() -> None:
    events_bool = np.array([[True], [False]], dtype=bool)
    out = compute_adjacent_jaccard(events_bool)
    assert out.shape == (0,)
