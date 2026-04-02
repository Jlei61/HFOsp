from __future__ import annotations

from pathlib import Path

from src.preprocessing import (
    parse_seizure_annotation_events,
    build_recording_timeline,
    epoch_to_local_hour,
    fast_read_edf_annotations,
    parse_seizure_onsets_from_annotations,
    read_edf_record_info,
)


def _fw(text: str, width: int) -> bytes:
    raw = text.encode("ascii", errors="ignore")[:width]
    return raw.ljust(width, b" ")


def _write_minimal_edf_plus_with_annotations(
    path: Path,
    *,
    tal_text: str | None = None,
    n_records: int = 1,
    record_duration_sec: float = 1.0,
) -> None:
    n_signals = 2
    ns_per_record = [1, 64]  # signal + annotation bytes/2
    header_n_bytes = 256 + (n_signals * 256)

    fixed_header = b"".join(
        [
            _fw("0", 8),
            _fw("X", 80),
            _fw("X", 80),
            _fw("01.01.24", 8),
            _fw("00.00.00", 8),
            _fw(str(header_n_bytes), 8),
            _fw("EDF+C", 44),
            _fw(str(n_records), 8),
            _fw(str(record_duration_sec), 8),
            _fw(str(n_signals), 4),
        ]
    )
    assert len(fixed_header) == 256

    labels = [_fw("A1", 16), _fw("EDF Annotations", 16)]
    transducer = [_fw("", 80), _fw("", 80)]
    phys_dim = [_fw("uV", 8), _fw("", 8)]
    phys_min = [_fw("-1000", 8), _fw("-1", 8)]
    phys_max = [_fw("1000", 8), _fw("1", 8)]
    dig_min = [_fw("-32768", 8), _fw("-32768", 8)]
    dig_max = [_fw("32767", 8), _fw("32767", 8)]
    prefilter = [_fw("", 80), _fw("", 80)]
    samples = [_fw(str(ns_per_record[0]), 8), _fw(str(ns_per_record[1]), 8)]
    reserved = [_fw("", 32), _fw("", 32)]

    signal_header = b"".join(
        [
            b"".join(labels),
            b"".join(transducer),
            b"".join(phys_dim),
            b"".join(phys_min),
            b"".join(phys_max),
            b"".join(dig_min),
            b"".join(dig_max),
            b"".join(prefilter),
            b"".join(samples),
            b"".join(reserved),
        ]
    )
    assert len(signal_header) == n_signals * 256

    if tal_text is None:
        tal_text = (
            "+10\x1515\x14EEG SZ\x14\x00"
            "+40\x14SZ1\x14\x00"
            "+45\x14END\x14\x00"
            "+50\x155\x14artifact\x14\x00"
        )
    tal = tal_text.encode("latin1")
    ann_n_bytes = ns_per_record[1] * 2
    tal_payload = tal.ljust(ann_n_bytes, b"\x00")
    data_record = b"\x00\x00" + tal_payload

    with open(path, "wb") as f:
        f.write(fixed_header)
        f.write(signal_header)
        f.write(data_record)


def test_fast_read_edf_annotations_and_seizure_parse(tmp_path: Path) -> None:
    edf_path = tmp_path / "mini.edf"
    _write_minimal_edf_plus_with_annotations(edf_path)

    anns = fast_read_edf_annotations(edf_path)
    assert len(anns) >= 2
    assert anns[0][0] == 10.0
    assert anns[0][1] == 15.0
    assert anns[0][2] == "EEG SZ"

    intervals = parse_seizure_onsets_from_annotations(
        edf_path,
        target_labels=["EEG SZ"],
        start_t_epoch=1000.0,
    )
    assert intervals == [(1010.0, 1025.0)]

    paired = parse_seizure_onsets_from_annotations(
        edf_path,
        target_labels=["SZ1"],
        start_t_epoch=1000.0,
    )
    assert paired == [(1040.0, 1045.0)]


def test_epoch_to_local_hour_timezone_explicit() -> None:
    assert epoch_to_local_hour(1574123210, "Asia/Shanghai") == 8
    assert epoch_to_local_hour(1574123210, "Europe/Paris") == 1


def test_parse_seizure_annotations_merges_overlapping_duplicates(tmp_path: Path) -> None:
    edf_path = tmp_path / "dup.edf"
    _write_minimal_edf_plus_with_annotations(
        edf_path,
        tal_text=(
            "+10\x14SZ1\x14\x00"
            "+12\x14EEG onset\x14\x00"
            "+20\x14END\x14\x00"
        ),
    )

    intervals = parse_seizure_onsets_from_annotations(
        edf_path,
        target_labels=["SZ1", "EEG onset"],
        start_t_epoch=1000.0,
    )
    assert intervals == [(1010.0, 1020.0)]


def test_parse_seizure_annotations_preserves_orphan_markers_separately(tmp_path: Path) -> None:
    edf_path = tmp_path / "orphan.edf"
    _write_minimal_edf_plus_with_annotations(
        edf_path,
        tal_text="+10\x14SZ1\x14\x00",
    )

    intervals = parse_seizure_onsets_from_annotations(
        edf_path,
        target_labels=["SZ1"],
        start_t_epoch=1000.0,
    )
    assert intervals == []
    parsed = parse_seizure_annotation_events(
        edf_path,
        target_labels=["SZ1"],
        start_t_epoch=1000.0,
    )
    assert parsed["intervals"] == []
    assert parsed["orphan_onsets"] == [1010.0]


def test_parse_seizure_annotation_events_reports_offset_sources(tmp_path: Path) -> None:
    edf_path = tmp_path / "sources.edf"
    _write_minimal_edf_plus_with_annotations(
        edf_path,
        tal_text=(
            "+10\x155\x14EEG SZ\x14\x00"
            "+40\x14SZ1\x14\x00"
            "+45\x14END\x14\x00"
        ),
    )
    parsed = parse_seizure_annotation_events(
        edf_path,
        target_labels=["EEG SZ", "SZ1"],
        start_t_epoch=1000.0,
    )
    assert parsed["intervals"] == [(1010.0, 1015.0), (1040.0, 1045.0)]
    details = parsed["raw_interval_details"]
    assert [d["offset_source"] for d in details] == ["duration", "end_label"]
    assert details[1]["paired_end_label"] == "END"


def test_read_edf_record_info_and_timeline(tmp_path: Path) -> None:
    edf_a = tmp_path / "A.edf"
    edf_b = tmp_path / "B.edf"
    _write_minimal_edf_plus_with_annotations(
        edf_a,
        tal_text="",
        n_records=7200,
        record_duration_sec=1.0,
    )
    _write_minimal_edf_plus_with_annotations(
        edf_b,
        tal_text="",
        n_records=3600,
        record_duration_sec=1.0,
    )

    # Rewrite B start time to be 2h + 30s after A.
    with open(edf_b, "r+b") as f:
        f.seek(168)
        f.write(_fw("01.01.24", 8))
        f.write(_fw("02.00.30", 8))

    info_a = read_edf_record_info(edf_a)
    info_b = read_edf_record_info(edf_b)
    assert info_a["duration_sec"] == 7200.0
    assert info_b["duration_sec"] == 3600.0

    timeline = build_recording_timeline([edf_b, edf_a], continuity_gap_tolerance_sec=10.0)
    assert timeline["n_records"] == 2
    assert timeline["sum_duration_sec"] == 10800.0
    assert timeline["n_gap_violations"] == 1
    assert timeline["is_continuous"] is False
    assert timeline["records"][1]["gap_prev_sec"] == 30.0
