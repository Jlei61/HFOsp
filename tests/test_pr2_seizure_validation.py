import pytest

from scripts.pr2_seizure_validation import _subject_level_match


def test_subject_level_match_deduplicates_epilepsiae_cross_block_segments() -> None:
    results = [
        {
            "manual_abs_intervals": [
                {
                    "seizure_id": "s1",
                    "abs_onset_epoch": 100.0,
                    "abs_offset_epoch": 120.0,
                }
            ],
            "manual_abs_onset_only": [],
            "detected_abs_intervals": [(105.0, 119.0)],
        },
        {
            "manual_abs_intervals": [
                {
                    "seizure_id": "s1",
                    "abs_onset_epoch": 120.0,
                    "abs_offset_epoch": 140.0,
                }
            ],
            "manual_abs_onset_only": [],
            "detected_abs_intervals": [(120.0, 138.0)],
        },
    ]
    out = _subject_level_match(results, "epilepsiae")
    assert len(out["manual_pairs"]) == 1
    assert out["manual_pairs"][0] == (100.0, 140.0)
    assert len(out["detected_pairs"]) == 1
    assert out["recall"] == 1.0
    assert out["precision"] == 1.0


def test_subject_level_match_keeps_yuquan_onset_only_events() -> None:
    results = [
        {
            "manual_abs_intervals": [],
            "manual_abs_onset_only": [100.0],
            "detected_abs_intervals": [(95.0, 110.0)],
        }
    ]
    out = _subject_level_match(results, "yuquan")
    assert out["manual_onset_only_abs"] == [100.0]
    assert len(out["tp"]) == 1
    assert out["fn"] == []
    assert out["fp"] == []
