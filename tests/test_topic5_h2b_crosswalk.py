"""B0.1 contract tests: patient -> recording -> seizure crosswalk.

Every test here pins one clause of the prose contract in
``docs/archive/topic5/group_event_state_v0_2_h2b_spec_plan_2026-09-01.md`` §4
and the engineering invariant "Yuquan seizure ID 不能直接字符串连接;
用 recording code crosswalk 并逐发作核对 onset".
"""

from __future__ import annotations

import pytest

from src.topic5_h2b_transfer.crosswalk import (
    Disposition,
    build_recording_index,
    crosswalk_seizures,
    recording_code_of_record_name,
)


def _block(subject, dataset, record_name, start, end):
    return {
        "subject": subject,
        "dataset": dataset,
        "record_name": record_name,
        "block_start_epoch": str(start),
        "block_end_epoch": str(end),
        "status": "PASS",
    }


def _seizure(subject, record, sid, onset, offset):
    return {
        "subject": subject,
        "recording_id": record,
        "record": record,
        "seizure_id": sid,
        "eeg_onset_epoch": str(onset),
        "eeg_offset_epoch": str(offset),
    }


# --- C1: recording code is derived per dataset, not guessed ------------------


def test_epilepsiae_record_name_splits_off_the_block_suffix():
    assert recording_code_of_record_name("epilepsiae", "107300102_0000") == "107300102"
    assert recording_code_of_record_name("epilepsiae", "107300102_0231") == "107300102"


def test_yuquan_record_name_is_itself_the_recording_code():
    assert recording_code_of_record_name("yuquan", "FC10477Q") == "FC10477Q"


# --- C1 + C2: the join is by recording code AND onset containment ------------


def test_onset_inside_a_different_recording_does_not_count_as_matched():
    """A seizure is matched only when its OWN recording contains the onset.

    Time-only containment would match it against the neighbouring recording;
    the recording-code clause forbids that.
    """
    blocks = [
        _block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0),
        _block("yuquan_gaolan", "yuquan", "FA0013KS", 2000.0, 3000.0),
    ]
    index = build_recording_index(blocks)
    # onset 2500 lies in FA0013KS, but the seizure row claims FA0013KQ
    rows = [_seizure("gaolan", "FA0013KQ", "sz1", 2500.0, 2530.0)]
    result = crosswalk_seizures(rows, index, "yuquan", {"yuquan_gaolan"})
    (entry,) = result.entries
    assert entry.disposition is Disposition.ONSET_OUTSIDE_RECORDING


def test_seizure_whose_onset_is_inside_its_own_recording_is_matched():
    blocks = [_block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0)]
    index = build_recording_index(blocks)
    rows = [_seizure("gaolan", "FA0013KQ", "sz1", 1500.0, 1530.0)]
    result = crosswalk_seizures(rows, index, "yuquan", {"yuquan_gaolan"})
    (entry,) = result.entries
    assert entry.disposition is Disposition.MATCHED
    assert entry.subject == "yuquan_gaolan"
    assert entry.block_record_name == "FA0013KQ"


# --- C1 negative: the failure mode the invariant was written against ---------


def test_recording_without_group_event_artifact_is_reported_not_dropped():
    """The Yuquan silent-drop mode: subject exists, recording does not.

    A subject-string inner join would happily attach this seizure to the
    patient; the recording-code join must surface it as unmatched instead.
    """
    blocks = [_block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0)]
    index = build_recording_index(blocks)
    rows = [_seizure("gaolan", "FA0099ZZ", "sz1", 1500.0, 1530.0)]
    result = crosswalk_seizures(rows, index, "yuquan", {"yuquan_gaolan"})
    (entry,) = result.entries
    assert entry.disposition is Disposition.RECORDING_ABSENT
    assert entry.seizure_id == "sz1"


# --- C3: no silent drops; dispositions reconcile -----------------------------


def test_every_input_row_gets_exactly_one_disposition_and_counts_reconcile():
    blocks = [
        _block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0),
        _block("yuquan_xuxinyi", "yuquan", "FA0012BD", 1000.0, 2000.0),
    ]
    index = build_recording_index(blocks)
    rows = [
        _seizure("gaolan", "FA0013KQ", "a", 1500.0, 1530.0),  # matched
        _seizure("gaolan", "FA0099ZZ", "b", 1500.0, 1530.0),  # recording absent
        _seizure("gaolan", "FA0013KQ", "c", 9999.0, 9999.0),  # onset outside
        _seizure("nobody", "FA0013KQ", "d", 1500.0, 1530.0),  # not in dataset
    ]
    result = crosswalk_seizures(
        rows, index, "yuquan", {"yuquan_gaolan", "yuquan_xuxinyi"}
    )
    assert len(result.entries) == len(rows)
    assert sum(result.disposition_counts.values()) == len(rows)
    assert result.disposition_counts[Disposition.MATCHED.value] == 1


# --- C4: ambiguity is surfaced, never resolved by picking one ----------------


def test_duplicate_seizure_id_within_a_subject_is_flagged_ambiguous():
    blocks = [_block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0)]
    index = build_recording_index(blocks)
    rows = [
        _seizure("gaolan", "FA0013KQ", "dup", 1500.0, 1530.0),
        _seizure("gaolan", "FA0013KQ", "dup", 1600.0, 1630.0),
    ]
    result = crosswalk_seizures(rows, index, "yuquan", {"yuquan_gaolan"})
    assert all(e.disposition is Disposition.DUPLICATE_SEIZURE_ID for e in result.entries)


def test_overlapping_recordings_make_the_onset_ambiguous():
    blocks = [
        _block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0),
        _block("yuquan_gaolan", "yuquan", "FA0013KS", 1400.0, 2400.0),
    ]
    index = build_recording_index(blocks)
    rows = [_seizure("gaolan", "FA0013KQ", "sz1", 1500.0, 1530.0)]
    result = crosswalk_seizures(rows, index, "yuquan", {"yuquan_gaolan"})
    (entry,) = result.entries
    assert entry.disposition is Disposition.AMBIGUOUS_MULTIPLE_RECORDINGS


# --- C5: zero-seizure dataset subjects stay visible --------------------------


def test_dataset_subject_with_no_seizure_rows_is_reported_explicitly():
    """'未检出' must not be silently indistinguishable from '无发作'."""
    blocks = [
        _block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0),
        _block("yuquan_chengshuai", "yuquan", "FC10477Q", 1000.0, 2000.0),
    ]
    index = build_recording_index(blocks)
    rows = [_seizure("gaolan", "FA0013KQ", "sz1", 1500.0, 1530.0)]
    result = crosswalk_seizures(
        rows, index, "yuquan", {"yuquan_gaolan", "yuquan_chengshuai"}
    )
    assert result.dataset_subjects_without_seizure_rows == ("yuquan_chengshuai",)


# --- C6: subject-set symmetric difference, never assumed equal ---------------


def test_subjects_present_in_only_one_of_the_two_collections_are_reported():
    blocks = [_block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0)]
    index = build_recording_index(blocks)
    rows = [
        _seizure("gaolan", "FA0013KQ", "sz1", 1500.0, 1530.0),
        _seizure("litengsheng", "FA0011AA", "sz2", 1500.0, 1530.0),
    ]
    result = crosswalk_seizures(
        rows, index, "yuquan", {"yuquan_gaolan", "yuquan_chengshuai"}
    )
    assert result.inventory_subjects_not_in_dataset == ("yuquan_litengsheng",)
    assert result.dataset_subjects_without_seizure_rows == ("yuquan_chengshuai",)


# --- C7: degenerate intervals are flagged, not dropped and not silently kept --


def test_zero_duration_interval_is_flagged_but_still_carried():
    blocks = [_block("yuquan_chenziyang", "yuquan", "FC1047Y7", 1000.0, 2000.0)]
    index = build_recording_index(blocks)
    rows = [_seizure("chenziyang", "FC1047Y7", "sz1", 1500.0, 1500.0)]
    result = crosswalk_seizures(rows, index, "yuquan", {"yuquan_chenziyang"})
    (entry,) = result.entries
    assert entry.disposition is Disposition.MATCHED
    assert "zero_duration" in entry.flags


def test_non_finite_onset_is_flagged_and_cannot_be_matched():
    blocks = [_block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0)]
    index = build_recording_index(blocks)
    rows = [_seizure("gaolan", "FA0013KQ", "sz1", "", "")]
    result = crosswalk_seizures(rows, index, "yuquan", {"yuquan_gaolan"})
    (entry,) = result.entries
    assert entry.disposition is Disposition.INCOMPLETE_INTERVAL
    assert "onset_not_finite" in entry.flags


# --- index construction guards ------------------------------------------------


def test_recording_index_rejects_a_recording_shared_by_two_subjects():
    blocks = [
        _block("yuquan_gaolan", "yuquan", "FA0013KQ", 1000.0, 2000.0),
        _block("yuquan_xuxinyi", "yuquan", "FA0013KQ", 1000.0, 2000.0),
    ]
    with pytest.raises(ValueError, match="shared by"):
        build_recording_index(blocks)


# --- C9: the inventory's own block_id is checked, not trusted ----------------


def _blk(bid, start, end):
    return {"block_id": bid, "block_start_epoch": str(start), "block_end_epoch": str(end)}


def test_block_id_that_contains_the_onset_is_kept_unchanged():
    from src.topic5_h2b_transfer.crosswalk import resolve_block_for_onset

    blocks = [_blk("b0", 0.0, 100.0), _blk("b1", 100.0, 200.0)]
    bid, status = resolve_block_for_onset(150.0, blocks, claimed_block_id="b1")
    assert (bid, status) == ("b1", "claim_ok")


def test_a_stale_block_id_is_repaired_to_the_block_that_holds_the_onset():
    """Two canonical Epilepsiae rows name a block ~14 h before their own onset.

    The onset epoch is the trustworthy field, so the denormalised block_id is
    repaired rather than the seizure being dropped -- but the repair is
    reported, never silent.
    """
    from src.topic5_h2b_transfer.crosswalk import resolve_block_for_onset

    blocks = [_blk("b0", 0.0, 100.0), _blk("b1", 100.0, 200.0)]
    bid, status = resolve_block_for_onset(150.0, blocks, claimed_block_id="b0")
    assert (bid, status) == ("b1", "claim_repaired")


def test_an_onset_in_no_block_at_all_is_not_repaired():
    from src.topic5_h2b_transfer.crosswalk import resolve_block_for_onset

    blocks = [_blk("b0", 0.0, 100.0)]
    bid, status = resolve_block_for_onset(999.0, blocks, claimed_block_id="b0")
    assert bid is None and status == "no_block_contains_onset"
