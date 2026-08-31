from pathlib import Path

import numpy as np

from src.topic5_group_event_state.source_audit import (
    _packed_path,
    _record_name,
    coverage_sessions,
    ictal_masks,
    lagpat_variant_of,
    subject_artifact_dir,
)


def test_record_name_handles_both_lagpat_suffixes():
    assert _record_name(Path("abc_lagPat_withFreqCent.npz")) == "abc"
    assert _record_name(Path("abc_lagPat.npz")) == "abc"


def test_subject_roots_are_dataset_specific():
    dataset, patient, root = subject_artifact_dir("epilepsiae_958")
    assert dataset == "epilepsiae"
    assert patient == "958"
    assert str(root).endswith("/958/all_recs")
    dataset, patient, root = subject_artifact_dir("yuquan_chengshuai")
    assert dataset == "yuquan"
    assert patient == "chengshuai"
    assert str(root).endswith("/chengshuai")


def test_packed_file_follows_the_lagpat_variant_not_file_existence(tmp_path):
    # Both packers wrote output for this record and they disagree on the event
    # list; pairing the old lagPat with the new packedTimes silently misaligns
    # every event.
    (tmp_path / "R_packedTimes.npy").write_bytes(b"")
    (tmp_path / "R_packedTimes_withFreqCent.npy").write_bytes(b"")
    old = tmp_path / "R_lagPat.npz"
    new = tmp_path / "R_lagPat_withFreqCent.npz"
    assert _packed_path(old, "R").name == "R_packedTimes.npy"
    assert _packed_path(new, "R").name == "R_packedTimes_withFreqCent.npy"
    assert lagpat_variant_of(old).name == "legacy"
    assert lagpat_variant_of(new).name == "withFreqCent"


class _FakeBlock:
    def __init__(self, name, start, end, n_events):
        self.record_name = name
        self.block_start_epoch = float(start)
        self.block_end_epoch = float(end)
        self.n_events = int(n_events)


def test_unobserved_recorded_block_breaks_the_session():
    # b0 and b2 abut in the clock only because b1 was recorded but never packed.
    # Bridging them would carry state across an hour with no observed events.
    blocks = [_FakeBlock("b0", 0, 3600, 10), _FakeBlock("b2", 7200, 10800, 10)]
    inventory = [
        {"block_stem": "b0", "block_start_epoch": "0", "block_end_epoch": "3600"},
        {"block_stem": "b1", "block_start_epoch": "3600", "block_end_epoch": "7200"},
        {"block_stem": "b2", "block_start_epoch": "7200", "block_end_epoch": "10800"},
    ]
    sessions = coverage_sessions(blocks, inventory)
    assert len(sessions) == 2

    # With b1 absent from the recording too, the clock gap alone still splits.
    sessions = coverage_sessions(blocks, inventory[:1] + inventory[2:])
    assert len(sessions) == 2


def test_contiguous_blocks_join_within_seam_tolerance():
    blocks = [_FakeBlock("b0", 0, 3600, 10), _FakeBlock("b1", 3600.5, 7200, 10)]
    inventory = [
        {"block_stem": "b0", "block_start_epoch": "0", "block_end_epoch": "3600"},
        {"block_stem": "b1", "block_start_epoch": "3600.5", "block_end_epoch": "7200"},
    ]
    sessions = coverage_sessions(blocks, inventory)
    assert len(sessions) == 1
    assert sessions[0]["n_events"] == 20


def test_ictal_mask_excludes_overlap_and_keeps_preictal():
    start = np.array([0.0, 90.0, 100.0, 150.0, 260.0])
    end = start + 0.25
    seizures = [{"onset_epoch": 100.0, "offset_epoch": 200.0}]
    masks = ictal_masks(start, end, seizures)
    assert masks["is_ictal"].tolist() == [False, False, True, True, False]
    # the event 10 s before onset survives and knows its lead time
    assert masks["time_to_next_seizure_sec"][1] == 10.0
    assert np.isinf(masks["time_to_next_seizure_sec"][4])
    assert masks["time_since_prev_seizure_sec"][4] == 60.0
