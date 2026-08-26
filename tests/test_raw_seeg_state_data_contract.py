"""TDD suite for the Raw-SEEG R0.1 data contract layer (Worker A).

Every test pins one invariant from
``docs/archive/topic5/raw_seeg_state_scientific_spec_2026-08-21.md`` §4 and
would fail if the corresponding rule were violated.  All fixtures are small
synthetic frames — no ``/mnt`` access — except the two integration tests at the
bottom, which are marked ``integration``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.topic5_raw_seeg_state import contract
from src.topic5_raw_seeg_state import data_contract as dc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

T0 = 1_600_000_000.0


def _minute_frame(
    n_minutes: int,
    *,
    covered=None,
    guard_free=None,
    session_id=None,
    split=None,
    first_epoch: float = T0,
) -> pd.DataFrame:
    """Build a synthetic window-index frame with the A3 columns."""
    idx = np.arange(n_minutes, dtype=np.int64)
    covered = np.ones(n_minutes, bool) if covered is None else np.asarray(covered, bool)
    guard_free = (
        np.ones(n_minutes, bool) if guard_free is None else np.asarray(guard_free, bool)
    )
    session_id = (
        np.zeros(n_minutes, np.int64) if session_id is None else np.asarray(session_id, np.int64)
    )
    split = np.array(["train"] * n_minutes, dtype=object) if split is None else np.asarray(split, dtype=object)
    minute_usable = covered & guard_free & (session_id >= 0)
    df = pd.DataFrame(
        {
            "subject": "synthetic",
            "minute_index": idx,
            "minute_start_epoch": first_epoch + 60.0 * idx,
            "session_id": session_id,
            "split": split,
            "covered": covered,
            "guard_free": guard_free,
            "n_valid_contacts": np.full(n_minutes, -1, np.int64),
            "minute_usable": minute_usable,
        }
    )
    flags = dc.compute_eligibility_flags(
        session_id=df["session_id"].to_numpy(),
        split=df["split"].to_numpy(),
        minute_usable=df["minute_usable"].to_numpy(),
        guard_free=df["guard_free"].to_numpy(),
    )
    for key, value in flags.items():
        df[key] = value
    return df[list(contract.WINDOW_INDEX_COLUMNS)]


# ---------------------------------------------------------------------------
# 1. session splitting
# ---------------------------------------------------------------------------


def test_session_join_299s_gap_keeps_one_session():
    starts = np.array([0.0, 600.0 + 299.0])
    ends = np.array([600.0, 600.0 + 299.0 + 600.0])
    gap, opens, sid = dc.assign_sessions(starts, ends)
    assert np.isnan(gap[0])  # first block has no predecessor
    assert gap[1] == pytest.approx(299.0)
    assert opens.tolist() == [True, False]
    assert sid.tolist() == [0, 0]


def test_session_join_301s_gap_opens_second_session():
    starts = np.array([0.0, 600.0 + 301.0])
    ends = np.array([600.0, 600.0 + 301.0 + 600.0])
    gap, opens, sid = dc.assign_sessions(starts, ends)
    assert gap[1] == pytest.approx(301.0)
    assert opens.tolist() == [True, True]
    assert sid.tolist() == [0, 1]


def test_session_boundary_uses_contract_constant():
    """Exactly SESSION_JOIN_SECONDS must NOT open a session (rule is ``> 300``)."""
    j = contract.SESSION_JOIN_SECONDS
    starts = np.array([0.0, 600.0 + j])
    ends = np.array([600.0, 600.0 + j + 600.0])
    _, opens, sid = dc.assign_sessions(starts, ends)
    assert opens.tolist() == [True, False]
    assert sid.tolist() == [0, 0]


def test_assign_sessions_sorts_unsorted_input():
    starts = np.array([10_000.0, 0.0])
    ends = np.array([10_600.0, 600.0])
    gap, opens, sid = dc.assign_sessions(starts, ends)
    # returned in chronological order
    assert opens.tolist() == [True, True]
    assert sid.tolist() == [0, 1]
    assert gap[1] == pytest.approx(9400.0)


# ---------------------------------------------------------------------------
# 2. minute coverage
# ---------------------------------------------------------------------------


def test_minute_with_57s_coverage_is_not_covered():
    # block covers [0, 57) of the minute [0, 60)
    sec = dc.minute_covered_seconds(np.array([0.0]), np.array([0.0]), np.array([57.0]))
    assert sec[0] == pytest.approx(57.0)
    assert bool(dc.covered_from_seconds(sec)[0]) is False


def test_minute_with_58s_coverage_is_covered():
    sec = dc.minute_covered_seconds(np.array([0.0]), np.array([0.0]), np.array([58.0]))
    assert sec[0] == pytest.approx(58.0)
    assert bool(dc.covered_from_seconds(sec)[0]) is True


def test_minute_coverage_merges_overlapping_blocks():
    """Slightly overlapping EDF blocks must not double-count coverage."""
    starts = np.array([0.0, 59.94])
    ends = np.array([60.0, 120.0])
    sec = dc.minute_covered_seconds(np.array([0.0]), starts, ends)
    assert sec[0] == pytest.approx(60.0)


def test_minute_coverage_sums_across_a_micro_gap():
    starts = np.array([0.0, 40.0])
    ends = np.array([20.0, 60.0])
    sec = dc.minute_covered_seconds(np.array([0.0]), starts, ends)
    assert sec[0] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# 3. seizure guard
# ---------------------------------------------------------------------------


def test_guard_marks_minute_inside_preictal_postictal_window():
    onset, offset = 100_000.0, 100_120.0
    guards = dc.guard_intervals_from_seizures(
        np.array([onset]), np.array([offset])
    )
    assert guards.tolist() == [
        [onset - contract.PREICTAL_GUARD_SECONDS, offset + contract.POSTICTAL_GUARD_SECONDS]
    ]
    # a minute overlapping the guard
    inside = dc.minute_guard_free(np.array([onset - 600.0]), guards)
    assert bool(inside[0]) is False
    # the minute that starts exactly at the guard onset
    edge = dc.minute_guard_free(
        np.array([onset - contract.PREICTAL_GUARD_SECONDS]), guards
    )
    assert bool(edge[0]) is False


def test_minute_ending_3601s_before_onset_is_guard_free():
    onset, offset = 100_000.0, 100_120.0
    guards = dc.guard_intervals_from_seizures(np.array([onset]), np.array([offset]))
    # minute [onset-3661, onset-3601): ends 3601 s before onset -> no overlap
    free = dc.minute_guard_free(np.array([onset - 3661.0]), guards)
    assert bool(free[0]) is True
    # one minute later it clips the guard and must fail
    clipped = dc.minute_guard_free(np.array([onset - 3601.0]), guards)
    assert bool(clipped[0]) is False


def test_missing_seizure_offset_falls_back_to_onset_plus_120s():
    onset = np.array([100_000.0, 200_000.0])
    offset = np.array([100_050.0, np.nan])
    guards = dc.guard_intervals_from_seizures(onset, offset)
    assert guards[1, 1] == pytest.approx(
        200_000.0 + dc.MISSING_OFFSET_FALLBACK_SECONDS + contract.POSTICTAL_GUARD_SECONDS
    )
    assert dc.count_offset_fallbacks(onset, offset) == 1


# ---------------------------------------------------------------------------
# 4 / 5. horizon eligibility
# ---------------------------------------------------------------------------


def test_h100_false_when_a_guard_minute_sits_between_t_and_t_plus_100():
    n = 200
    guard_free = np.ones(n, bool)
    guard_free[60] = False  # strictly between t=20 and t+100=120
    df = _minute_frame(n, guard_free=guard_free)
    t = 20
    assert bool(df.loc[t, "ctx_ok"]) is True
    assert bool(df.loc[t + 100, "minute_usable"]) is True
    assert bool(df.loc[t, "h100_ok"]) is False
    # a t whose whole [t-9, t+100] window avoids minute 60 is still eligible
    t_ok = 70
    assert bool(df.loc[t_ok, "h100_ok"]) is True


def test_uncovered_intervening_minute_does_not_block_the_horizon():
    """Spec §4.4: intervening minutes may be unrecorded (micro-gap)."""
    n = 60
    covered = np.ones(n, bool)
    covered[25] = False  # strictly between t=20 and t+10=30
    df = _minute_frame(n, covered=covered)
    assert bool(df.loc[20, "h10_ok"]) is True
    # but the target minute itself must be usable
    assert bool(df.loc[15, "h10_ok"]) is False  # target 25 unusable


def test_horizon_false_across_a_session_boundary():
    n = 60
    session = np.zeros(n, np.int64)
    session[30:] = 1
    df = _minute_frame(n, session_id=session)
    assert bool(df.loc[25, "h5_ok"]) is False   # target 30 in session 1
    assert bool(df.loc[20, "h10_ok"]) is False  # target 30 in session 1
    assert bool(df.loc[20, "h5_ok"]) is True    # entirely inside session 0
    assert bool(df.loc[45, "h5_ok"]) is True    # entirely inside session 1
    # ctx_ok itself must fail where the context straddles the boundary
    assert bool(df.loc[32, "ctx_ok"]) is False
    assert bool(df.loc[39, "ctx_ok"]) is True


def test_horizon_false_across_a_split_boundary():
    n = 60
    split = np.array(["train"] * 30 + ["validation"] * 30, dtype=object)
    df = _minute_frame(n, split=split)
    assert bool(df.loc[25, "h5_ok"]) is False
    assert bool(df.loc[20, "h5_ok"]) is True


def test_horizon_false_when_target_is_outside_the_grid():
    df = _minute_frame(50)
    assert bool(df.loc[49, "h1_ok"]) is False
    assert bool(df.loc[48, "h1_ok"]) is True
    assert not df["h100_ok"].any()


def test_ctx_ok_requires_full_context_history():
    df = _minute_frame(40)
    assert not df.loc[: contract.CONTEXT_MINUTES - 2, "ctx_ok"].any()
    assert bool(df.loc[contract.CONTEXT_MINUTES - 1, "ctx_ok"]) is True


def test_minute_session_id_is_minus_one_inside_a_long_gap():
    starts = np.array([0.0, 7200.0])
    ends = np.array([600.0, 7800.0])
    _, _, sid = dc.assign_sessions(starts, ends)
    extents = dc.session_extents(starts, ends, sid)
    minute_starts = np.array([0.0, 3600.0, 7200.0])
    got = dc.minute_session_ids(minute_starts, extents)
    assert got.tolist() == [0, -1, 1]


# ---------------------------------------------------------------------------
# 6. sealed gate
# ---------------------------------------------------------------------------


def test_minute_grid_never_reaches_the_sealed_bound():
    dev_end = T0 + 60.0 * 10 + 17.0
    starts = dc.minute_grid_starts(T0, dev_end)
    assert starts[-1] < dev_end
    assert starts[-1] + 60.0 >= dev_end  # grid is maximal
    assert starts[0] == pytest.approx(T0)


def test_assert_not_sealed_rejects_a_violating_epoch_for_a_real_subject():
    subject = contract.cohort_subjects()[0]
    bound = contract.dev_end_epoch(subject)
    contract.assert_not_sealed(subject, np.array([bound - 1.0]))
    with pytest.raises(ValueError, match="SEALED-PARTITION VIOLATION"):
        contract.assert_not_sealed(subject, np.array([bound]))


def test_block_split_labels_mark_post_seal_blocks_sealed():
    train_end, dev_end = 1000.0, 2000.0
    labels = dc.block_split_labels(
        np.array([0.0, 999.0, 1000.0, 1999.0, 2000.0, 5000.0]), train_end, dev_end
    )
    assert labels.tolist() == [
        "train",
        "train",
        "validation",
        "validation",
        "sealed",
        "sealed",
    ]


# ---------------------------------------------------------------------------
# 7. bipolar pairing
# ---------------------------------------------------------------------------


def test_bipolar_pairs_are_adjacent_index_only():
    labels = ["A1", "A2", "A3", "B1", "B2"]
    pairs = dc.bipolar_pairs_from_labels(labels)
    assert [p["channel_name"] for p in pairs] == ["A1-A2", "A2-A3", "B1-B2"]
    assert [(p["native_index_anode"], p["native_index_cathode"]) for p in pairs] == [
        (0, 1),
        (1, 2),
        (3, 4),
    ]
    assert [p["shaft"] for p in pairs] == ["A", "A", "B"]
    # shaft_index is the 0-based position ALONG the shaft, not the contact number
    assert [p["shaft_index"] for p in pairs] == [0, 1, 0]
    assert [p["anode_ordinal"] for p in pairs] == [1, 2, 1]


def test_bipolar_pairing_does_not_skip_across_a_missing_contact():
    labels = ["A1", "A3", "B1", "B2"]
    pairs = dc.bipolar_pairs_from_labels(labels)
    names = [p["channel_name"] for p in pairs]
    assert "A1-A3" not in names
    assert names == ["B1-B2"]


def test_bipolar_pairing_respects_keep_indices_and_native_order():
    labels = ["ECG", "A1", "SCALP", "A2", "A3"]
    keep = [1, 3, 4]
    pairs = dc.bipolar_pairs_from_labels(labels, keep_indices=keep)
    assert [p["channel_name"] for p in pairs] == ["A1-A2", "A2-A3"]
    assert [(p["native_index_anode"], p["native_index_cathode"]) for p in pairs] == [
        (1, 3),
        (3, 4),
    ]


def test_bipolar_pairing_handles_prime_shafts_and_label_prefixes():
    labels = ["POL D'1", "POL D'2", "EEG A1-Ref", "EEG A2-Ref", "POL A3", "POL E"]
    pairs = dc.bipolar_pairs_from_labels(labels)
    assert [p["channel_name"] for p in pairs] == ["A1-A2", "A2-A3", "D'1-D'2"]


def test_shaft_index_is_dense_across_a_contact_gap():
    """A1 A2 A4 A5 -> two pairs that must occupy categorical slots 0 and 1."""
    pairs = dc.bipolar_pairs_from_labels(["A1", "A2", "A4", "A5"])
    assert [p["channel_name"] for p in pairs] == ["A1-A2", "A4-A5"]
    assert [p["shaft_index"] for p in pairs] == [0, 1]
    assert [p["anode_ordinal"] for p in pairs] == [1, 4]


def test_dense_shaft_index_skips_invalid_rows_and_marks_them_minus_one():
    shafts = ["A", "A", "A", "B", "B"]
    valid = [True, False, True, True, True]
    got = dc.dense_shaft_index(shafts, valid)
    assert got.tolist() == [0, -1, 1, 0, 1]


# ---------------------------------------------------------------------------
# 9. coordinate / contact decoupling  (coordinator ruling 1)
# ---------------------------------------------------------------------------


def _pairs(n_a: int = 3, n_b: int = 2):
    labels = [f"A{i}" for i in range(1, n_a + 1)] + [f"B{i}" for i in range(1, n_b + 1)]
    return dc.bipolar_pairs_from_labels(labels)


def test_missing_coordinate_does_not_clear_contact_valid():
    pairs = _pairs()                       # A1-A2, A2-A3, B1-B2
    coords = np.array([[1.0, 2.0, 3.0], [np.nan] * 3, [4.0, 5.0, 6.0]])
    mapped = np.array([True, False, True])
    df = dc.assemble_contact_rows(
        "synthetic", "yuquan", pairs, coords, mapped, "fs_native_ras_mm[mm]",
        addressable=np.ones(3, bool),
    )
    assert list(df.columns) == list(contract.CONTACT_METADATA_COLUMNS)
    assert df["contact_valid"].tolist() == [True, True, True]
    assert df["coord_valid"].tolist() == [True, False, True]
    assert df.loc[1, "drop_reason"] == "missing_coordinate"
    assert df.loc[0, "drop_reason"] == ""
    assert np.isnan(df.loc[1, "x_mm"])                 # NaN stays NaN in the parquet
    # one mapped contact is enough to keep the subject in mm mode
    assert (df["coord_mode"] == contract.COORD_MODE_FULL).all()
    # every contact keeps a categorical slot even without a coordinate
    assert df["shaft_index"].tolist() == [0, 1, 0]


def test_subject_with_no_coordinates_at_all_is_topology_only_but_still_valid():
    pairs = _pairs()
    coords = np.full((3, 3), np.nan)
    df = dc.assemble_contact_rows(
        "synthetic", "yuquan", pairs, coords, np.zeros(3, bool), "unavailable",
        addressable=np.ones(3, bool),
    )
    assert df["contact_valid"].all()                   # recordings are fine
    assert not df["coord_valid"].any()
    assert (df["coord_mode"] == contract.COORD_MODE_TOPOLOGY_ONLY).all()
    assert (df["drop_reason"] == "missing_coordinate").all()
    assert df["shaft_index"].tolist() == [0, 1, 0]


def test_unaddressable_contact_is_the_only_thing_that_clears_contact_valid():
    pairs = _pairs()
    coords = np.zeros((3, 3))
    df = dc.assemble_contact_rows(
        "synthetic", "epilepsiae", pairs, coords, np.ones(3, bool), "mni152_1mm[mm]",
        addressable=np.array([True, False, True]),
    )
    assert df["contact_valid"].tolist() == [True, False, True]
    assert df.loc[1, "drop_reason"] == "inconsistent_native_index"
    assert df.loc[1, "coord_valid"]                    # coords survive independently
    # the dropped channel must not consume a categorical slot
    assert df["shaft_index"].tolist() == [0, -1, 0]


# ---------------------------------------------------------------------------
# 10. seizure guard source union  (coordinator ruling 2)
# ---------------------------------------------------------------------------


def test_merge_seizure_sources_adds_a_zero_duration_onset_the_inventory_dropped():
    inv_on = np.array([100_000.0, 300_000.0])
    inv_off = np.array([100_180.0, 300_150.0])
    # the annotation scan repeats both (within 1 s) and adds a zero-duration mark
    scan_on = np.array([100_000.4, 300_000.0, 200_000.0])
    scan_off = np.array([100_180.0, 300_150.0, 200_000.0])
    on, off, n_new = dc.merge_seizure_sources((inv_on, inv_off), (scan_on, scan_off))
    assert n_new == 1
    assert on.tolist() == [100_000.0, 200_000.0, 300_000.0]

    guards = dc.guard_intervals_from_seizures(on, off)
    # the zero-duration onset must still produce a full guard via the 120 s fallback
    assert guards[1, 0] == pytest.approx(200_000.0 - contract.PREICTAL_GUARD_SECONDS)
    assert guards[1, 1] == pytest.approx(
        200_000.0
        + contract.SEIZURE_OFFSET_FALLBACK_SECONDS
        + contract.POSTICTAL_GUARD_SECONDS
    )
    assert dc.count_offset_fallbacks(on, off) == 1
    # and it actually removes minutes
    assert not dc.minute_guard_free(np.array([200_000.0 + 3660.0]), guards)[0]
    assert dc.minute_guard_free(np.array([200_000.0 + 3781.0]), guards)[0]


def test_merge_seizure_sources_dedups_within_one_second_only():
    inv = (np.array([500.0]), np.array([560.0]))
    near = (np.array([500.9]), np.array([560.0]))
    far = (np.array([502.0]), np.array([560.0]))
    assert dc.merge_seizure_sources(inv, near)[2] == 0
    assert dc.merge_seizure_sources(inv, far)[2] == 1


def test_merge_seizure_sources_handles_an_empty_inventory():
    on, off, n_new = dc.merge_seizure_sources(
        (np.zeros(0), np.zeros(0)), (np.array([42.0]), np.array([np.nan]))
    )
    assert n_new == 1 and on.tolist() == [42.0]
    assert dc.count_offset_fallbacks(on, off) == 1


def test_zero_duration_inventory_offset_uses_the_fallback_not_a_zero_length_seizure():
    guards = dc.guard_intervals_from_seizures(np.array([1000.0]), np.array([1000.0]))
    assert guards[0, 1] == pytest.approx(
        1000.0 + contract.SEIZURE_OFFSET_FALLBACK_SECONDS + contract.POSTICTAL_GUARD_SECONDS
    )


# ---------------------------------------------------------------------------
# 8. artifact refinement hook
# ---------------------------------------------------------------------------


def test_refine_minute_index_with_artifacts_flips_usability_and_flags():
    n = 40
    n_contacts = 10
    df = _minute_frame(n)
    assert bool(df.loc[20, "minute_usable"]) is True
    assert bool(df.loc[20, "h5_ok"]) is True

    mask = np.ones((n, n_contacts), bool)
    # minute 20 keeps 6/10 = 0.60 < 0.70 -> unusable
    mask[20, :4] = False
    # minute 21 keeps 7/10 = 0.70 == threshold -> still usable
    mask[21, :3] = False

    out = dc.refine_minute_index_with_artifacts(df, mask)
    assert out is not df                     # non-destructive
    assert bool(df.loc[20, "minute_usable"]) is True   # original untouched
    assert out.loc[20, "n_valid_contacts"] == 6
    assert out.loc[21, "n_valid_contacts"] == 7
    assert bool(out.loc[20, "minute_usable"]) is False
    assert bool(out.loc[21, "minute_usable"]) is True
    # ctx_ok / h*_ok must be recomputed, not carried over
    assert bool(out.loc[20, "ctx_ok"]) is False
    assert bool(out.loc[25, "ctx_ok"]) is False   # context 16..25 contains 20
    assert bool(out.loc[30, "ctx_ok"]) is True
    assert bool(out.loc[15, "h5_ok"]) is False    # target 20 unusable
    assert bool(out.loc[19, "h1_ok"]) is False


def test_refine_rejects_a_mask_with_the_wrong_number_of_minutes():
    df = _minute_frame(10)
    with pytest.raises(ValueError, match="artifact_mask"):
        dc.refine_minute_index_with_artifacts(df, np.ones((9, 4), bool))


def test_refine_keeps_guard_and_coverage_rules():
    n = 30
    covered = np.ones(n, bool)
    covered[5] = False
    guard = np.ones(n, bool)
    guard[6] = False
    df = _minute_frame(n, covered=covered, guard_free=guard)
    out = dc.refine_minute_index_with_artifacts(df, np.ones((n, 8), bool))
    assert bool(out.loc[5, "minute_usable"]) is False
    assert bool(out.loc[6, "minute_usable"]) is False
    assert out.loc[5, "n_valid_contacts"] == 8


# ---------------------------------------------------------------------------
# schema conformance
# ---------------------------------------------------------------------------


def test_window_index_columns_match_the_frozen_schema():
    df = _minute_frame(20)
    assert list(df.columns) == list(contract.WINDOW_INDEX_COLUMNS)


def test_contact_metadata_columns_match_the_frozen_schema():
    df = dc.assemble_contact_rows(
        "synthetic", "yuquan", _pairs(), np.zeros((3, 3)), np.ones(3, bool),
        "fs_native_ras_mm[mm]", np.ones(3, bool),
    )
    assert list(df.columns) == list(contract.CONTACT_METADATA_COLUMNS)
    assert "coord_mode" in contract.CONTACT_METADATA_COLUMNS


def test_compute_eligibility_flags_emits_one_column_per_horizon():
    flags = dc.compute_eligibility_flags(
        session_id=np.zeros(5, np.int64),
        split=np.array(["train"] * 5, dtype=object),
        minute_usable=np.ones(5, bool),
        guard_free=np.ones(5, bool),
    )
    expected = {"ctx_ok"} | {f"h{h}_ok" for h in contract.HORIZONS_MIN}
    assert set(flags) == expected


# ---------------------------------------------------------------------------
# integration (touches /mnt and the frozen inventories)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_build_subject_blocks_for_one_epilepsiae_subject():
    blocks = dc.build_subject_blocks("epilepsiae_253")
    assert list(blocks.columns) == list(contract.DATASET_MANIFEST_COLUMNS)
    assert len(blocks) > 0
    assert (blocks["block_end_epoch"] >= blocks["block_start_epoch"]).all()
    assert blocks["block_start_epoch"].is_monotonic_increasing
    dev_end = contract.dev_end_epoch("epilepsiae_253")
    dev = blocks[blocks["split"] != "sealed"]
    assert (dev["block_start_epoch"] < dev_end).all()


@pytest.mark.integration
def test_coordinate_less_yuquan_subject_keeps_all_its_contacts():
    df = dc.build_contact_metadata("yuquan_gaolan")
    assert len(df) > 20
    assert df["contact_valid"].all()
    assert not df["coord_valid"].any()
    assert (df["coord_mode"] == contract.COORD_MODE_TOPOLOGY_ONLY).all()


@pytest.mark.integration
def test_yuquan_annotation_scan_recovers_the_dropped_onsets():
    for subject, native in (("yuquan_zhangbichen", "zhangbichen"), ("yuquan_chenziyang", "chenziyang")):
        onsets, _, meta = dc.load_seizure_table(subject)
        assert meta["seizure_guard_source"] == "annotation_scan_only"
        assert meta["n_seizures_from_supplement"] == 1
        assert onsets.size == 1
        assert onsets[0] < contract.dev_end_epoch(subject)


@pytest.mark.integration
def test_build_minute_index_for_one_yuquan_subject_respects_the_seal():
    subject = "yuquan_chengshuai"
    minutes = dc.build_minute_index(subject)
    assert list(minutes.columns) == list(contract.WINDOW_INDEX_COLUMNS)
    contract.assert_not_sealed(subject, minutes["minute_start_epoch"].to_numpy())
    assert minutes["minute_index"].is_monotonic_increasing
    assert (minutes["n_valid_contacts"] == -1).all()


# ---------------------------------------------------------------------------
# Regression: a minute straddling a block boundary is NOT covered
# ---------------------------------------------------------------------------


def test_minute_straddling_a_block_gap_is_not_covered():
    """Found during pilot integration on epilepsiae_620, minute 6967.

    Epilepsiae's hourly blocks abut with ~1 s gaps. A minute sitting on such a
    boundary sums to 59 of 60 recorded seconds and would pass a summed-coverage
    test, yet it contains a recorder discontinuity: splicing across the gap puts
    a step into the 60 s Welch estimate and splatters broadband power into every
    band of that minute's target. The rule therefore uses the longest UNBROKEN
    stretch, so such a minute is dropped (~1 minute per block boundary).
    """
    import numpy as np
    from src.topic5_raw_seeg_state import data_contract as dc

    m = np.array([1158987515.0])                     # 30 s in block A, 29 s in block B
    bs = np.array([1158983945.0, 1158987546.0])
    be = np.array([1158987545.0, 1158991146.0])

    assert dc.minute_covered_seconds(m, bs, be)[0] == pytest.approx(59.0)
    assert dc.minute_max_contiguous_seconds(m, bs, be)[0] == pytest.approx(30.0)
    assert not bool(dc.covered_from_seconds(dc.minute_max_contiguous_seconds(m, bs, be))[0])

    # a minute wholly inside one block is untouched by the tightening
    inside = np.array([1158985000.0])
    assert dc.minute_max_contiguous_seconds(inside, bs, be)[0] == pytest.approx(60.0)
    assert bool(dc.covered_from_seconds(dc.minute_max_contiguous_seconds(inside, bs, be))[0])

    # and the 57 / 58 s single-block boundary still behaves
    one = np.array([0.0])
    assert not bool(dc.covered_from_seconds(
        dc.minute_max_contiguous_seconds(one, np.array([0.0]), np.array([57.0])))[0])
    assert bool(dc.covered_from_seconds(
        dc.minute_max_contiguous_seconds(one, np.array([0.0]), np.array([58.0])))[0])
