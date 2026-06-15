"""TDD for src.topic5_axis_alignment (pure A-line join + null-shuffle helpers)."""
import numpy as np
import pytest

from src.topic5_axis_alignment import (
    matched_channels, make_field_record, interictal_and_ictal_values,
    channel_shuffle, within_shaft_shuffle, anchor_matched_shuffle,
    within_shaft_anchor_shuffle, effective_shuffle_n, along_axis_sign)


def test_within_shaft_anchor_shuffle_stays_in_shaft_and_anchor_bin():
    # A-shaft (A1,A2 low-anchor; A3,A4 high-anchor), B-shaft (B1,B2)
    names = ["A1", "A2", "A3", "A4", "B1", "B2"]
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    anchor = np.array([10.0, 11.0, 90.0, 91.0, 50.0, 51.0])
    out = within_shaft_anchor_shuffle(v, names, anchor, np.random.default_rng(3), n_bins=2)
    # within (A-shaft, low bin) = {1,2}; (A-shaft, high bin) = {3,4}; (B-shaft) one bin = {5,6}
    assert sorted(out[:2].tolist()) == [1.0, 2.0]
    assert sorted(out[2:4].tolist()) == [3.0, 4.0]
    assert sorted(out[4:].tolist()) == [5.0, 6.0]


def test_effective_shuffle_n_flags_degenerate_nulls():
    names = ["A1", "A2", "B1"]                 # A-shaft has 2, B-shaft singleton
    anchor = np.array([1.0, 2.0, 3.0])
    assert effective_shuffle_n(names, anchor, "channel") == 3          # all in one group
    assert effective_shuffle_n(names, anchor, "within_shaft") == 2     # only A1,A2 move; B1 singleton
    # joint with n_bins=3 -> every (shaft,bin) cell is a singleton -> 0 effective
    assert effective_shuffle_n(names, anchor, "joint", n_bins=3) == 0


def _axis_record():
    return {"channels": [
        {"name": "A1", "x_norm": 0.0, "y_norm": 0.0, "support": 1.0, "typical_rank": 0.1},
        {"name": "A2", "x_norm": 0.3, "y_norm": 0.0, "support": 1.0, "typical_rank": 0.5},
        {"name": "B1", "x_norm": 0.6, "y_norm": 0.2, "support": 1.0, "typical_rank": 0.9},
        {"name": "B2", "x_norm": 0.9, "y_norm": 0.2, "support": 1.0, "typical_rank": 0.7},
    ]}


def test_matched_channels_drops_unmatched_keeps_order():
    rec = _axis_record()
    m = matched_channels(rec, {"A1": 1.0, "B1": 2.0, "ZZ": 9.0})  # ZZ not in record
    assert [c["name"] for c in m] == ["A1", "B1"]               # A2/B2 dropped, order kept


def test_make_field_record_swaps_only_the_scalar():
    rec = _axis_record()
    m = matched_channels(rec, {c["name"]: 0 for c in rec["channels"]})
    fr = make_field_record(m, [10.0, 20.0, 30.0, 40.0])
    assert [c["typical_rank"] for c in fr["channels"]] == [10.0, 20.0, 30.0, 40.0]
    # x_norm/y_norm/support preserved from the axis record
    assert fr["channels"][2]["x_norm"] == 0.6 and fr["channels"][2]["support"] == 1.0


def test_make_field_record_length_mismatch_raises():
    with pytest.raises(ValueError):
        make_field_record([{"name": "A1"}], [1.0, 2.0])


def test_interictal_and_ictal_values_aligned():
    rec = _axis_record()
    vbn = {"A1": 5.0, "A2": 6.0, "B1": 7.0, "B2": 8.0}
    m = matched_channels(rec, vbn)
    inter, ict = interictal_and_ictal_values(m, vbn)
    assert np.allclose(inter, [0.1, 0.5, 0.9, 0.7])   # the axis record's own ranks
    assert np.allclose(ict, [5.0, 6.0, 7.0, 8.0])     # the joined activation values


def test_channel_shuffle_is_a_permutation():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = channel_shuffle(v, np.random.default_rng(0))
    assert sorted(out.tolist()) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert not np.array_equal(out, v)                 # seed 0 actually permutes


def test_within_shaft_shuffle_preserves_per_shaft_multiset():
    names = ["A1", "A2", "A3", "B1", "B2"]
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = within_shaft_shuffle(v, names, np.random.default_rng(1))
    assert sorted(out[:3].tolist()) == [1.0, 2.0, 3.0]   # A-shaft values stay on A
    assert sorted(out[3:].tolist()) == [4.0, 5.0]        # B-shaft values stay on B


def test_anchor_matched_shuffle_permutes_within_anchor_bins():
    # anchor splits into a low bin {1,2,3} and a high bin {4,5,6} (n_bins=2)
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    anchor = np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
    out = anchor_matched_shuffle(v, anchor, np.random.default_rng(2), n_bins=2)
    assert sorted(out[:3].tolist()) == [1.0, 2.0, 3.0]   # low-anchor values stay low-anchor
    assert sorted(out[3:].tolist()) == [4.0, 5.0, 6.0]   # high-anchor values stay high-anchor


def test_anchor_matched_shuffle_length_mismatch_raises():
    with pytest.raises(ValueError):
        anchor_matched_shuffle(np.array([1.0, 2.0]), np.array([1.0]), np.random.default_rng(0))


def test_along_axis_sign_source_end_hotter_is_negative():
    # low rank (source/early) carries high activation -> forward -> sign -1
    rank = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    activation = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    out = along_axis_sign(rank, activation)
    assert out["sign"] == -1
    assert out["signed_corr"] < 0
    assert out["n"] == 5


def test_along_axis_sign_sink_end_hotter_is_positive():
    # high rank (sink/late) carries high activation -> reverse -> sign +1
    rank = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    activation = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out = along_axis_sign(rank, activation)
    assert out["sign"] == 1
    assert out["signed_corr"] > 0


def test_along_axis_sign_too_few_finite_channels_returns_zero_no_crash():
    # only 2 finite-pair channels -> below the 3-channel floor -> sign 0, nan corr, no crash
    rank = np.array([0.0, 1.0, np.nan, np.nan, 4.0])
    activation = np.array([4.0, 3.0, 2.0, 1.0, np.nan])  # finite-pair only at idx 0,1
    out = along_axis_sign(rank, activation)
    assert out["sign"] == 0
    assert np.isnan(out["signed_corr"])
    assert out["n"] == 2 and out["n"] < 3


def test_along_axis_sign_is_mirror_irrelevant_1d():
    # along_axis_sign is a purely 1D rank-vs-activation quantity: no y/lateral coordinate
    # enters, so any spatial mirror flip is irrelevant by construction. The result depends
    # ONLY on the 1D (rank, activation) pairing, so it is stable for a fixed 1D input.
    rank = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    activation = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    out_a = along_axis_sign(rank, activation)
    out_b = along_axis_sign(rank, activation)
    assert out_a == out_b
    assert out_a["sign"] == -1


def test_seizure_parity_subsets_split_by_position_not_value():
    from src.topic5_axis_alignment import seizure_parity_subsets
    even, odd = seizure_parity_subsets([10, 21, 32, 43, 54])
    # positions 0,2,4 -> values 10,32,54 ; positions 1,3 -> 21,43
    assert even == {10, 32, 54}
    assert odd == {21, 43}
    # disjoint and exhaustive
    assert even & odd == set()
    assert even | odd == {10, 21, 32, 43, 54}


def test_seizure_parity_subsets_single_seizure_leaves_one_half_empty():
    from src.topic5_axis_alignment import seizure_parity_subsets
    even, odd = seizure_parity_subsets([7])
    assert even == {7}
    assert odd == set()
