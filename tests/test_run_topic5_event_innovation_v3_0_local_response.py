from __future__ import annotations

import numpy as np

from scripts.run_topic5_event_innovation_v3_0_local_response import (
    ResponseRows,
    chronology_nulls,
    contiguous_group_codes,
    group_balanced_weights,
    innovation_lookup,
    state_matched_donor,
    state_matched_donor_pool,
)
from src.topic5_event_innovation_response_v3_0 import fit_weighted_local_projection


def test_innovation_lookup_rejects_duplicate_events():
    with np.testing.assert_raises(ValueError):
        innovation_lookup(
            np.array([1, 1]), np.zeros((2, 2)), np.ones((2, 2), dtype=bool)
        )


def test_group_balanced_response_weights_equalize_units():
    weight = group_balanced_weights(np.array([0, 0, 0, 1]))
    np.testing.assert_allclose(weight[:3].sum(), 1.0)
    np.testing.assert_allclose(weight[3:].sum(), 1.0)


def test_state_matched_donor_is_a_derangement_and_preserves_values():
    pre = np.arange(20, dtype=float)[:, None]
    innovation = np.column_stack([np.arange(20), -np.arange(20)])
    donor, audit = state_matched_donor(
        pre,
        innovation,
        np.zeros(20, dtype=int),
        np.linspace(0, 1, 20),
        seed=5,
        top_k=2,
    )
    assert all(any(np.array_equal(row, original) for original in innovation) for row in donor)
    assert not np.any(np.all(donor == innovation, axis=1))
    assert audit["eligible_anchor_fraction"] == 1.0
    pools, _ = state_matched_donor_pool(
        pre, np.zeros(20, dtype=int), np.linspace(0, 1, 20), top_k=2
    )
    assert all(index not in pool for index, pool in enumerate(pools))


def test_contiguous_group_codes_split_dropped_event_gaps():
    codes = contiguous_group_codes(
        np.array([0, 0, 0, 1, 1]), np.array([10, 11, 14, 20, 21])
    )
    assert codes[0] == codes[1]
    assert codes[2] != codes[1]
    assert codes[3] == codes[4]


def test_chronology_nulls_break_known_event_update_without_wraparound():
    rng = np.random.default_rng(31)
    pre = rng.normal(size=(120, 1))
    innovation = rng.normal(size=(120, 1))
    future = 0.7 * pre + 0.4 * innovation
    fit = fit_weighted_local_projection(pre, future, innovation, alpha=1e-8)
    rows = ResponseRows(
        event_index=np.arange(120),
        group=np.zeros(120, dtype=int),
        pre_state=pre,
        future_state=future,
        past_state=np.zeros_like(pre),
        innovation_state=innovation,
        nuisance=np.empty((120, 0)),
        observed_future_field=np.zeros((120, 2)),
        future_support=np.ones((120, 2)),
        future_windows=[np.array([index]) for index in range(120)],
    )
    null = chronology_nulls(
        fit,
        rows,
        block_sizes=[1, 20],
        safe_shift_events=[2],
        draws=20,
        seed=9,
    )
    assert null["block"]["1"]["median_true_minus_null_gain"] > 0
    assert null["safe_shift"]["2"]["true_minus_shift_gain"] > 0
    assert null["safe_shift"]["2"]["no_wraparound"] is True
