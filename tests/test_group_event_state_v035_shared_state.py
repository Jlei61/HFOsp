from pathlib import Path

import numpy as np

from src.topic5_group_event_state.v035.contracts import RateTrainConfig
from src.topic5_group_event_state.v035.frozen_shared_evaluator import (
    _negative_binomial_nll, _non_overlapping_window_count, _poisson_nll,
    _ridge_fit, _shift_donors,
)
from src.topic5_group_event_state.v035.grammar_targets import (
    GrammarDictionary, aggregate_grammar_blocks,
)
from src.topic5_group_event_state.v035.stepwise_train import (
    _combine_pairs, _take_anchor_phase,
)
from src.topic5_group_event_state.v035.full_mark_state import (
    _assert_strict_future_target_indices,
)
from src.topic5_group_event_state.v033_training_lab.sg_o2 import GrammarPairs


def test_shared_observed_support_requires_multiple_horizons_and_shared_split():
    cfg = RateTrainConfig(
        horizons_seconds=(7200.0, 21600.0, 28800.0),
        window_contract="observed_support",
        split_contract="shared_multi_horizon_observed_time",
    )
    assert cfg.validate().horizons_seconds[-1] == 28800.0


def test_grammar_block_decomposes_occupancy_coupling_and_mixture():
    part = np.asarray([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=bool)
    tied = np.asarray([[0, -1, 1], [-1, 0, 1], [0, 1, -1]], dtype=np.int16)
    dictionary = GrammarDictionary(
        community_of_contact=np.asarray([0, 1, 1]),
        repertoire_centres=np.zeros((2, 2), dtype=np.float32),
        event_repertoire_embedding=np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
        event_repertoire_label=np.asarray([0, 1, 0]),
        n_communities=2, n_repertoires=2, fit_event_count=3, provenance={"fit_rows_only": True},
    )
    targets = aggregate_grammar_blocks(
        grid_time=np.asarray([0.0]), horizons_seconds=(30.0,),
        future_valid=np.ones((1, 1), dtype=bool), event_time=np.asarray([1.0, 2.0, 3.0]),
        participation=part, tied_group_id=tied, dictionary=dictionary,
    )
    assert targets.community_valid[0, 0]
    assert targets.coupling_valid[0, 0]
    assert targets.repertoire_valid[0, 0]
    np.testing.assert_allclose(targets.community_occupancy[0, 0].sum(), 1.0)
    np.testing.assert_allclose(targets.cross_community_coupling[0, 0].sum(), 1.0)
    np.testing.assert_allclose(targets.repertoire_mixture[0, 0].sum(), 1.0)


def test_block_shift_is_distant_and_keeps_real_states():
    time = np.arange(20, dtype=float) * 3600.0
    rows = np.arange(20, dtype=np.int64)
    target, donor = _shift_donors(time, rows, 6 * 3600.0)
    assert target.size > 0
    assert np.all(np.abs(time[target] - time[donor]) >= 6 * 3600.0)
    assert set(donor).issubset(set(rows))


def test_long_window_support_is_not_grid_anchor_count():
    times = np.asarray([0.0, 60.0, 120.0, 3600.0, 3660.0])
    assert _non_overlapping_window_count(times, 1800.0) == 2


def test_frozen_ridge_is_invariant_to_duplicate_anchor_rows():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(23, 5))
    y = rng.normal(size=(23, 2))
    beta = _ridge_fit(x, y, 0.1)
    duplicated = _ridge_fit(np.repeat(x, 7, axis=0), np.repeat(y, 7, axis=0), 0.1)
    np.testing.assert_allclose(beta, duplicated, rtol=1e-10, atol=1e-10)


def test_negative_binomial_count_score_has_poisson_limit():
    log_rate = np.log(np.asarray([12.0, 20.0, 5.0]))
    count = np.asarray([10.0, 23.0, 4.0])
    exposure = np.full(3, 3600.0)
    got = _negative_binomial_nll(log_rate, count, exposure, 1e8)
    want = _poisson_nll(log_rate, count, exposure)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


def _pairs(anchors, events):
    anchors = np.asarray(anchors, dtype=np.int64)
    pair_anchor = np.arange(len(anchors), dtype=np.int64)
    return GrammarPairs(
        anchor_rows=anchors, pair_anchor=pair_anchor,
        pair_event=np.asarray(events, dtype=np.int64),
        pair_weight=np.full(len(anchors), 1.0 / len(anchors)),
    ).validate()


def test_shared_decoder_adapter_uses_rate_phases_not_legacy_split():
    combined = _combine_pairs((_pairs([0, 1], [10, 11]), _pairs([2, 3], [12, 13])))
    phases = np.asarray(["FIT", "FIT", "INNER", "SELECTION"])
    fit = _take_anchor_phase(combined, phases, "FIT")
    inner = _take_anchor_phase(combined, phases, "INNER")
    selection = _take_anchor_phase(combined, phases, "SELECTION")
    np.testing.assert_array_equal(fit.anchor_rows, [0, 1])
    np.testing.assert_array_equal(inner.anchor_rows, [2])
    np.testing.assert_array_equal(selection.anchor_rows, [3])


def test_same_prefix_state_anchor_is_strictly_before_target_event():
    time = np.asarray([1.0, 2.0, 3.0])
    _assert_strict_future_target_indices(np.asarray([0, 1]), np.asarray([1, 2]), time)
    with np.testing.assert_raises_regex(ValueError, "strictly after"):
        _assert_strict_future_target_indices(np.asarray([1]), np.asarray([1]), time)
