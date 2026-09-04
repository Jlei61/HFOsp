import numpy as np

from src.topic5_group_event_state.v034_h3 import (
    CoveragePiece,
    audit_replacement_event_overlap,
    audit_event_count_design,
    audit_optimizer_trace,
    audit_physical_window_design,
    build_feedback_arm_contracts,
    event_window_overlap_fraction,
    fit_scale_stable_ridge,
    interval_overlap_fraction,
    optimizer_scale_equivalent,
    rolling_prefix_slow_level,
    selection_period_mean_oracle,
    validate_arm_contracts,
)
from src.topic5_group_event_state.v034_h3.synthetic import run_synthetic_canary


def test_event_count_support_never_crosses_real_coverage_piece():
    times = np.r_[np.arange(0.0, 30.0), np.arange(100.0, 130.0)]
    segment = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
    pieces = [
        CoveragePiece(0, "state_train", 0.0, 30.0),
        CoveragePiece(1, "inner_val", 100.0, 130.0),
    ]
    got = audit_event_count_design(
        times, segment, pieces, n_events=5, future_seconds=2.0,
        min_blocks_by_phase={"state_train": 3, "inner_val": 3},
    )
    assert got["state_train"].core_eligible
    assert got["inner_val"].core_eligible
    for support in got.values():
        assert support.n_complete_candidates == 24
        for block in support.nonoverlap_blocks:
            piece = pieces[block.segment_id]
            assert piece.start <= block.exposure_start < block.boundary < block.future_stop <= piece.stop


def test_physical_support_counts_combined_exposure_plus_future_not_anchors():
    pieces = [CoveragePiece(0, "state_train", 0.0, 30.0)]
    got = audit_physical_window_design(
        pieces, exposure_seconds=5.0, future_seconds=5.0, anchor_step_seconds=1.0,
        min_blocks_by_phase={"state_train": 3},
    )["state_train"]
    assert got.n_complete_candidates == 21
    assert got.n_nonoverlap_blocks == 3
    assert got.core_eligible


def test_exposure_may_use_causal_prefix_but_future_stays_in_evaluation_phase():
    times = np.arange(0.0, 30.0)
    segment = np.zeros(30, dtype=int)
    piece = CoveragePiece(
        0, "inner_val", 10.0, 30.0, coverage_start=0.0, coverage_stop=30.0,
    )
    event = audit_event_count_design(
        times, segment, [piece], n_events=5, future_seconds=2.0,
        min_blocks_by_phase={"inner_val": 1},
    )["inner_val"]
    physical = audit_physical_window_design(
        [piece], exposure_seconds=5.0, future_seconds=2.0, anchor_step_seconds=1.0,
        min_blocks_by_phase={"inner_val": 1},
    )["inner_val"]
    assert event.n_complete_candidates == 19
    assert physical.n_complete_candidates == 19
    assert min(block.exposure_start for block in event.nonoverlap_blocks) < piece.start
    assert min(block.exposure_start for block in physical.nonoverlap_blocks) < piece.start
    assert all(block.future_stop <= piece.stop for block in event.nonoverlap_blocks)


def test_120_minute_future_is_exploratory_even_when_supported():
    pieces = [CoveragePiece(0, "state_train", 0.0, 100_000.0)]
    got = audit_physical_window_design(
        pieces, exposure_seconds=300.0, future_seconds=7200.0,
        min_blocks_by_phase={"state_train": 1},
    )["state_train"]
    assert got.estimable
    assert not got.core_eligible
    assert got.tier == "exploratory_long_horizon"


def test_rolling_prefix_excludes_same_time_event_and_resets_at_segment():
    values, audit = rolling_prefix_slow_level(
        event_times=np.array([1.0, 2.0, 101.0]),
        event_segments=np.array([0, 0, 1]),
        anchor_times=np.array([1.0, 2.5, 101.0, 102.0]),
        anchor_segments=np.array([0, 0, 1, 1]),
        half_life_seconds=10.0,
    )
    assert values[0, 0] == 0.0
    assert values[1, 0] > 0.0
    assert values[2, 0] == 0.0
    assert values[3, 0] > 0.0
    assert audit["causal_at_anchor"]


def test_period_mean_is_never_a_primary_causal_control():
    repeated, definition = selection_period_mean_oracle(np.array([[0.0], [2.0]]))
    assert np.all(repeated == 1.0)
    assert not definition.causal_at_anchor
    assert not definition.allowed_primary_comparator
    assert definition.uses_future_inputs_for_earlier_anchors


def test_delayed_and_interval_overlap_are_explicit():
    assert event_window_overlap_fraction(0, 10_000, 1_000, 11_000) == 0.9
    assert event_window_overlap_fraction(0, 100, 1000, 1100) == 0.0
    assert interval_overlap_fraction(0.0, 100.0, 50.0, 150.0) == 0.5
    audit = audit_replacement_event_overlap(
        np.array([0, 100]), np.array([100, 200]), np.array([0, 0]),
        np.array([50, 0]), np.array([150, 100]), np.array([0, 1]),
    )
    assert not audit["passed"]
    assert audit["per_pair_overlap_fraction"] == [0.5, 0.0]


def test_m0_m1_m2_share_intercept_rank_and_parameter_count():
    arms = build_feedback_arm_contracts(state_dim=8, source_rank=3)
    validate_arm_contracts(arms)
    assert {a.trainable_parameters for a in arms} == {32}
    assert {a.fitted_state_intercept for a in arms} == {True}
    assert sum(a.source_is_event_feedback for a in arms) == 2


def test_scaled_ridge_is_invariant_to_per_column_units():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 2))
    y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.01, size=200)
    p1, f1 = fit_scale_stable_ridge(x[:140], y[:140], x[140:], y[140:])
    scaled = x * np.array([1e8, 1e-7])
    p2, f2 = fit_scale_stable_ridge(scaled[:140], y[:140], scaled[140:], y[140:])
    assert np.allclose(p1, p2, atol=1e-9, rtol=1e-9)
    assert f1.selected_lambda == f2.selected_lambda


def test_divergent_fit_is_not_counted_as_negative_scientific_result():
    x_train = np.linspace(-1.0, 1.0, 100)[:, None]
    y_train = x_train[:, 0]
    x_val = np.linspace(-1.0, 1.0, 100)[:, None]
    y_val = -0.2 * x_val[:, 0]
    _pred, fit = fit_scale_stable_ridge(
        x_train, y_train, x_val, y_val, lambdas=(0.0,), divergence_factor=4.0,
    )
    assert not fit.estimable
    assert any(reason.startswith("validation_mse_over_intercept") for reason in fit.divergence_reasons)


def test_cpu_synthetic_canary_recovers_feedback_and_nulls():
    result = run_synthetic_canary()
    assert result["passed"]
    assert result["n_passed"] == result["n_total"] == 9


def test_optimizer_no_learning_and_divergence_are_not_scientific_zeroes():
    audit = audit_optimizer_trace(
        steps=np.array([0, 100, 200]),
        inner_validation_loss=np.array([1.0, 10.0, 11.0]),
        intercept_inner_validation_loss=1.0,
        update_norm=np.zeros(3),
        parameter_norm=np.ones(3),
        selected_step=0,
        budget_steps=200,
    )
    assert audit.no_learning
    assert not audit.estimable

    divergent = audit_optimizer_trace(
        steps=np.array([0, 100, 200]),
        inner_validation_loss=np.array([1.0, 5.0, 6.0]),
        intercept_inner_validation_loss=1.0,
        update_norm=np.array([0.0, 0.1, 0.1]),
        parameter_norm=np.ones(3),
        selected_step=100,
        budget_steps=200,
    )
    assert divergent.divergent
    assert not divergent.estimable


def test_optimizer_unit_rescaling_canary_has_explicit_verdict():
    same = optimizer_scale_equivalent(
        np.array([2.0, 1.0]), np.array([2.0, 1.0]),
        np.array([0.1, 0.05]), np.array([0.1, 0.05]),
    )
    changed = optimizer_scale_equivalent(
        np.array([2.0, 1.0]), np.array([2.0, 1.5]),
        np.array([0.1, 0.05]), np.array([0.1, 0.05]),
    )
    assert same["passed"]
    assert not changed["passed"]
