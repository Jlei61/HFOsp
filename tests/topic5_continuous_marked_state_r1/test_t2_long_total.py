from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm
import torch

from src.topic5_continuous_marked_state_r1.t2_s1 import (
    fit_participation_innovation,
)
from src.topic5_continuous_marked_state_r1.t2_long_total import (
    build_long_window_design,
    count_windows_crossing_segment,
    decoder_readout,
    delayed_control_overlap,
    delayed_union_start_index,
    effective_memory_audit,
    endpoint_support_audit,
    estimability_guard,
    fit_decoder_space_edge,
    intercept_operator,
    nonoverlapping_window_audit,
    predict_state,
    state_prediction_metrics,
    target_shift_audit,
)


def _linear(value):
    value = np.asarray(value, dtype=np.float32)
    layer = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(value))
    return layer


def _readout_model(dim=2):
    return SimpleNamespace(
        state_timing=_linear([[1.0, 0.2]]),
        state_size=_linear([[0.3, 0.7], [0.8, -0.1], [-0.2, 0.4]]),
        state_contact=_linear([[0.6, 0.1], [-0.4, 0.9]]),
    )


def test_prefix_operator_equals_explicit_event_rollout() -> None:
    time = np.arange(12, dtype=np.float64) * 60.0
    split = np.asarray([0] * 8 + [1] * 4, dtype=np.int8)
    segment = np.zeros(12, dtype=np.int64)
    state = np.column_stack([np.linspace(0, 1, 12), np.linspace(1, 0, 12)])
    innovation = np.linspace(-1, 1, 12)
    matrix = np.asarray([[-0.03, 0.02], [-0.02, -0.01]])
    design = build_long_window_design(
        time, split, segment, state, innovation, matrix, np.zeros(2),
        window_kind="event_count_10000", scale_events=3, delay_events=1,
        coverage_start=np.asarray([-1.0]),
    )
    row = 0
    start, end = int(design.start_index[row]), int(design.end_index[row])
    theta = np.asarray([0.4, -0.3, 0.2, 0.1])
    value = np.zeros(2)
    scale = np.sqrt(3.0)
    for event in range(start, end):
        jump = theta[:2] + theta[2:] * innovation[event]
        value = value + jump / scale
        value = expm(matrix * ((time[event + 1] - time[event]) / 60.0)) @ value
    got = design.real_operator[row] @ theta
    assert np.allclose(got, value, atol=1e-10)


def test_boxcar_operator_uses_every_event_without_generator_decay() -> None:
    time = np.arange(12, dtype=np.float64) * 3600.0
    split = np.asarray([0] * 8 + [1] * 4, dtype=np.int8)
    segment = np.zeros(12, dtype=np.int64)
    state = np.zeros((12, 2), dtype=np.float64)
    innovation = np.arange(12, dtype=np.float64)
    design = build_long_window_design(
        time, split, segment, state, innovation, -np.eye(2), np.zeros(2),
        window_kind="event_count_3000", scale_events=3, delay_events=1,
        coverage_start=np.asarray([-1.0]), exposure_memory="boxcar",
    )
    start, end = int(design.start_index[0]), int(design.end_index[0])
    scale = np.sqrt(3.0)
    assert (start, end) == (1, 4)
    assert np.allclose(design.real_operator[0, :, :2], np.eye(2) * 3 / scale)
    assert np.allclose(
        design.real_operator[0, :, 2:],
        np.eye(2) * innovation[start:end].sum() / scale,
    )
    assert np.allclose(
        design.delayed_operator[0, :, 2:],
        np.eye(2) * innovation[start - 1:end - 1].sum() / scale,
    )


def test_delayed_union_support_starts_before_nominal_real_window() -> None:
    segment = np.asarray([0] * 8 + [1] * 7, dtype=np.int64)
    start = np.asarray([3, 6, 11, 14], dtype=np.int64)
    got = delayed_union_start_index(start, segment, delay_events=2)
    assert got.tolist() == [1, 4, 9, 12]

    with pytest.raises(ValueError, match="lacks the requested"):
        delayed_union_start_index(
            np.asarray([1], dtype=np.int64), segment, delay_events=2
        )


def test_boxcar_operator_accepts_multidimensional_repertoire_innovation() -> None:
    time = np.arange(12, dtype=np.float64) * 60.0
    split = np.asarray([0] * 8 + [1] * 4, dtype=np.int8)
    segment = np.zeros(12, dtype=np.int64)
    state = np.zeros((12, 2), dtype=np.float64)
    innovation = np.column_stack([
        np.arange(12, dtype=np.float64),
        -np.arange(12, dtype=np.float64),
    ])
    design = build_long_window_design(
        time, split, segment, state, innovation, -np.eye(2), np.zeros(2),
        window_kind="event_count_3000", scale_events=3, delay_events=1,
        coverage_start=np.asarray([-1.0]), exposure_memory="boxcar",
    )
    start, end = int(design.start_index[0]), int(design.end_index[0])
    scale = np.sqrt(3.0)
    assert design.real_operator.shape[2] == 6
    assert np.allclose(
        design.real_operator[0, :, 2:4],
        np.eye(2) * innovation[start:end, 0].sum() / scale,
    )
    assert np.allclose(
        design.real_operator[0, :, 4:6],
        np.eye(2) * innovation[start:end, 1].sum() / scale,
    )


def test_participation_innovation_is_train_only_composition_not_load() -> None:
    rng = np.random.default_rng(19)
    n, contacts = 120, 6
    state = rng.normal(size=(n, 2))
    history = rng.normal(size=(n, 12))
    participation = np.zeros((n, contacts), dtype=bool)
    for row in range(n):
        take = 1 + row % 4
        choice = rng.choice(contacts, size=take, replace=False)
        participation[row, choice] = True
    train = np.arange(n) < 90
    score, audit = fit_participation_innovation(
        state, history, participation, train, n_components=2,
    )
    assert score.shape == (n, 2)
    assert np.allclose(score[train].std(0), 1.0, atol=1e-6)
    assert audit["composition_removes_total_load"] is True
    assert audit["uses_validation_outcome"] is False
    # SVD sign is fixed by the largest-magnitude loading for reproducibility.
    components = np.asarray(audit["components"])
    anchor = np.argmax(np.abs(components), axis=1)
    assert np.all(components[np.arange(len(components)), anchor] >= 0.0)


def test_windows_never_cross_coverage_segment() -> None:
    time = np.asarray([0, 1, 2, 3, 100, 101, 102, 103], dtype=np.float64) * 60
    split = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    segment = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    state = np.zeros((8, 2))
    design = build_long_window_design(
        time, split, segment, state, np.arange(8), -np.eye(2) * 0.01,
        np.zeros(2), window_kind="event_count_10000", scale_events=2,
        delay_events=1, coverage_start=np.asarray([0.0, 100 * 60.0]),
    )
    assert np.array_equal(segment[design.start_index], segment[design.end_index])
    assert np.all(design.end_index - design.start_index == 2)


def test_nonoverlapping_window_audit_does_not_count_sliding_rows_as_replicates() -> None:
    time = np.arange(10, dtype=np.float64)
    start = np.asarray([0, 1, 2, 5], dtype=np.int64)
    end = np.asarray([5, 6, 7, 9], dtype=np.int64)
    split = np.asarray([1, 1, 1, 1], dtype=np.int8)
    audit = nonoverlapping_window_audit(time, start, end, split)
    assert audit["validation"]["windows"] == 4
    assert audit["validation"]["nonoverlapping_full_windows"] == 2


def test_decoder_space_ridge_recovers_true_edge() -> None:
    rng = np.random.default_rng(8)
    n, dim = 500, 2
    operator = rng.normal(size=(n, dim, 2 * dim))
    theta = np.asarray([0.4, -0.2, 0.3, 0.1])
    delta = np.einsum("ndp,p->nd", operator, theta)
    split = np.asarray([0] * 400 + [1] * 100, dtype=np.int8)
    readout = decoder_readout(_readout_model(), delta, split == 0)
    fitted, audit = fit_decoder_space_edge(operator, delta, split, readout)
    assert audit["development_validation_used_for_selection"] is False
    assert np.allclose(fitted, theta, atol=1e-4)


def test_decoder_space_ridge_recovers_multifeature_repertoire_edge() -> None:
    rng = np.random.default_rng(81)
    n, dim = 600, 2
    # occurrence plus two repertoire-composition components
    operator = rng.normal(size=(n, dim, 3 * dim))
    theta = np.asarray([0.4, -0.2, 0.3, 0.1, -0.25, 0.35])
    delta = np.einsum("ndp,p->nd", operator, theta)
    split = np.asarray([0] * 480 + [1] * 120, dtype=np.int8)
    readout = decoder_readout(_readout_model(), delta, split == 0)
    fitted, audit = fit_decoder_space_edge(operator, delta, split, readout)
    assert audit["development_validation_used_for_selection"] is False
    assert np.allclose(fitted, theta, atol=1e-4)


def test_decoder_metric_prefers_recovered_prediction() -> None:
    rng = np.random.default_rng(2)
    n, dim = 100, 2
    natural = rng.normal(size=(n, dim))
    operator = rng.normal(size=(n, dim, 2 * dim))
    theta = np.asarray([0.2, -0.1, 0.4, 0.3])
    target = natural + np.einsum("ndp,p->nd", operator, theta)
    split = np.asarray([0] * 80 + [1] * 20, dtype=np.int8)
    readout = decoder_readout(_readout_model(), target - natural, split == 0)
    fitted, _ = fit_decoder_space_edge(operator, target - natural, split, readout)
    recovered = natural + np.einsum("ndp,p->nd", operator, fitted)
    rows = np.flatnonzero(split == 1)
    full = state_prediction_metrics(recovered, target, rows, readout)
    null = state_prediction_metrics(natural, target, rows, readout)
    assert full["decoder_total_equal_block_mse"] < null["decoder_total_equal_block_mse"]
    assert readout.rank == dim


def _six_hour_stream(n=400, spacing_seconds=120.0):
    time = np.arange(n, dtype=np.float64) * spacing_seconds
    split = np.where(np.arange(n) < n // 2, 0, 1).astype(np.int8)
    segment = np.zeros(n, dtype=np.int64)
    state = np.column_stack([np.linspace(0, 1, n), np.linspace(1, 0, n)])
    innovation = np.sin(np.arange(n) / 7.0)
    matrix = np.asarray([[-0.03, 0.02], [-0.02, -0.01]])
    return time, split, segment, state, innovation, matrix


def test_physical_six_hour_windows_respect_coverage_and_duration() -> None:
    time, split, segment, state, innovation, matrix = _six_hour_stream()
    design = build_long_window_design(
        time, split, segment, state, innovation, matrix, np.zeros(2),
        window_kind="physical_6h", duration_hours=6.0, delay_events=5,
        coverage_start=np.asarray([time[0] - 1.0]),
    )
    assert design.window_kind == "physical_6h"
    # No window may reach back before recorded coverage opened.
    assert np.all(time[design.end_index] - 6.0 * 3600.0 >= time[0] - 1e-6)
    # The window is the first event at or after endpoint minus six hours, so it
    # is never longer than six hours and never empty.
    assert np.all(design.duration_hours <= 6.0 + 1e-9)
    assert np.all(design.duration_hours > 0)
    for start, end in zip(design.start_index, design.end_index):
        requested = time[end] - 6.0 * 3600.0
        assert time[start] >= requested - 1e-6
        assert start == 0 or time[start - 1] < requested


def test_physical_six_hour_windows_never_cross_a_gap() -> None:
    time, split, segment, state, innovation, matrix = _six_hour_stream()
    segment = np.where(np.arange(len(time)) < 250, 0, 1).astype(np.int64)
    coverage_start = np.asarray([time[0] - 1.0, time[250] - 1.0])
    design = build_long_window_design(
        time, split, segment, state, innovation, matrix, np.zeros(2),
        window_kind="physical_6h", duration_hours=6.0, delay_events=5,
        coverage_start=coverage_start,
    )
    assert count_windows_crossing_segment(
        design.start_index, design.end_index, segment
    ) == 0


def test_count_windows_crossing_segment_detects_a_crossing() -> None:
    assert count_windows_crossing_segment(
        np.asarray([0, 2]), np.asarray([1, 5]),
        np.asarray([0, 0, 0, 0, 0, 1]),
    ) == 1


def test_delayed_arm_shares_the_occurrence_block_exactly() -> None:
    time, split, segment, state, innovation, matrix = _six_hour_stream(n=120)
    design = build_long_window_design(
        time, split, segment, state, innovation, matrix, np.zeros(2),
        window_kind="event_count_10000", scale_events=20, delay_events=5,
        coverage_start=np.asarray([time[0] - 1.0]),
    )
    dim = design.start_state.shape[1]
    assert np.allclose(
        design.real_operator[:, :, :dim], design.delayed_operator[:, :, :dim]
    )
    # ...and differ only in the load columns, so real-minus-delayed is both
    # parameter-matched and intercept-matched.
    assert not np.allclose(
        design.real_operator[:, :, dim:], design.delayed_operator[:, :, dim:]
    )


def test_intercept_arm_absorbs_an_exposure_free_offset() -> None:
    time, split, segment, state, innovation, matrix = _six_hour_stream(n=200)
    design = build_long_window_design(
        time, split, segment, state, innovation, matrix, np.zeros(2),
        window_kind="event_count_10000", scale_events=20, delay_events=5,
        coverage_start=np.asarray([time[0] - 1.0]),
    )
    offset = intercept_operator(design)
    assert np.allclose(offset, offset[0])
    rng = np.random.default_rng(11)
    # Target carries a constant state offset and no exposure information.
    target_delta = np.asarray([0.5, -0.3]) + rng.normal(
        scale=0.01, size=design.natural_state.shape
    )
    target = design.natural_state + target_delta
    readout = decoder_readout(_readout_model(), target_delta, design.split == 0)
    rows = np.flatnonzero(design.split == 1)
    fitted = {}
    for name, operator in (
        ("real", design.real_operator), ("intercept", offset),
    ):
        theta, _ = fit_decoder_space_edge(
            operator, target_delta, design.split, readout
        )
        fitted[name] = state_prediction_metrics(
            predict_state(design, operator, theta), target, rows, readout
        )["decoder_total_equal_block_mse"]
    no_edge = state_prediction_metrics(
        design.natural_state, target, rows, readout
    )["decoder_total_equal_block_mse"]
    # The artefact: the exposure arm crushes raw no-edge on a target that holds
    # no exposure information at all, purely through its saturated occurrence
    # block.  Against the intercept-matched reference essentially all of that
    # apparent gain disappears, which is the whole point of the extra arm.
    artefact = no_edge - fitted["intercept"]
    assert artefact > 1.0
    assert abs(fitted["real"] - fitted["intercept"]) < 0.05 * artefact


def test_readout_with_no_train_variation_is_degenerate() -> None:
    delta = np.zeros((50, 2))
    readout = decoder_readout(_readout_model(), delta, np.ones(50, dtype=bool))
    assert readout.blocks_at_scale_floor
    assert readout.degenerate is True


def test_a_single_floored_block_is_a_caveat_not_a_blocked_run() -> None:
    # One block whose TRAIN target variation vanishes must not force the whole
    # instrument to UNTESTABLE; the other blocks still separate the arms.
    model = SimpleNamespace(
        state_timing=_linear([[0.0, 0.0]]),
        state_size=_linear([[0.3, 0.7], [0.8, -0.1], [-0.2, 0.4]]),
        state_contact=_linear([[0.6, 0.1], [-0.4, 0.9]]),
    )
    rng = np.random.default_rng(5)
    delta = rng.normal(size=(200, 2))
    readout = decoder_readout(model, delta, np.ones(200, dtype=bool))
    assert readout.blocks_at_scale_floor == ("timing",)
    assert readout.degenerate is False


def test_numerically_dead_readout_does_not_pass_the_admissibility_gate() -> None:
    model = SimpleNamespace(
        state_timing=_linear([[1e-10, 2e-10]]),
        state_size=_linear([[1e-10, 1e-10], [1e-10, 2e-10], [3e-10, 1e-10]]),
        state_contact=_linear([[2e-10, 1e-10], [1e-10, 3e-10]]),
    )
    rng = np.random.default_rng(3)
    delta = rng.normal(size=(200, 2))
    readout = decoder_readout(model, delta, np.ones(200, dtype=bool))
    # An absolute rank tolerance on the rescaled matrix used to call this
    # instrument usable; every block's TRAIN variation is on the scale floor.
    assert readout.blocks_at_scale_floor
    assert readout.degenerate is True


def test_effective_memory_reports_the_generator_time_constant() -> None:
    # Six hours of events, but the slowest generator mode decays in one hour.
    time = np.arange(3000, dtype=np.float64) * 7.2
    matrix = -np.eye(2) * (1.0 / 60.0)
    audit = effective_memory_audit(
        time, np.zeros(1, dtype=np.int64), np.asarray([2999]), matrix,
    )
    assert audit["slowest_mode_time_constant_minutes"] == pytest.approx(60.0)
    assert audit["median_nominal_events"] == 2999
    assert audit["median_effective_weighted_events"] < 600
    assert audit["median_hours_holding_ninety_percent_weight"] < 2.5


def test_endpoint_support_reports_the_independent_window_budget() -> None:
    time = np.arange(200, dtype=np.float64) * 60.0
    end = np.arange(100, 200, dtype=np.int64)
    split = np.ones(100, dtype=np.int8)
    matrix = -np.eye(2) * (1.0 / 60.0)
    audit = endpoint_support_audit(time, end, split, matrix)
    assert audit["validation"]["windows"] == 100
    assert audit["validation"]["endpoint_span_hours"] == pytest.approx(99 / 60)
    # 100 overlapping windows inside 1.65 h of one-hour memory are not 100
    # independent measurements.
    assert audit["validation"]["effective_independent_windows"] < 2.0


def test_boxcar_endpoint_support_uses_the_window_not_the_generator() -> None:
    # One-hour generator mode, but an equal-weight boxcar six hours long: two
    # endpoints stay dependent until they are a window apart, not a tau apart.
    time = np.arange(600, dtype=np.float64) * 60.0
    end = np.arange(360, 600, dtype=np.int64)
    start = end - 360
    split = np.ones(len(end), dtype=np.int8)
    matrix = -np.eye(2) * (1.0 / 60.0)
    generator = endpoint_support_audit(time, end, split, matrix)
    boxcar = endpoint_support_audit(
        time, end, split, matrix, exposure_memory="boxcar", start_index=start,
    )
    assert generator["decorrelation_minutes"] == pytest.approx(60.0)
    assert boxcar["decorrelation_minutes"] == pytest.approx(360.0)
    assert (boxcar["validation"]["effective_independent_windows"]
            < generator["validation"]["effective_independent_windows"] / 5.0)


def test_delayed_control_overlap_is_reported() -> None:
    time, split, segment, state, innovation, matrix = _six_hour_stream(n=200)
    design = build_long_window_design(
        time, split, segment, state, innovation, matrix, np.zeros(2),
        window_kind="event_count_10000", scale_events=40, delay_events=10,
        coverage_start=np.asarray([time[0] - 1.0]),
    )
    overlap = delayed_control_overlap(design)
    # A ten-event delay inside a forty-event window leaves three quarters of the
    # exposure shared between the real and delayed arms.
    assert overlap["median_shared_exposure_fraction"] == pytest.approx(0.75)
    assert overlap["delay_events"] == 10


def test_estimability_guard_rejects_a_diverged_arm() -> None:
    reference = {"decoder_total_equal_block_mse": 1.0}
    assert estimability_guard(
        {"decoder_total_equal_block_mse": 0.8}, reference)["estimable"] is True
    assert estimability_guard(
        {"decoder_total_equal_block_mse": 3.9}, reference)["estimable"] is True
    # The archived very-long readouts land 68x and 2375x above the constant they
    # nest; that is extrapolation, not an exposure null.
    assert estimability_guard(
        {"decoder_total_equal_block_mse": 94.9}, reference)["estimable"] is False


def test_target_shift_audit_sees_a_validation_scale_break() -> None:
    rng = np.random.default_rng(17)
    readout = decoder_readout(
        _readout_model(), rng.normal(size=(200, 2)), np.ones(200, dtype=bool)
    )
    split = np.asarray([0] * 100 + [1] * 100, dtype=np.int8)
    stable = rng.normal(size=(200, 2))
    calm = target_shift_audit(stable, split, readout)
    shifted = stable.copy()
    shifted[100:] += 50.0
    broken = target_shift_audit(shifted, split, readout)
    assert calm["validation_minus_train_mean_shift_in_train_sd"] < 1.0
    assert broken["validation_minus_train_mean_shift_in_train_sd"] > 10.0


def test_ridge_selection_is_invariant_to_operator_rescaling() -> None:
    # An absolute `ridge * I` penalty made the grid meaningless as soon as the
    # operator or the window count changed scale, so every archived fit pinned
    # itself at the grid maximum.  The selection must depend on the fit, not on
    # the units the operator happens to be in.
    rng = np.random.default_rng(23)
    n, dim = 400, 2
    operator = rng.normal(size=(n, dim, 2 * dim))
    theta = np.asarray([0.4, -0.2, 0.3, 0.1])
    delta = np.einsum("ndp,p->nd", operator, theta) + rng.normal(scale=0.2, size=(n, dim))
    split = np.asarray([0] * 320 + [1] * 80, dtype=np.int8)
    readout = decoder_readout(_readout_model(), delta, split == 0)
    _, small = fit_decoder_space_edge(operator, delta, split, readout)
    _, large = fit_decoder_space_edge(operator * 1000.0, delta, split, readout)
    assert small["selected_ridge"] == large["selected_ridge"]
    assert small["penalty_scaling"].startswith("ridge x mean diagonal")
