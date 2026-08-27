import numpy as np
import pytest

from src.topic4_node_accessibility import (
    FieldGatedZeroSumNodeRecovery,
    build_spatial_shift_permutation,
)


def _controller(mode="zero_sum", **kwargs):
    defaults = {
        "support_g": np.array([1.0, 0.75, 0.25, 0.0]),
        "dt_ms": 0.1,
        "tau_ms": 250.0,
        "a_ref_mV": 0.2,
        "r_ref_hz": 50.0,
        "trace_dt_ms": 0.1,
        "mode": mode,
    }
    defaults.update(kwargs)
    return FieldGatedZeroSumNodeRecovery(**defaults)


def test_q_is_derived_from_reference_amplitude_rate_and_tau():
    controller = _controller()
    expected = (
        controller.a_ref_mV
        * (1.0 - np.exp(-controller.dt_ms / controller.tau_ms))
        / (controller.r_ref_hz * controller.dt_ms / 1000.0)
    )
    assert controller.q_mV_per_spike == pytest.approx(expected)
    assert controller.a_max_mV == pytest.approx(2.0 * controller.a_ref_mV)


def test_zero_sum_threshold_is_conserved_and_does_not_alias_or_mutate_base():
    controller = _controller()
    controller.state_mV[:] = [0.4, 0.2, 0.0, 0.4]
    base = np.array([18.0, 18.5, 19.0, 19.5, 20.0])
    frozen = base.copy()

    effective = controller.threshold(base)
    delta = effective[: controller.n_e] - base[: controller.n_e]

    assert np.array_equal(base, frozen)
    assert not np.shares_memory(effective, base)
    assert np.sum(delta) == pytest.approx(0.0, abs=1e-14)
    assert effective[-1] == base[-1]
    assert controller.diagnostics()["delta_sum_mV"] == pytest.approx(0.0, abs=1e-14)


def test_trace_is_bounded_and_step_only_affects_next_threshold():
    controller = _controller(
        support_g=np.array([1.0, 1.0]),
        a_ref_mV=0.05,
    )
    base = np.array([18.0, 18.0])

    before = controller.threshold(base)
    controller.step(np.array([True, False]), 0.1)
    after = controller.threshold(base)

    assert np.array_equal(before, base)
    assert after[0] > base[0]
    assert after[1] < base[1]
    for _ in range(1000):
        controller.step(np.array([True, False]), 0.1)
    assert controller.state_mV[0] <= controller.a_max_mV
    assert controller.state_mV[0] == pytest.approx(controller.a_max_mV)
    assert controller.state_mV[1] == 0.0


def test_raise_only_never_lowers_threshold_and_is_not_zero_sum():
    controller = _controller(mode="raise_only")
    controller.state_mV[:] = [0.2, 0.1, 0.3, 0.4]
    base = np.full(4, 18.0)
    delta = controller.threshold(base) - base

    assert np.all(delta >= 0.0)
    assert np.sum(delta) > 0.0
    assert delta[-1] == 0.0


def test_stratified_shuffle_uses_frozen_constructor_permutation():
    permutation = np.array([1, 0, 2, 3])
    controller = _controller(
        mode="stratified_shuffle",
        support_g=np.ones(4),
        shuffle_permutation=permutation,
    )
    controller.state_mV[:] = [0.3, 0.0, 0.0, 0.0]
    base = np.full(4, 18.0)
    delta = controller.threshold(base) - base

    assert np.argmax(delta) == 1
    assert np.argmin(delta) != 1
    assert np.sum(delta) == pytest.approx(0.0, abs=1e-14)
    assert not controller.shuffle_permutation.flags.writeable


def _grid_positions(n_side=24):
    x, y = np.meshgrid(
        np.linspace(0.0, 1.0, n_side, endpoint=False),
        np.linspace(0.0, 1.0, n_side, endpoint=False),
    )
    return np.column_stack([x.ravel(), y.ravel()])


def _grid_neighbor_correlation(values, n_side):
    grid = np.asarray(values).reshape(n_side, n_side)
    pairs = np.concatenate(
        [
            np.column_stack([grid[:, :-1].ravel(), grid[:, 1:].ravel()]),
            np.column_stack([grid[:-1, :].ravel(), grid[1:, :].ravel()]),
        ],
        axis=0,
    )
    return float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])


def _support_weighted_sd(values, support):
    mean = np.dot(support, values) / np.sum(support)
    return float(np.sqrt(np.dot(support, np.square(values - mean)) / np.sum(support)))


def test_spatial_shift_mapping_is_deterministic_complete_and_support_preserving():
    positions = _grid_positions(20)
    support = np.ones(positions.shape[0])
    support[positions[:, 1] >= 0.75] = 0.0

    first = build_spatial_shift_permutation(
        positions,
        support_g=support,
        block_shape=(8, 8),
        shift_blocks=(4, 0),
    )
    second = build_spatial_shift_permutation(
        positions.copy(),
        support_g=support.copy(),
        block_shape=(8, 8),
        shift_blocks=(4, 0),
    )

    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(positions.shape[0]))
    assert np.array_equal(support[first] > 0.0, support > 0.0)
    assert not np.array_equal(first, np.arange(positions.shape[0]))


def test_spatial_shift_preserves_coarse_smoothness_but_disrupts_alignment():
    n_side = 40
    positions = _grid_positions(n_side)
    permutation = build_spatial_shift_permutation(
        positions,
        block_shape=(8, 8),
        shift_blocks=(4, 0),
    )
    smooth_field = np.cos(2.0 * np.pi * positions[:, 0]) + 0.3 * np.sin(
        2.0 * np.pi * positions[:, 1]
    )
    shifted = smooth_field[permutation]

    original_spatial_r = _grid_neighbor_correlation(smooth_field, n_side)
    shifted_spatial_r = _grid_neighbor_correlation(shifted, n_side)
    alignment_r = float(np.corrcoef(smooth_field, shifted)[0, 1])

    assert original_spatial_r > 0.98
    assert shifted_spatial_r > 0.98
    assert abs(shifted_spatial_r - original_spatial_r) < 0.01
    assert alignment_r < -0.75


def test_spatial_shift_is_zero_sum_and_matches_unshifted_dynamic_amplitude():
    positions = _grid_positions(24)
    support = 0.2 + 0.8 * np.exp(-np.square(positions[:, 1] - 0.45) / (2.0 * 0.22**2))
    permutation = build_spatial_shift_permutation(
        positions,
        support_g=support,
        block_shape=(8, 8),
        shift_blocks=(4, 0),
    )
    kwargs = {
        "support_g": support,
        "dt_ms": 0.1,
        "tau_ms": 250.0,
        "a_ref_mV": 0.2,
        "r_ref_hz": 50.0,
        "trace_dt_ms": 0.1,
    }
    reference = FieldGatedZeroSumNodeRecovery(mode="zero_sum", **kwargs)
    shifted = FieldGatedZeroSumNodeRecovery(
        mode="spatial_shift",
        spatial_shift_permutation=permutation,
        **kwargs,
    )
    state = (
        0.25
        + 0.12 * np.cos(2.0 * np.pi * positions[:, 0])
        + 0.04 * np.sin(4.0 * np.pi * positions[:, 1])
    )
    reference.state_mV[:] = state
    shifted.state_mV[:] = state

    for step_index in range(8):
        reference_delta = reference.delta_theta()
        shifted_delta = shifted.delta_theta()

        assert np.sum(reference_delta) == pytest.approx(0.0, abs=1e-14)
        assert np.sum(shifted_delta) == pytest.approx(0.0, abs=1e-14)
        assert _support_weighted_sd(shifted_delta, support) == pytest.approx(
            _support_weighted_sd(reference_delta, support), rel=1e-14, abs=1e-14
        )
        if step_index == 0:
            assert np.corrcoef(reference_delta, shifted_delta)[0, 1] < -0.5

        spikes = np.zeros(positions.shape[0], dtype=bool)
        spikes[step_index::37] = True
        reference.step(spikes, 0.1)
        shifted.step(spikes, 0.1)
    assert not shifted.spatial_shift_permutation.flags.writeable


def test_spatial_shift_checkpoint_round_trip_preserves_mapping_and_delta():
    positions = _grid_positions(8)
    support = np.ones(positions.shape[0])
    permutation = build_spatial_shift_permutation(
        positions,
        support_g=support,
        block_shape=(4, 4),
        shift_blocks=(2, 0),
    )
    kwargs = {
        "support_g": support,
        "dt_ms": 0.1,
        "tau_ms": 250.0,
        "a_ref_mV": 0.2,
        "r_ref_hz": 50.0,
        "trace_dt_ms": 0.1,
        "mode": "spatial_shift",
        "spatial_shift_permutation": permutation,
    }
    controller = FieldGatedZeroSumNodeRecovery(**kwargs)
    spikes = np.zeros(positions.shape[0], dtype=bool)
    spikes[[0, 3, 17, 42]] = True
    controller.step(spikes, 0.1)
    controller.threshold(np.full(positions.shape[0], 18.0))
    payload = controller.checkpoint_state()

    restored = FieldGatedZeroSumNodeRecovery(**kwargs)
    restored.restore_checkpoint_state(payload)

    assert restored.config_sha256 == controller.config_sha256
    assert np.array_equal(
        restored.spatial_shift_permutation,
        controller.spatial_shift_permutation,
    )
    assert np.array_equal(restored.state_mV, controller.state_mV)
    assert np.array_equal(restored.delta_theta(), controller.delta_theta())
    assert restored.diagnostics() == controller.diagnostics()


def test_checkpoint_round_trip_is_exact_and_non_aliasing():
    controller = _controller()
    controller.step(np.array([True, False, True, False]), 0.1)
    controller.threshold(np.full(4, 18.0))
    payload = controller.checkpoint_state()

    restored = _controller()
    restored.restore_checkpoint_state(payload)

    assert restored.step_index == controller.step_index
    assert np.array_equal(restored.state_mV, controller.state_mV)
    assert restored.diagnostics() == controller.diagnostics()
    for key, expected in controller.trace_arrays().items():
        assert np.array_equal(restored.trace_arrays()[key], expected)

    payload["state_mV"][0] = 0.0
    payload["trace__time_ms"][0] = 999.0
    assert restored.state_mV[0] != 0.0
    assert restored.trace_arrays()["time_ms"][0] != 999.0


def test_checkpoint_rejects_different_frozen_configuration():
    source = _controller()
    payload = source.checkpoint_state()
    target = _controller(tau_ms=100.0)

    with pytest.raises(ValueError, match="configuration mismatch"):
        target.restore_checkpoint_state(payload)


def test_invalid_support_mode_and_permutation_are_rejected():
    with pytest.raises(ValueError, match="support_g"):
        _controller(support_g=np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="mode"):
        _controller(mode="unknown")
    with pytest.raises(ValueError, match="requires"):
        _controller(mode="stratified_shuffle")
    with pytest.raises(ValueError, match="complete permutation"):
        _controller(
            mode="stratified_shuffle",
            shuffle_permutation=np.array([0, 0, 2, 3]),
        )
    with pytest.raises(ValueError, match="requires"):
        _controller(mode="spatial_shift")
    with pytest.raises(ValueError, match="preserve positive support"):
        _controller(
            mode="spatial_shift",
            spatial_shift_permutation=np.array([3, 1, 2, 0]),
        )
