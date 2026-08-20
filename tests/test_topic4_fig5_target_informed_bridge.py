import numpy as np

from src.topic4_fig5_target_informed_bridge import (
    bootstrap_patient_summary,
    exact_contact_reorder,
    lse,
    nonoverlap_log_power_windows,
    robust_z_against_reference,
    select_state_defined_readout,
    shaft_balanced_scaled_l1,
    smooth_rate,
)


def _oscillation(freq, seconds, *, fs=1000.0, contacts=2, amplitude=1.0):
    t = np.arange(int(seconds * fs)) / fs
    return np.stack([
        amplitude * np.sin(2 * np.pi * freq * t + phase)
        for phase in np.linspace(0.0, 0.4, contacts)
    ], axis=1)


def test_exact_contact_reorder_is_name_based_and_fail_closed():
    values = np.array([[1.0, 2.0, 3.0]])
    got = exact_contact_reorder(values, ["b", "c", "a"], ["a", "b", "c"])
    np.testing.assert_array_equal(got, [[3.0, 1.0, 2.0]])
    try:
        exact_contact_reorder(values, ["b", "c", "x"], ["a", "b", "c"])
    except ValueError as error:
        assert "exact contact mismatch" in str(error)
    else:
        raise AssertionError("contact mismatch did not fail")


def test_reference_robust_z_does_not_use_candidate_values():
    reference = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    candidate = np.array([100.0, 100.0])
    z1, audit1 = robust_z_against_reference(reference, candidate)
    z2, audit2 = robust_z_against_reference(reference, candidate * 100.0)
    np.testing.assert_allclose(audit1["median"], audit2["median"])
    np.testing.assert_allclose(audit1["mad"], audit2["mad"])
    assert np.all(z2 > z1)


def test_nonoverlap_power_keeps_independent_windows():
    trace = _oscillation(40.0, 2.0)
    powers = nonoverlap_log_power_windows(trace, 1.0, window_ms=500.0)
    assert powers.shape == (4, 2)
    np.testing.assert_allclose(powers, np.tile(powers[0], (4, 1)), atol=1e-8)


def test_readout_is_earliest_state_window_not_later_better_target_frame():
    baseline = _oscillation(12.0, 1.0)
    trace = np.concatenate([
        _oscillation(12.0, 1.0),
        _oscillation(40.0, 0.5, amplitude=1.0),
        _oscillation(80.0, 0.5, amplitude=20.0),
    ])
    field_t = np.arange(20.0, 2000.1, 20.0)
    active = np.where(field_t >= 1000.0, 0.9, 0.1)
    spatial = active.copy()
    selected = select_state_defined_readout(
        trace=trace,
        dt_ms=1.0,
        full_field_time_ms=field_t,
        active_fraction=active,
        spatial_fraction=spatial,
        t_ictal_ms=1000.0,
        baseline_trace=baseline,
    )
    assert selected is not None
    assert selected.start_ms == 1000.0


def test_shaft_balance_prevents_long_shaft_from_hiding_short_shaft_failure():
    model = np.array([0.0] * 11 + [10.0] * 4)
    target = np.zeros(15)
    scale = np.ones(15)
    shafts = np.array(["ICL"] * 11 + ["SCL"] * 4)
    score, by_shaft = shaft_balanced_scaled_l1(model, target, scale, shafts)
    assert by_shaft == {"ICL": 0.0, "SCL": 10.0}
    assert score > 9.0
    assert score > np.mean(np.abs(model - target))


def test_patient_bootstrap_excludes_display_seizure_upstream_and_is_deterministic():
    pre = np.arange(60.0).reshape(4, 15)
    early = pre + np.arange(15.0)
    first = bootstrap_patient_summary(pre, early, draws=128, seed=7)
    second = bootstrap_patient_summary(pre, early, draws=128, seed=7)
    np.testing.assert_array_equal(first["early"]["q025"], second["early"]["q025"])
    assert len(first["global_early_per_seizure"]) == 4


def test_lse_is_equal_to_common_value_for_equal_inputs():
    assert lse([3.0, 3.0, 3.0]) == 3.0


def test_smoothed_rate_makes_sparse_reference_median_resolvable():
    rate = np.zeros(1000)
    rate[::50] = 1000.0
    assert np.median(rate) == 0.0
    assert np.median(smooth_rate(rate, 0.1, 20.0)) > 0.0
