import numpy as np
import pytest

from src.topic4_ou_runtime_audit import (
    OUAuditProxy,
    spatial_correlation_length_mm,
    stationarity_report,
    temporal_autocorrelation_time_ms,
)
from src.topic4_spatial_ou_drive import SpatialOUConfig, SpatialOUDrive


def _drive(sigma=0.1, tau=20.0, ell=0.38, seed=11, n=900, sheet=6.0, dt=0.1):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, sheet, size=(n, 2))
    return SpatialOUDrive(
        positions, sheet, dt,
        SpatialOUConfig(mode="local", sigma_rate_per_ms=sigma, tau_ms=tau,
                        ell_mm=ell, update_interval_ms=1.0,
                        grid_spacing_mm=0.4, seed=seed))


def test_proxy_is_observation_only_and_reproduces_the_bare_drive():
    """The audit wrapper must not perturb the field the SNN would have seen."""
    bare = _drive()
    wrapped = OUAuditProxy(_drive(), dt_ms=0.1, snapshot_interval_ms=1.0)
    for step in range(500):
        time_ms = step * 0.1
        np.testing.assert_array_equal(bare.step(time_ms), wrapped.step(time_ms))
    evidence = wrapped.runtime_evidence(500)
    assert evidence["n_step_calls"] == 500
    assert evidence["called_every_membrane_step"] is True
    assert evidence["call_step_gap_max"] == 1
    assert evidence["call_step_gap_min"] == 1


def test_runtime_evidence_reports_a_drive_that_was_not_stepped_every_step():
    """A drive queried on a subset of steps must not be certified as running."""
    wrapped = OUAuditProxy(_drive(), dt_ms=0.1, snapshot_interval_ms=1.0)
    for step in range(0, 500, 5):
        wrapped.step(step * 0.1)
    evidence = wrapped.runtime_evidence(500)
    assert evidence["called_every_membrane_step"] is False
    assert evidence["call_step_gap_max"] == 5


def test_measured_tau_recovers_the_declared_ou_time_constant():
    declared_tau = 25.0
    wrapped = OUAuditProxy(_drive(tau=declared_tau), dt_ms=0.1,
                           snapshot_interval_ms=1.0)
    for step in range(40000):
        wrapped.step(step * 0.1)
    got = temporal_autocorrelation_time_ms(
        wrapped.snapshot_arrays()["ou_grid_snapshots"], 1.0)
    assert got["tau_hat_ms"] == pytest.approx(declared_tau, rel=0.25)


def test_measured_spatial_length_tracks_the_declared_smoothing_scale():
    """Doubling the smoothing kernel must roughly double the measured length."""
    lengths = []
    for ell in (0.4, 0.8):
        wrapped = OUAuditProxy(_drive(ell=ell), dt_ms=0.1,
                               snapshot_interval_ms=1.0)
        for step in range(4000):
            wrapped.step(step * 0.1)
        got = spatial_correlation_length_mm(
            wrapped.snapshot_arrays()["ou_grid_snapshots"], 0.4)
        lengths.append(got["correlation_length_mm_1_over_e"])
    assert lengths[0] < lengths[1]
    assert lengths[1] / lengths[0] == pytest.approx(2.0, rel=0.35)


def test_stationarity_report_flags_an_amplitude_change_across_the_split():
    times = np.arange(0.0, 200.0, 1.0)
    values = np.ones((len(times), 50))
    rng = np.random.default_rng(3)
    values *= rng.standard_normal(values.shape)
    values[times >= 100.0] *= 4.0
    report = stationarity_report(values, times, 100.0)
    assert report["sd_ratio_after_over_before"] == pytest.approx(4.0, rel=0.2)

    steady = rng.standard_normal((len(times), 50))
    steady_report = stationarity_report(steady, times, 100.0)
    assert steady_report["sd_ratio_after_over_before"] == pytest.approx(
        1.0, abs=0.1)


def test_reseed_keeps_statistics_and_only_changes_the_realisation():
    """Same amplitude, tau and length; a different innovation stream."""
    from src.topic4_ou_runtime_audit import OUProtocolProxy

    plain = OUProtocolProxy(_drive(), dt_ms=0.1, snapshot_interval_ms=1.0)
    reseeded = OUProtocolProxy(_drive(), dt_ms=0.1, snapshot_interval_ms=1.0,
                               reseed_at_ms=200.0, reseed_seed=4242)
    for step in range(20000):
        time_ms = step * 0.1
        before = np.array(plain.step(time_ms))
        after = np.array(reseeded.step(time_ms))
        if time_ms < 200.0:
            np.testing.assert_array_equal(before, after)
    assert reseeded.protocol_evidence()["reseed_applied_ms"] == 200.0
    plain_grid = plain.snapshot_arrays()["ou_grid_snapshots"]
    reseed_grid = reseeded.snapshot_arrays()["ou_grid_snapshots"]
    late = slice(250, None)
    assert not np.allclose(plain_grid[late], reseed_grid[late])
    plain_tau = temporal_autocorrelation_time_ms(plain_grid[late], 1.0)
    reseed_tau = temporal_autocorrelation_time_ms(reseed_grid[late], 1.0)
    assert reseed_tau["tau_hat_ms"] == pytest.approx(
        plain_tau["tau_hat_ms"], rel=0.3)
    assert float(reseed_grid[late].std()) == pytest.approx(
        float(plain_grid[late].std()), rel=0.2)


def test_amplitude_dip_is_bounded_and_exactly_restored():
    from src.topic4_ou_runtime_audit import OUProtocolProxy

    proxy = OUProtocolProxy(_drive(), dt_ms=0.1, snapshot_interval_ms=1.0,
                            dip_start_ms=20.0, dip_duration_ms=10.0,
                            dip_factor=0.0)
    reference = _drive()
    inside, outside = [], []
    for step in range(500):
        time_ms = step * 0.1
        got = np.array(proxy.step(time_ms))
        want = np.array(reference.step(time_ms))
        if 20.0 <= time_ms < 30.0:
            inside.append(np.allclose(got, 0.0))
        else:
            outside.append(np.allclose(got, want))
    assert all(inside) and all(outside)
    assert proxy.protocol_evidence()["n_steps_inside_dip"] == 100
