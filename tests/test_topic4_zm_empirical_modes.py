import numpy as np

from src.topic4_zm_empirical_modes import fit_axial_dmd, project_on_left_mode


def test_traveling_wave_recovers_complex_nonuniform_mode_and_phase_gradient():
    n, nt, dt_ms = 24, 600, 10.0
    x = np.arange(n)[:, None]
    t = np.arange(nt)[None, :]
    field = 5.0 + np.cos(2 * np.pi * 12.0 * t * dt_ms / 1000.0 - 0.35 * x)
    out = fit_axial_dmd(field, dt_ms=dt_ms)
    mode = out["pathological_mode_candidate"]
    assert abs(mode["frequency_hz"] - 12.0) < 1.0
    assert mode["uniform_overlap"] < 0.3
    assert abs(mode["phase_gradient_rad_per_bin"]) > 0.2
    assert mode["phase_gradient_r2"] > 0.9


def test_uniform_oscillation_has_dominant_common_mode():
    n, nt, dt_ms = 24, 500, 10.0
    t = np.arange(nt)
    spatial = np.ones(n)[:, None]
    field = 3.0 + spatial * np.sin(2 * np.pi * 8.0 * t[None, :] * dt_ms / 1000.0)
    field += 1e-5 * np.arange(n)[:, None] * np.cos(2 * np.pi * 3.0 * t[None, :] * dt_ms / 1000.0)
    out = fit_axial_dmd(field, dt_ms=dt_ms)
    assert out["pc1_fraction"] > 0.999
    assert out["leading_mode"]["uniform_overlap"] > 0.99


def test_left_mode_projection_preserves_time_dimension():
    n, nt, dt_ms = 24, 400, 10.0
    x = np.arange(n)[:, None]
    t = np.arange(nt)[None, :]
    field = np.cos(0.2 * x - 0.1 * t)
    out = fit_axial_dmd(field, dt_ms=dt_ms)
    amplitude = project_on_left_mode(field, out["pathological_mode_candidate"], out["mean_field"])
    assert amplitude.shape == (nt,)
    assert np.isfinite(amplitude).all()
