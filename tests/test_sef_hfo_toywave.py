# tests/test_sef_hfo_toywave.py
"""TDD for src/sef_hfo_toywave — Increment 1 analytic sources (no model)."""
import numpy as np

from src.sef_hfo_toywave import (
    traveling_wave,
    radial_source,
    synchronous_amplitude_source,
)


def test_traveling_wave_peak_moves_along_n_hat():
    src = traveling_wave(n=64, L=64.0, angle_rad=0.0, c=0.5, dt=0.25,
                         t_max=200.0, width=8.0)
    frames, coords, window, n_hat = (src["frames"], src["grid_xy"],
                                     src["window"], src["n_hat"])
    assert frames.shape[1] == coords.shape[0]
    np.testing.assert_allclose(n_hat, [1.0, 0.0], atol=1e-9)
    # peak-position projection on n_hat increases over time (wave moves +x)
    proj = coords @ n_hat
    early = proj[frames[10].argmax()]
    late = proj[frames[-10].argmax()]
    assert late > early
    assert window[1] > window[0]


def test_radial_source_is_centered_and_isotropic():
    src = radial_source(n=64, L=64.0, c=0.4, dt=0.25, t_max=150.0, width=6.0)
    frames, coords = src["frames"], src["grid_xy"]
    # centroid of activity stays ~at origin (no preferred axis)
    f = frames[frames.shape[0] // 2]
    centroid = (coords * f[:, None]).sum(0) / f.sum()
    np.testing.assert_allclose(centroid, [0.0, 0.0], atol=1.0)


def test_synchronous_amplitude_source_has_no_arrival_gradient():
    src = synchronous_amplitude_source(n=64, L=64.0, dt=0.25, t_max=120.0,
                                       width=10.0, ramp_axis_rad=0.0)
    frames = src["frames"]
    # every pixel peaks at the SAME time frame (h(t) shared) -> no arrival order
    peak_frames = frames.argmax(axis=0)
    active = frames.max(axis=0) > 0.1 * frames.max()
    assert np.ptp(peak_frames[active]) == 0
