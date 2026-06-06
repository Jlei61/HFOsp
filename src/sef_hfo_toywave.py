# src/sef_hfo_toywave.py
"""Analytic synthetic sources for the Increment-1 known-direction contract gate.

Each returns dict(frames=(n_time, n_pix), grid_xy=(n_pix,2), window=(t_on,t_off) ms,
n_hat=(2,) or None, dt, pitch_hint). No model dynamics; coords match
src.sef_hfo_observation.grid_coords(n, L).
"""
from __future__ import annotations

import numpy as np

from src.sef_hfo_observation import grid_coords


def _times(dt, t_max):
    return np.arange(0.0, t_max, dt)


def traveling_wave(n, L, angle_rad, c, dt, t_max, width, t0=None):
    """Smooth bell a(x,t)=exp(-(s - c(t-t0))^2 / 2 width^2), s = x·n_hat."""
    coords = grid_coords(n, L)
    n_hat = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    s = coords @ n_hat
    t = _times(dt, t_max)
    if t0 is None:
        t0 = 0.0
    s0 = s.min() - 2 * width                     # start just outside the sheet
    front = s0 + c * (t - t0)
    frames = np.exp(-((s[None, :] - front[:, None]) ** 2) / (2.0 * width ** 2))
    return dict(frames=frames, grid_xy=coords, window=(0.0, t_max), n_hat=n_hat,
                dt=dt, pitch_hint=L / n)


def radial_source(n, L, c, dt, t_max, width):
    """Centered expanding ring a(x,t)=exp(-(r - c t)^2 / 2 width^2), r=||x||.
    Real arrival gradient (outward) but NO preferred axis -> direction must be no-axis."""
    coords = grid_coords(n, L)
    r = np.linalg.norm(coords, axis=1)
    t = _times(dt, t_max)
    frames = np.exp(-((r[None, :] - (c * t)[:, None]) ** 2) / (2.0 * width ** 2))
    return dict(frames=frames, grid_xy=coords, window=(0.0, t_max), n_hat=None,
                dt=dt, pitch_hint=L / n)


def synchronous_amplitude_source(n, L, dt, t_max, width, ramp_axis_rad=0.0):
    """a(x,t)=b(x)·h(t): all pixels rise/fall together (shared bell h(t)); only
    spatial amplitude b(x) (linear ramp along ramp_axis) differs. NO arrival
    gradient -> tests whether electrode layout + threshold fabricate fake order."""
    coords = grid_coords(n, L)
    axis = np.array([np.cos(ramp_axis_rad), np.sin(ramp_axis_rad)])
    proj = coords @ axis
    b = 0.5 + 0.5 * (proj - proj.min()) / max(np.ptp(proj), 1e-12)  # in [0.5,1]
    t = _times(dt, t_max)
    h = np.exp(-((t - t_max / 2.0) ** 2) / (2.0 * width ** 2))
    frames = h[:, None] * b[None, :]
    return dict(frames=frames, grid_xy=coords, window=(0.0, t_max), n_hat=None,
                dt=dt, pitch_hint=L / n)
