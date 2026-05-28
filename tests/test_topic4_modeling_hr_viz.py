"""Tests for src/topic4_modeling/hr_viz.py."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def test_nullcline_x_formula():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_viz import compute_x_nullcline
    p = HRParams()
    x = np.array([0.0, 1.0, -1.0])
    out = compute_x_nullcline(x, p, z=0.5, I=-1.6)
    expected = p.a * x**3 - p.b * x**2 + 0.5 - (-1.6)
    np.testing.assert_allclose(out, expected)


def test_nullcline_y_formula():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_viz import compute_y_nullcline
    p = HRParams()
    x = np.array([0.0, 1.0, -1.0])
    out = compute_y_nullcline(x, p)
    np.testing.assert_allclose(out, p.c - p.d * x**2)


def test_plot_phase_portrait_smoke(tmp_path):
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    from src.topic4_modeling.hr_viz import plot_phase_portrait
    p = HRParams()
    _, traj = simulate_trajectory(p, I=-1.6, T=100.0, dt=0.05,
                                   sigma_ou=0.1, tau_ou=10.0, seed=0)
    fig = plot_phase_portrait(traj, p, I=-1.6)
    out = tmp_path / "phase.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_regime_map_smoke(tmp_path):
    from src.topic4_modeling.hr_viz import plot_regime_map
    rows = []
    for I in [-2.0, -1.6, -1.0]:
        for r in [0.004, 0.006]:
            for sigma in [0.0, 0.1]:
                regime = "silent" if sigma == 0.0 else "excitable"
                rows.append({"I": I, "r_used": r, "sigma_ou": sigma,
                             "seed": 0, "regime": regime, "n_bursts": 1,
                             "mean_burst_duration": 5.0, "mean_ibi": 100.0})
    df = pd.DataFrame(rows)
    fig = plot_regime_map(df)
    out = tmp_path / "regime.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 1000
