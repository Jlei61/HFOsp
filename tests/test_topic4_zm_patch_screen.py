"""Path-B cheap-first screen: reduced K-patch inhibitory-containment model (task §7 Path B).

Characterization tests: the model was calibrated by exploration into the relaxation-oscillation regime
(single patch oscillates for I0 in ~[0.6,1.2] with w_rec=2); these tests LOCK that regime + the mechanics
+ the load-bearing DIRECTIONAL screen result (a single GLOBAL scalar pool keeps the patches synchronized,
a PATCHWISE local pool desynchronizes them). They would fail if the model regressed out of this regime.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_patch_screen import (  # noqa: E402
    PatchParams, simulate, population_signal, population_occupancy, synchrony, screen_metrics, _smooth_ring)


def test_no_drive_activity_dies():
    P = population_signal(simulate(PatchParams(K=4, I0=0.0, w_rec=0.0), T_ms=2000.0)["a"])
    assert P[len(P) // 2:].mean() < 0.02


def test_single_patch_relaxation_oscillation_in_calibrated_band():
    P = population_signal(simulate(PatchParams(K=1, I0=1.0, w_rec=2.0), T_ms=3000.0)["a"])
    ps = P[len(P) // 4:]
    depth = (ps.max() - ps.min()) / (ps.mean() + 1e-9)
    assert depth > 0.8 and ps.std() > 0.02          # genuinely oscillating, not a fixed point


def test_high_drive_damps_to_fixed_point():
    P = population_signal(simulate(PatchParams(K=1, I0=1.5, w_rec=2.0), T_ms=3000.0)["a"])
    ps = P[len(P) // 4:]
    assert ps.std() < 0.01                           # above the oscillatory band -> stable fixed point


def test_simulate_shapes():
    r = simulate(PatchParams(K=8), T_ms=1000.0, dt=0.5)
    assert r["a"].shape == (2000, 8) and r["s_loc"].shape == (2000, 8) and r["s_glob"].shape == (2000,)


def test_population_occupancy_sustained_vs_bursty():
    n = 4000
    sustained = np.full(n, 1.0) + 0.01 * np.random.default_rng(0).standard_normal(n)
    bursty = np.zeros(n)
    bursty[::200] = 1.0                              # sparse spikes -> low occupancy
    for i in range(0, n, 200):
        bursty[i:i + 10] = 1.0
    assert population_occupancy(sustained) > 0.9
    assert population_occupancy(bursty) < 0.3


def test_synchrony_identical_vs_antiphase():
    t = np.linspace(0, 20 * np.pi, 2000)
    inphase = np.stack([np.sin(t) + 2, np.sin(t) + 2], axis=1)
    antiphase = np.stack([np.sin(t) + 2, -np.sin(t) + 2], axis=1)
    assert synchrony(inphase) > 0.9
    assert synchrony(antiphase) < -0.5


def test_pool_smoothing_sigma_zero_is_identity():
    a = np.array([0.1, 0.9, 0.2, 0.7, 0.3])
    assert np.array_equal(_smooth_ring(a, 0.0), a)
    sm = _smooth_ring(a, 1.0)
    assert abs(sm.sum() - a.sum()) < 1e-9           # smoothing conserves mass


def test_global_pool_keeps_patches_more_synchronized_than_patchwise():
    """The load-bearing screen result: with heterogeneity, a single global scalar pool locks the patches
    together (high sync, low population occupancy = burst train) while patchwise pools desynchronize them
    (low sync, higher occupancy). This is the mechanism the full-SNN Path-B would exploit."""
    kw = dict(K=16, I0=1.0, w_rec=2.0, sigma_I=0.4, w_c=0.05, seed=1)
    g = screen_metrics(simulate(PatchParams(mode="global", **kw), T_ms=6000.0))
    p = screen_metrics(simulate(PatchParams(mode="patchwise", **kw), T_ms=6000.0))
    assert g["synchrony"] > p["synchrony"] + 0.5     # global far more synchronized
    assert p["occupancy"] > 0.9 and p["patch_osc"] > 0.01   # patchwise: sustained population + still oscillating
    assert p["carrier_proxy"] is True                # patchwise passes the reduced-model carrier proxy...
    # ...while a homogeneous global scalar pool is a synchronized burst train (population collapses to OFF)
    gh = screen_metrics(simulate(PatchParams(mode="global", K=16, I0=1.0, w_rec=2.0, seed=1), T_ms=6000.0))
    assert gh["occupancy"] < 0.7 and gh["carrier_proxy"] is False
