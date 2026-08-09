"""The separation gate decides whether a whole family of terminators is possible, so the
integration behind it has to match the engine's own step and the verdict has to be able to say no.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.topic4_fcxr_lc3_percell_load import (  # noqa: E402
    aggregate_activation,
    replay_load,
    separation,
    stationary_load,
)
from src.topic4_mz_slowvars import replay_adaptation_peak  # noqa: E402


def _raster(nsteps, ncell, every, rng=None):
    E = np.zeros((nsteps, ncell), dtype=bool)
    E[::every, :] = True
    return E


def test_the_peak_agrees_with_the_existing_replay():
    """Two integrations of one ODE must not drift apart."""
    E = _raster(400, 5, 7)
    mine = replay_load(E, dt_ms=0.05, tau_ms=120.0)["peak"]
    theirs = replay_adaptation_peak(E, dt=0.05, tau_adp=120.0)
    assert np.allclose(mine, theirs, rtol=0, atol=1e-12)


def test_it_decays_before_it_increments_like_the_engine():
    """One spike at step 0: the engine decays the (zero) load first, so the peak is exactly 1."""
    E = np.zeros((1, 3), dtype=bool)
    E[0] = True
    out = replay_load(E, dt_ms=0.05, tau_ms=100.0)
    assert np.allclose(out["peak"], 1.0)
    assert np.allclose(out["final"], 1.0)


def test_a_seeded_start_is_not_a_charging_ramp():
    """Starting at the stationary load must leave a steadily-firing cell where it started."""
    dt, tau, every = 0.05, 200.0, 20          # a spike every 1 ms -> 1000 Hz
    E = _raster(4000, 2, every)               # 200 ms of it
    r = 1000.0 / (every * dt)
    seeded = replay_load(E, dt, tau, init=stationary_load(np.full(2, r), tau))["settled"]
    cold = replay_load(E, dt, tau)["settled"]
    assert np.all(seeded > cold), "seeding must remove the charge ramp, not add one"
    assert np.allclose(seeded, stationary_load(np.full(2, r), tau), rtol=0.05)


def test_the_stationary_load_is_rate_times_tau_whatever_the_pattern():
    """Bursty and even firing at the same rate settle at the same load -- this is why the time
    constant alone cannot separate two states, and the gate exists because of it."""
    dt, tau = 0.05, 500.0
    even = _raster(20000, 1, 40)                       # 500 Hz, evenly spaced
    bursty = np.zeros((20000, 1), dtype=bool)
    for start in range(0, 20000, 400):                 # same count, clumped into bursts
        bursty[start:start + 10, 0] = True
    a = replay_load(even, dt, tau, settle_from_ms=600.0)["settled"]
    b = replay_load(bursty, dt, tau, settle_from_ms=600.0)["settled"]
    assert np.allclose(a, b, rtol=0.02)


def test_init_shape_is_checked():
    with pytest.raises(ValueError, match="init has shape"):
        replay_load(_raster(10, 4, 3), 0.05, 100.0, init=np.zeros(3))


def test_overlap_is_reported_as_a_gap_below_one_not_hidden_by_the_percentile():
    """The failure this replaces: a rare interictal excursion above the discharge's floor reads as
    a comfortable ratio against the 99.9th percentile while the distributions actually overlap."""
    quiet = np.concatenate([np.full(3998, 1.0), np.full(2, 40.0)])   # 0.05%: below the percentile
    out = separation(quiet, np.full(4000, 20.0), quiet_q=99.9)
    assert out["headroom"] > 1.0, "the percentile ratio looks fine"
    assert out["gap"] < 1.0, "...while the extremes overlap, which is the number that matters"
    assert out["overlap_frac"] > 0.0


def test_a_clean_gap_reports_above_one():
    assert separation(np.linspace(0.0, 1.0, 4000), np.full(4000, 50.0))["gap"] == 50.0


def test_the_negative_is_scoped_to_a_memoryless_threshold_on_one_filtered_load():
    """A small gap must not be reported as ruling out every mechanism that reads a cell's own
    spikes -- duration, two timescales and joint gating are untouched by this measurement."""
    b = separation(np.full(10, 1.0), np.full(10, 1.1))["boundary"]
    for escape in ("duration", "two timescales", "state-dependent"):
        assert escape in b


def test_a_tail_that_crosses_need_not_mean_the_population_is_engaged():
    """Why the boolean was the wrong instrument: the same distribution whose tail crosses K can
    carry almost no activation, and that is what the interictal side has to satisfy."""
    quiet = np.concatenate([np.full(3990, 1.0), np.full(10, 40.0)])
    a = aggregate_activation(quiet, K=20.0, n=4)
    assert a["frac_half_on"] > 0.0, "the tail does cross"
    assert a["mean"] < 0.01, "yet the population carries essentially none of it"


def test_cooperativity_sharpens_the_same_two_distributions():
    quiet, ictal = np.full(100, 1.0), np.full(100, 2.0)
    ratios = [aggregate_activation(ictal, 1.4, n)["mean"] / aggregate_activation(quiet, 1.4, n)["mean"]
              for n in (2, 4, 8)]
    assert ratios[0] < ratios[1] < ratios[2]
