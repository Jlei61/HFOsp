"""Task 7 (spec rev3.1 §4.4/§6.3): source metrics, rest distance and the probabilistic taxonomy.

All synthetic: every taxonomy outcome the spec names must be reachable and must be reached for the
right reason, so a real fork result cannot be talked into a class it does not earn.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_minimal_carrier as MC  # noqa: E402
import src.topic4_zm_anchor_states as AS  # noqa: E402

DT = 0.1
BIN = MC.BIN_MS


# ---------------------------------------------------------------- source metrics
def _raster(nsteps, NE, pattern):
    E = np.zeros((nsteps, NE), bool)
    pattern(E)
    return E


def test_source_metrics_separate_extent_from_rate():
    """A hotspot and a spread state can have the SAME population rate; A_active and H_spatial must
    tell them apart, otherwise 'spatially organised' is untestable."""
    NE, nsteps = 400, 1000
    pos = np.stack(np.meshgrid(np.linspace(0, 20, 20), np.linspace(0, 20, 20)), -1).reshape(-1, 2)
    core = np.zeros(NE, bool)
    core[:40] = True

    def hotspot(E):
        E[:, :20] = True                      # 20 neurons firing every step

    def spread(E):
        for t in range(nsteps):
            E[t, (t * 7) % NE::20] = True     # same total spikes, spread over the sheet

    a = MC.source_metrics(_raster(nsteps, NE, hotspot), core, pos, 20.0, DT)
    b = MC.source_metrics(_raster(nsteps, NE, spread), core, pos, 20.0, DT)
    assert b["A_active"].mean() > 3 * a["A_active"].mean()
    assert b["H_spatial"].mean() > a["H_spatial"].mean()
    assert b["n_grid_active"].mean() > a["n_grid_active"].mean()


def test_rest_distance_is_zero_at_the_reference_and_grows_with_deviation():
    n = 40
    met = {k: np.zeros(n) for k in MC.REST_KEYS}
    met["n_bins"] = n
    for k in MC.REST_KEYS:
        met[k][:] = 1.0 + 0.01 * np.random.default_rng(0).standard_normal(n)
    ref = MC.rest_reference(met, 0, 20)
    d = MC.rest_distance(met, ref)
    assert d[:20].mean() < 2.0
    met2 = {k: v.copy() for k, v in met.items() if k != "n_bins"}
    met2["n_bins"] = n
    met2["r_core"] = met2["r_core"] + 10.0
    assert MC.rest_distance(met2, ref)[0] > 5 * d[0]


def test_a_brief_trough_is_not_a_rest_return():
    """Spec §4.4: troughs are allowed; only a dwell below threshold counts as a basin return."""
    d = np.full(200, 5.0)
    d[50:53] = 0.2                                  # 75 ms dip
    assert MC.first_rest_return(d, BIN, 1.0, 200.0) is None
    d[100:120] = 0.2                                # 500 ms dwell
    assert MC.first_rest_return(d, BIN, 1.0, 200.0) == 100


# ---------------------------------------------------------------- taxonomy
def _rep(survived, life, end=None, resets=0):
    return dict(survived=survived, lifetime_ms=life, end_reason=end, rest_returns=resets)


def test_stable_carrier_requires_high_posterior_and_lifetime_beyond_ied():
    reps = [_rep(True, 8000.0) for _ in range(6)]
    out = MC.classify_replicas(reps, ied_lifetime_ms=120.0)
    assert out["klass"] == "stable_carrier" and out["posterior"]["median"] > 0.8
    short = [_rep(True, 150.0) for _ in range(6)]
    assert MC.classify_replicas(short, ied_lifetime_ms=120.0)["klass"] == "transient_carrier_like"


def test_metastable_carrier_band():
    reps = [_rep(True, 6000.0)] * 3 + [_rep(False, 3000.0)] * 3
    out = MC.classify_replicas(reps, ied_lifetime_ms=120.0)
    assert out["klass"] in ("metastable_carrier", "probabilistically_indeterminate")
    reps = [_rep(True, 6000.0)] * 7 + [_rep(False, 3000.0)] * 3
    assert MC.classify_replicas(reps, ied_lifetime_ms=120.0)["klass"] == "metastable_carrier"


def test_repeated_rest_basin_reset_is_an_hfo_like_train_not_a_carrier():
    reps = [_rep(True, 8000.0, resets=5) for _ in range(6)]
    reps = [dict(r, survived=False, lifetime_ms=2000.0) for r in reps]
    assert MC.classify_replicas(reps, 120.0)["klass"] == "hfo_like_relaxation_train"


def test_runaway_and_plateau_outrank_the_posterior():
    assert MC.classify_replicas([_rep(False, 500.0, "runaway")] * 5, 120.0)["klass"] == "runaway"
    assert MC.classify_replicas([_rep(False, 500.0, "saturated_plateau")] * 5,
                                120.0)["klass"] == "saturated_plateau"


def test_threshold_edge_returns_indeterminate_not_a_forced_class():
    out = MC.classify_replicas([_rep(True, 9000.0)] * 2 + [_rep(False, 800.0)], 120.0)
    assert out["klass"] == "probabilistically_indeterminate", out
    assert out["posterior"]["lo"] < 0.8 < out["posterior"]["hi"]


def test_empty_evidence_is_no_evidence_not_a_negative():
    assert MC.classify_replicas([], 120.0)["klass"] == "no_evidence"


def test_jeffreys_posterior_is_calibrated():
    p = MC.jeffreys_posterior(0, 3)
    assert p["median"] < 0.3 and p["hi"] > 0.3, "3/3 failures must not read as a certain zero"
    q = MC.jeffreys_posterior(9, 9)
    assert q["median"] > 0.8 and q["lo"] < 1.0


# ---------------------------------------------------------------- partial order
def test_smallest_positive_subsystem_partial_order():
    assert MC.smallest_positive_subsystem({"freeze_all": "stable_carrier"}) == ["carrier_fast_only"]
    both = MC.smallest_positive_subsystem({"freeze_all": "transient_carrier_like",
                                           "freeze_zm": "metastable_carrier",
                                           "freeze_zsg": "stable_carrier"})
    assert both == ["carrier_fast_plus_m", "carrier_fast_plus_sg"], "ties must be reported, not broken"
    assert MC.smallest_positive_subsystem({"freeze_all": "transient_carrier_like",
                                           "freeze_z": "stable_carrier"}) == \
        ["carrier_fast_plus_m_sg"]
    assert MC.smallest_positive_subsystem({"freeze_all": "hfo_like_relaxation_train"}) is None


# ---------------------------------------------------------------- anchor selection
def test_arclength_bins_resist_uneven_trajectory_speed():
    """A trajectory that dawdles then sprints must not put all three bounded bins in the dawdle."""
    n = 300
    slow_part = np.linspace(0, 1, 200)
    fast_part = np.linspace(1, 10, 100)
    q = np.concatenate([slow_part, fast_part])[:, None].repeat(7, axis=1)
    bins, arc = AS.arclength_bins(q, 0, n)
    idx = [bins[k] for k in AS.BOUNDED_BINS]
    assert idx[0] < idx[1] < idx[2]
    assert idx[2] > 200, "the late bin must land in the fast leg, not in the slow leg"
    by_time = [int(n * AS.BOUNDED_QUANTILES[k]) for k in AS.BOUNDED_BINS]
    assert idx != by_time, "arc-length binning must differ from wall-clock binning here"


def test_fast_phase_uses_local_derivatives_not_the_clock():
    bin_ms = 25.0
    t = np.arange(400)
    r = 10 + 9 * np.sin(2 * np.pi * t / 40.0)
    peak = AS.natural_fast_phase(r, bin_ms, 200, "peak")
    trough = AS.natural_fast_phase(r, bin_ms, 200, "trough")
    rising = AS.natural_fast_phase(r, bin_ms, 200, "rising")
    assert r[peak] > r[rising] > r[trough]
    assert np.gradient(r)[rising] > 0


def test_anchor_eligibility_rejects_runaway_and_short_containment():
    n = 800
    met = dict(r_core=np.concatenate([np.tile([0.5, 8.0, 0.5, 0.5], 25), np.full(700, 30.0)]))
    met["n_bins"] = n
    ok, info = AS.anchor_eligibility(met, BIN, runaway_ms=None)
    assert ok and info["bounded_ms"] >= AS.MIN_BOUNDED_MS
    ok2, info2 = AS.anchor_eligibility(met, BIN, runaway_ms=3000.0)
    assert not ok2 and any("runaway" in r for r in info2["reasons"])
    short = dict(r_core=np.concatenate([np.tile([0.5, 8.0, 0.5, 0.5], 175), np.full(100, 30.0)]))
    short["n_bins"] = 800
    ok3, info3 = AS.anchor_eligibility(short, BIN, runaway_ms=None)
    assert not ok3 and any("contained segment" in r for r in info3["reasons"])


def test_returning_events_are_measured_before_escalation_only():
    """Sparse returning events on a quiet baseline, then a sustained contained state. Only the
    pre-escalation window may contribute to the IED reference the carrier has to beat."""
    quiet = np.tile(np.concatenate([[6.0], np.full(9, 0.2)]), 20)      # one event per 250 ms
    r = np.concatenate([quiet, np.full(400, 25.0)])
    st = AS.returning_event_stats(r, BIN, hi_bin=200)
    assert st["n_events"] >= 10, st
    assert 0 < st["median_duration_ms"] <= 300.0, st
    assert st["median_peak_hz"] < 25.0, "the contained state must not leak into the IED reference"
