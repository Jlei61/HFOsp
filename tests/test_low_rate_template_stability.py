"""TDD for low-rate-window template stability (the GOOD question, scale-relocated).

Question: in low-event (quiet) time windows where the firing-COUNT ranking gets jittery,
does the propagation TEMPLATE (source->sink axis) still reproduce the full-recording
template better than count reproduces its own full ranking? Universe = all lagPat channels
(NOT SOZ-restricted). Reversal handled by flipping reverse-template events onto a common
axis via global per-event labels (NO per-window winner pick). See task spec 2026-06-07.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.low_rate_template_stability import (
    align_template_events,
    window_reproductions,
    stratify_by_event_count,
    count_matched_null_gap,
    m_bucket,
)


# ---------- LR-1: reversal-aware axis alignment ----------

def test_align_flips_reverse_template_onto_common_axis():
    # template A (events 0,1): ch0 source; template B (events 2,3): reversed (ch0 sink)
    masked = np.array([[0.00, 0.00, 1.00, 1.00],
                       [0.33, 0.33, 0.66, 0.66],
                       [0.66, 0.66, 0.33, 0.33],
                       [1.00, 1.00, 0.00, 0.00]])
    labels = np.array([0, 0, 1, 1])
    aligned, meta = align_template_events(masked, labels)
    full_axis = np.nanmean(aligned, axis=1)
    assert meta["reversed"] is True
    assert full_axis[0] == pytest.approx(0.0)        # ch0 source after alignment (NOT blurred to 0.5)
    assert full_axis[0] < full_axis[3]               # clean source->sink gradient
    # un-aligned mean WOULD be blurred to ~0.5 (the failure mode we're avoiding)
    assert np.nanmean(masked, axis=1)[0] == pytest.approx(0.5)


def test_align_no_flip_when_single_direction():
    masked = np.array([[0.0, 0.0, 0.0],
                       [0.5, 0.5, 0.5],
                       [1.0, 1.0, 1.0]])
    labels = np.array([0, 0, 1])      # both clusters same direction (corr > 0)
    aligned, meta = align_template_events(masked, labels)
    assert meta["reversed"] is False
    assert np.allclose(aligned, masked, equal_nan=True)


# ---------- LR-2: window reproductions + stratification ----------

def test_window_reproductions_template_and_rate_to_full():
    # 5 channels, stable axis; window = subset of events
    n_ev = 20
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, 5)[:, None]                       # ch0 source ... ch4 sink
    masked = np.clip(base + rng.normal(0, 0.05, (5, n_ev)), 0, 1)
    # per-channel participation rate varies -> count ranking has variance (Spearman defined)
    bools = rng.random((5, n_ev)) < np.linspace(0.5, 0.95, 5)[:, None]
    aligned, _ = align_template_events(masked, np.zeros(n_ev, dtype=int))
    full_axis = np.nanmean(aligned, axis=1)
    full_count = bools.sum(axis=1).astype(float)
    rep = window_reproductions(aligned, bools, full_axis, full_count,
                               window_ev=np.arange(10), min_ch=3)
    assert rep["template_repro"] > 0.8        # stable axis reproduces well
    assert np.isfinite(rep["rate_repro"])


def test_window_reproductions_rate_on_COMMON_channels_not_penalized_by_silent_zeros():
    # ch3 is silent in the window (no propagation rank -> template drops it). The FAIR rate_repro
    # must also drop it (not count it as a tied 0), so rate is not over-penalized vs template.
    aligned = np.array([[0.0] * 6, [0.4] * 6, [0.8] * 6, [np.nan] * 6])   # ch3 NaN (silent) in window
    bools = np.zeros((4, 6), dtype=bool)
    bools[0, :] = True; bools[1, :4] = True; bools[2, :2] = True          # ch3 never participates
    full_axis = np.array([0.0, 0.4, 0.8, 1.0])                            # ch3 HAS a full-recording rank
    full_count = np.array([6.0, 4.0, 2.0, 20.0])                          # ch3 busy overall, silent here
    rep = window_reproductions(aligned, bools, full_axis, full_count, np.arange(6), min_ch=3)
    # FAIR (common channels ch0,1,2): both rankings are [6,4,2] vs [6,4,2] -> perfect
    assert rep["template_repro"] == pytest.approx(1.0)
    assert rep["rate_repro"] == pytest.approx(1.0)
    # the OLD all-channel rate (ch3 as 0 vs full 20) is reported separately and is strictly worse
    assert rep["rate_repro_allch"] < rep["rate_repro"]


def test_stratify_by_event_count_low_mid_high():
    counts = [10, 12, 50, 55, 200, 210]       # 6 windows
    strata = stratify_by_event_count(counts, n_strata=3)
    assert strata[0] == "low" and strata[1] == "low"
    assert strata[4] == "high" and strata[5] == "high"


def test_m_bucket_edges():
    assert m_bucket(2) == "<=2(unresolvable)"
    assert m_bucket(4) == "3-4"
    assert m_bucket(20) == "5-20"
    assert m_bucket(500) == ">100"


def test_count_matched_null_gap_finite_and_small_for_stable_axis():
    # stable axis + per-channel-varied count -> null gap finite, modest (estimator-smoothness floor)
    rng = np.random.default_rng(0)
    n_ev = 200
    base = np.linspace(0, 1, 6)[:, None]
    masked = np.clip(base + rng.normal(0, 0.05, (6, n_ev)), 0, 1)
    bools = rng.random((6, n_ev)) < np.linspace(0.5, 0.95, 6)[:, None]
    aligned, _ = align_template_events(masked, np.zeros(n_ev, dtype=int))
    full_axis = np.array([np.nanmean(r) for r in aligned])
    full_count = bools.sum(axis=1).astype(float)
    g = count_matched_null_gap(aligned, bools, full_axis, full_count, m=30, n_ev=n_ev,
                               rng=np.random.default_rng(1), n_null=50, min_ch=3)
    assert np.isfinite(g)
    assert abs(g) < 0.5    # estimator-smoothness floor is modest, not a huge gap
