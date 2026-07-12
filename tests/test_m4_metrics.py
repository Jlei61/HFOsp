"""Unit tests for M4 per-cell metric extraction (src.sef_hfo_m4_metrics) + runner import/refuse guard.
Synthetic arrays only — NO simulation is run."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.sef_hfo_m4_metrics import (  # noqa: E402
    onset_gradient_r2, off_axis_fraction, core_overlap_spikes, globality_pr,
    branching_ratio, monotonic_saturation, finite_energy_ok, spatial_self_limited,
    active_mask_post_kick, extract_cell_metrics,
)
from src.sef_hfo_m4_phaseplane import CellMetrics  # noqa: E402


def test_onset_gradient_r2():
    posE = np.random.default_rng(0).uniform(0, 6, size=(60, 2))
    onset = 10.0 + 2.0 * posE[:, 0]                    # perfect x-gradient
    assert onset_gradient_r2(posE, onset) > 0.999
    assert onset_gradient_r2(posE, np.full(60, 5.0)) == 0.0   # simultaneous -> 0
    assert onset_gradient_r2(posE, np.full(60, np.nan)) == 0.0  # nothing activated


def test_off_axis_fraction():
    posE = np.array([[3.0, 3.0], [3.0, 3.5], [3.0, 5.0], [3.0, 0.5]])   # x=3 line; 2 near, 2 far in y
    active = np.ones(4, bool)
    axis = np.array([0.0, 1.0])                          # axis along y through center (3,3)
    # perpendicular distance from a y-axis line = |x - 3| = 0 for all -> none off-axis
    assert off_axis_fraction(posE, active, axis, np.array([3.0, 3.0]), band_half=1.0) == 0.0
    axis_x = np.array([1.0, 0.0])                        # axis along x -> perp dist = |y-3|
    assert off_axis_fraction(posE, active, axis_x, np.array([3.0, 3.0]), band_half=1.0) == 0.5  # 2/4 far
    assert np.isnan(off_axis_fraction(posE, active, None, np.array([3.0, 3.0]), 1.0))


def test_core_overlap_and_globality():
    NE = 10; nsteps = 30
    spk = np.zeros((nsteps, NE), bool)
    spk[10:20, :3] = True                               # only the 3 "core" neurons fire
    core = np.zeros(NE, bool); core[:3] = True
    assert np.isclose(core_overlap_spikes(spk, dt=1.0, t_kick=0.0, core_neuron_mask=core), 1.0)
    assert globality_pr(spk, 1.0, 0.0) < 0.35           # concentrated on 3/10 -> low
    spk_uniform = np.ones((nsteps, NE), bool)
    assert np.isclose(globality_pr(spk_uniform, 1.0, 0.0), 1.0)   # uniform -> 1


def test_branching_and_saturation_and_energy():
    dt = 1.0
    growing = np.concatenate([np.zeros(5), np.array([12., 14., 16., 18., 20.])])
    assert branching_ratio(growing, dt, t_kick=5.0, thresh_hz=10.0) > 1.0
    steady = np.concatenate([np.zeros(5), np.full(10, 15.0)])
    assert abs(branching_ratio(steady, dt, 5.0, 10.0) - 1.0) < 1e-6
    sat = np.concatenate([np.zeros(5), np.linspace(10, 95, 5), np.full(10, 95.0)])
    assert monotonic_saturation(sat, dt, 5.0, sat_ceiling=100.0)          # pinned tail near ceiling
    runaway = np.concatenate([np.zeros(5), np.full(10, 105.0)])           # peak >= ceiling -> not finite
    assert not finite_energy_ok(runaway, dt, 5.0, sat_ceiling=100.0)
    decaying = np.concatenate([np.zeros(5), np.array([50., 40., 20., 5., 2., 2., 2., 2.])])
    assert not monotonic_saturation(decaying, dt, 5.0, sat_ceiling=100.0)
    assert finite_energy_ok(decaying, dt, 5.0, sat_ceiling=100.0)


def test_spatial_self_limited():
    NE = 20; dt = 1.0
    spk = np.zeros((200, NE), bool)
    spk[50:70, :] = True                                 # peak: all fire near t=60
    spk[150:170, :3] = True                              # late: only 3 fire (retreated to core)
    assert spatial_self_limited(spk, dt, t_kick=40.0, peak_t=60.0, late_after=80.0, win=40.0,
                                retreat_factor=0.5)
    spk_sustained = np.zeros((200, NE), bool)
    spk_sustained[50:170, :] = True                      # all keep firing -> no retreat
    assert not spatial_self_limited(spk_sustained, dt, 40.0, 60.0, 80.0, 40.0, 0.5)


def test_extract_cell_metrics_end_to_end():
    # synthetic localized-core event: core fires early, spreads a little, then rate falls (no retreat).
    NE = 40; nsteps = 400; dt = 1.0
    rng = np.random.default_rng(1)
    posE = rng.uniform(0, 6, size=(NE, 2))
    core = posE[:, 0] < 2.0                              # left-edge core
    spk = np.zeros((nsteps, NE), bool)
    spk[60:200, core] = True                             # sustained core activity
    rate = spk.sum(axis=1).astype(float)
    res = {"E_spk_bool": spk, "rate_E": rate}
    m = extract_cell_metrics(res, posE, dt, t_kick=50.0, core_neuron_mask=core,
                             center=posE[core].mean(0), T_min=60.0, band_half=1.0,
                             sat_ceiling=0.5 * NE, thresh_hz=2.0, retreat_factor=0.5)
    assert isinstance(m, CellMetrics)
    assert m.persist                                     # burst_duration long
    assert 0.0 <= m.core_overlap <= 1.0 and m.core_overlap > 0.9   # all spikes from core
    assert 0.0 <= m.globality <= 1.0


def test_extract_uses_per_neuron_hz_for_ceiling():
    # fix P1-2: rate_E is a per-step COUNT; the ceiling is per-neuron Hz. 3 of NE=10 firing each step at
    # dt=1.0 -> 3/10/1*1e3 = 300 Hz. With a 250 Hz ceiling -> finite_energy False. If extract compared the
    # raw COUNT 3 to 250 it would be True -> asserting False proves the Hz conversion is applied.
    NE = 10; nsteps = 60; dt = 1.0
    posE = np.random.default_rng(0).uniform(0, 6, (NE, 2))
    core = np.zeros(NE, bool); core[:2] = True
    spk = np.zeros((nsteps, NE), bool)
    spk[10:40, :3] = True
    res = {"E_spk_bool": spk, "rate_E": spk.sum(axis=1).astype(float)}
    m = extract_cell_metrics(res, posE, dt, t_kick=5.0, core_neuron_mask=core, center=posE[core].mean(0),
                             T_min=5.0, band_half=1.0, sat_ceiling=250.0, thresh_hz=10.0, retreat_factor=0.5)
    assert not m.finite_energy                     # 300 Hz > 250 Hz ceiling -> Hz conversion applied


def test_r50_from_peak_fail_closed():
    # fix P1-1: r50 from the rE_fast TIME-peak trace; fail closed on empty / no-activity traces.
    import importlib
    import pytest
    R = importlib.import_module("scripts.run_m4_phaseplane")
    with pytest.raises(R.CalibrationError):
        R._r50_from_peak([])                        # empty trace -> fail closed
    with pytest.raises(R.CalibrationError):
        R._r50_from_peak([0.0, 0.0])                # peak below R50_MIN_PEAK -> fail closed
    assert abs(R._r50_from_peak([0.5, 1.0, 0.3]) - 1.0 * R.R50_FRAC) < 1e-9   # r50 = R50_FRAC * time-peak


def test_runner_imports_without_running_and_refuses():
    # importing the runner must NOT run any simulation; main() without --confirm-run must refuse.
    import importlib
    R = importlib.import_module("scripts.run_m4_phaseplane")
    assert hasattr(R, "main") and hasattr(R, "sweep") and hasattr(R, "run_cell")
    old = sys.argv
    try:
        sys.argv = ["run_m4_phaseplane.py"]              # no --confirm-run
        R.main()                                         # returns immediately (REFUSED), no sim
    finally:
        sys.argv = old
