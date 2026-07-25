"""FCXR-HEO3 TDD — joint sliding-window gate + source-space audit (HEO2.1 review P1-b/P1-c closeout).

The point of these instruments is to REJECT things HEO2.1's whole-window summaries would have accepted:
an on-off burst train must not read as 'phase-dispersed', and one loud core seen by every contact must
not read as whole-tissue recruitment. Each test encodes one of those bad-data regressions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.topic4_mz_fcxr_heo3 as H3  # noqa: E402

FS = 1000.0


def _sines(n, f, C=15, phases=None, fs=FS):
    t = np.arange(n) / fs
    ph = np.zeros(C) if phases is None else np.asarray(phases, float)
    return np.stack([np.sin(2 * np.pi * f * t + ph[c]) for c in range(C)], axis=1)


# ============================== per-window spectral field ==============================
def test_band_power_windows_short_window_resolves_more_windows_than_1s():
    x = _sines(4000, 16.0)
    bp250, t250 = H3.band_power_windows(x, FS, win_ms=250.0, hop_ms=50.0)
    assert bp250.shape[1:] == (15, 6)
    assert bp250.shape[0] > 40                       # ~4s at 50ms hop -> many windows (1s win gives ~31)
    assert np.all(np.diff(t250) > 0)
    # a 16 Hz tone must put its power in the 13-30 band (index 3), not the 1-4 band
    assert bp250[:, 0, 3].mean() > 50 * bp250[:, 0, 0].mean()


# ============================== phase order parameter ==============================
def test_order_parameter_high_for_in_phase_low_for_scattered():
    n = 4000
    sync = _sines(n, 16.0)                                            # all contacts in phase
    rng = np.random.default_rng(0)
    scat = _sines(n, 16.0, phases=rng.uniform(0, 2 * np.pi, 15))      # phases scattered
    R_sync = H3.phase_order_parameter(sync, FS)[500:-500].mean()
    R_scat = H3.phase_order_parameter(scat, FS)[500:-500].mean()
    assert R_sync > 0.95
    assert R_scat < 0.6
    assert R_sync > R_scat


def test_order_parameter_traveling_wave_is_partially_dispersed():
    """A phase-ramped travelling wave (HEO1's ~184° span) is NOT fully synchronous."""
    n = 4000
    ramp = np.linspace(0.0, np.pi, 15)
    R = H3.phase_order_parameter(_sines(n, 16.0, phases=ramp), FS)[500:-500].mean()
    assert 0.4 < R < 0.95


# ============================== the joint gate (bad-data regressions) ==============================
def _gate(bp, ref, rate, R, **kw):
    return H3.joint_target_windows(bp, ref, rate, R, fs_win=20.0, **kw)


def test_joint_gate_requires_all_four_simultaneously():
    nw, C, B = 50, 15, 6
    ref = np.ones((C, B))
    hot = np.full((nw, C, B), 10.0)                    # all bands 10x baseline -> +10 dB
    rate = np.full(nw, 100.0); R = np.full(nw, 0.5)
    assert _gate(hot, ref, rate, R)["frac_target"] == 1.0
    # drop each criterion in turn -> target must fail
    assert _gate(hot, ref, np.full(nw, 5.0), R)["frac_target"] == 0.0          # low energy
    assert _gate(hot, ref, rate, np.full(nw, 0.99))["frac_target"] == 0.0      # fully synchronous
    narrow = np.ones((nw, C, B)); narrow[:, :, 3] = 10.0                       # only beta up
    assert _gate(narrow, ref, rate, R)["frac_target"] == 0.0                   # not broadband
    few = np.ones((nw, C, B)); few[:, :5, :] = 10.0                            # only 5 contacts up
    assert _gate(few, ref, rate, R)["frac_target"] == 0.0                      # not recruited


def test_joint_gate_longest_run_measures_persistence_not_total():
    """Two regimes with the SAME total target fraction but different persistence must score
    differently — this is the whole point of HEO3 (sustain, not merely reach)."""
    nw, C, B = 40, 15, 6
    ref = np.ones((C, B)); hot = np.full((C, B), 10.0)
    def _mk(mask):
        bp = np.ones((nw, C, B)); bp[mask] = hot
        rate = np.where(mask, 100.0, 1.0); R = np.where(mask, 0.5, 0.99)
        return _gate(bp, ref, rate, R)
    flicker = np.zeros(nw, bool); flicker[::2] = True          # 20 scattered target windows
    block = np.zeros(nw, bool); block[:20] = True              # 20 consecutive target windows
    a, b = _mk(flicker), _mk(block)
    assert abs(a["frac_target"] - b["frac_target"]) < 1e-9     # same total
    assert a["longest_run_ms"] < b["longest_run_ms"] / 5       # very different persistence
    assert b["longest_run_ms"] == 1000.0                       # 20 windows @ 20/s = 1000 ms


def test_joint_gate_reports_which_criterion_blocks():
    nw, C, B = 30, 15, 6
    ref = np.ones((C, B)); narrow = np.ones((nw, C, B)); narrow[:, :, 3] = 10.0
    out = _gate(narrow, ref, np.full(nw, 100.0), np.full(nw, 0.5))
    assert out["frac_by_criterion"]["recruited"] == 1.0        # beta alone still counts as recruited
    assert out["frac_by_criterion"]["broadband"] == 0.0        # but not as broadband
    assert out["frac_by_criterion"]["high_energy"] == 1.0
    assert out["frac_by_criterion"]["dispersed"] == 1.0


# ============================== source-space audit (bad-data regressions) ==============================
def _grid(n=20):
    xs, ys = np.meshgrid(np.linspace(0, 19, n), np.linspace(0, 19, n))
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def test_build_regions_partitions_every_cell_exactly_once():
    pos = _grid()
    reg = H3.build_regions(pos, [2.0, 10.0], [17.0, 10.0], core_r=3.0)
    masks = [reg[k] for k in ("core_source", "core_sink", "axis_corridor", "off_axis")]
    stacked = np.stack(masks)
    assert np.all(stacked.sum(axis=0) == 1)                    # a partition: exactly one region per cell
    assert reg["core_source"].sum() > 0 and reg["core_sink"].sum() > 0
    assert reg["axis_corridor"].sum() > 0 and reg["off_axis"].sum() > 0
    assert 0.0 <= reg["axis_coord"].min() and reg["axis_coord"].max() <= 1.0


def test_participation_ratio_separates_one_loud_core_from_uniform():
    """THE P1-b regression: 15/15 sensor coverage can hide 'one core is loud'. PR must expose it.
    The load-bearing case is a loud core PLUS weak activity everywhere — there the active FRACTION is
    1.0 (so any active-fraction-style metric says 'whole tissue recruited') while the activity is in
    fact concentrated in 5% of cells. PR must report the concentration, not the active count."""
    N = 400
    uniform = np.ones(N)
    one_core = np.zeros(N); one_core[:20] = 1.0                # 5% of cells carry everything
    loud_core_weak_rest = np.full(N, 0.01); loud_core_weak_rest[:20] = 1.0   # EVERY cell is active
    assert H3.participation_ratio(uniform) == 1.0
    assert abs(H3.participation_ratio(one_core) - 0.05) < 1e-9
    assert H3.participation_ratio(one_core) < H3.participation_ratio(uniform)
    assert float((loud_core_weak_rest > 0).mean()) == 1.0      # active fraction is blind here ...
    assert H3.participation_ratio(loud_core_weak_rest) < 0.15  # ... but PR exposes the concentration
    assert H3.participation_ratio(np.zeros(N)) == 0.0          # silence -> 0, not nan


def test_source_space_audit_detects_core_only_vs_whole_field():
    pos = _grid(20)                                            # NE=400
    reg = H3.build_regions(pos, [2.0, 10.0], [17.0, 10.0], core_r=3.0)
    nsteps = 400                                               # dt=1ms -> 400ms
    core_only = np.zeros((nsteps, len(pos)), bool)
    core_only[::4, reg["core_source"]] = True                  # only the source core fires
    whole = np.zeros((nsteps, len(pos)), bool)
    whole[::4, :] = True                                       # everything fires
    a = H3.source_space_audit(core_only, pos, reg, dt=1.0, win_ms=200.0, hop_ms=100.0)[0]
    b = H3.source_space_audit(whole, pos, reg, dt=1.0, win_ms=200.0, hop_ms=100.0)[0]
    assert a["participation_ratio"] < 0.2                      # one core -> low PR ...
    assert b["participation_ratio"] > 0.95                     # ... whole field -> ~1
    assert a["rate_off_axis"] == 0.0 and a["rate_core_source"] > 0
    assert b["rate_off_axis"] > 0
    assert a["centroid_axis_coord"] < 0.2                      # centroid sits at the source end
