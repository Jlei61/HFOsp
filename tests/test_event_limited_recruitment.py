"""ELR layer contract tests (HYB2 plan section 2 + section 11).

The load-bearing one is `test_HYB1_RATCHET_REGRESSION...`: the judge must be able to REPRODUCE
HYB1's failure before it is allowed to pass HYB2.
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import src.snn_engine.event_limited_recruitment as E     # noqa: E402

NG, NV = 8, 64
I_R_MAX = 4.134151260609386


def _cfg(**over):
    b = np.full(NV, 100.0)
    d = dict(b_v=b, eps_s=0.1 * 100.0, tau_R_ms=26.55, Q_on=2.0, Q_scale=2.0, eps_q=0.2,
             I_R_max=I_R_MAX, n_grid=NG, dt_R_ms=0.5)
    d.update(over)
    return E.ELRConfig(**d)


def _layer(N=NV * 4, **over):
    cfg = _cfg(**over)
    voxel = (np.arange(N) % NV).astype(np.int32)
    return E.EventLimitedRecruitment(N, voxel, cfg), cfg


def _drive(lay, load_hz, n_blocks):
    """Drive every occupied voxel at a given per-cell rate for n blocks."""
    per_vox = lay.n_per_voxel[0]
    k = int(round(load_hz * per_vox * lay.cfg.dt_R_ms * 1e-3))
    for _ in range(n_blocks):
        lay._counts[:] = k
        lay.update()


# ------------------------------------------------------------------ strict below-background zero
def test_envelope_source_is_exactly_zero_at_and_below_background():
    lay, cfg = _layer()
    _drive(lay, 100.0, 200)                      # exactly at b_v
    assert lay.q.max() == 0.0


def test_envelope_source_is_exactly_zero_below_background():
    lay, _ = _layer()
    _drive(lay, 40.0, 200)
    assert lay.q.max() == 0.0


def test_actuator_is_exactly_zero_while_q_is_at_or_below_Q_on():
    cfg = _cfg()
    for q in (0.0, 0.5 * cfg.Q_on, cfg.Q_on):
        assert float(E.recruit_current(np.array([q]), cfg)[0]) == 0.0


def test_a_softplus_source_would_break_the_strict_zero():
    leak = 0.1 * np.log1p(np.exp(-60.0 / 10.0))
    assert leak > 0.0
    assert E.deadband_positive(np.array([-60.0]), 10.0)[0] == 0.0


# ------------------------------------------------------------------ bounded amplitude
def test_actuator_is_bounded_by_I_R_max_for_any_sustained_input():
    cfg = _cfg()
    r = E.recruit_current(np.array([1.05 * cfg.Q_on, 1e2, 1e4, 1e8]), cfg)
    assert np.all(r <= cfg.I_R_max) and np.all(np.diff(r) >= 0)
    assert 0.0 < r[0] < cfg.I_R_max                 # just above Q_on: on the rising part
    assert r[-1] == pytest.approx(cfg.I_R_max, rel=1e-12)   # saturates, never exceeds


def test_a_very_long_sustained_drive_reaches_a_PLATEAU_not_an_unbounded_ramp():
    lay, cfg = _layer()
    _drive(lay, 100.0 + 400.0, 20000)            # 10 s of a 400 Hz excess
    a = float(E.recruit_current(lay.q, cfg).max())
    _drive(lay, 100.0 + 400.0, 20000)            # another 10 s
    b = float(E.recruit_current(lay.q, cfg).max())
    assert b <= cfg.I_R_max and (b - a) < 1e-9 * cfg.I_R_max


def test_the_illustrative_gain_at_twice_Q_on_is_the_deadband_corrected_value():
    """Records the number the contract quotes: the deadband subtracts ~eps_q first, so it is
    tanh(0.9091) = 0.7207, NOT tanh(1) = 0.7616."""
    cfg = _cfg(Q_on=1.0, Q_scale=1.0, eps_q=0.1)
    assert float(E.recruit_current(np.array([2.0]), cfg)[0] / cfg.I_R_max) == pytest.approx(
        0.7207, abs=1e-4)


# ------------------------------------------------------------------ exact exponential update
def test_envelope_step_is_the_EXACT_solution_for_piecewise_constant_drive():
    q, e, tau, dt = 0.0, 3.0, 26.55, 0.5
    for k in range(1, 400):
        q = E.envelope_step(q, e, dt_ms=dt, tau_ms=tau)
        want = e * (1.0 - np.exp(-k * dt / tau))
        assert float(q) == pytest.approx(want, rel=1e-12, abs=1e-14)


def test_envelope_step_is_invariant_to_the_block_size_for_constant_drive():
    """dt is not a hidden parameter: halving it reproduces the same trajectory at matched times."""
    for dt in (0.5, 0.25, 0.125):
        q, n = 0.0, int(round(20.0 / dt))
        for _ in range(n):
            q = E.envelope_step(q, 3.0, dt_ms=dt, tau_ms=26.55)
        assert float(q) == pytest.approx(3.0 * (1 - np.exp(-20.0 / 26.55)), rel=1e-12)


def test_envelope_decays_autonomously_with_tau_R_when_the_drive_stops():
    """tau chosen so an integer number of 0.5 ms blocks lands exactly on one time constant."""
    tau = 26.5
    q = 5.0
    for _ in range(int(round(tau / 0.5))):
        q = E.envelope_step(q, 0.0, dt_ms=0.5, tau_ms=tau)
    assert float(q) == pytest.approx(5.0 * np.exp(-1.0), rel=1e-12)


def test_non_positive_tau_fails_closed():
    with pytest.raises(ValueError):
        E.envelope_step(1.0, 0.0, dt_ms=0.5, tau_ms=0.0)


# ------------------------------------------------------------------ no script, no event labels
def test_the_module_never_reads_an_offline_event_label_or_resets():
    src = inspect.getsource(E)
    for banned in ("detect_events", "event_bar", "onsets", "t_off", "reset("):
        assert banned not in src, f"ELR must be causal and autonomous; found {banned!r}"


def test_there_is_no_hard_reset_path_the_state_only_relaxes():
    lay, _ = _layer()
    _drive(lay, 100.0 + 300.0, 60)
    hot = float(lay.q.max())
    lay._counts[:] = 0
    lay.update()
    after = float(lay.q.max())
    assert 0.0 < after < hot
    assert after == pytest.approx(hot * np.exp(-0.5 / 26.55), rel=1e-9)


# ------------------------------------------------------------------ cross-event non-accumulation
def _burst_train(lay, *, excess_hz, burst_ms, gap_ms, n):
    """n bursts of `burst_ms` separated by `gap_ms` of silence; returns q_max just before each
    burst onset (the plan 7.1 clause-2 statistic)."""
    nb, ng = int(round(burst_ms / 0.5)), int(round(gap_ms / 0.5))
    pre = []
    for i in range(n):
        pre.append(float(lay.q.max()))
        _drive(lay, lay.cfg.b_v[0] + excess_hz, nb)
        for _ in range(ng):
            lay._counts[:] = 0
            lay.update()
    return pre


def _calibrated_layer(excess_hz=300.0, burst_ms=22.0, **over):
    """A layer whose Q_on is derived the way the real calibration derives it: 1.10 x the peak q
    of one interictal-scale event.  A fixture with an arbitrary Q_on is not self-consistent and
    tests nothing about the contract."""
    probe, _ = _layer(**over)
    _drive(probe, probe.cfg.b_v[0] + excess_hz, int(round(burst_ms / 0.5)))
    q_pk = float(probe.q.max())
    return _layer(Q_on=1.10 * q_pk, Q_scale=1.10 * q_pk, eps_q=0.10 * 1.10 * q_pk, **over)


def test_no_cross_event_accumulation_at_the_registered_GAP():
    """Two trains separated by GAP_05: q_v before the second is <= 1% of Q_on."""
    lay, cfg = _calibrated_layer()
    pre = _burst_train(lay, excess_hz=300.0, burst_ms=22.0, gap_ms=147.5, n=6)
    assert max(pre[1:]) <= 0.01 * cfg.Q_on


def test_the_B0_residual_clause_is_the_SAME_inequality_as_the_tau_R_rule():
    """Structural fact found while writing these tests, recorded so it is not mistaken for
    evidence: Q_on = 1.10 * (max interictal event peak), so

        q_peak * exp(-GAP/tau_R) <= 0.01 * Q_on   <=>   exp(-GAP/tau_R) <= 0.011

    which is the tau_R selection rule itself (<= 0.01) up to the 1.10 margin.  The residual clause
    therefore passes BY CONSTRUCTION for any event at or below the calibration maximum, and can
    only fail on gaps SHORTER than GAP_05.  It is not independent evidence.
    """
    tau, gap = 26.55, 147.5
    assert np.exp(-gap / tau) <= 0.011
    lay, cfg = _calibrated_layer()
    pre = _burst_train(lay, excess_hz=300.0, burst_ms=22.0, gap_ms=gap, n=4)
    assert max(pre[1:]) / cfg.Q_on == pytest.approx(np.exp(-gap / tau) / 1.10, rel=0.05)
    # ... and it DOES fail on a short gap, which is why GAP_01 must be reported
    lay2, cfg2 = _calibrated_layer()
    pre2 = _burst_train(lay2, excess_hz=300.0, burst_ms=22.0, gap_ms=40.0, n=4)
    assert max(pre2[1:]) > 0.01 * cfg2.Q_on


def test_the_pre_event_floor_does_NOT_ratchet_at_the_registered_tau():
    lay, cfg = _calibrated_layer()
    pre = _burst_train(lay, excess_hz=300.0, burst_ms=22.0, gap_ms=147.5, n=12)
    drift = (pre[-1] - pre[1]) / cfg.Q_on
    assert drift <= 0.01


def test_HYB1_RATCHET_REGRESSION_the_judge_must_reproduce_the_old_failure():
    """Same synthetic input, tau_R put back to HYB1's 654 ms concentration memory.

    plan section 11: a judge that cannot fail HYB1 has no standing to pass HYB2.  The pre-event
    floor must ratchet past the 1%-of-Q_on drift clause.
    """
    lay, cfg = _calibrated_layer(tau_R_ms=654.0)
    pre = _burst_train(lay, excess_hz=300.0, burst_ms=22.0, gap_ms=147.5, n=12)
    drift = (pre[-1] - pre[1]) / cfg.Q_on
    assert drift > 0.01, f"the 654 ms memory must ratchet; got drift {drift:.4g}"


# ------------------------------------------------------------------ membrane coupling
def test_the_current_reaches_BOTH_E_and_I_cells_and_is_never_a_conductance():
    lay, cfg = _layer(N=16)
    lay.q[:] = 10.0
    lay._cur = E.recruit_current(lay.q, cfg)[lay.cell_voxel]

    class _MZ:
        NE = 10
        def membrane_terms(self, n):
            return np.zeros(n), None, None
        def step(self, *a):
            pass
    a = E.ELRMZAdapter(_MZ(), lay)
    drive, g_rel, g_rev = a.membrane_terms(16)
    assert np.all(drive > 0) and drive.size == 16
    assert g_rel is None and g_rev is None


def test_the_open_arm_zeroes_the_CURRENT_but_keeps_the_sensor_state_evolving():
    """t_gate must be a counterfactual sensor tracked identically in both arms (plan 5.2)."""
    on, _ = _layer(enabled=True)
    off, _ = _layer(enabled=False)
    for lay in (on, off):
        _drive(lay, 100.0 + 300.0, 60)
    assert np.array_equal(on.q, off.q)
    assert on.t_gate_block == off.t_gate_block is not None
    assert np.all(off.membrane_current() == 0.0) and np.any(on.membrane_current() > 0.0)


def test_the_adapter_does_not_synthesise_absent_engine_attributes():
    lay, _ = _layer(N=16)

    class _MZ:
        NE = 10
    a = E.ELRMZAdapter(_MZ(), lay)
    assert not hasattr(a, "q_I") and not hasattr(a, "uses_shunt") and not hasattr(a, "nE")


# ------------------------------------------------------------------ bookkeeping / safety
def test_voxel_load_is_a_per_cell_rate_and_empty_voxels_never_produce_a_source():
    N = 8
    lay = E.EventLimitedRecruitment(N, np.zeros(N, np.int32), _cfg())
    lay.accumulate(np.ones(N, bool))
    lay.update()
    assert np.all(lay.q[1:] == 0.0)


def test_active_occupancy_counts_only_occupied_voxels():
    N = 8
    lay = E.EventLimitedRecruitment(N, np.zeros(N, np.int32), _cfg(Q_on=1e-9, Q_scale=1e-9,
                                                                  eps_q=1e-10))
    for _ in range(5):
        lay.accumulate(np.ones(N, bool)); lay.update()
    assert lay.active_occupancy() == pytest.approx(1.0)


def test_t_gate_is_the_first_block_whose_max_q_exceeds_Q_on():
    lay, cfg = _calibrated_layer()
    assert lay.t_gate_ms() is None
    _drive(lay, cfg.b_v[0] + 60.0, 4)                  # too weak: never crosses
    assert lay.t_gate_ms() is None
    _drive(lay, cfg.b_v[0] + 900.0, 400)               # strong: crosses at a LATER block
    t = lay.t_gate_ms()
    assert t is not None and t > 0.0
    assert lay.t_gate_block == int(t / cfg.dt_R_ms)


def test_two_identical_layers_step_identically():
    a, _ = _layer(); b, _ = _layer()
    rng = np.random.default_rng(4)
    for _ in range(80):
        c = rng.integers(0, 60, NV).astype(float)
        for L in (a, b):
            L._counts[:] = c; L.update()
    assert np.array_equal(a.q, b.q)


def test_snapshot_restart_reproduces_the_continuous_run():
    a, _ = _layer(); b, _ = _layer()
    rng = np.random.default_rng(7)
    seq = [rng.integers(0, 60, NV).astype(float) for _ in range(60)]
    for c in seq:
        a._counts[:] = c; a.update()
    for c in seq[:25]:
        b._counts[:] = c; b.update()
    r, _ = _layer(); r.load_state_dict(b.state_dict())
    for c in seq[25:]:
        r._counts[:] = c; r.update()
    assert np.allclose(a.q, r.q, rtol=0, atol=0) and r.n_updates == a.n_updates
    assert r.t_gate_block == a.t_gate_block


def test_out_of_band_q_fails_closed_instead_of_clamping():
    lay, _ = _layer(q_bounds=(-1e-12, 1e-6))
    with pytest.raises(E.ELRSafetyError):
        _drive(lay, 100.0 + 300.0, 20)


def test_mismatched_shapes_fail_closed():
    with pytest.raises(ValueError):
        E.EventLimitedRecruitment(10, np.zeros(9, np.int32), _cfg())
    with pytest.raises(ValueError):
        E.EventLimitedRecruitment(10, np.zeros(10, np.int32), _cfg(b_v=np.full(NV + 1, 1.0)))


def test_there_is_no_diffusion_term_so_a_silent_voxel_stays_silent():
    """Plan 7.2: without diffusion the actuator amplifies already-active tissue only."""
    lay, _ = _layer()
    for _ in range(200):
        lay._counts[:] = 0.0
        lay._counts[0] = 400.0 * lay.n_per_voxel[0] * 0.5e-3
        lay.update()
    assert lay.q[0] > 0.0 and np.all(lay.q[1:] == 0.0)
