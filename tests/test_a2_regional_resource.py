"""M3A-A2 RegionalResource: off-parity, region partition, depletion ODE, bounds, per-core isolation."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from slow_vars import RegionalResource, RegionalResourceConfig  # noqa: E402


def _mk(N=10, NE=8, mode="two_tank", **kw):
    core_mask_E = np.zeros(N, bool); core_mask_E[:3] = True       # first 3 E cells are core
    cfg = RegionalResourceConfig(mode=mode, **kw)
    return RegionalResource(N, 18.0, core_mask_E, cfg, NE=NE), core_mask_E


# ---- Task 1: apply_currents parity + region partition ----
def test_full_tank_is_parity():
    rr, _ = _mk(k_use=0.0)
    I_E = np.arange(10, dtype=float) + 1.0
    I_I = np.arange(10, dtype=float) * 0.5
    out = rr.apply_currents(I_E, I_I, labels=None)
    assert np.array_equal(out, I_E - I_I)                         # q=1 everywhere -> exact


def test_q_global_is_true_global_multiplier():
    rr, _ = _mk(); rr.q_global = 0.5; rr.q_core = 1.0
    I_E = np.ones(10); I_I = np.ones(10)
    out = rr.apply_currents(I_E, I_I, None)
    assert np.allclose(out[:8], 1.0 - 0.5 * 1.0)                  # ALL E (core+surround) scaled by 0.5
    assert np.allclose(out[8:], 1.0 - 1.0)                        # I cells unscaled


def test_q_core_is_core_extra():
    rr, _ = _mk(); rr.q_global = 1.0; rr.q_core = 0.5
    I_E = np.ones(10); I_I = np.ones(10)
    out = rr.apply_currents(I_E, I_I, None)
    assert np.allclose(out[:3], 1.0 - 0.5)                        # core E: q_global*q_core = 0.5
    assert np.allclose(out[3:8], 1.0 - 1.0)                       # surround E: q_global = 1.0
    assert np.allclose(out[8:], 1.0 - 1.0)                        # I cells unscaled


# ---- Task 2: step ODE exactness, bounds, frozen, no-NaN ----
def test_ode_exact_one_step():
    rr, _ = _mk(mode="core_only", k_use=0.002, tau_rec=5000.0, tau_a=100.0)
    dt = 0.1
    spk = np.zeros(10, bool); spk[:3] = True                      # all 3 core E spike -> a_core frac=1.0
    rr.step(spk, labels=None, dt=dt)
    alpha = 1.0 - np.exp(-dt / 100.0)
    ema = alpha * (1.0 - 0.0)
    q_exp = 1.0 + dt * ((1.0 - 1.0) / 5000.0 - 0.002 * ema * 1.0)
    assert abs(rr.q_core - q_exp) < 1e-12
    assert rr.q_global == 1.0                                     # core_only: q_global frozen at 1
    assert abs(rr.trace_a_core[-1] - ema) < 1e-12                 # [P1-3] EMA traced


def test_bounded_floor():
    rr, _ = _mk(mode="core_only", k_use=10.0, q_min=0.25, tau_rec=5000.0, tau_a=1.0)
    spk = np.zeros(10, bool); spk[:3] = True
    for _ in range(2000):
        rr.step(spk, None, 0.1)
    assert 0.25 <= rr.q_core <= 1.0


def test_frozen_holds_q_but_traces():
    rr, _ = _mk(mode="core_only", k_use=5.0, frozen=True, q_core_init=0.6)
    spk = np.ones(10, bool)
    for _ in range(100):
        rr.step(spk, None, 0.1)
    assert rr.q_core == 0.6 and rr.q_global == 1.0               # [P0-2] q held
    assert len(rr.trace_core) == 100 and rr.trace_core[-1] == 0.6 # [P0-2] q traced
    assert len(rr.trace_a_core) == 100 and rr.trace_a_core[-1] > 0.0  # [P1-3] activity still traced


def test_no_nan_long():
    rr, _ = _mk(mode="two_tank", k_use=0.003)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        spk = rng.random(10) < 0.1
        rr.step(spk, None, 0.1)
        assert np.isfinite(rr.q_core) and np.isfinite(rr.q_global)


# ---- Task 3: per-core depletion isolation ----
def test_per_core_isolation():
    N, NE = 12, 10
    core_mask_E = np.zeros(N, bool); core_mask_E[:4] = True       # cores = E cells 0..3
    left = np.zeros(N, bool); left[:2] = True                     # left core = E 0,1
    right = np.zeros(N, bool); right[2:4] = True                  # right core = E 2,3
    cfg = RegionalResourceConfig(mode="per_core", k_use=0.05, tau_a=1.0, tau_rec=1e9)
    rr = RegionalResource(N, 18.0, core_mask_E, cfg, NE=NE, left_core_E=left, right_core_E=right)
    spk = np.zeros(N, bool); spk[:2] = True                       # ONLY left core fires
    for _ in range(500):
        rr.step(spk, None, 0.1)
    assert rr.q_L < 0.99                                          # left depletes
    assert rr.q_R == 1.0                                          # right untouched (no right activity)
    assert rr.q_global < 1.0                                      # global sees the (left) activity


# ---- optional g_K (sAHP) recovery term: off-by-default parity + builds/decays ----
def test_gk_off_by_default_parity():
    rr, _ = _mk(k_use=0.0)                                        # gk_max default 0
    spk = np.zeros(10, bool); spk[:3] = True
    rr.step(spk, None, 0.1)                                       # would build gK if gk_max>0
    I_E = np.arange(10, dtype=float) + 1.0; I_I = np.arange(10, dtype=float) * 0.5
    assert np.array_equal(rr.apply_currents(I_E, I_I, None), I_E - I_I)   # gK==0 -> parity holds


def test_gk_builds_subtracts_decays():
    rr, _ = _mk(mode="core_only", k_use=0.0, gk_max=0.5, tau_k=5000.0)
    spk = np.zeros(10, bool); spk[:3] = True                     # E cells 0,1,2 spike
    rr.step(spk, None, 0.1)
    assert rr.gK[0] > 0 and rr.gK[5] == 0                        # builds on spiking E, not on quiet E
    out = rr.apply_currents(np.ones(10), np.zeros(10), None)
    assert abs(out[0] - (1.0 - rr.gK[0])) < 1e-12               # gK subtracts from I_net
    g0 = float(rr.gK[0])
    for _ in range(200):
        rr.step(np.zeros(10, bool), None, 0.1)                   # quiet -> decay
    assert rr.gK[0] < g0 and len(rr.trace_gk) == 201
