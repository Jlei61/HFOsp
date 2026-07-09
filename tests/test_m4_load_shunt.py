import numpy as np
import pytest
from src.sef_hfo_m4_load_shunt import LoadShuntParams, hill_pi, load_shunt_step


def _p(**kw):
    base = dict(tau_n=20000.0, k_n=1.0, rho_n=0.0, n_base=0.0, n50=0.5,
                hill_h=2.0, a_max=1.0, u_n0=0.0, n_min=0.0, n_max=10.0)
    base.update(kw)
    return LoadShuntParams(**base)


def test_hill_pi_monotone_and_bounded():
    p = _p()
    xs = np.array([0.0, 0.25, 0.5, 1.0, 4.0])
    pi = hill_pi(xs, p)
    assert np.all(np.diff(pi) > 0)          # strictly increasing in n
    assert pi[0] == 0.0                       # at n_base -> 0
    assert np.all(pi < 1.0) and pi[-1] > 0.9  # saturates toward 1


def test_baseline_center_rectifies_subthreshold_drive():
    # u_n below the set-point u_n0 must not build load (rectified to 0)
    p = _p(u_n0=0.3, k_n=5.0)
    n, a = load_shunt_step(np.array(0.0), np.array(0.2), dt=1.0, p=p)
    assert n == 0.0 and a == 0.0              # drive 0.2 < u_n0 0.3 -> no build, decays/stays at n_base


def test_quiet_baseline_decays_to_n_base_and_a_zero():
    p = _p(n_base=0.0, u_n0=0.0)
    n = np.array(2.0)                          # start elevated
    for _ in range(200000):                    # many ms of zero drive
        n, a = load_shunt_step(n, np.array(0.0), dt=1.0, p=p)
    assert n == pytest.approx(0.0, abs=1e-3)
    assert a == pytest.approx(0.0, abs=1e-3)


def test_sustained_drive_accumulates_load_and_shunt():
    p = _p(u_n0=0.0, k_n=1.0, tau_n=20000.0)
    n = np.array(0.0)
    for _ in range(5000):                       # 5 s of sustained drive
        n, a = load_shunt_step(n, np.array(1.0), dt=1.0, p=p)
    assert n > 0.3 and a > 0.1                  # load and shunt rose measurably


def test_clamps_hold():
    p = _p(k_n=1e6, n_max=3.0, a_max=0.8)
    n, a = load_shunt_step(np.array(0.0), np.array(1.0), dt=1.0, p=p)
    assert n <= 3.0 and 0.0 <= a <= 0.8


def test_validate_rejects_bad_params():
    with pytest.raises(ValueError):
        _p(tau_n=0.0).validate()
    with pytest.raises(ValueError):
        _p(n50=0.0).validate()


def test_a_uses_updated_n_not_pre_step_n():
    # a = a_max * Pi(n_new), NOT Pi(pre-step n). One step moves n 0 -> 0.8-ish; Pi(0)=0 discriminates.
    p = _p(k_n=1.0, rho_n=0.0, tau_n=1e9)          # huge tau -> leak negligible this step
    n0 = np.array(0.0)
    n_new, a = load_shunt_step(n0, np.array(1.0), dt=1.0, p=p)
    assert a == pytest.approx(p.a_max * hill_pi(n_new, p))   # uses the UPDATED n
    assert a != pytest.approx(p.a_max * hill_pi(n0, p))      # and NOT the pre-step n (which gives 0)


def test_subthreshold_drive_does_not_reduce_load_below_base():
    # drive below u_n0 must rectify to 0 (no build AND no spurious negative pull). n starts at n_base,
    # away from n_min, so an unrectified negative dn would visibly drop n (not be masked by the clamp).
    p = _p(k_n=1.0, rho_n=0.0, n_base=2.0, u_n0=0.5, n_min=0.0)
    n_new, a = load_shunt_step(np.array(2.0), np.array(0.2), dt=1.0, p=p)   # 0.2 < u_n0=0.5
    assert n_new == pytest.approx(2.0)             # rectified -> stays at n_base (unrectified would give 1.7)


def test_rho_n_consumption_decays_load_faster_than_leak():
    # the -rho_n*Pi(n) term must subtract extra decay beyond the bare leak (verifies it is wired, right sign).
    n0 = 1.0
    p_leak = _p(k_n=0.0, rho_n=0.0, tau_n=5000.0, n_base=0.0)
    p_cons = _p(k_n=0.0, rho_n=0.5, tau_n=5000.0, n_base=0.0)
    n_leak, _ = load_shunt_step(np.array(n0), np.array(0.0), 1.0, p_leak)
    n_cons, _ = load_shunt_step(np.array(n0), np.array(0.0), 1.0, p_cons)
    assert n_cons < n_leak                          # consumption decays load faster than leak alone


def test_load_shunt_step_elementwise_on_1d_array():
    # the interface promises elementwise behavior on 1D/2D arrays (used later by SpatialSlowField).
    p = _p(k_n=1.0, rho_n=0.0)
    n = np.array([0.0, 5.0, 9.9]); u = np.array([1.0, 0.0, 1.0])
    n_new, a = load_shunt_step(n, u, dt=1.0, p=p)
    assert n_new.shape == (3,) and a.shape == (3,)
    assert np.all((n_new >= p.n_min) & (n_new <= p.n_max))   # all clamped in range


# Task 2: a-response metrics (Δa_IED / R_A)
from src.sef_hfo_m4_load_shunt import event_triggered_a_response, compute_R_A


def test_event_triggered_a_response_positive_bump():
    dt = 1.0
    a = np.zeros(1000)
    a[500:600] = 0.4                            # a bump right after t=500
    delta = event_triggered_a_response(a, [500], dt, pre_ms=100, post0_ms=10, post1_ms=90)
    assert delta == pytest.approx(0.4, abs=1e-9)


def test_event_triggered_skips_events_without_full_window():
    dt = 1.0
    a = np.zeros(200)
    # event at 10 has no room for pre=100 -> skipped; event at 100 is fine
    a[100:150] = 0.2
    delta = event_triggered_a_response(a, [10, 100], dt, pre_ms=100, post0_ms=0, post1_ms=50)
    assert delta == pytest.approx(0.2, abs=1e-9)


def test_event_triggered_raises_when_no_usable_event():
    a = np.zeros(50)
    with pytest.raises(ValueError):
        event_triggered_a_response(a, [10], 1.0, pre_ms=100, post0_ms=0, post1_ms=50)


def test_R_A_ratio_and_soft_gate_flag():
    assert compute_R_A(0.5, 0.05) == pytest.approx(10.0)
    assert compute_R_A(0.5, 0.0) == float("inf")     # IED did not move a -> soft ictal gate
