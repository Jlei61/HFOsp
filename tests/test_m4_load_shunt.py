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
