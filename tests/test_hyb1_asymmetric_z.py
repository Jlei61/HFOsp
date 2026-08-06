"""Asymmetric Z recovery (plan section 4).  Off-by-default byte parity is the load-bearing test."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig       # noqa: E402

NE, NI, DT = 40, 10, 0.05
N = NE + NI


def _mk(**over):
    d = dict(use_z=True, tau_z=5000.0, I_th_EI=5.0)
    d.update(over)
    s = MZSlowVars(N, 18.0, MZSlowVarsConfig(**d), NE=NE)
    return s


def _drive(s, sensor, steps):
    """Drive the Z sensor directly: this isolates the kinetics from the network."""
    for _ in range(steps):
        s._z_sensor_last_E = np.full(NE, float(sensor))
        s.step(np.zeros(N, bool), None, DT)
    return s.z[s.is_E].copy()


def test_off_by_default_is_byte_identical_to_the_symmetric_tau():
    a = _drive(_mk(), 9.0, 400)
    b = _drive(_mk(tau_z_down=None, tau_z_up=None), 9.0, 400)
    assert np.array_equal(a, b)


def test_equal_taus_reduce_exactly_to_the_symmetric_branch():
    a = _drive(_mk(), 9.0, 400)
    b = _drive(_mk(tau_z_down=5000.0, tau_z_up=5000.0), 9.0, 400)
    assert np.allclose(a, b, rtol=0, atol=0)


def test_high_load_depletes_on_tau_z_down():
    """Sensor above threshold -> z_inf = 0 < z -> the DOWN tau selects."""
    fast = _drive(_mk(tau_z_down=500.0, tau_z_up=50000.0), 9.0, 2000)
    slow = _drive(_mk(tau_z_down=5000.0, tau_z_up=50000.0), 9.0, 2000)
    assert fast.mean() < slow.mean()


def test_recovery_after_the_load_falls_uses_tau_z_up():
    """Sensor back below threshold -> z_inf = 1 >= z -> the UP tau selects."""
    out = {}
    for tu in (1000.0, 40000.0):
        s = _mk(tau_z_down=500.0, tau_z_up=tu)
        _drive(s, 9.0, 4000)                       # deplete under load
        out[tu] = _drive(s, 0.0, 4000).mean()      # then release the load
    assert out[1000.0] > out[40000.0]


def test_recovery_is_gradual_not_an_instantaneous_reset():
    s = _mk(tau_z_down=500.0, tau_z_up=20000.0)
    _drive(s, 9.0, 6000)
    depleted = s.z[s.is_E].mean()
    after_one = _drive(s, 0.0, 1).mean()
    assert after_one > depleted
    assert after_one - depleted < 0.01 * (1.0 - depleted)      # far from a jump to z=1


def test_z_stays_inside_the_unit_interval_under_extreme_taus():
    z = _drive(_mk(tau_z_down=1e-3, tau_z_up=1e-3), 9.0, 50)
    assert np.all(z >= 0.0) and np.all(z <= 1.0)


def test_one_sided_configuration_fails_closed():
    for kw in (dict(tau_z_down=500.0), dict(tau_z_up=500.0)):
        with pytest.raises(ValueError):
            MZSlowVars(N, 18.0, MZSlowVarsConfig(use_z=True, tau_z=5000.0, I_th_EI=5.0, **kw),
                       NE=NE)


def test_non_positive_taus_fail_closed():
    with pytest.raises(ValueError):
        MZSlowVars(N, 18.0, MZSlowVarsConfig(use_z=True, tau_z=5000.0, I_th_EI=5.0,
                                             tau_z_down=0.0, tau_z_up=1.0), NE=NE)


def test_asymmetric_z_is_deterministic():
    a = _drive(_mk(tau_z_down=700.0, tau_z_up=9000.0), 9.0, 500)
    b = _drive(_mk(tau_z_down=700.0, tau_z_up=9000.0), 9.0, 500)
    assert np.array_equal(a, b)


def test_snapshot_restart_reproduces_a_continuous_asymmetric_run():
    cont = _mk(tau_z_down=700.0, tau_z_up=9000.0)
    _drive(cont, 9.0, 300); _drive(cont, 0.0, 300)
    half = _mk(tau_z_down=700.0, tau_z_up=9000.0)
    _drive(half, 9.0, 300)
    state = half.z.copy()
    resumed = _mk(tau_z_down=700.0, tau_z_up=9000.0)
    resumed.z[:] = state
    _drive(resumed, 0.0, 300)
    assert np.allclose(cont.z, resumed.z, rtol=0, atol=0)


def test_the_selector_is_the_load_indicator_not_an_explicit_X_dependency():
    """Plan section 4 records this as a deliberate deviation: Z must not read X's state, or the
    responsibility separation the sprint tests would already be broken in the implementation."""
    import inspect
    src = inspect.getsource(MZSlowVars.step)
    i = src.index("tau_z_down is None and c.tau_z_up is None")
    seg = src[i:i + 900]
    assert "z_inf_E < zE" in seg
    assert "x_relay" not in seg and "y_" not in seg
