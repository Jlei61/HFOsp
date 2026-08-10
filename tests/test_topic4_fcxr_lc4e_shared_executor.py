from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_topic4_fcxr_lc4_lifecycle as LIFE  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402


def _mk(mode="local"):
    cfg = MZSlowVarsConfig(
        use_m=True, tau_adp=1000.0, eta_m=0.0,
        membrane_mode="full_conductance", m_hill_K=2.0,
        m_hill_deadzone=1.0, m_hill_n=4.0,
        tau_a_on=100.0, tau_a_off=10000.0, g_m_max=20.0,
        m_hill_spatial_mode=mode,
    )
    return MZSlowVars(6, 18.0, cfg, NE=4, core_mask_E=np.zeros(4, bool))


def test_local_mode_is_the_literal_historical_current():
    slow = _mk("local")
    slow.a[:4] = [0.0, 0.2, 0.5, 1.0]
    assert np.array_equal(slow._cooperative_current_E(), 20.0 * slow.a[:4])


def test_shared_mode_is_uniform_and_mean_dose_matched():
    local = _mk("local")
    shared = _mk("shared")
    field = np.asarray([0.0, 0.2, 0.5, 1.0])
    local.a[:4] = field
    shared.a[:4] = field
    i_local = local._cooperative_current_E()
    i_shared = shared._cooperative_current_E()
    assert np.all(i_shared == i_shared[0])
    assert float(i_shared[0]) == float(i_local.mean())
    assert float(i_shared.mean()) == pytest.approx(float(i_local.mean()), abs=1e-15)


def test_exact_deadzone_zero_stays_exactly_zero_in_both_modes():
    for mode in ("local", "shared"):
        slow = _mk(mode)
        slow.a[:4] = 0.0
        assert np.array_equal(slow._cooperative_current_E(), np.zeros(4))


def test_shared_current_reaches_only_the_E_conductance_path():
    slow = _mk("shared")
    slow.a[:4] = [0.0, 0.2, 0.5, 1.0]
    I_E = np.zeros(6)
    I_I = np.zeros(6)
    drive, g_rel, _ = slow.membrane_terms(I_E, I_I, I_E_rec=np.zeros(6))
    assert np.all(g_rel[:4] > 0.0)
    assert np.all(g_rel[:4] == g_rel[0])
    assert np.array_equal(g_rel[4:], np.zeros(2))
    assert np.array_equal(drive[4:], np.zeros(2))


def test_invalid_spatial_mode_fails_loudly():
    with pytest.raises(ValueError, match="m_hill_spatial_mode"):
        _mk("regional_magic")


def test_lifecycle_adapter_defaults_old_candidates_to_local_and_threads_shared():
    c = dict(tau_adp_ms=1000.0, K=2.0, n=4.0, tau_a_on_ms=100.0,
             tau_a_off_ms=10000.0, g_m_max=20.0)
    assert LIFE._cfg(c)["m_hill_spatial_mode"] == "local"
    assert LIFE._cfg(dict(c, m_hill_spatial_mode="shared"))["m_hill_spatial_mode"] == "shared"
