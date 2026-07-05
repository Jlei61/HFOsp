"""Contract unit tests for the M3A-A1 runner's pure helpers (no simulation).

Locks the two load-bearing science-contract clauses (CLAUDE.md §6):
  - inactive slow variables are reported as 'NA', NEVER 0 (plan §4);
  - the z/φ/g_K slow= path and the e_GABA membrane-shunt path are kept SEPARATE
    (slow path: slow set, shunt off, V_th_per_neuron bypassed; e_GABA path: slow None,
    shunt on, V_th_per_neuron honored) — the runner never combines them.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_m3a_quasistatic_slowvars import (_frozen_slow_scalar, _slow_state_fields,  # noqa: E402
                                          _build_slow_membrane, _SLOW_KEYS, _LANDMARKS)


def test_off_slow_state_all_NA():
    f = _slow_state_fields("off", 1.0, 18.0)
    assert all(v == "NA" for v in f.values())
    assert len(f) == len(_SLOW_KEYS) * len(_LANDMARKS)


def test_z_active_others_NA_not_zero():
    f = _slow_state_fields("z", 0.9, 18.0)
    assert f["z_pre"] == f["z_onset"] == f["z_peak"] == f["z_end"] == 0.9
    for k in ("phi", "gK", "e_gaba"):
        for lm in _LANDMARKS:
            assert f[f"{k}_{lm}"] == "NA"          # NA, never 0


def test_egaba_active_others_NA():
    f = _slow_state_fields("egaba", 14.0, 18.0)
    assert all(f[f"e_gaba_{lm}"] == 14.0 for lm in _LANDMARKS)
    for k in ("z", "phi", "gK"):
        assert all(f[f"{k}_{lm}"] == "NA" for lm in _LANDMARKS)


def test_phi_recorded_as_absolute_threshold():
    # phi offset +2 on vth0=18 -> absolute adaptive threshold 20.
    assert _frozen_slow_scalar("phi", 2.0, 18.0) == ("phi", 20.0)
    f = _slow_state_fields("phi", 2.0, 18.0)
    assert all(f[f"phi_{lm}"] == 20.0 for lm in _LANDMARKS)


def test_build_membrane_z_path():
    slow, vth_pn, shunt, e_gaba, g = _build_slow_membrane("z", 0.8, 50, 18.0, 1.0)
    assert slow is not None and vth_pn is None        # slow= path bypasses V_th_per_neuron
    assert shunt is False and e_gaba is None           # shunt OFF on the slow path


def test_build_membrane_egaba_path():
    slow, vth_pn, shunt, e_gaba, g = _build_slow_membrane("egaba", 14.0, 50, 18.0, 1.0)
    assert slow is None                                # e_GABA uses membrane shunt, not slow=
    assert shunt is True and e_gaba == 14.0 and g == 1.0
    np.testing.assert_array_equal(vth_pn, np.full(50, 18.0))   # V_th_per_neuron honored here


def test_build_membrane_off_path():
    slow, vth_pn, shunt, e_gaba, g = _build_slow_membrane("off", 1.0, 50, 18.0, 1.0)
    assert slow is None and shunt is False
    np.testing.assert_array_equal(vth_pn, np.full(50, 18.0))
