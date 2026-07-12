"""TDD for M4-3A n(x,t) load -> a(x,t) shunt field added to SpatialSlowField (Task 4).

`n_load` is the activity-load field (elementwise ODE from src.sef_hfo_m4_load_shunt,
Task 1); `a_shunt = a_max*Pi(n_load)` is the shunt strength it produces. `use_A` gates
the whole mechanism off by default (byte-parity with the pre-M4-3A engine). Per P1-1,
`uses_shunt()` requires use_A AND k_n!=0 AND alpha_A!=0 -- k_n=0 leaves a==0 forever,
so kick_probe (Task 5) must stay on its literal parity path in that case.
"""
import numpy as np
from src.snn_engine.slow_field import SpatialSlowField, SpatialSlowFieldConfig

L = 4.0
NE = 64
NI = 16


def _mk(**kw):
    # minimal field; reuse existing required ctor args as the codebase defines them.
    # (grep SpatialSlowField.__init__ for the exact signature: n_grid, L, posE, posI, nE, nI ...)
    cfg = SpatialSlowFieldConfig(use_qI=True, k_q=0.0, use_gK=False, k_K=0.0, **kw)
    return cfg


def _make_field(cfg):
    # Tiny field (n_grid=8, nE=64/nI=16 on a 4x4mm sheet) for speed -- mirrors the
    # existing M4 pool helper `_pool_field` in tests/test_m4_shared_inhibition.py.
    cfg.n_grid = 8
    rng = np.random.default_rng(0)
    posE = rng.uniform(0, L, size=(NE, 2))
    posI = rng.uniform(0, L, size=(NI, 2))
    return SpatialSlowField(NE + NI, 16.5, posE, posI, L, cfg=cfg)


def _zero_spikes(fld):
    return np.zeros(fld.N, dtype=bool)


def _driving_spikes(fld):
    # All E cells "spike" every step -> drives rE (EMA rate field) -> u_n=K_n*rE > 0.
    spk = np.zeros(fld.N, dtype=bool)
    spk[:fld.nE] = True
    return spk


def test_shunt_off_by_default_is_byte_parity():
    # use_A default False -> a_shunt stays 0, n_load stays n_base, shunt_g_at_E all zero
    cfg = _mk()
    assert cfg.use_A is False
    fld = _make_field(cfg)
    for _ in range(100):
        fld.step(_zero_spikes(fld), labels=None, dt=0.1)
    assert np.all(fld.a_shunt == 0.0)
    assert np.allclose(fld.n_load, cfg.n_base)
    assert np.all(fld.shunt_g_at_E() == 0.0)
    assert fld.uses_shunt() is False


def test_k_n_zero_is_parity_path_even_with_alpha_A():
    # P1-1: k_n=0 -> a==0 forever -> uses_shunt() MUST be False so kick_probe stays literal.
    cfg = _mk(use_A=True, k_n=0.0, alpha_A=2.0)
    fld = _make_field(cfg)
    for _ in range(100):
        fld.step(_driving_spikes(fld), labels=None, dt=0.1)
    assert np.allclose(fld.n_load, cfg.n_base)   # k_n=0 -> load never evolves
    assert np.all(fld.a_shunt == 0.0)
    assert fld.uses_shunt() is False             # P1-1: k_n=0 keeps the byte-parity path


def test_uses_shunt_true_only_when_all_three_on():
    assert _make_field(_mk(use_A=True, k_n=1.0, alpha_A=2.0)).uses_shunt() is True
    assert _make_field(_mk(use_A=True, k_n=1.0, alpha_A=0.0)).uses_shunt() is False   # no conductance
    assert _make_field(_mk(use_A=False, k_n=1.0, alpha_A=2.0)).uses_shunt() is False  # field off


def test_shunt_g_clips_and_tracks_a():
    cfg = _mk(use_A=True, k_n=1.0, alpha_A=5.0, a_max=1.0, g_A_max=3.0)
    fld = _make_field(cfg)
    for _ in range(20000):                        # drive load up
        fld.step(_driving_spikes(fld), labels=None, dt=0.1)
    g = fld.shunt_g_at_E()
    assert g.shape == (fld.nE,)
    assert np.all((g >= 0.0) & (g <= 3.0))        # clipped to g_A_max
    assert g.max() > 0.0                          # shunt engaged


def test_eta_A_subtractive_term_only_when_on():
    # apply_currents subtracts eta_A*a on E cells; with a==0 it's a no-op (parity)
    cfg = _mk(use_A=True, k_n=0.0, eta_A=1.0)     # a stays 0 -> no subtraction
    fld = _make_field(cfg)
    I_E = np.ones(fld.N); I_I = np.zeros(fld.N)
    out0 = fld.apply_currents(I_E.copy(), I_I.copy())
    assert np.allclose(out0[:fld.nE], I_E[:fld.nE])  # a==0 -> unchanged
