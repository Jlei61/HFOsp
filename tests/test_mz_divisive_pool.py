"""Contract tests for current-based per-neuron Z/M plus the M4 recurrent-E divisor."""
import hashlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from connectivity_rot import build_connectivity_rot  # noqa: E402
from connectivity import place_neurons  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from mz_divisive_pool import (  # noqa: E402
    MZDivisivePoolConfig,
    MZDivisivePoolSlowVars,
    slow_gate_drive,
)
from params import Params  # noqa: E402


def _adapter(**kw):
    nE, nI, L = 32, 8, 4.0
    rng = np.random.default_rng(0)
    posE = rng.uniform(0, L, size=(nE, 2))
    posI = rng.uniform(0, L, size=(nI, 2))
    core = np.zeros(nE, bool)
    core[:4] = True
    cfg = MZDivisivePoolConfig(n_grid=8, **kw)
    return MZDivisivePoolSlowVars(
        nE + nI, 18.0, posE, posI, L, cfg=cfg, NE=nE, core_mask_E=core
    )


def test_active_formula_divides_recurrent_e_only_and_keeps_i_cells_unchanged():
    slow = _adapter(use_z=True, use_m=True, use_SG=True, alpha_G=2.0, eta_m=0.25)
    slow.mz.z[: slow.NE] = 0.5
    slow.mz.m[: slow.NE] = 2.0
    slow.pool.S_G = 0.5
    n = slow.N
    I_E = np.full(n, 3.0)
    I_I = np.ones(n)
    I_E_rec = np.ones(n)
    out = slow.apply_currents(I_E, I_I, I_E_rec=I_E_rec)
    # E: 3 - z*1 - eta*m - rec*(alpha*S)/(1+alpha*S) = 3-.5-.5-.5 = 1.5
    assert np.allclose(out[: slow.NE], 1.5)
    assert np.allclose(out[slow.NE :], 2.0)  # I cells retain I_E-I_I


def test_active_divisor_requires_recurrent_current():
    slow = _adapter(use_SG=True, alpha_G=2.0)
    slow.pool.S_G = 0.5
    with pytest.raises(RuntimeError, match="I_E_rec"):
        slow.apply_currents(np.ones(slow.N), np.zeros(slow.N))


def test_composite_and_standalone_mz_states_match_for_identical_inputs():
    cfg = dict(
        use_z=True,
        use_m=True,
        I_th_EI=2.0,
        tau_z=1000.0,
        tau_adp=500.0,
        eta_m=0.1,
    )
    comp = _adapter(use_SG=True, alpha_G=0.0, **cfg)
    standalone = MZSlowVars(
        comp.N,
        18.0,
        MZSlowVarsConfig(**cfg),
        NE=comp.NE,
        core_mask_E=np.r_[np.ones(4, bool), np.zeros(comp.NE - 4, bool)],
    )
    rng = np.random.default_rng(4)
    for _ in range(300):
        I_E = rng.uniform(0, 4, comp.N)
        I_I = rng.uniform(0, 4, comp.N)
        I_E_rec = rng.uniform(0, 2, comp.N)
        spk = rng.random(comp.N) < 0.1
        out_c = comp.apply_currents(I_E, I_I, I_E_rec=I_E_rec)
        out_m = standalone.apply_currents(I_E, I_I)
        assert np.array_equal(out_c, out_m)
        comp.step(spk, None, 0.1)
        standalone.step(spk, None, 0.1)
    assert np.array_equal(comp.z, standalone.z)
    assert np.array_equal(comp.m, standalone.m)
    assert np.array_equal(comp.mz.trace_z_mean, standalone.trace_z_mean)
    assert comp.pool.S_G > 0.0  # neutral divisor was nevertheless observed and evolved


def test_pool_builds_and_remains_bounded():
    slow = _adapter(
        use_SG=True,
        alpha_G=4.0,
        r50_psi=0.3,
        tau_mu=30.0,
        tau_S=80.0,
    )
    spk = np.zeros(slow.N, bool)
    spk[: slow.NE] = True
    for _ in range(3000):
        slow.apply_currents(np.ones(slow.N), np.zeros(slow.N), I_E_rec=np.ones(slow.N))
        slow.step(spk, None, 0.1)
    assert 0.0 < slow.pool.S_G <= slow.cfg.S_max
    assert len(slow.trace_SG) == 3000


def test_slow_gate_drive_has_hard_ied_floor_and_bounded_hill_response():
    assert slow_gate_drive(0.149, A0=0.15, A50=0.10, exponent=4.0) == 0.0
    assert slow_gate_drive(0.15, A0=0.15, A50=0.10, exponent=4.0) == 0.0
    assert slow_gate_drive(0.25, A0=0.15, A50=0.10, exponent=4.0) == pytest.approx(0.5)
    assert 0.5 < slow_gate_drive(0.40, A0=0.15, A50=0.10, exponent=4.0) < 1.0


def test_slow_gate_builds_above_threshold_decays_below_and_stays_bounded():
    slow = _adapter(
        use_SG=True,
        alpha_G=2.0,
        p_pool=1.0,
        r50_psi=0.3,
        use_TG=True,
        alpha_TG=4.0,
        AG0_TG=0.15,
        AG50_TG=0.10,
        n_TG=4.0,
        tau_TG=750.0,
    )
    high = np.zeros(slow.N, bool)
    high[: slow.NE] = True
    for _ in range(2000):
        slow.step(high, None, 0.1)
    built = slow.T_G
    assert 0.0 < built <= slow.cfg.TG_max
    assert max(slow.trace_UTG) > 0.0
    quiet = np.zeros(slow.N, bool)
    for _ in range(2000):
        slow.step(quiet, None, 0.1)
    assert 0.0 <= slow.T_G < built
    assert len(slow.trace_TG) == 4000


def test_slow_gate_adds_only_to_recurrent_e_divisor():
    slow = _adapter(use_SG=True, alpha_G=0.0, use_TG=True, alpha_TG=4.0)
    slow.pool.S_G = 0.0
    slow.T_G = 0.5
    I_E = np.full(slow.N, 3.0)
    I_I = np.ones(slow.N)
    I_E_rec = np.ones(slow.N)
    out = slow.apply_currents(I_E, I_I, I_E_rec=I_E_rec)
    # recurrent component 1 is divided by 1 + 4*0.5 = 3, so 2/3 is removed.
    assert np.allclose(out[: slow.NE], 3.0 - 1.0 - 2.0 / 3.0)
    assert np.allclose(out[slow.NE :], 2.0)


def test_alpha_TG_zero_is_literal_neutral_path():
    base = _adapter(use_z=True, use_SG=True, alpha_G=2.0, use_TG=False)
    gated = _adapter(use_z=True, use_SG=True, alpha_G=2.0, use_TG=True, alpha_TG=0.0)
    base.pool.S_G = gated.pool.S_G = 0.25
    gated.T_G = 0.8
    rng = np.random.default_rng(18)
    I_E = rng.uniform(0, 4, base.N)
    I_I = rng.uniform(0, 2, base.N)
    rec = rng.uniform(0, 2, base.N)
    assert np.array_equal(
        base.apply_currents(I_E, I_I, I_E_rec=rec),
        gated.apply_currents(I_E, I_I, I_E_rec=rec),
    )


def test_config_rejects_nonpositive_timescale_and_p_below_one():
    with pytest.raises(ValueError, match="tau_S"):
        _adapter(use_SG=True, tau_S=0.0)
    with pytest.raises(ValueError, match="p_pool"):
        _adapter(use_SG=True, p_pool=0.5)
    with pytest.raises(ValueError, match="tau_TG"):
        _adapter(use_SG=True, use_TG=True, tau_TG=0.0)
    with pytest.raises(ValueError, match="requires use_SG"):
        _adapter(use_SG=False, use_TG=True)


def test_full_engine_alpha_zero_composite_is_byte_identical_to_active_mz():
    seed = 9
    p = Params(L=2.0, density=250.0, T=180.0, dt=0.1, seed=seed, nu_ext_ratio=0.8)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(
        p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0
    )
    n = NE + NI
    core = np.zeros(NE, bool)
    core[: max(1, NE // 10)] = True
    mz_cfg = MZSlowVarsConfig(
        use_z=True,
        use_m=True,
        I_th_EI=1.0,
        tau_z=2000.0,
        tau_adp=1000.0,
        eta_m=0.01,
    )
    comp_cfg = MZDivisivePoolConfig(
        use_z=True,
        use_m=True,
        use_SG=True,
        alpha_G=0.0,
        I_th_EI=1.0,
        tau_z=2000.0,
        tau_adp=1000.0,
        eta_m=0.01,
        n_grid=8,
        r50_psi=0.3,
    )
    gated_neutral_cfg = MZDivisivePoolConfig(
        use_z=True,
        use_m=True,
        use_SG=True,
        alpha_G=0.0,
        use_TG=True,
        alpha_TG=0.0,
        I_th_EI=1.0,
        tau_z=2000.0,
        tau_adp=1000.0,
        eta_m=0.01,
        n_grid=8,
        r50_psi=0.3,
    )

    def run(slow):
        net["rng"] = np.random.default_rng(seed)
        return simulate_kick(
            p,
            net,
            5.0,
            slow=slow,
            kick_center=np.array([p.L / 2, p.L / 2]),
            r_kick=0.5,
            t_kick=50.0,
            V_th_per_neuron=np.full(n, 17.0),
        )

    mz = MZSlowVars(n, 18.0, mz_cfg, NE=NE, core_mask_E=core)
    comp = MZDivisivePoolSlowVars(
        n,
        18.0,
        pos[:NE],
        pos[NE:],
        p.L,
        cfg=comp_cfg,
        NE=NE,
        core_mask_E=core,
    )
    gated_neutral = MZDivisivePoolSlowVars(
        n,
        18.0,
        pos[:NE],
        pos[NE:],
        p.L,
        cfg=gated_neutral_cfg,
        NE=NE,
        core_mask_E=core,
    )
    a = run(mz)
    b = run(comp)
    c = run(gated_neutral)
    assert hashlib.sha1(a["E_spk_bool"].tobytes()).hexdigest() == hashlib.sha1(
        b["E_spk_bool"].tobytes()
    ).hexdigest()
    assert np.array_equal(a["rate_E"], b["rate_E"])
    assert np.array_equal(a["E_spk_bool"], c["E_spk_bool"])
    assert np.array_equal(a["rate_E"], c["rate_E"])
    assert a["E_spk_bool"].sum() > 0


def test_full_engine_active_fast_pool_matches_neutral_slow_gate_for_e_and_i():
    seed = 11
    p = Params(L=2.0, density=250.0, T=180.0, dt=0.1, seed=seed, nu_ext_ratio=0.8)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(
        p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0
    )
    n = NE + NI
    core = np.zeros(NE, bool)
    core[: max(1, NE // 10)] = True

    def build(use_tg):
        return MZDivisivePoolSlowVars(
            n,
            18.0,
            pos[:NE],
            pos[NE:],
            p.L,
            cfg=MZDivisivePoolConfig(
                use_z=True,
                I_th_EI=1.0,
                tau_z=2000.0,
                use_SG=True,
                alpha_G=2.0,
                use_TG=use_tg,
                alpha_TG=0.0,
                n_grid=8,
                r50_psi=0.3,
            ),
            NE=NE,
            core_mask_E=core,
        )

    def run(slow):
        net["rng"] = np.random.default_rng(seed)
        return simulate_kick(
            p,
            net,
            5.0,
            slow=slow,
            kick_center=np.array([p.L / 2, p.L / 2]),
            r_kick=0.5,
            t_kick=50.0,
            V_th_per_neuron=np.full(n, 17.0),
        )

    base = run(build(False))
    neutral_tg = run(build(True))
    assert np.array_equal(base["E_spk_bool"], neutral_tg["E_spk_bool"])
    assert np.array_equal(base["rate_E"], neutral_tg["rate_E"])
    assert np.array_equal(base["rate_I"], neutral_tg["rate_I"])
