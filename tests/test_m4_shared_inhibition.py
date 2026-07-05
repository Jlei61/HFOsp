"""TDD for M4 divisive shared inhibitory pool S_G (spec 2026-07-05 rev4).

Pool: r_E -> Psi_G (per-location recruitment) -> A_G=[<Psi^p>]^(1/p) -> mu_G (fast low-pass)
-> S_G (low-pass). Membrane: I_net_E = I_ff + I_rec/(1+alpha_G*S_G) - q_I*I_I - eta_K*g_K, implemented
as I_E - I_E_rec*(alpha_G*S_G/(1+alpha_G*S_G)) - ... so alpha_G*S_G=0 is EXACT byte-parity.
OFF-by-default: use_SG=False -> no pool, apply_currents unchanged, engine byte-identical to slow=None.
"""
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params  # noqa: E402
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from slow_field import (  # noqa: E402
    SpatialSlowField, SpatialSlowFieldConfig, psi_recruit, pnorm_pool,
)

DT = 0.1


def _net(L=6.0, T=250.0, seed=1, density=100.0, nu=0.6):
    p = Params(L=L, density=density, T=T, dt=DT, nu_ext_ratio=nu, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    return p, net, NE, NI


def _slow_for(p, net, **cfgkw):
    NE, NI = net["NE"], net["NI"]
    posE = net["pos"][net["labels"] == 0]
    posI = net["pos"][net["labels"] == 1]
    cfg = SpatialSlowFieldConfig(n_grid=8, **cfgkw)
    return SpatialSlowField(NE + NI, p.V_th, posE, posI, p.L, cfg=cfg)


def _pool_field(use_SG=True, alpha_G=0.0, **cfgkw):
    L = 4.0; nE = 64; nI = 16
    rngp = np.random.default_rng(0)
    posE = rngp.uniform(0, L, size=(nE, 2)); posI = rngp.uniform(0, L, size=(nI, 2))
    cfg = SpatialSlowFieldConfig(n_grid=8, use_SG=use_SG, alpha_G=alpha_G, **cfgkw)
    return SpatialSlowField(nE + nI, 16.5, posE, posI, L, cfg=cfg), nE, nI


# ---------------------------------------------------------------- Task 2: sensor helpers
def test_psi_recruit_hill_shape():
    r = np.array([0.0, 1.0, 2.0, 100.0])
    z = psi_recruit(r, r0=0.0, r50=1.0, n=2.0)
    assert np.isclose(z[0], 0.0)                       # background not recruited
    assert np.isclose(z[1], 0.5)                       # r=r50 -> half recruited
    assert z[3] > 0.99 and np.all(z <= 1.0)            # saturates to 1
    assert np.isclose(psi_recruit(0.4, 0.5, 1.0, 2.0), 0.0)   # sub-threshold clipped


def test_pnorm_pool_mean_and_focal_limits():
    z = np.zeros(100); z[:1] = 1.0                     # one hot cell among 100
    assert np.isclose(pnorm_pool(z, 1.0), 0.01)        # area/mean
    assert pnorm_pool(z, 4.0) > pnorm_pool(z, 1.0)     # focal-sensitive
    assert np.isclose(pnorm_pool(z, 4.0), 0.01 ** 0.25)
    z2 = np.full(100, 0.5)
    assert np.isclose(pnorm_pool(z2, 3.0), 0.5)        # uniform field: p-invariant


# ---------------------------------------------------------------- Task 3: pool state + step
def test_SG_builds_and_is_bounded_under_activity():
    f, nE, nI = _pool_field(tau_mu=40.0, tau_S=120.0, r0_psi=0.0, r50_psi=1.0, n_psi=2.0, p_pool=3.0)
    spk = np.zeros(nE + nI, dtype=bool); spk[:nE] = True    # all E "spike" each step -> A_G high
    for _ in range(4000):
        f.step(spk, labels=None, dt=DT)
    assert 0.0 < f.S_G <= f.cfg.S_max + 1e-9
    assert 0.0 < f.mu_G <= 1.0 + 1e-9
    assert f.S_G > 0.1                                  # genuinely built, not stuck at 0


def test_SG_off_by_default_no_pool_evolution():
    f, nE, nI = _pool_field(use_SG=False)
    spk = np.zeros(nE + nI, dtype=bool); spk[:nE] = True
    for _ in range(1000):
        f.step(spk, labels=None, dt=DT)
    assert f.S_G == 0.0 and f.mu_G == 0.0               # OFF -> no evolution


# ---------------------------------------------------------------- Task 4: divisive membrane + arms
def test_apply_currents_exact_parity_when_alpha_zero():
    f, nE, nI = _pool_field(use_SG=True, alpha_G=0.0)
    f.S_G = 0.7                                         # even S_G>0: alpha_G=0 -> ZERO divisive term
    N = nE + nI
    I_E = np.linspace(1, 2, N); I_I = np.linspace(0, 1, N); I_E_rec = np.linspace(0, 0.5, N)
    out_with = f.apply_currents(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    out_none = f.apply_currents(I_E, I_I, labels=None, I_E_rec=None)
    assert np.array_equal(out_with, out_none)          # BYTE-exact


def test_apply_currents_divides_recurrent_E_only():
    f, nE, nI = _pool_field(use_SG=True, alpha_G=2.0)
    f.S_G = 0.5                                         # D_G = 1 + 2*0.5 = 2.0 ; frac = 1/2
    N = nE + nI
    I_E = np.full(N, 3.0); I_I = np.zeros(N); I_E_rec = np.full(N, 1.0)
    out = f.apply_currents(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    assert np.allclose(out[:nE], 2.5)                  # 3 - 1*(1/2) = I_ff + I_rec/D_G = 2 + 0.5
    assert np.allclose(out[nE:], 3.0)                  # I cells untouched


def test_beta_SG_subtractive_arm():
    f, nE, nI = _pool_field(use_SG=True, alpha_G=0.0, beta_SG=0.4)
    f.S_G = 0.5
    N = nE + nI
    I_E = np.full(N, 3.0); I_I = np.zeros(N); I_E_rec = np.full(N, 1.0)
    out = f.apply_currents(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    assert np.allclose(out[:nE], 3.0 - 0.4 * 0.5)      # only the subtractive pool term
    assert np.allclose(out[nE:], 3.0)


# ---------------------------------------------------------------- Task 5: end-to-end smoke
def test_use_SG_off_matches_slow_none_engine_output():
    p, net, NE, NI = _net(T=150.0)
    net["rng"] = np.random.default_rng(3)
    a = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9)
    p, net, NE, NI = _net(T=150.0)
    net["rng"] = np.random.default_rng(3)
    slow = _slow_for(p, net)                            # defaults: all mechanisms off, use_SG=False
    b = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, slow=slow)
    assert hashlib.sha1(a["E_spk_bool"].tobytes()).hexdigest() == \
           hashlib.sha1(b["E_spk_bool"].tobytes()).hexdigest()


def test_alpha_G_does_not_increase_recruitment():
    # divisive pool can only reduce recurrent E gain -> total E spikes non-increasing vs neutral pool.
    def total(alpha_G):
        p, net, NE, NI = _net(T=300.0)
        net["rng"] = np.random.default_rng(7)
        slow = _slow_for(p, net, use_SG=True, alpha_G=alpha_G, tau_mu=30.0, tau_S=80.0,
                         r0_psi=0.0, r50_psi=1.0, n_psi=2.0, p_pool=3.0, S_max=1.0)
        res = simulate_kick(p, net, KICK_BOOST=8.0, r_kick=1.5,
                            V_th_per_neuron=np.full(NE + NI, 16.5), slow=slow)
        return float(res["E_spk_bool"].sum())
    n0 = total(0.0)
    n_big = total(6.0)
    assert n_big <= n0 + 1e-9                           # divisive pool never increases recruitment
