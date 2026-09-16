import copy
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src/snn_engine')]
from src.topic4_continuous_core_state import ContinuousIState, coupled_counts, ou_path, poisson_quantile
from kick_probe import simulate_kick
from model import build_network
from params import Params


def test_quantile_matches_reference_and_coupling_is_monotone():
    rng = np.random.default_rng(391)
    u = rng.uniform(1e-12, 1-1e-12, 10000)
    mu = rng.uniform(0, 20, len(u))
    assert np.array_equal(poisson_quantile(u, mu), poisson.ppf(u, mu))
    base = rng.poisson(mu)
    assert np.array_equal(coupled_counts(base, mu, np.ones(len(u)), u), base)
    low = coupled_counts(base, mu, np.full(len(u), .8), u)
    high = coupled_counts(base, mu, np.full(len(u), 1.2), u)
    assert np.all(low <= base) and np.all(high >= base)
    # Actual marginals over independent cells; large discrepancy would expose
    # a wrong randomized-PIT interval, not mere implementation differences.
    n = 200000; m = np.full(n, .7); k = rng.poisson(m); v = rng.random(n)
    out = coupled_counts(k, m, np.full(n, 1.2), v)
    assert abs(out.mean()-.84) < .01
    assert abs(out.var()-.84) < .02


def test_ou_stationarity_autocorrelation_and_prefix():
    x = ou_path(200000, 1., 50., 29)
    assert abs(x.mean()) < .08 and abs(x.var()-1) < .08
    assert abs(np.corrcoef(x[:-50], x[50:])[0,1]-np.exp(-1)) < .04
    assert np.array_equal(x[:1000], ou_path(1000, 1., 50., 29))


def test_engine_off_parity_and_unchanged_background_in_active_arm():
    p = Params(L=1., density=1500., C_EE=100, C_IE=100, C_EI=50, C_II=50,
               T=25., dt=.1, seed=71, nu_ext_ratio=.9)
    net = build_network(p, verbose=False)
    idx = np.arange(net['NE'], net['NE']+30)
    loading = np.r_[np.ones(10), np.full(20, -.5)]
    def run(amp, z, plain=False):
        net['rng'] = np.random.default_rng(41)
        control = ContinuousIState(idx, loading, np.full(250,z), amplitude=amp,
                                  dt_ms=.1, seed=9, warmup_ms=0, ramp_ms=.1)
        out = simulate_kick(p, net, KICK_BOOST=0., t_kick=1e9,
                            external_i_state=None if plain else control)
        return out, control, copy.deepcopy(net['rng'].bit_generator.state)
    plain, _, rng0 = run(0., 1., True)
    off, co, rng1 = run(0., 1.)
    active, ca, rng2 = run(.2, 1.)
    assert np.array_equal(plain['E_spk_bool'], off['E_spk_bool'])
    assert np.array_equal(plain['rate_E'], off['rate_E'])
    assert rng0 == rng1 == rng2
    assert co.audit()['legacy_external_counts_sha256'] == ca.audit()['legacy_external_counts_sha256']
    assert ca.audit()['maximum_relative_total_rate_error'] < 1e-12
    assert ca.audit()['changed_cell_steps'] > 0
