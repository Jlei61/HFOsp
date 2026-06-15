"""Equivalence gate for the opt-in tail-bounded `prune_radius` in
src/snn_engine/connectivity_rot._sample_partners_rot (Stage 4 Phase 0).

Net is sized so the prune radius R = 8*l_par genuinely truncates: L=14 mm >> 2R
(~8.6 mm) so a central target's R-ball is a strict subset of the sheet, while
density=35 keeps >= C_EE=800 weighted E-candidates inside the ball (gate (1)).
"""
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "src/snn_engine")
from params import Params                       # noqa: E402
from connectivity import place_neurons          # noqa: E402
import connectivity_rot as cr                    # noqa: E402


def _small_net(L=12.0, density=100.0, seed=1):
    # PRODUCTION density (100): the prune is only equivalent when the fixed in-degree
    # C_EE=800 is satisfied well INSIDE the prune radius. C_EE-radius = sqrt(C_EE/(pi*0.8*D))
    # ~ 1.8 mm at D=100 << R = 8*l_par ~ 4.3 mm, so the truncated cells are never selected.
    # At low density (e.g. 35) 800 partners reach ~3 mm ~ R and the prune DOES distort -- that
    # regime is out of scope (the production sheet is d100). L=12 > 2R so the ball restricts.
    p = Params(L=L, density=density, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    return p, pos, labels, NE, NI


def _kernel_args(p, AR=2.0):
    return p.l_EE * np.sqrt(AR), p.l_EE / np.sqrt(AR), np.radians(45.0)


def test_pruned_in_degree_exact_and_within_radius():
    p, pos, labels, NE, NI = _small_net()
    posE = pos[:NE]
    l_par, l_perp, th = _kernel_args(p)
    R = 8.0 * l_par
    t = int(np.argmin(np.linalg.norm(posE - posE.mean(0), axis=1)))  # central target
    n_local = int((np.linalg.norm(posE - posE[t], axis=1) <= R).sum()) - 1  # exclude self
    assert n_local < NE, "prune radius covers the whole sheet — net too small to test restriction"
    assert n_local >= p.C_EE, "prune ball holds < C_EE candidates — widen R or raise density (gate 1)"
    cols = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                   np.random.default_rng(0), self_local=t, prune_radius=R)
    assert cols.size == min(p.C_EE, n_local)
    assert (np.linalg.norm(posE[cols] - posE[t], axis=1) <= R + 1e-9).all()


def test_prune_none_is_bit_identical():
    p, pos, labels, NE, NI = _small_net()
    posE = pos[:NE]
    l_par, l_perp, th = _kernel_args(p)
    t = 7
    a = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                np.random.default_rng(3), self_local=t)
    b = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                np.random.default_rng(3), self_local=t, prune_radius=None)
    assert np.array_equal(np.sort(a), np.sort(b))


def _partner_dists_offs(p, posE, prune, seed=11, step=8):
    l_par, l_perp, th = _kernel_args(p)
    tree = cKDTree(posE); rng = np.random.default_rng(seed); ds = []; offs = []
    for t in range(0, len(posE), step):
        cols = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                       rng, self_local=t, prune_radius=prune, src_tree=tree)
        if cols.size:
            ds.append(np.linalg.norm(posE[cols] - posE[t], axis=1))
            offs.append(posE[cols] - posE[t])
    return np.concatenate(ds), np.concatenate(offs)


def _cov_axis(offs):
    C = np.cov(offs, rowvar=False); ev, evec = np.linalg.eigh(C)
    vmaj = evec[:, 1]
    ang = np.degrees(np.arctan2(vmaj[1], vmaj[0])) % 180.0
    return ang, float(np.sqrt(ev[1] / ev[0]))


def test_equivalence_ks_delay_ar_tailmass():
    from scipy import stats
    p, pos, labels, NE, NI = _small_net(seed=2)
    posE = pos[:NE]
    l_par = p.l_EE * np.sqrt(2.0)
    R = 8.0 * l_par
    d_naive, o_naive = _partner_dists_offs(p, posE, None)
    d_prune, o_prune = _partner_dists_offs(p, posE, R)
    # (2) partner-distance EFFECT SIZE small: a tail-bounded approximation can never be
    # p-value-identical (it has a hard cutoff at R), so gate on the KS statistic (max CDF
    # gap), not the p-value. At d100 the truncated cells are never selected -> gap ~ 0.
    ks = stats.ks_2samp(d_naive, d_prune)
    assert ks.statistic < 0.03, f"partner-distance KS statistic {ks.statistic:.4f} (p={ks.pvalue:.2g})"
    # (3) BULK delay quantiles match tightly (delay = tau0 + d/v_axon, monotone in d).
    # The extreme tail (q>=0.99) is INTENTIONALLY bounded by R -> check the bulk, then
    # assert the tail is bounded (not that it matches the unbounded naive tail).
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    assert np.allclose(np.quantile(d_naive, qs), np.quantile(d_prune, qs), rtol=0.03)
    assert np.quantile(d_prune, 0.99) <= np.quantile(d_naive, 0.99) + 1e-9  # tail bounded by R
    # (4) realized anisotropy preserved: cov major-axis ~ theta_EE=45 deg, ratio within 10%
    a_n, r_n = _cov_axis(o_naive); a_p, r_p = _cov_axis(o_prune)
    assert min(abs(a_p - 45.0), 180.0 - abs(a_p - 45.0)) < 8.0
    assert abs(r_p - r_n) / r_n < 0.10
    # (5) 2D tail-mass bound < 1% at R=8*l_par; the WRONG bare exp(-R/l) would pass even at 6*l_par
    tail = (1.0 + R / l_par) * np.exp(-R / l_par)
    assert tail < 0.01, f"2D tail-mass {tail:.4f} not < 1% (use R>=8*l_par)"
    assert (1.0 + 6.0) * np.exp(-6.0) > 0.01   # documents why 6*l_par is inadequate
