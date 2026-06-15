"""Equivalence gate for the opt-in tail-bounded `prune_radius` in
src/snn_engine/connectivity_rot._sample_partners_rot (Stage 4 Phase 0).

Net is sized so the prune radius R = 8*l_par genuinely truncates: L=14 mm >> 2R
(~8.6 mm) so a central target's R-ball is a strict subset of the sheet, while
density=35 keeps >= C_EE=800 weighted E-candidates inside the ball (gate (1)).
"""
import sys
import numpy as np

sys.path.insert(0, "src/snn_engine")
from params import Params                       # noqa: E402
from connectivity import place_neurons          # noqa: E402
import connectivity_rot as cr                    # noqa: E402


def _small_net(L=14.0, density=35.0, seed=1):
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
