"""Correctness gate for the flat-edge spike scatter in kick_probe.simulate_kick
(Stage 4 Phase 0 perf fix, 2026-06-15).

The optimization replaces a dense N-vector add per delay bin with a single
source-indexed np.add.at over the firing edges. These tests prove (a) the flat
scatter equals the original dense per-bin formula exactly on the supported
entries, and (b) simulate_kick stays deterministic.
"""
import sys
import numpy as np

sys.path.insert(0, "src/snn_engine")
from params import Params, compute_nu_theta          # noqa: E402
from connectivity import place_neurons                # noqa: E402
from connectivity_rot import build_connectivity_rot   # noqa: E402
from kick_probe import _flatten_by_source, simulate_kick  # noqa: E402


def _net(L=6.0, density=100.0, seed=1):
    p = Params(L=L, density=density, T=100.0, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.radians(45), AR=2.0, prune_radius=4.3)
    return p, net, NE, NI


def test_flat_scatter_equals_dense_per_bin():
    """The flat source-indexed scatter must reproduce the original dense
    `ampa[d][:, spE].sum(axis=1)` per-bin contribution exactly (to FP)."""
    p, net, NE, NI = _net()
    ampa = net["ampa_by_delay"]; M = net["max_delay_steps"] + 1; N = NE + NI
    bins = [d for d in range(M) if ampa[d].nnz > 0]
    spE = np.random.default_rng(7).choice(NE, size=40, replace=False)
    t = 3
    # dense per-bin reference (the pre-optimization formula)
    ring_dense = np.zeros((M, N))
    for d in bins:
        ring_dense[(t + d) % M] += np.asarray(ampa[d][:, spE].sum(axis=1)).ravel()
    # flat scatter (the optimization)
    indptr, dst, dly, w = _flatten_by_source(ampa, bins, NE)
    ring_flat = np.zeros((M, N))
    st = indptr[spE]; cnt = indptr[spE + 1] - st; tot = int(cnt.sum())
    idx = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt)
    np.add.at(ring_flat, ((t + dly[idx]) % M, dst[idx]), w[idx])
    assert np.array_equal(ring_dense != 0, ring_flat != 0)      # identical support
    assert np.allclose(ring_dense, ring_flat, rtol=0, atol=1e-12)


def test_simulate_kick_deterministic():
    """Same seed -> bit-identical spike output (no hidden nondeterminism)."""
    p, net, NE, NI = _net()
    boost = 4 * compute_nu_theta(p)[0]
    net["rng"] = np.random.default_rng(1)
    a = simulate_kick(p, net, KICK_BOOST=boost, kick_center=[3, 3], r_kick=1.0)
    net["rng"] = np.random.default_rng(1)
    b = simulate_kick(p, net, KICK_BOOST=boost, kick_center=[3, 3], r_kick=1.0)
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert np.array_equal(a["rate_E"], b["rate_E"])
