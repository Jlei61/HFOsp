import numpy as np
import pytest
from scipy import sparse
from src.topic4_degnorm import ee_degree, degnorm_vth_delta


def _net_from_EE(W_EE, NE, NI=0):
    """Fake net: ampa_by_delay = [one full (NE+NI) matrix] with W_EE in the E->E block."""
    N = NE + NI
    full = np.zeros((N, N))
    full[:NE, :NE] = W_EE
    return {"ampa_by_delay": [sparse.csr_matrix(full)]}


def test_out_and_in_strength_track_column_and_row_sums():
    # W[target, source]: source 0 broadcasts to targets 1 and 2; target 2 receives most.
    W = np.array([[0, 0, 0], [2, 0, 0], [3, 0, 0.]])
    net = _net_from_EE(W, NE=3)
    out = ee_degree(net, 3, "out_strength")   # raw column sums [5, 0, 0]
    in_ = ee_degree(net, 3, "in_strength")    # raw row sums    [0, 2, 3]
    assert out[0] > out[1] and out[0] > out[2]     # source 0 = the broadcaster
    assert in_[2] > in_[1] and in_[1] > in_[0]     # target 2 receives most


def test_alpha_zero_is_noop():
    net = _net_from_EE(np.zeros((3, 3)), NE=3, NI=2)
    d = degnorm_vth_delta(net, 3, 2, alpha=0.0)
    assert d.shape == (5,) and np.all(d == 0)


def test_delta_raises_high_outdegree_cells_more_and_spares_I():
    W = np.array([[0, 0, 0], [2, 0, 0], [3, 0, 0.]])
    net = _net_from_EE(W, NE=3, NI=2)
    d = degnorm_vth_delta(net, 3, 2, alpha=1.0, scheme="out_strength")
    assert d.shape == (5,)
    assert np.all(d[3:] == 0)              # I cells untouched
    assert d[0] > d[1] and d[0] > d[2]     # broadcaster source raised most


def test_hybrid_is_elementwise_max():
    W = np.array([[0, 0, 0], [2, 0, 0], [3, 0, 0.]])
    net = _net_from_EE(W, NE=3)
    h = ee_degree(net, 3, "hybrid")
    o = ee_degree(net, 3, "out_strength")
    i = ee_degree(net, 3, "in_strength")
    assert np.allclose(h, np.maximum(o, i))


def test_unknown_scheme_raises():
    net = _net_from_EE(np.zeros((2, 2)), NE=2)
    with pytest.raises(ValueError):
        ee_degree(net, 2, "bogus")
