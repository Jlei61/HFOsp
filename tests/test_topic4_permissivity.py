"""TDD tests for topic4_permissivity.permissivity_vth_delta (Task 2, plan §Task 2 Step 1).

Contract (verbatim from plan):
  - I cells (last NI entries): delta = 0 (never moved).
  - E cells (first NE entries): delta[i] = -delta_theta * mu * h_eff[bin_of_cell[i]]
  - control='none'     -> h_eff = h
  - control='uniform'  -> h_eff = ones (uniform-mu control, ignores h's spatial shape)
  - control='shuffle'  -> h_eff = h permuted across bins using rng
  - mu=0 => all zeros (bit-parity precondition for engine baseline match).
"""
import numpy as np
import pytest
from src.topic4_permissivity import permissivity_vth_delta


# --- verbatim from plan Task 2 Step 1 ---

def test_mu0_zero_delta():
    h = np.array([1., 2., 0.5]); boc = np.array([0, 0, 1, 1, 2, 2])
    d = permissivity_vth_delta(h, boc, NE=6, NI=2, mu=0.0, delta_theta=3.0)
    assert d.shape == (8,) and np.allclose(d, 0)         # mu=0 -> all 0 (bit-parity)


def test_sign_negative_and_high_h_lowers_more():
    h = np.array([1., 2.]); boc = np.array([0, 1])
    d = permissivity_vth_delta(h, boc, NE=2, NI=1, mu=0.5, delta_theta=4.0)
    assert (d[:2] <= 0).all()                            # lowers threshold
    assert d[1] < d[0]                                   # higher h -> lowers more
    assert d[2] == 0                                     # I cell untouched


def test_uniform_control_ignores_h_shape():
    h = np.array([1., 9.]); boc = np.array([0, 1])
    d = permissivity_vth_delta(h, boc, NE=2, NI=0, mu=0.5, delta_theta=2.0, control='uniform')
    assert np.allclose(d[0], d[1])                       # uniform -> independent of h shape


# --- 4th test: shuffle control ---

def test_shuffle_control_permutes_h():
    """shuffle control uses rng to permute the bin->value map;
    with a seeded rng the result is a permutation of 'none' delta values
    and differs from it for a non-uniform h."""
    h = np.array([1., 3., 7., 2.])   # non-uniform, 4 bins
    boc = np.array([0, 1, 2, 3])     # each E cell in its own bin (NE=4, NI=0)
    rng = np.random.default_rng(42)

    d_none = permissivity_vth_delta(h, boc, NE=4, NI=0, mu=0.5, delta_theta=2.0,
                                     control='none')
    d_shuf = permissivity_vth_delta(h, boc, NE=4, NI=0, mu=0.5, delta_theta=2.0,
                                     control='shuffle', rng=rng)

    # shuffled values must be a permutation of 'none' values
    assert np.allclose(np.sort(d_shuf), np.sort(d_none)), \
        "shuffle must be a permutation of none delta values"
    # for a non-uniform h with random seed, the order must differ from none
    assert not np.allclose(d_shuf, d_none), \
        "shuffle should reorder values (non-uniform h, seeded rng)"
