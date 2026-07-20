"""FCXR Stage D — D1 frozen fast-branch map + D2 mode analysis unit tests.

Fast unit tests only (synthetic observable rows / tiny fields). The real 40k
`build_substrate` alignment check and SNN cell runs are exercised by the runner
scripts, not here (build_substrate ~137s / 6.8GB).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.topic4_mz_fcxr_dynamics import (
    load_onset_depletion_pi,
    assert_field_substrate_aligned,
)

SNAP = "results/topic4_sef_hfo/state_conditioned_susceptibility/snapshots/zA_q75_tz5000/seed_1.npz"


# ---------------- D0.1: locked p_i loader ----------------

def test_pi_is_mean_one_and_nonneg():
    pk = load_onset_depletion_pi(SNAP)
    assert pk["p_i"].shape == (32000,)
    assert np.isclose(float(pk["p_i"].mean()), 1.0, atol=1e-6)   # mean-depletion normalization
    assert (pk["p_i"] >= 0).all()
    assert pk["pos_E"].shape == (32000, 2)
    assert pk["vth_E"].shape == (32000,)


# ---------------- D0.1: substrate-alignment gate (synthetic S, fast) ----------------

def _fake_pack(NE=50, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(NE, 2))
    vth = rng.normal(18.0, 1.0, size=NE)
    return dict(p_i=np.ones(NE), pos_E=pos, vth_E=vth,
                src_xy=np.zeros(2), snk_xy=np.ones(2), axis_unit=np.array([1.0, 0.0]), L=20.0)


def _fake_S(pk, NI=10):
    NE = pk["pos_E"].shape[0]
    posI = np.zeros((NI, 2))
    vth_full = np.concatenate([pk["vth_E"], np.full(NI, 18.0)])   # E-then-I, length N
    return dict(NE=NE, NI=NI, N=NE + NI, posE=pk["pos_E"].copy(),
                posI=posI, vth=vth_full)


def test_alignment_passes_when_field_matches_substrate():
    pk = _fake_pack()
    S = _fake_S(pk)
    assert_field_substrate_aligned(pk, S)   # must not raise


def test_alignment_rejects_shuffled_field():
    pk = _fake_pack()
    S = _fake_S(pk)
    bad = dict(pk); bad["pos_E"] = pk["pos_E"][::-1].copy()   # neuron order reversed
    with pytest.raises(ValueError):
        assert_field_substrate_aligned(bad, S)


def test_alignment_rejects_NE_mismatch():
    pk = _fake_pack(NE=50)
    S = _fake_S(_fake_pack(NE=40))   # substrate has 40 E cells, field has 50
    with pytest.raises(ValueError):
        assert_field_substrate_aligned(pk, S)
