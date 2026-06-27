"""Contract tests for src/sef_hfo_mini_w_event.py — mini-W_event pilot helpers.

Design: docs/archive/topic4/sef_hfo/m3_mini_w_event_design_2026-06-23.md
Pilot scope ONLY (§8 step 3): K_min(q) extraction + center-source W_shape
reproducibility (B1a). B1b/c/d (axis fit, ordering predictivity, matched shape) are
NOT in this module yet — they are post-review step 4.

Contract clauses pinned here (no SNN; pure functions on synthetic data):
  C6  success seed = EA-local-returned AND NOT spontaneous-ignition (reuses upstream flags)
  C7  build_w_shape: source-excluded + per-seed normalized + mean over SUCCESS seeds only;
      NaN (no-event) rows excluded; raise on zero success
  C8  extract_kmin = min kick with P_EA >= 0.7 AND n_seeds >= 6; extract_k50 linear interp
  C9  w_shape_reproducibility: observed cross-seed similarity vs spatial-bin-shuffled null,
      pass iff observed >= null p95; deterministic rng (no Date/random)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import sef_hfo_mini_w_event as mwe  # noqa: E402


# --------------------------------------------------------------------------- #
# C8 — K_min / K50 extraction                                                  #
# --------------------------------------------------------------------------- #
def test_extract_kmin_first_kick_at_or_above_threshold():
    kicks = [0.8, 1.0, 1.2, 1.4]
    p_ea = [0.2, 0.5, 0.7, 0.9]
    n_seeds = [12, 12, 12, 12]
    assert mwe.extract_kmin(kicks, p_ea, n_seeds) == 1.2     # first P_EA >= 0.7


def test_extract_kmin_skips_kicks_below_min_seeds():
    # 1.2 reaches 0.7 but only has 4 seeds (< MIN_SEEDS=6) -> not eligible; 1.4 qualifies
    kicks = [0.8, 1.0, 1.2, 1.4]
    p_ea = [0.2, 0.5, 0.7, 0.8]
    n_seeds = [12, 12, 4, 12]
    assert mwe.extract_kmin(kicks, p_ea, n_seeds) == 1.4


def test_extract_kmin_none_qualifies_is_inf():
    kicks = [0.8, 1.0, 1.2]
    p_ea = [0.1, 0.3, 0.5]
    n_seeds = [12, 12, 12]
    assert mwe.extract_kmin(kicks, p_ea, n_seeds) == float("inf")


def test_extract_k50_linear_interp_crossing():
    # P_EA crosses 0.5 between kick 1.0 (0.4) and 1.2 (0.6) -> 1.1
    kicks = [0.8, 1.0, 1.2, 1.4]
    p_ea = [0.2, 0.4, 0.6, 0.9]
    assert mwe.extract_k50(kicks, p_ea) == pytest.approx(1.1)


def test_extract_k50_never_reaches_is_nan():
    assert np.isnan(mwe.extract_k50([0.8, 1.0], [0.1, 0.3]))


# --------------------------------------------------------------------------- #
# C6 — success seed selection (EA-local AND NOT spontaneous)                    #
# --------------------------------------------------------------------------- #
def _rec(seed, r95_ea, far_ea, returned):
    return {"seed": float(seed), "r95_ea": r95_ea, "far_ea": far_ea,
            "returned": returned, "runaway": 0.0}


def test_success_seeds_require_ea_local_and_not_spontaneous():
    recs = [
        _rec(0, 3.0, 0.1, 1),    # EA-local
        _rec(1, 9.0, 0.1, 1),    # r95 too big -> NOT EA-local
        _rec(2, 3.0, 0.1, 1),    # EA-local but spontaneous -> excluded
        _rec(3, 3.0, 0.9, 1),    # far too big -> NOT EA-local
        _rec(4, 3.0, 0.1, 0),    # not returned -> NOT EA-local
    ]
    spont = {2}
    got = mwe.success_seeds_at_kick(recs, spont)
    assert got == {0}            # only seed 0 passes both gates


# --------------------------------------------------------------------------- #
# C7 — build_w_shape                                                            #
# --------------------------------------------------------------------------- #
def test_build_w_shape_excludes_source_bin_and_normalizes_per_seed():
    # 3 bins, source = bin 0 (big), shape lives in bins 1,2
    src = 0
    ea = np.array([
        [100.0, 3.0, 1.0],     # seed 0
        [ 80.0, 6.0, 2.0],     # seed 1
    ])
    per_seed, mean_w, used = mwe.build_w_shape(ea, {0, 1}, src_bin_idx=src, normalize="l1")
    assert per_seed.shape == (2, 2)            # source bin dropped -> 2 non-source bins
    assert np.allclose(per_seed.sum(axis=1), 1.0)   # each seed L1-normalized
    # seed 0 non-source = [3,1] -> [0.75,0.25]; seed 1 = [6,2] -> [0.75,0.25]
    assert np.allclose(per_seed[0], [0.75, 0.25])
    assert np.allclose(mean_w, [0.75, 0.25])
    assert sorted(used) == [0, 1]


def test_build_w_shape_uses_only_success_seeds():
    src = 0
    ea = np.array([
        [10.0, 9.0, 1.0],      # seed 0 (success)
        [10.0, 1.0, 9.0],      # seed 1 (NOT success -> ignored)
    ])
    per_seed, mean_w, used = mwe.build_w_shape(ea, {0}, src_bin_idx=src, normalize="l1")
    assert per_seed.shape == (1, 2)
    assert used == [0]
    assert np.allclose(mean_w, [0.9, 0.1])


def test_build_w_shape_excludes_nan_rows():
    src = 0
    ea = np.array([
        [10.0, 9.0, 1.0],
        [np.nan, np.nan, np.nan],   # no-event seed -> NaN row, must be dropped
    ])
    per_seed, mean_w, used = mwe.build_w_shape(ea, {0, 1}, src_bin_idx=src, normalize="l1")
    assert per_seed.shape == (1, 2)             # NaN seed dropped even though in success set
    assert used == [0]


def test_build_w_shape_raises_on_zero_success():
    src = 0
    ea = np.array([[10.0, 9.0, 1.0]])
    with pytest.raises(ValueError):
        mwe.build_w_shape(ea, set(), src_bin_idx=src, normalize="l1")


# --------------------------------------------------------------------------- #
# C9 — B1a reproducibility vs spatial-bin-shuffled null                         #
# --------------------------------------------------------------------------- #
def test_reproducibility_identical_localized_shapes_pass():
    # 3 seeds, IDENTICAL localized shape -> observed cosine ~1, beats shuffled-bin null
    base = np.array([0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0])
    per_seed = np.vstack([base, base, base])
    res = mwe.w_shape_reproducibility(per_seed, n_null=500, metric="cosine", rng_seed=0)
    assert res["observed"] == pytest.approx(1.0, abs=1e-9)
    assert res["pass"] is True
    assert res["observed"] >= res["null_p95"]


def test_reproducibility_disjoint_shapes_do_not_pass():
    # seeds active in DISJOINT bins -> observed cross-seed cosine = 0, below null p95
    per_seed = np.array([
        [1.0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1.0, 0],
    ])
    res = mwe.w_shape_reproducibility(per_seed, n_null=500, metric="cosine", rng_seed=0)
    assert res["observed"] == pytest.approx(0.0, abs=1e-9)
    assert res["pass"] is False


def test_reproducibility_is_deterministic():
    rng = np.random.default_rng(1)
    per_seed = rng.random((4, 10))
    per_seed /= per_seed.sum(axis=1, keepdims=True)
    a = mwe.w_shape_reproducibility(per_seed, n_null=300, metric="cosine", rng_seed=7)
    b = mwe.w_shape_reproducibility(per_seed, n_null=300, metric="cosine", rng_seed=7)
    assert a["observed"] == b["observed"]
    assert a["null_p95"] == b["null_p95"]
