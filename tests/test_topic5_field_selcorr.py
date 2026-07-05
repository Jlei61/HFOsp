"""Tests for src/topic5_field_selcorr.py (selection-corrected field-concordance null)."""
import numpy as np

import src.topic5_field_selcorr as sc
from src.propagation_contact_plane_readout import make_plane_grid, R_smooth_rank, S_THRESH


def test_smoother_matches_R_smooth_rank():
    X, Y = make_plane_grid()
    names = [f"C{i}" for i in range(6)]; xs = np.linspace(0.1, 0.9, 6)
    chans = [{"name": n, "x_norm": float(x), "y_norm": 0.0, "support": 0.5 + 0.1 * i}
             for i, (n, x) in enumerate(zip(names, xs))]
    vals = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sigma = 0.2
    # reference via smooth_field
    rec = {"channels": [dict(c, typical_rank=float(v)) for c, v in zip(chans, vals)]}
    F_ref = R_smooth_rank(rec, X, Y, sigma, S_THRESH)
    # matmul smoother
    sm = sc.precompute_smoother(chans, X, Y, sigma)
    F = sc.field_from_values(sm, vals)
    m = np.isfinite(F_ref["T"]) & np.isfinite(F["T"])
    assert m.sum() > 100
    assert np.allclose(F["T"][m], F_ref["T"][m], atol=1e-9)
    assert np.allclose(F["S"], F_ref["S"], atol=1e-9)


def test_null_aligns_vectorized_matches_scalar():
    from src.propagation_contact_plane_readout import corr_pair_mirror_invariant, OVERLAP_MIN
    X, Y = make_plane_grid()
    rng = np.random.default_rng(7)
    names = [f"C{i}" for i in range(8)]
    xs = np.linspace(0.05, 0.95, 8); ys = rng.uniform(-0.4, 0.4, 8)
    chans = [{"name": n, "x_norm": float(x), "y_norm": float(y), "support": float(0.4 + 0.1 * i)}
             for i, (n, x, y) in enumerate(zip(names, xs, ys))]
    sigma = 0.25
    # interictal field (rank) fixed
    F_inter = R_smooth_rank({"channels": [dict(c, typical_rank=float(i) / 7) for i, c in enumerate(chans)]},
                            X, Y, sigma, S_THRESH)
    sm = sc.precompute_smoother(chans, X, Y, sigma)
    v = rng.uniform(0, 1, 8)
    V = np.array([v[rng.permutation(8)] for _ in range(20)])           # 20 permuted draws
    vec = sc.null_aligns_vectorized(F_inter, sm, V, S_THRESH, OVERLAP_MIN)
    # scalar reference per draw
    for b in range(20):
        Fj = sc.field_from_values(sm, V[b])
        r = corr_pair_mirror_invariant(F_inter["T"], F_inter["S"], Fj["T"], Fj["S"], S_THRESH, OVERLAP_MIN)["corr"]
        ref = abs(r) if (r is not None and np.isfinite(r)) else np.nan
        if np.isnan(ref):
            assert np.isnan(vec[b])
        else:
            assert abs(vec[b] - ref) < 1e-9, (b, vec[b], ref)


def test_selcorr_passes_when_real_beats_max_null():
    rng = np.random.default_rng(0)
    real = {"a": 0.9, "b": 0.4}
    nd = {"a": list(rng.normal(0.3, 0.05, 500)), "b": list(rng.normal(0.35, 0.05, 500))}
    out = sc.selection_corrected_pvalue(real, nd)
    assert out["status"] == "ok" and out["best_candidate"] == "a"
    assert out["pass_selcorr"] and out["p_selcorr"] < 0.05


def test_selcorr_fails_when_max_null_exceeds_real():
    rng = np.random.default_rng(1)
    real = {"a": 0.5, "b": 0.5}
    # both nulls routinely exceed 0.5 -> selection-corrected should NOT pass
    nd = {"a": list(rng.normal(0.6, 0.05, 500)), "b": list(rng.normal(0.6, 0.05, 500))}
    out = sc.selection_corrected_pvalue(real, nd)
    assert not out["pass_selcorr"] and out["p_selcorr"] > 0.05


def test_selcorr_max_null_ge_single_candidate_null():
    # the selection-corrected null is harder than any single candidate's null (p larger or equal)
    rng = np.random.default_rng(2)
    real = {"a": 0.7, "b": 0.7, "c": 0.7}
    nd = {c: list(rng.normal(0.5, 0.08, 800)) for c in ("a", "b", "c")}
    out = sc.selection_corrected_pvalue(real, nd)
    # single-candidate p for 'a'
    a = np.asarray(nd["a"]); p_single = (np.sum(a >= 0.7) + 1) / (len(a) + 1)
    assert out["p_selcorr"] >= p_single - 1e-9      # selection makes it no easier
