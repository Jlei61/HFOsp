"""Contract tests for the D_AB 3D gradient axis (methods_axis_gradient_rewrite.md).

Each test pins one contract clause from the spec:
  C1 axis = least-squares gradient beta direction, oriented so axial projection
     correlates POSITIVELY with D_AB (B-lead pole -> A-lead pole).
  C2 rotation-equivariant / translation-invariant (it is a spatial direction).
  C3 extreme-tercile pole centroids are DISPLAY ONLY -- they never enter beta;
     the display arrow (p_B -> p_A) is parallel to u, not the raw-centroid connector.
  C4 fail-closed degeneracy: no D_AB variance / |beta|~0 / <6 contacts -> not defined.
  C5 numeric QC present: n, sd_dab, beta_norm, R2, matrix rank, condition number,
     within-shaft fraction, leave-one-shaft-out cosine.
  C6 provenance tag axis_definition == 'dab_gradient_v1'.
  C7 rank-deficient (single-shaft/collinear) coords are FLAGGED (full_rank False),
     axis still returned as the min-norm gradient in the sampled subspace.
"""
import numpy as np
import pytest

from src.dab_gradient_axis import compute_dab_gradient_axis, dab_from_ranks


def _synthetic(n=18, direction=(0.0, 1.0, 0.0), noise=0.05, seed=0):
    """n contacts on 3 shafts; D_AB is a linear ramp along `direction` + small noise."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-30, 30, size=(n, 3))
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    dab = (coords - coords.mean(0)) @ u + noise * rng.standard_normal(n)
    shafts = np.array([f"S{i % 3}" for i in range(n)])
    return coords, dab, shafts, u


# ---- C1: axis = beta direction, oriented positive with D_AB --------------------
def test_axis_matches_lstsq_beta_direction():
    coords, dab, shafts, _ = _synthetic()
    r = compute_dab_gradient_axis(coords, dab, shafts)
    Xc = coords - coords.mean(0)
    beta, *_ = np.linalg.lstsq(Xc, dab - dab.mean(), rcond=None)
    u_beta = beta / np.linalg.norm(beta)
    assert abs(float(np.dot(r["u"], u_beta))) > 0.999  # parallel (sign handled below)


def test_orientation_positive_with_dab():
    coords, dab, shafts, _ = _synthetic()
    r = compute_dab_gradient_axis(coords, dab, shafts)
    along = (coords - coords.mean(0)) @ r["u"]
    assert np.corrcoef(along, dab)[0, 1] > 0  # B-lead(low D_AB) -> A-lead(high D_AB)


def test_recovers_planted_direction():
    coords, dab, shafts, u_true = _synthetic(noise=0.01)
    r = compute_dab_gradient_axis(coords, dab, shafts)
    assert abs(float(np.dot(r["u"], u_true))) > 0.98


# ---- C2: rotation-equivariance / translation-invariance ------------------------
def test_rotation_equivariance():
    coords, dab, shafts, _ = _synthetic()
    r0 = compute_dab_gradient_axis(coords, dab, shafts)
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    r1 = compute_dab_gradient_axis(coords @ R.T, dab, shafts)
    assert abs(float(np.dot(r1["u"], R @ r0["u"]))) > 0.999


def test_translation_invariance():
    coords, dab, shafts, _ = _synthetic()
    r0 = compute_dab_gradient_axis(coords, dab, shafts)
    r1 = compute_dab_gradient_axis(coords + np.array([100.0, -50.0, 20.0]), dab, shafts)
    assert abs(float(np.dot(r0["u"], r1["u"]))) > 0.999


# ---- C3: poles display-only; arrow parallel to u -------------------------------
def test_pole_arrow_is_parallel_to_axis():
    coords, dab, shafts, _ = _synthetic()
    r = compute_dab_gradient_axis(coords, dab, shafts)
    arrow = np.asarray(r["p_A"]) - np.asarray(r["p_B"])
    assert abs(float(np.dot(arrow / np.linalg.norm(arrow), r["u"]))) > 0.999
    # raw centroids reported separately + pole separation is 3D (not the projected span)
    assert "mu_A" in r and "mu_B" in r
    assert r["L_poles"] >= np.linalg.norm(arrow) - 1e-6


# ---- C4: fail-closed degeneracy ------------------------------------------------
def test_constant_dab_not_defined():
    coords, _, shafts, _ = _synthetic()
    r = compute_dab_gradient_axis(coords, np.zeros(len(coords)), shafts)
    assert r["status"] != "ok" and r["u"] is None


def test_insufficient_contacts():
    coords, dab, shafts, _ = _synthetic(n=5)
    r = compute_dab_gradient_axis(coords, dab, shafts)
    assert r["status"] == "insufficient_contacts" and r["u"] is None


def test_nan_coords_dropped_then_gated():
    coords, dab, shafts, _ = _synthetic(n=7)
    coords[:3] = np.nan  # only 4 valid -> below the 6-contact floor
    r = compute_dab_gradient_axis(coords, dab, shafts)
    assert r["status"] == "insufficient_contacts"


# ---- C5 / C7: QC outputs + rank-deficient flag ---------------------------------
def test_qc_fields_present():
    coords, dab, shafts, _ = _synthetic()
    r = compute_dab_gradient_axis(coords, dab, shafts)
    for k in ("n", "sd_dab", "beta_norm", "R2", "matrix_rank", "condition_number",
              "within_shaft_frac", "loso_cosine", "n_shafts"):
        assert k in r and r[k] is not None


def test_rank_deficient_single_shaft_flagged():
    """All contacts collinear (single electrode) -> rank 1, full_rank False, but u
    is still returned as the min-norm gradient (spec numerical-QC clause)."""
    rng = np.random.default_rng(1)
    t = np.linspace(-20, 20, 10)
    line = np.array([1.0, 2.0, -1.0]); line /= np.linalg.norm(line)
    coords = np.outer(t, line) + np.array([3, -2, 5])
    dab = t / 10 + 0.01 * rng.standard_normal(10)
    shafts = np.array(["A"] * 10)
    r = compute_dab_gradient_axis(coords, dab, shafts)
    assert r["status"] == "ok"
    assert r["matrix_rank"] == 1 and r["full_rank"] is False
    assert r["u"] is not None


def test_loso_cosine_high_for_clean_axis():
    coords, dab, shafts, _ = _synthetic(noise=0.02)
    r = compute_dab_gradient_axis(coords, dab, shafts)
    assert r["loso_cosine"] > 0.9  # dropping any one shaft barely moves the axis


# ---- C6: provenance ------------------------------------------------------------
def test_provenance_tag():
    coords, dab, shafts, _ = _synthetic()
    r = compute_dab_gradient_axis(coords, dab, shafts)
    assert r["axis_definition"] == "dab_gradient_v1"


# ---- reuse: dab_from_ranks wraps build_D_AB ------------------------------------
def test_dab_from_ranks_matches_build_D_AB():
    from src.topic5_scaffold_ab_contrast import build_D_AB
    rng = np.random.default_rng(2)
    ra = rng.permutation(12).astype(float)
    rb = rng.permutation(12).astype(float)
    assert np.allclose(dab_from_ranks(ra, rb), build_D_AB(ra, rb)["D_AB"])
