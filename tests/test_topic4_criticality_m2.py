"""TDD tests for Topic 4 M3-v2.2 criticality Milestone 2 (Task 0, plan Step 2).

Contract (verbatim from task brief .superpowers/sdd/task-0-brief.md Step 2):
  - shape scores behave sanely on synthetic anisotropic blobs (port of pilot4).
  - basis_vectors returns unit-norm, mutually orthogonal e_global / e_axis_gradient.
  - load_m2_config resolves the "THETA_EE" sentinel to a float.
"""
import numpy as np

from src.topic4_criticality import load_crit_config, _crit_op_context
import src.topic4_criticality_m2 as m2


def _ctx():
    grid, kernels, core, b_core = _crit_op_context(load_crit_config())
    return grid, kernels, core, b_core


def _gauss(grid, theta, sig_par, sig_perp, ang):
    X, Y = grid.coords(); cx, cy = X.mean(), Y.mean()
    u = (X - cx) * np.cos(ang) + (Y - cy) * np.sin(ang)
    w = -(X - cx) * np.sin(ang) + (Y - cy) * np.cos(ang)
    return np.exp(-0.5 * ((u / sig_par) ** 2 + (w / sig_perp) ** 2))


def test_shape_scores_sanity_on_synthetic_blobs():
    grid, kernels, core, _ = _ctx(); th = kernels.theta
    import src.topic4_m3b_spectral_phase as spm
    along = _gauss(grid, th, 1.6, 0.5, th)
    perp = _gauss(grid, th, 1.6, 0.5, th + np.pi / 2)
    uniform = np.ones_like(along)
    assert spm.elongation_axis_score(along, grid, th) > 0.3
    assert spm.elongation_axis_score(along, grid, th) > spm.off_axis_score(along, grid, th)
    assert spm.off_axis_score(perp, grid, th) > 0.3 and spm.elongation_axis_score(perp, grid, th) < 0
    assert spm.globality(uniform, grid) > 0.9 and abs(spm.elongation_axis_score(uniform, grid, th)) < 0.05


def test_basis_vectors_unit_norm_and_orthogonal():
    grid, kernels, _, _ = _ctx()
    b = m2.basis_vectors(grid, kernels.theta)
    assert abs(np.linalg.norm(b["e_global"]) - 1.0) < 1e-9
    assert abs(np.linalg.norm(b["e_axis_gradient"]) - 1.0) < 1e-9
    assert abs(float(b["e_global"] @ b["e_axis_gradient"])) < 1e-9   # axis grad is zero-mean


def test_load_m2_config_resolves_theta():
    cfg = m2.load_m2_config()
    assert cfg["basis"]["off_axis_score_tol"] == 0.05
    assert isinstance(cfg["basis"]["theta"], float)      # "THETA_EE" -> np.pi/4
