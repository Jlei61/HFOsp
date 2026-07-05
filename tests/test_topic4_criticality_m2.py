"""TDD tests for Topic 4 M3-v2.2 criticality Milestone 2 (Task 0, plan Step 2).

Contract (verbatim from task brief .superpowers/sdd/task-0-brief.md Step 2):
  - shape scores behave sanely on synthetic anisotropic blobs (port of pilot4).
  - basis_vectors returns unit-norm, mutually orthogonal e_global / e_axis_gradient.
  - load_m2_config resolves the "THETA_EE" sentinel to a float.
"""
import json

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


# --- Task 1: dense alpha0 localization (bracket -> coarse scan -> bisect), verbatim from
# task brief .superpowers/sdd/task-1-brief.md Step 1 ---
def _points():
    p = m2._REPO / "results/topic4_criticality/trajectory_verdict.json"   # M1 deliverable
    return json.loads(p.read_text())["points"]


def test_localize_alpha0_crossing_brackets_zero():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    out = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    assert out["crossing_status"] in ("single", "multiple_alpha0_crossings")
    assert out["alpha_left"] < 0.0                       # last neg before crossing
    assert out["crossing_frac"] is not None
    assert 470.0 < out["alpha0_crossing_time_ms"] < 520.0  # M1 idx14->idx15 window


# --- T1 review FIX 1 (Critical): op_solve_quality is a FOLD-APPROPRIATE residual bar, not the
# solver's strict 1e-9 converged flag. Near-fold ops (residual ~1e-3-4e-3) never hit converged=True
# yet read a stable spectrum, so both sides must still be quality=True. ---
def test_op_solve_quality_is_residual_based_not_strict_converged():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    out = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    assert out["op_solve_quality_left"] is True
    assert out["op_solve_quality_right"] is True
    op = out["_crossing_op"]
    assert op is not None
    # documents the rule is residual-based: the crossing op is NOT strictly converged (>1e-9) yet
    # sits within the fold-appropriate tolerance -> quality=True despite converged=False.
    assert float(op.residual) > 1e-9
    assert float(op.residual) <= m2cfg["densification"]["op_residual_tol"]
    assert op.converged is False


# --- T1 review FIX 2 (spec §3.1 output gap): branch_identity_clean must be present in the return
# (feeds the §5.0 ignition base gate) and True on the real crossing, where M1's continuation check
# reports the low branch stays continuous and reaches alpha0 (no branch jump/fold). ---
def test_branch_identity_clean_present_and_true_on_real_crossing():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    out = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    assert "branch_identity_clean" in out
    assert out["branch_identity_clean"] is True
    assert out["_branch_continuation_status"] == "low_branch_reaches_alpha0_before_jump"


# --- Task 2: linear_ignition readout + two-core symmetry-break confirmation, verbatim from
# task brief .superpowers/sdd/task-2-brief.md Step 1 ---
def test_linear_ignition_core_localized_on_real_crossing():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config(); pts = _points()
    crossing = m2.localize_alpha0_crossing(pts, grid, kernels, core, cfg, m2cfg)
    ig = m2.read_linear_ignition(crossing, grid, kernels, core, cfg, m2cfg, pts)
    assert ig["class"] == "core_localized"
    assert ig["core_overlap"] >= m2cfg["ignition"]["core_localized_overlap_thresh"]
    assert ig["globality"] <= m2cfg["ignition"]["core_localized_globality_thresh"]
    assert ig["two_core_symmetry_break"] is True
    assert ig["corridor_power"] <= m2cfg["two_core_confirm"]["corridor_dark_thresh"]
    assert "post-fold" in ig["near_fold_note"] and "symmetric" in ig["near_fold_note"]


def test_ignition_class_delocalized_on_global_loading():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    cls, sub = m2._classify_ignition(core_overlap=0.2, globality=0.8,
                                     axis_elongation=0.0, off_axis=0.0,
                                     corridor_power=0.0, n_core_peaks=1, m2cfg=m2cfg)
    assert cls == "delocalized" and sub == "global_like"


# --- Task 3: projected gain/leak + nonaxis off_axis sentinel, verbatim from task brief
# .superpowers/sdd/task-3-brief.md Step 1 ---
def test_off_axis_sentinel_absent_on_core_localized_crossing():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    crossing = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    s = m2.off_axis_sentinel(crossing, grid, kernels, core, m2cfg)
    assert s["off_axis"] == "absent"
    assert "core-compactness residual" in s["annotation"]


def test_off_axis_present_requires_both_gates():
    m2cfg = m2.load_m2_config()
    # score gate open but gain gate closed -> NOT present
    v = m2._off_axis_decision(off_axis_score=0.09, gain_nonaxis=0.10,
                              gain_axis=0.20, gain_global=0.05, m2cfg=m2cfg)
    assert v == "undetermined"
    v2 = m2._off_axis_decision(off_axis_score=0.09, gain_nonaxis=0.40,
                               gain_axis=0.10, gain_global=0.05, m2cfg=m2cfg)
    assert v2 == "present"
    v3 = m2._off_axis_decision(off_axis_score=0.01, gain_nonaxis=0.01,
                               gain_axis=0.20, gain_global=0.05, m2cfg=m2cfg)
    assert v3 == "absent"
