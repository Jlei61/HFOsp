"""Topic 4 M3-v2.2 criticality Milestone 2 — two-stage ignition/spread readout.

Productionizes the M2 de-risk pilots (results/topic4_criticality_m2/pilots/*.py).
Spec: docs/superpowers/specs/2026-07-04-topic4-m3v2-2-m2-critical-mode-decomposition-design.md (rev2.1).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import yaml
import src.topic4_m3b_spectral_phase as spm

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _REPO / "config/topic4_criticality_m2.yaml"


def load_m2_config(path=None) -> dict:
    cfg = yaml.safe_load(Path(path or _DEFAULT_CFG).read_text())
    if cfg["basis"].get("theta") == "THETA_EE":
        cfg["basis"]["theta"] = float(spm.THETA_EE)
    return cfg


def basis_vectors(grid, theta) -> dict:
    X, Y = grid.coords()
    e_global = np.ones(X.size); e_global /= np.linalg.norm(e_global)
    s = (X * np.cos(theta) + Y * np.sin(theta)).ravel(); s = s - s.mean()
    e_axis = s - (s @ e_global) * e_global
    e_axis /= (np.linalg.norm(e_axis) + 1e-300)
    return {"e_global": e_global, "e_axis_gradient": e_axis}


def nonaxis_direction(loading, grid, theta, min_norm):
    v = np.asarray(loading, float).ravel(); nv = float(np.linalg.norm(v))
    b = basis_vectors(grid, theta)
    proj_g = (v @ b["e_global"]) * b["e_global"]
    proj_a = (v @ b["e_axis_gradient"]) * b["e_axis_gradient"]
    residual = v - proj_g - proj_a
    rn = float(np.linalg.norm(residual))
    frac = lambda x: float(np.linalg.norm(x) / (nv + 1e-300))
    e_nonaxis = residual / rn if rn >= min_norm else None
    return e_nonaxis, frac(residual), frac(proj_g), frac(proj_a)


def shape_scores_at(res, grid, kernels, core) -> dict:
    idxs = spm.leading_subspace_indices(res.eigenvalues, min_sep=1e-3, imag_tol=1e-3)
    loading = spm.pair_loading(res.right, idxs, grid)
    th = kernels.theta; lead = res.eigenvalues[0]
    return {
        "axis_elongation": float(spm.elongation_axis_score(loading, grid, th)),
        "axis_wavevector_alignment": float(spm.phase_gradient_axis_score(loading, grid, th)),
        "off_axis": float(spm.off_axis_score(loading, grid, th)),
        "globality": float(spm.globality(loading, grid)),
        "core_overlap": float(spm.core_overlap(loading, grid, core)),
        "leading_subspace_dim": int(len(idxs)),
        "leading_is_complex_pair": bool(len(idxs) == 2 and abs(lead.imag) > 1e-3),
        "leading_eigenvalue_real": float(lead.real),
        "leading_eigenvalue_imag": float(lead.imag),
        "_loading": loading,
    }
