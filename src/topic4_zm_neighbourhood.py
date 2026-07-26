"""Local slow-state neighbourhood audit and the Branch T / Branch F split
(spec rev3.1 §7, plan Task 8).

The question this phase exists to answer: when the visited states show no carrier, is that because
the fast/slow neighbourhood has none (Branch F), or because the actual slow trajectory merely misses
a nearby window (Branch T)? Getting that backwards would send the next mechanism spec to repair the
wrong object, so the spec demands THREE locked representations and blocks Branch F when they
disagree:

  1. coarse decision representation -- the seven [z_core, z_surround, dz, m_core, m_surround, dm,
     S_G] summaries, robustly standardized, reduced to the first two trajectory directions;
  2. full-field PCA over vectorized [z_i, m_i, S_G];
  3. pathology-axis projections of z_i and m_i (parallel/perpendicular, axial gradient,
     core-boundary displacement).

Neighbourhood states are built by REPLACING the slow fields of a real visited snapshot -- the fast
membrane/synaptic/refractory/delay state stays a naturally occurring microstate, and the slow fields
come from trajectory interpolation or PCA reconstruction, never from independent scalar edits.
"""
from __future__ import annotations

import numpy as np

NEIGHBOURHOOD_VERSION = "zm_neighbourhood_v1.1_2026-07-27_fail_closed"

MAX_SD = 1.0            # §7.2: no coordinate more than one robust trajectory SD from observed range
LATTICE = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-0.7, -0.7), (0.7, 0.7))


# ================================================================ representations
def fit_pca(X, n_modes=2):
    """Deterministic PCA: mean-centred SVD with a fixed sign convention (largest |loading| > 0)."""
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    comps = Vt[:n_modes].copy()
    for i in range(comps.shape[0]):
        j = int(np.argmax(np.abs(comps[i])))
        if comps[i, j] < 0:
            comps[i] *= -1.0
    var = (S ** 2) / max(1, X.shape[0] - 1)
    return dict(mean=mu, components=comps, singular_values=S[:n_modes],
                explained_variance_ratio=(var[:n_modes] / var.sum()).tolist() if var.sum() > 0
                else [0.0] * n_modes)


def coarse_representation(Q_std, n_modes=2):
    """Representation 1: first two directions of the standardized 7-summary trajectory."""
    return fit_pca(Q_std, n_modes=n_modes)


def full_field_representation(states, n_modes=3):
    """Representation 2: PCA over vectorized [z_i (E cells), m_i (E cells), S_G].

    `states` is a list of dicts with keys 'z', 'm', 'S_G' (already restricted to E cells).
    With few snapshots the modes are few; the explained variance is reported so the reader can see
    exactly how thin the basis is instead of it being implied.
    """
    X = np.array([np.concatenate([s["z"], s["m"], [s["S_G"]]]) for s in states], float)
    n_modes = min(n_modes, max(1, X.shape[0] - 1))
    pca = fit_pca(X, n_modes=n_modes)
    pca["n_samples"] = int(X.shape[0])
    pca["dim"] = int(X.shape[1])
    return pca


def pathology_axis_projection(z, m, axis_coord, core_mask):
    """Representation 3: preregistered parallel/perpendicular structure of the slow fields.

    axial_gradient      slope of the field along the source->sink axis (per mm)
    core_boundary_disp  core-minus-surround contrast, i.e. how far the field's edge sits from the
                        low-threshold core
    """
    a = np.asarray(axis_coord, float)
    out = {}
    for name, v in (("z", np.asarray(z, float)), ("m", np.asarray(m, float))):
        A = np.vstack([a, np.ones_like(a)]).T
        slope, _ = np.linalg.lstsq(A, v, rcond=None)[0]
        out[f"{name}_axial_gradient"] = float(slope)
        out[f"{name}_core"] = float(v[core_mask].mean())
        out[f"{name}_surround"] = float(v[~core_mask].mean())
        out[f"{name}_core_boundary_disp"] = float(v[core_mask].mean() - v[~core_mask].mean())
        resid = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        out[f"{name}_perp_sd"] = float(np.std(resid))
    return out


# ================================================================ neighbourhood construction
def trajectory_scale(Q_std):
    """Robust per-direction SD of the standardized trajectory (the §7.2 displacement budget)."""
    return float(np.median(np.linalg.norm(Q_std - Q_std.mean(axis=0), axis=1)))


def build_lattice(anchor_q, pca, scale, lattice=LATTICE, max_sd=MAX_SD):
    """q_test = q_anchor + a*u1 + b*u2, clipped to <= max_sd robust trajectory SDs.

    Clipping that would silently change the DIRECTION is rejected: a displacement is either kept at
    its requested direction with a reduced magnitude, or dropped.
    """
    u1, u2 = pca["components"][0], pca["components"][1]
    out = []
    for a, b in lattice:
        d = a * u1 + b * u2
        nrm = float(np.linalg.norm(d))
        if nrm <= 0:
            continue
        budget = max_sd * scale
        mag = min(nrm, budget)
        out.append(dict(a=float(a), b=float(b), direction=(d / nrm).tolist(),
                        magnitude=float(mag), clipped=bool(mag < nrm),
                        q=(np.asarray(anchor_q, float) + d / nrm * mag).tolist()))
    return out


def interpolate_fields(field_a, field_b, lam):
    """Trajectory interpolation between two REAL visited slow fields (spec §7.2 / §10.1)."""
    lam = float(np.clip(lam, 0.0, 1.0))
    return (1.0 - lam) * np.asarray(field_a, float) + lam * np.asarray(field_b, float)


def reconstruct_full_field(pca, coeffs):
    """Full-field reconstruction from the field PCA: mean + sum_k c_k u_k (never a scalar edit)."""
    v = np.asarray(pca["mean"], float).copy()
    for k, c in enumerate(coeffs):
        if k < pca["components"].shape[0]:
            v = v + float(c) * pca["components"][k]
    return v


def split_full_field(vec, nE):
    return dict(z=np.clip(vec[:nE], 0.0, 1.0), m=np.maximum(vec[nE:2 * nE], 0.0),
                S_G=float(np.clip(vec[2 * nE], 0.0, 1.0)))


# ================================================================ branch verdict
def branch_verdict(visited_positive, local_positive_seeds, eligible_seeds,
                   representations_agree, n_required_seeds=2, *,
                   local_negative_seeds=(), evidence_complete=False):
    """The §7.3 pure decision. Unknown / missing evidence NEVER falls through to Branch F."""
    if visited_positive:
        return dict(verdict="carrier_at_visited_states", reason="carrier found at visited states")
    if not representations_agree:
        return dict(verdict="representation_sensitive_no_branch",
                    reason="coarse and spatial representations disagree; Branch F is blocked")
    n_local = len(set(local_positive_seeds))
    if n_local >= n_required_seeds:
        return dict(verdict="branch_T_slow_trajectory_repair",
                    reason=f"local carrier window in {n_local} eligible primary seeds")
    n_negative = len(set(local_negative_seeds))
    if (evidence_complete and len(set(eligible_seeds)) >= 3 and
            n_negative >= 3 and n_local == 0):
        return dict(verdict="branch_F_fast_carrier_repair",
                    reason="no carrier at visited states or in the local neighbourhood, "
                           "replicated across three eligible seeds, representations agree")
    return dict(verdict="no_evidence",
                reason=f"incomplete local audit: {len(set(eligible_seeds))} eligible seeds, "
                       f"{n_local} local-positive, {n_negative} fully local-negative, "
                       f"evidence_complete={bool(evidence_complete)}")
