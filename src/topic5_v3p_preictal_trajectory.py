"""Topic 5 V3p — preictal-only non-axial trajectory (pure math).

READ-ONLY reuse of the V3a stack (src.topic5_v3_mode_transition,
scripts._topic5_v3_io). V3p adds only the preictal-restriction + slope +
residualization + N->N self-sustain layer. See
docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md.
Exploratory; no forecasting.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Mapping, Sequence
import numpy as np
import yaml
from scipy import stats

from src.topic5_v2_criticality import activations_from_z
from src.topic5_v3_mode_transition import (
    atm_offdiag,
    atm_lag0,
    net_offaxis_flux,
    demean_window,
    lowrank_var,
    dominant_right_singular_vector,
    map_lowrank_vector_to_contacts,
    subspace_mode_shift,
    project_2d,
    direct_2d_var,
    beta_axis,
)

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config" / "topic5_v3p.yaml"

def load_v3p_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path is not None else _DEFAULT_CFG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"V3p config must be a mapping: {cfg_path}")
    return dict(cfg)

def _finite_pairs(y, t):
    y = np.asarray(y, float); t = np.asarray(t, float)
    m = np.isfinite(y) & np.isfinite(t)
    return y[m], t[m]

def theil_sen_slope(y, t) -> float:
    y, t = _finite_pairs(y, t)
    if y.size < 2 or np.unique(t).size < 2:
        return float("nan")
    return float(stats.theilslopes(y, t)[0])

def spearman_trend(y, t) -> float:
    y, t = _finite_pairs(y, t)
    if y.size < 3 or np.unique(y).size < 2 or np.unique(t).size < 2:
        return float("nan")
    return float(stats.spearmanr(y, t).correlation)

def slope_over_windows(values, centers, estimator) -> float:
    if estimator == "theil_sen":
        return theil_sen_slope(values, centers)
    if estimator == "spearman":
        return spearman_trend(values, centers)
    if estimator == "ols":
        y, t = _finite_pairs(values, centers)
        if y.size < 2 or np.unique(t).size < 2:
            return float("nan")
        return float(np.polyfit(t, y, 1)[0])
    raise ValueError(f"unknown estimator: {estimator!r}")

def within_compartment_flux(atm, idx, normalization="source_mean") -> float:
    """Self-sustain: mean over ACTIVE sources i in idx of that source's
    outgoing mass into the SAME set idx. Mirrors V3a compartment_flux
    source_mean but for the N x N block. Requires a diagonal-free ATM."""
    mat = np.asarray(atm, float)
    if not np.allclose(np.diag(mat), 0.0):
        raise ValueError("within_compartment_flux requires a diagonal-free ATM")
    idx = np.asarray(idx, int)
    if idx.size == 0:
        return 0.0
    active = mat[idx].sum(axis=1) > 0.0
    if not np.any(active):
        return 0.0
    block_mass = mat[np.ix_(idx, idx)].sum(axis=1)   # into the same set (diag already 0)
    if normalization == "source_mean":
        return float(block_mass[active].mean())
    if normalization == "sum":
        return float(block_mass.sum())
    raise ValueError(f"unknown normalization: {normalization!r}")

def global_axial_energy(env_win, axis_rows) -> tuple[float, float]:
    """Per-window energy scalars: mean over rows of the within-window mean
    |envelope|. global = all rows; axial = axis rows only (0.0 if none)."""
    env = np.asarray(env_win, float)
    row_energy = np.nanmean(np.abs(env), axis=1)
    g = float(np.nanmean(row_energy)) if row_energy.size else float("nan")
    axis_rows = np.asarray(axis_rows, int)
    a = float(np.nanmean(row_energy[axis_rows])) if axis_rows.size else 0.0
    return g, a

def residualize_slope(values, centers, covariates, estimator) -> float:
    """Slope of the residual of `values` after OLS-regressing on
    `covariates` (each an array aligned to `values`). Conservative: if a
    covariate is collinear with time, the shared trend is absorbed and the
    residual slope shrinks toward 0 — this is the documented floor (spec
    Sec 7), NOT evidence the non-axis rise is absent. NaN-safe: windows
    with any non-finite value/covariate are dropped; rank-deficient design
    or fewer than `len(covariates)+2` surviving windows -> nan."""
    y = np.asarray(values, float); t = np.asarray(centers, float)
    cov = [np.asarray(c, float) for c in covariates]
    m = np.isfinite(y) & np.isfinite(t)
    for c in cov:
        m &= np.isfinite(c)
    if m.sum() < len(cov) + 2:
        return float("nan")
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in cov])
    try:
        beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    resid = y[m] - X @ beta
    return slope_over_windows(resid, t[m], estimator)

_WINDOW_METRIC_KEYS = (
    "net_offaxis_flux_lag1", "net_offaxis_flux_lag0",
    "mode_shift_density", "mode_singular_gap",
    "nonaxis_activation_rate", "n_activation_events",
    "global_energy", "axial_energy",
    "N_self_sustain_lag1", "N_self_sustain_lag0",
    "gain_axis", "gain_nonaxis", "beta_axis_strength",
)

def extract_window_metrics(env_win, geom, v3cfg) -> dict:
    """One preictal window's bb-envelope -> a dict of scalar metrics (Task 7 trajectory atom).

    ``env_win`` is ``(n_all_clean, n_t)``, rows ordered by ``geom["names"]``.
    Reuses the frozen V3a dynamics/avalanche math
    (``src.topic5_v3_mode_transition``) + V2 ``activations_from_z`` + this
    module's Task-3 ``within_compartment_flux``/``global_axial_energy``.
    Every estimator below is independently try/except-guarded: a degenerate
    window (``n_t < 2``, an empty axis/non-axis index, a rank-0 SVD) leaves
    only the affected key(s) ``nan`` rather than raising -- Task 7 loops
    this over many windows per subject and one bad window must not kill the
    whole trajectory.
    """
    env = np.asarray(env_win, dtype=float)
    names = geom["names"]
    axis_idx = np.asarray(geom["axis_idx"], dtype=int)
    nonaxis_idx = np.asarray(geom["nonaxis_idx"], dtype=int)
    P_A, P_N = geom["P_A"], geom["P_N"]
    e_axis_mean, e_nonaxis_mean = geom["e_axis_mean"], geom["e_nonaxis_mean"]
    rank_fwd = geom["rank_forward"]

    dyn = v3cfg["dynamics"]
    lowrank = int(dyn["lowrank"])
    k_star = int(dyn["finite_horizon_k"])
    alpha = float(dyn["var_ridge_alpha"])
    z_thr = float(v3cfg["avalanche"]["z_threshold"])

    out = {k: float("nan") for k in _WINDOW_METRIC_KEYS}

    active = None
    try:
        active = activations_from_z(env, z_thr)
        out["n_activation_events"] = int(active.sum())
    except Exception:
        active = None

    if active is not None:
        try:
            out["nonaxis_activation_rate"] = float(active[nonaxis_idx].mean())
        except Exception:
            pass

        atm1 = None
        try:
            atm1 = atm_offdiag(active)
        except Exception:
            atm1 = None
        if atm1 is not None:
            try:
                out["net_offaxis_flux_lag1"] = net_offaxis_flux(atm1, axis_idx, nonaxis_idx, "source_mean")
            except Exception:
                pass
            try:
                out["N_self_sustain_lag1"] = within_compartment_flux(atm1, nonaxis_idx)
            except Exception:
                pass

        atm0 = None
        try:
            atm0 = atm_lag0(active)
        except Exception:
            atm0 = None
        if atm0 is not None:
            try:
                out["net_offaxis_flux_lag0"] = net_offaxis_flux(atm0, axis_idx, nonaxis_idx, "source_mean")
            except Exception:
                pass
            try:
                out["N_self_sustain_lag0"] = within_compartment_flux(atm0, nonaxis_idx)
            except Exception:
                pass

    A_lr = U_r = None
    try:
        A_lr, U_r = lowrank_var(env, lowrank, alpha)
    except Exception:
        A_lr, U_r = None, None
    if A_lr is not None:
        try:
            sv = np.linalg.svd(np.linalg.matrix_power(A_lr, k_star), compute_uv=False)
            if sv.size >= 2 and sv[1] != 0:
                out["mode_singular_gap"] = float(sv[0] / sv[1])
        except Exception:
            pass
        try:
            u_r = dominant_right_singular_vector(A_lr, k_star)
            u_c = map_lowrank_vector_to_contacts(u_r, U_r)
            out["mode_shift_density"] = subspace_mode_shift(u_c, P_N, P_A, "density")
        except Exception:
            pass

    try:
        out["global_energy"], out["axial_energy"] = global_axial_energy(env, axis_idx)
    except Exception:
        pass

    try:
        Z = project_2d(demean_window(env), e_axis_mean, e_nonaxis_mean)
        B = direct_2d_var(Z, alpha)
        out["gain_axis"] = float(np.linalg.norm(B[:, 0]))
        out["gain_nonaxis"] = float(np.linalg.norm(B[:, 1]))
    except Exception:
        pass

    try:
        metric_by_name = {}
        for i, name in enumerate(names):
            diff_abs = np.abs(np.diff(env[i]))
            finite = diff_abs[np.isfinite(diff_abs)]
            metric_by_name[name] = float(np.mean(finite)) if finite.size else float("nan")
        out["beta_axis_strength"] = abs(beta_axis(metric_by_name, rank_fwd))
    except Exception:
        pass

    return out

def null_slope_distribution(resample_traj_fn, estimator, n_perm, rng) -> np.ndarray:
    out = np.empty(int(n_perm), float)
    for p in range(int(n_perm)):
        per_sz = resample_traj_fn(rng)
        slopes = [slope_over_windows(v, c, estimator) for v, c in per_sz]
        slopes = [s for s in slopes if np.isfinite(s)]
        out[p] = float(np.median(slopes)) if slopes else float("nan")
    return out

def surplus_and_p(obs_slope, null_slopes, direction) -> dict:
    null = np.asarray(null_slopes, float)
    null = null[np.isfinite(null)]
    n = null.size
    if n == 0 or not np.isfinite(obs_slope):
        return {"surplus": float("nan"), "p": float("nan"), "z": float("nan")}
    med = float(np.median(null))
    if direction == "greater":
        p = (1 + int(np.sum(null >= obs_slope))) / (1 + n)
    elif direction == "less":
        p = (1 + int(np.sum(null <= obs_slope))) / (1 + n)
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    mad = 1.4826 * float(np.median(np.abs(null - med)))
    z = (obs_slope - med) / mad if mad > 0 else float("nan")
    return {"surplus": float(obs_slope - med), "p": float(p), "z": float(z)}
