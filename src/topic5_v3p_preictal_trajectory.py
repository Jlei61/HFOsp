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
    or <2 surviving windows -> nan."""
    y = np.asarray(values, float); t = np.asarray(centers, float)
    cov = [np.asarray(c, float) for c in covariates]
    m = np.isfinite(y) & np.isfinite(t)
    for c in cov:
        m &= np.isfinite(c)
    if m.sum() < 3:
        return float("nan")
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in cov])
    try:
        beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    resid = y[m] - X @ beta
    return slope_over_windows(resid, t[m], estimator)
