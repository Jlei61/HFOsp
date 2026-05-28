"""Stage 1 parameter sweep + baseline picker.

evaluate_cell composes sim+detect+classify into one call.
sweep_hr_parameters runs Cartesian product over (I, r, sigma, seed) with
joblib parallel.
pick_excitable_baseline applies Stage 1 exit-contract criteria.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .hr_config import BurstConfig, RegimeConfig
from .hr_core import HRParams
from .hr_dynamics import classify_regime, detect_bursts, simulate_trajectory


def evaluate_cell(
    params: HRParams,
    I: float, sigma_ou: float, tau_ou: float,
    r_override: float | None,
    T: float, dt: float, seed: int,
    burst_cfg: BurstConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
    burn_in: float = 100.0,
) -> dict:
    """Run one cell: sim + detect bursts + classify regime.

    burn_in (v3.2 lock, user-return round-4 catch): discards the first
    burn_in HR time units before burst detection / regime classification.
    Default 100.0 is ~6 × slow-var τ at r=0.006 — enough for HR to
    relax from arbitrary initial conditions (x0=-1.6, y0=-10, z0=2) into
    the (I, r)-specific attractor. Without burn-in, the picker would
    treat the relaxation transient as steady-state burst behavior and
    silently accept cells that only LOOK excitable.
    """
    if burst_cfg is None:
        burst_cfg = BurstConfig()
    if regime_cfg is None:
        regime_cfg = RegimeConfig()
    p = params if r_override is None else replace(params, r=r_override)
    t, traj = simulate_trajectory(p, I, T, dt, sigma_ou, tau_ou, seed,
                                    burn_in=burn_in)
    bursts = detect_bursts(traj[:, 0], t, burst_cfg)
    durations = [e - s for s, e in bursts]
    ibis = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
    regime = classify_regime(bursts, T, regime_cfg)
    return {
        "regime": regime,
        "n_bursts": len(bursts),
        "mean_burst_duration": float(np.mean(durations)) if durations else 0.0,
        "mean_ibi": float(np.mean(ibis)) if ibis else float("inf"),
        "I": I, "r_used": p.r, "sigma_ou": sigma_ou, "seed": seed, "T": T,
        "burn_in": burn_in,
    }


def sweep_hr_parameters(
    I_grid: Sequence[float],
    r_grid: Sequence[float],
    sigma_grid: Sequence[float],
    seeds: Sequence[int],
    T: float, dt: float, n_jobs: int = 1,
    params_base: HRParams | None = None,
) -> pd.DataFrame:
    """Cartesian sweep. Returns DataFrame with one row per (I, r, sigma, seed)."""
    if params_base is None:
        params_base = HRParams()
    cells = [
        (I, r, sigma, seed)
        for I in I_grid for r in r_grid for sigma in sigma_grid for seed in seeds
    ]
    def _eval(cell):
        I, r, sigma, seed = cell
        return evaluate_cell(params_base, I=I, sigma_ou=sigma, tau_ou=10.0,
                              r_override=r, T=T, dt=dt, seed=seed)
    if n_jobs == 1:
        results = [_eval(c) for c in cells]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval)(c) for c in cells
        )
    return pd.DataFrame(results)


def pick_excitable_baseline(df: pd.DataFrame) -> dict | None:
    """Pick Stage 1 baseline (I*, r*, sigma*) per spec §3 stage 1 exit contract.

    Criteria (spec §3 Stage 1: regime stable to ±50% noise perturbation):
        (a) modal regime at (I, r, sigma) == "excitable", sigma > 0
        (b) same (I, r) at sigma=0 modal regime == "silent"
        (c) **lower** sigma in [0.4 * sigma, 0.6 * sigma] (closest to 0.5σ):
            modal regime still "excitable" (NOT silent, NOT repetitive-burst)
        (d) **upper** sigma in [1.4 * sigma, 1.6 * sigma] (closest to 1.5σ):
            modal regime still "excitable" (NOT repetitive-burst, NOT unstable)

    A candidate's grid must contain BOTH lower (in window) AND upper (in
    window) neighbors AND both must be modal excitable. This enforces the
    spec's "±50% noise robust" requirement and rules out threshold-edge
    baselines (v3 fix: v2 only checked upper-side, missed lower).

    Among surviving candidates, picks the one with median sigma_star
    (sorted, len//2 index).

    Returns dict {I_star, r_star, sigma_star, noise_robust=True,
    lower_sigma, upper_sigma} or None.
    """
    all_sigmas = sorted(df["sigma_ou"].unique())
    candidates = []
    for (I, r, sigma), group in df.groupby(["I", "r_used", "sigma_ou"]):
        if sigma <= 0.0:
            continue
        if group["regime"].mode().iloc[0] != "excitable":
            continue
        # (b) zero-noise silent
        zn = df[(df["I"] == I) & (df["r_used"] == r) & (df["sigma_ou"] == 0.0)]
        if zn.empty or zn["regime"].mode().iloc[0] != "silent":
            continue
        # (c) lower-side robustness: need sigma' in [0.4σ, 0.6σ]
        lower_window = [s for s in all_sigmas if 0.4 * sigma <= s <= 0.6 * sigma]
        if not lower_window:
            continue
        # Closest to 0.5σ (take the one with min |s - 0.5σ|)
        lo_sigma = min(lower_window, key=lambda s: abs(s - 0.5 * sigma))
        lo_cell = df[
            (df["I"] == I) & (df["r_used"] == r) & (df["sigma_ou"] == lo_sigma)
        ]
        if lo_cell.empty or lo_cell["regime"].mode().iloc[0] != "excitable":
            continue
        # (d) upper-side robustness: need sigma' in [1.4σ, 1.6σ]
        upper_window = [s for s in all_sigmas if 1.4 * sigma <= s <= 1.6 * sigma]
        if not upper_window:
            continue
        hi_sigma = min(upper_window, key=lambda s: abs(s - 1.5 * sigma))
        hi_cell = df[
            (df["I"] == I) & (df["r_used"] == r) & (df["sigma_ou"] == hi_sigma)
        ]
        if hi_cell.empty or hi_cell["regime"].mode().iloc[0] != "excitable":
            continue
        candidates.append({
            "I_star": float(I), "r_star": float(r), "sigma_star": float(sigma),
            "noise_robust": True,
            "lower_sigma": float(lo_sigma),
            "upper_sigma": float(hi_sigma),
        })
    if not candidates:
        return None
    candidates.sort(key=lambda c: c["sigma_star"])
    return candidates[len(candidates) // 2]
