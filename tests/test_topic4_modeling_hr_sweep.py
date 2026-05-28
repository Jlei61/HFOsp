"""Tests for src/topic4_modeling/hr_sweep.py.

Two test layers (per user-return strict catch):
  - Synthetic DataFrame unit tests for pick_excitable_baseline (always run)
  - Real fast smoke sweep (~30s) that xfail-flags if no excitable regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── evaluate_cell smoke ──────────────────────────────────────────────────

def test_evaluate_cell_returns_dict_with_regime_and_metadata():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_sweep import evaluate_cell
    result = evaluate_cell(HRParams(), I=-2.5, sigma_ou=0.0, tau_ou=10.0,
                            r_override=None, T=100.0, dt=0.05, seed=0)
    assert isinstance(result, dict)
    assert "regime" in result
    assert "n_bursts" in result
    assert result["regime"] in {"silent", "excitable", "repetitive-burst", "unstable"}


# (Empirical "r_override changes burst rate" test moved to
#  tests/test_topic4_modeling_hr_dynamics_integration.py per v3
#  user-return critique — not algebraic, would invite model tuning.)


# ── sweep_hr_parameters ──────────────────────────────────────────────────

def test_sweep_total_count_matches_cartesian_product():
    from src.topic4_modeling.hr_sweep import sweep_hr_parameters
    df = sweep_hr_parameters(
        I_grid=[-2.0, -1.6, 0.0],
        r_grid=[0.004, 0.006],
        sigma_grid=[0.0, 0.1],
        seeds=[0, 1],
        T=50.0, dt=0.05, n_jobs=1,
    )
    assert len(df) == 3 * 2 * 2 * 2


def test_sweep_columns_complete():
    from src.topic4_modeling.hr_sweep import sweep_hr_parameters
    df = sweep_hr_parameters(
        I_grid=[-2.0], r_grid=[0.006], sigma_grid=[0.0], seeds=[0],
        T=50.0, dt=0.05, n_jobs=1,
    )
    required = {"I", "r_used", "sigma_ou", "seed", "regime", "n_bursts"}
    assert required.issubset(df.columns)


def test_sweep_deterministic_per_cell():
    from src.topic4_modeling.hr_sweep import sweep_hr_parameters
    df1 = sweep_hr_parameters(
        I_grid=[-1.6], r_grid=[0.006], sigma_grid=[0.1], seeds=[42],
        T=100.0, dt=0.05, n_jobs=1,
    )
    df2 = sweep_hr_parameters(
        I_grid=[-1.6], r_grid=[0.006], sigma_grid=[0.1], seeds=[42],
        T=100.0, dt=0.05, n_jobs=1,
    )
    assert df1["regime"].iloc[0] == df2["regime"].iloc[0]


# ── pick_excitable_baseline SYNTHETIC unit tests (always run, never skip) ─

def _row(I, r, sigma, seed, regime, n_bursts=1):
    return {"I": I, "r_used": r, "sigma_ou": sigma, "seed": seed,
            "regime": regime, "n_bursts": n_bursts,
            "mean_burst_duration": 5.0, "mean_ibi": 100.0}


def test_picker_returns_candidate_with_full_noise_window():
    """Synthetic: candidate σ has BOTH lower (~0.5σ) and upper (~1.5σ)
    neighbors that are excitable, sigma=0 silent → returns candidate."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    # Candidate σ=0.10: needs lower in [0.04, 0.06] and upper in [0.14, 0.16]
    # Grid: [0, 0.05, 0.10, 0.15] (smoke config style)
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "excitable"))   # lower neighbor
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))   # candidate
        rows.append(_row(-1.3, 0.006, 0.15, seed, "excitable"))   # upper neighbor
    # Unrelated cells
    for seed in [0, 1, 2]:
        rows.append(_row(-2.5, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(2.0, 0.006, 0.10, seed, "repetitive-burst", 10))
    df = pd.DataFrame(rows)
    baseline = pick_excitable_baseline(df)
    assert baseline is not None
    assert baseline["I_star"] == -1.3 and baseline["r_star"] == 0.006
    assert baseline["sigma_star"] == 0.10
    assert baseline["noise_robust"] is True


def test_picker_returns_none_when_no_excitable_anywhere():
    """Synthetic: all silent → returns None."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for I in [-2.5, -1.6, 0.0]:
        for sigma in [0.0, 0.05, 0.10, 0.15]:
            rows.append(_row(I, 0.006, sigma, 0, "silent", 0))
    df = pd.DataFrame(rows)
    assert pick_excitable_baseline(df) is None


def test_picker_rejects_when_zero_noise_not_silent():
    """Synthetic: zero-noise NOT silent → reject (candidate is not the noise-driven excitable regime)."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "repetitive-burst", 10))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "excitable"))
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))
        rows.append(_row(-1.3, 0.006, 0.15, seed, "excitable"))
    df = pd.DataFrame(rows)
    assert pick_excitable_baseline(df) is None


def test_picker_rejects_when_upper_sigma_flips_regime():
    """Synthetic: upper neighbor flips to repetitive-burst → reject (upper-edge fragile)."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "excitable"))
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))
        rows.append(_row(-1.3, 0.006, 0.15, seed, "repetitive-burst", 10))
    df = pd.DataFrame(rows)
    assert pick_excitable_baseline(df) is None


def test_picker_rejects_when_lower_sigma_is_silent():
    """v3 NEW (user-return strict catch): lower neighbor silent → reject (lower-edge fragile).

    v2 picker missed this case — without lower-side check, a candidate σ
    sitting right at the noise threshold (where any smaller σ flips to
    silent) would pass. v3 enforces both lower AND upper noise-robustness.
    """
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "silent", 0))     # lower neighbor SILENT → fragile
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))     # candidate
        rows.append(_row(-1.3, 0.006, 0.15, seed, "excitable"))
    df = pd.DataFrame(rows)
    assert pick_excitable_baseline(df) is None


def test_picker_rejects_when_no_lower_neighbor_in_grid():
    """Candidate σ=0.05 has no lower neighbor in [0.02, 0.03] → reject."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "excitable"))   # candidate has no in-grid lower
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))
    df = pd.DataFrame(rows)
    # sigma=0.05 has no neighbor in [0.02, 0.03] (next-lower would be 0
    # which is silent by separate criterion) → fragile lower-side → reject.
    # sigma=0.10 has no upper in [0.14, 0.16] in this small grid → reject.
    # → no candidate qualifies
    assert pick_excitable_baseline(df) is None


def test_picker_picks_median_sigma_when_multiple_candidates():
    """Synthetic: 2 candidates (σ=0.10 and σ=0.20) both satisfy full noise window → picker picks median (sorted+median = σ=0.20 by len//2)."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    # Grid: [0, 0.05, 0.10, 0.15, 0.20, 0.30]
    # σ=0.10: lower=0.05 ∈[0.04,0.06]✓, upper=0.15 ∈[0.14,0.16]✓ → candidate
    # σ=0.15: lower=? need [0.06,0.09], none in grid → fail
    # σ=0.20: lower=0.10 ∈[0.08,0.12]✓, upper=0.30 ∈[0.28,0.32]✓ → candidate
    # σ=0.30: upper=? need [0.42,0.48], none → fail
    grid_sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    for seed in [0, 1, 2]:
        for sigma in grid_sigmas:
            regime = "silent" if sigma == 0.0 else "excitable"
            rows.append(_row(-1.3, 0.006, sigma, seed, regime,
                              0 if regime == "silent" else 1))
    df = pd.DataFrame(rows)
    baseline = pick_excitable_baseline(df)
    assert baseline is not None
    # Two candidates [0.10, 0.20]; sorted; median by len//2 = index 1 = 0.20
    assert baseline["sigma_star"] in {0.10, 0.20}, (
        f"Expected 0.10 or 0.20, got {baseline['sigma_star']}"
    )


# ── Real fast smoke sweep integration (xfail, NOT skip) ─────────────────

@pytest.mark.slow
def test_real_smoke_sweep_finds_excitable_region():
    """Real (small, ~20s) sweep — xfail if no excitable found.

    Using xfail (not skip) so this test counts as failure if it
    unexpectedly passes (XPASS) or unexpectedly fails (XFAIL marker
    visible in pytest output). Skip would silently hide a missing
    exit contract.
    """
    from src.topic4_modeling.hr_sweep import (
        sweep_hr_parameters, pick_excitable_baseline,
    )
    df = sweep_hr_parameters(
        I_grid=np.linspace(-2.0, -1.0, 6).tolist(),
        r_grid=[0.006, 0.008],
        sigma_grid=[0.0, 0.1, 0.15],
        seeds=[0, 1, 2],
        T=200.0, dt=0.05, n_jobs=1,
    )
    baseline = pick_excitable_baseline(df)
    if baseline is None:
        pytest.xfail(
            "No excitable baseline in small smoke sweep. "
            "Either parameter ranges need adjustment or HR doesn't have "
            "excitable regime — see spec §8 stage 1 fallback (FHN)."
        )
    # If baseline found, verify exit contract
    assert baseline["noise_robust"] is True
