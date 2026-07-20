"""Regression tests for the MZ onset-dynamics engineering fixes (task §4 + §5).

§4 focused-m aggregation contracts (pure, src-level): missing / duplicate / tau-mixed / schema-misaligned
   rows must be rejected; the tau-sensitivity assembler must reject missing/misplaced cells.
§5 counterfactual contracts (runner-level): `rotated_90` is a REAL rotation with a fail-closed path (never
   identity), and the uniform arm is honestly named `uniform_mean_matched` (== nanmean), never
   `uniform_current_matched`.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.topic4_mz_onset_dynamics import (
    validate_focused_m_grid, build_tau_sensitivity, tau_phenotype_denominators,
    FOCUSED_M_MAIN_SEEDS, FOCUSED_M_MAIN_A_FRACS,
)


# ------------------------------------------------------------------ helpers
def _row(seed, a, tau=2000.0, pheno="interictal_like", runaway=None):
    return dict(seed=seed, z_regime="zA_q75_tz5000", A_frac=a, tau_adp_ms=tau, eta_m=0.01,
                realized_a_max=0.0, D_max=0.05, phenotype=pheno, runaway_ms=runaway,
                n_events=10, event_bar=0.025)


def _full_grid():
    return [_row(s, a) for s in FOCUSED_M_MAIN_SEEDS for a in FOCUSED_M_MAIN_A_FRACS]


# ============================================================ §4.1 main-grid validation
def test_valid_18_row_grid_passes():
    rows = validate_focused_m_grid(_full_grid())
    assert len(rows) == 18
    assert [(r["seed"], r["A_frac"]) for r in rows] == sorted((r["seed"], r["A_frac"]) for r in rows)


def test_missing_cell_raises():
    grid = _full_grid()[:-1]                              # drop one (seed4, 0.01) cell
    with pytest.raises(ValueError, match="grid mismatch"):
        validate_focused_m_grid(grid)


def test_duplicate_cell_raises():
    grid = _full_grid() + [_row(1, 0.001)]               # duplicate (seed1, 0.001)
    with pytest.raises(ValueError, match="duplicate"):
        validate_focused_m_grid(grid)


def test_tau_contamination_raises():
    grid = _full_grid()
    grid[3] = _row(grid[3]["seed"], grid[3]["A_frac"], tau=500.0)   # a tau500 row sneaks into the 2000 grid
    with pytest.raises(ValueError, match="tau contamination"):
        validate_focused_m_grid(grid)


def test_schema_misalignment_missing_field_raises():
    grid = _full_grid()
    del grid[0]["event_bar"]                              # old-format / stale row missing a required field
    with pytest.raises(ValueError, match="missing required field"):
        validate_focused_m_grid(grid)


# ============================================================ §4.2 tau sensitivity
def test_tau_sensitivity_valid_and_denominators():
    by = {}
    plan = {2000.0: "interictal_like", 1000.0: "expanded_bounded", 500.0: "runaway"}
    for s in (1, 3, 4):
        for tau, ph in plan.items():
            by[(s, tau)] = _row(s, 0.001, tau=tau, pheno=ph, runaway=(12000.0 if ph == "runaway" else None))
    rows, denom = build_tau_sensitivity(by)
    assert len(rows) == 9
    assert denom["tau2000"]["n_runaway"] == 0 and denom["tau500"]["n_runaway"] == 3
    assert denom["tau1000"]["n"] == 3


def test_tau_sensitivity_missing_cell_raises():
    by = {(s, tau): _row(s, 0.001, tau=tau) for s in (1, 3, 4) for tau in (2000.0, 1000.0)}   # no tau500
    with pytest.raises(ValueError, match="missing cell"):
        build_tau_sensitivity(by)


def test_tau_sensitivity_wrong_a_frac_raises():
    by = {(s, tau): _row(s, 0.001, tau=tau) for s in (1, 3, 4) for tau in (2000.0, 1000.0, 500.0)}
    by[(1, 500.0)] = _row(1, 0.0025, tau=500.0)           # wrong A_frac in a cell
    with pytest.raises(ValueError, match="A_frac"):
        build_tau_sensitivity(by)


# ============================================================ §5 counterfactual contracts (runner-level)
@pytest.fixture(scope="module")
def R():
    for p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_topic4_mz_onset_dynamics as mod
    return mod


def _grid_filling_posE(n, L=1.0):
    """One E neuron per grid cell (fills every cell so rotation is well-defined)."""
    xs, ys, zc = [], [], []
    for iy in range(n):
        for ix in range(n):
            xs.append((ix + 0.5) / n * L)
            ys.append((iy + 0.5) / n * L)
            zc.append(float(ix))                          # gradient along x -> non-symmetric
    return np.column_stack([xs, ys]), np.asarray(zc, float)


def test_rotated90_is_real_rotation_not_identity(R):
    posE, z = _grid_filling_posE(4, L=1.0)
    z_rot = R._rotate90_coarse_field(z, posE, 1.0, 4)
    assert not np.array_equal(z_rot, z), "rotated_90 returned the input unchanged (identity fallback)"
    assert sorted(np.round(z_rot, 6)) == sorted(np.round(z, 6)), "rotation must preserve the value multiset"


def test_rotated90_fail_closed_on_empty_source_cell(R):
    n, L = 4, 1.0
    # all neurons on row iy=0 -> rows 1..3 empty -> rotation maps occupied cells to empty sources
    posE = np.column_stack([(np.arange(n) + 0.5) / n * L, np.full(n, 0.5 / n * L)])
    z = np.arange(n, dtype=float)
    with pytest.raises(ValueError, match="FAIL-CLOSED"):
        R._rotate90_coarse_field(z, posE, L, n)


def test_uniform_arm_is_mean_matched_and_named(R):
    posE, z = _grid_filling_posE(4, L=1.0)
    tf = R._counterfactual_transforms(z, {}, float(np.nanmean(z)), 42, posE=posE, L=1.0, grid_n=4)
    assert "uniform_mean_matched" in tf
    assert "uniform_current_matched" not in tf, "the misleading current-matched name must be gone"
    np.testing.assert_allclose(tf["uniform_mean_matched"](z), np.full_like(z, float(np.nanmean(z))))


def test_only_native_frozen_is_identity(R):
    posE, z = _grid_filling_posE(4, L=1.0)
    tf = R._counterfactual_transforms(z, {}, float(np.nanmean(z)), 42, posE=posE, L=1.0, grid_n=4)
    assert np.array_equal(tf["native_frozen"](z), z)                     # the ONLY identity arm
    assert not np.array_equal(tf["rotated_90"](z), z)                    # rotated is NOT identity
