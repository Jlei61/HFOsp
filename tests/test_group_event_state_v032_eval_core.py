"""Core numerics of the v0.3.2 measurement/evaluation package.

Every test here pins a contract clause from
``docs/archive/topic5/group_event_state_v0_3_2_measurement_contract_2026-09-02.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.topic5_group_event_state.v02.timeline import CoverageSegment
from src.topic5_group_event_state.v032_eval.partition import (
    EVAL_PHASES,
    EvalPartition,
    eval_partition,
)
from src.topic5_group_event_state.v032_eval.nb_glm import (
    NegativeBinomialRidge,
    nb_nll,
    select_and_refit,
)
from src.topic5_group_event_state.v032_eval.blocks import (
    block_ids_for_times,
    block_bootstrap_mean_ci,
    paired_gain_summary,
)
from src.topic5_group_event_state.v032_eval.shift import predefined_session_shifts


# ----------------------------------------------------------------------------
# partition: 60/70/80 boundaries on recorded time, refit = 0-70
# ----------------------------------------------------------------------------

def test_eval_partition_boundaries_follow_recorded_time_not_gaps():
    segments = [
        CoverageSegment(0, 0, 0.0, 100.0),
        CoverageSegment(1, 1, 1000.0, 1100.0),
    ]
    part = eval_partition(segments)
    # 200 recorded seconds; 60% = 120 -> 20 s into segment 1 (epoch 1020)
    assert np.allclose(part.boundary_epochs, [1020.0, 1040.0, 1060.0])
    labels = part.labels_of(np.array([10.0, 1019.0, 1020.0, 1050.0, 1090.0]))
    assert labels.tolist() == [0, 0, 1, 2, 3]
    assert EVAL_PHASES == ("base_fit", "inner_val", "dev_val", "dev_test")
    assert part.phase_of(1030.0) == "inner_val"
    # refit phase is the union of base_fit and inner_val
    mask = part.mask_for_phase(np.array([10.0, 1030.0, 1050.0, 1090.0]), "base_refit")
    assert mask.tolist() == [True, True, False, False]


def test_eval_partition_window_must_stay_inside_one_phase():
    segments = [CoverageSegment(0, 0, 0.0, 1000.0)]
    part = eval_partition(segments)  # boundaries 600 / 700 / 800
    ok = part.window_within_phase(np.array([500.0, 590.0, 610.0]), horizon=100.0)
    # [500,600) fits base_fit; [590,690) crosses 600; [610,710) crosses 700
    assert ok.tolist() == [True, False, False]


# ----------------------------------------------------------------------------
# negative-binomial ridge GLM
# ----------------------------------------------------------------------------

def _simulate_nb(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    beta = np.array([0.8, -0.5, 0.0])
    mu = np.exp(2.0 + x @ beta)
    alpha = 0.5
    # NB2 via Poisson-gamma mixture
    lam = rng.gamma(shape=1.0 / alpha, scale=mu * alpha)
    y = rng.poisson(lam)
    return x, y, beta, alpha


def test_nb_ridge_recovers_coefficients_and_dispersion_without_penalty():
    x, y, beta, alpha = _simulate_nb(6000)
    model = NegativeBinomialRidge(ridge=1e-8).fit(x, y)
    assert np.allclose(model.coef_, beta, atol=0.08)
    assert abs(model.intercept_ - 2.0) < 0.08
    assert abs(model.alpha_ - alpha) < 0.12
    assert model.converged_


def test_nb_nll_matches_scipy_logpmf():
    from scipy.stats import nbinom

    y = np.array([0, 3, 7, 20])
    mu = np.array([1.0, 2.5, 8.0, 15.0])
    alpha = 0.4
    ours = nb_nll(y, mu, alpha)
    r = 1.0 / alpha
    p = r / (r + mu)
    ref = -nbinom.logpmf(y, r, p)
    assert np.allclose(ours, ref, atol=1e-9)


def test_nb_ridge_standardises_on_fit_rows_only_and_never_refits_on_test():
    x, y, _beta, _alpha = _simulate_nb(3000, seed=1)
    fit = slice(0, 2000)
    test = slice(2000, 3000)
    model = NegativeBinomialRidge(ridge=1.0).fit(x[fit], y[fit])
    frozen = (model.coef_.copy(), model.intercept_, model.alpha_, model.x_mean_.copy(), model.x_scale_.copy())
    nll = model.nll(x[test], y[test])
    assert nll.shape == (1000,)
    assert np.isfinite(nll).all()
    # scoring must not mutate any fitted quantity
    assert np.array_equal(frozen[0], model.coef_)
    assert frozen[1] == model.intercept_ and frozen[2] == model.alpha_
    assert np.array_equal(frozen[3], model.x_mean_) and np.array_equal(frozen[4], model.x_scale_)
    # standardisation statistics come from the fit rows
    assert np.allclose(model.x_mean_, x[fit].mean(axis=0))


def test_select_and_refit_uses_inner_validation_and_flags_grid_edges():
    x, y, _beta, _alpha = _simulate_nb(4000, seed=2)
    base = np.arange(0, 2400)
    inner = np.arange(2400, 2800)
    refit_rows = np.arange(0, 2800)
    out = select_and_refit(
        x, y, fit_rows=base, select_rows=inner, refit_rows=refit_rows,
        ridge_grid=(1e-3, 1e-1, 10.0),
    )
    assert out["selected_ridge"] in (1e-3, 1e-1, 10.0)
    assert set(out["path"][0]) == {"ridge", "selection_nll"}
    assert isinstance(out["ridge_at_edge"], bool)
    assert out["model"].n_fit_rows_ == 2800
    # refit standardisation must remain the base_fit statistics (frozen constants)
    assert np.allclose(out["model"].x_mean_, x[base].mean(axis=0))


def test_nb_ridge_with_constant_column_and_zero_counts_stays_finite():
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(500), rng.normal(size=500)])
    y = np.zeros(500, dtype=int)
    y[:20] = rng.poisson(2.0, 20)
    model = NegativeBinomialRidge(ridge=1.0).fit(x, y)
    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.nll(x, y)).all()


# ----------------------------------------------------------------------------
# blocks and paired inference
# ----------------------------------------------------------------------------

def test_block_ids_are_non_overlapping_bins_inside_each_segment():
    t = np.array([300.0, 600.0, 1800.0, 2100.0, 5000.0, 5300.0])
    seg = np.array([0, 0, 0, 0, 1, 1])
    seg_start = {0: 0.0, 1: 4800.0}
    ids = block_ids_for_times(t, seg, seg_start, block_seconds=1800.0)
    # segment 0: [0,1800) -> block 0, [1800,3600) -> block 1 ; segment 1 own blocks
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
    assert ids[4] == ids[5] and ids[4] not in (ids[0], ids[2])


def test_block_bootstrap_ci_covers_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 1.0, size=400)
    blocks = np.repeat(np.arange(40), 10)
    a = block_bootstrap_mean_ci(values, blocks, n_boot=500, seed=1)
    b = block_bootstrap_mean_ci(values, blocks, n_boot=500, seed=1)
    assert a == b
    assert a["lower"] < values.mean() < a["upper"]
    assert a["n_blocks"] == 40


def test_paired_gain_summary_keeps_all_finite_pairs_and_reports_direction():
    control = np.array([2.0, 2.0, 2.0, 2.0, np.nan, 2.0])
    treated = np.array([1.5, 1.8, 2.2, 1.0, 1.0, np.nan])
    blocks = np.array([0, 0, 1, 1, 2, 2])
    out = paired_gain_summary(control, treated, blocks, n_boot=200, seed=0)
    # gain = control - treated on the 4 finite pairs
    assert out["n_pairs"] == 4
    assert np.isclose(out["mean_gain"], np.mean([0.5, 0.2, -0.2, 1.0]))
    assert out["n_blocks"] == 2
    assert out["direction"] == "favours_treated"


# ----------------------------------------------------------------------------
# predefined circular shifts within a session
# ----------------------------------------------------------------------------

def test_predefined_shifts_are_result_free_and_respect_minimum_distance():
    t = np.arange(0.0, 36000.0, 300.0)  # one 10 h session, 120 anchors
    session = np.zeros(t.size, dtype=int)
    shifts = predefined_session_shifts(t, session, n_shifts=5, denominator=6,
                                       min_distance_seconds=1800.0 + 300.0)
    assert len(shifts) == 5
    for spec in shifts:
        donor = spec["donor_index"]
        assert donor.shape == t.shape
        valid = donor >= 0
        # donor is a circular permutation of the session's own anchors
        assert np.array_equal(np.sort(donor[valid]), np.sort(np.flatnonzero(np.isin(np.arange(t.size), donor[valid]))))
        dt = np.abs(t[valid] - t[donor[valid]])
        assert (dt > 2100.0).all()
    # shifts are fixed fractions j/6 of the session length, never chosen from data
    assert [s["shift_anchors_by_session"][0] for s in shifts] == [20, 40, 60, 80, 100]


def test_predefined_shifts_never_cross_sessions():
    t = np.concatenate([np.arange(0.0, 6000.0, 300.0), np.arange(100000.0, 106000.0, 300.0)])
    session = np.repeat([0, 1], 20)
    shifts = predefined_session_shifts(t, session, n_shifts=5, denominator=6,
                                       min_distance_seconds=600.0)
    for spec in shifts:
        donor = spec["donor_index"]
        valid = donor >= 0
        assert (session[valid] == session[donor[valid]]).all()


def test_nb_ridge_winsorises_out_of_range_features_to_the_fit_range():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(400, 2))
    y = rng.poisson(np.exp(1.0 + 0.5 * x[:, 0]))
    model = NegativeBinomialRidge(ridge=1e-3).fit(x, y)
    far = np.array([[50.0, 0.0]])          # far outside the fit range
    edge = np.array([[x[:, 0].max(), 0.0]])
    assert np.allclose(model.predict_mu(far), model.predict_mu(edge))
    assert np.allclose(model.x_min_, x.min(axis=0)) and np.allclose(model.x_max_, x.max(axis=0))
