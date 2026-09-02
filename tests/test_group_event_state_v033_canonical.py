"""Canonical per-anchor evaluator (v0.3.3 Workstream A, plan Task 1, clauses C1-C11).

Every test names the production change that would make it fail: the canonical
score must be one pure function of (target, prediction, dispersion, mask,
weight) shared by the training branch, the independent evaluator and the
figure payload.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

from src.topic5_group_event_state.v033_evaluator import canonical as C


def _rows(seed: int = 0, n: int = 40, log_r: float = np.log(2.0)):
    rng = np.random.default_rng(seed)
    log_mu = rng.normal(2.0, 0.6, n)
    r = np.exp(log_r)
    y = rng.negative_binomial(r, r / (r + np.exp(log_mu))).astype(np.int64)
    return y, log_mu


def _table(seed: int = 0, n: int = 40, **kw):
    y, log_mu = _rows(seed, n)
    rng = np.random.default_rng(seed + 100)
    defaults = dict(
        subject="toy", seed=1, checkpoint_hash="deadbeef", split="dev_test",
        anchor_time=np.arange(n, dtype=np.float64) * 300.0, target=y,
        prediction_H=log_mu, prediction_H_plus_state=log_mu + rng.normal(0, 0.1, n),
        dispersion=np.log(2.0), mask=None, weight=None,
        eligibility="eligible", evidence_label="DIAGNOSTIC",
    )
    defaults.update(kw)
    return C.build_per_anchor_table(**defaults)


# --------------------------------------------------------------------------- C2 formula
def test_nb_nll_matches_scipy_logpmf():
    y, log_mu = _rows()
    log_r = np.log(3.5)
    r = np.exp(log_r)
    ref = -nbinom.logpmf(y, r, r / (r + np.exp(log_mu)))
    assert np.allclose(C.nb_nll(y, log_mu, log_r), ref, atol=1e-10)


def test_numpy_and_torch_branches_agree_within_1e9_nats():
    y, log_mu = _rows(seed=3)
    log_r = np.log(0.7)
    a = C.nb_nll(y, log_mu, log_r)
    b = C.nb_nll_torch(torch.from_numpy(y), torch.from_numpy(log_mu), torch.tensor(log_r))
    assert b.dtype == torch.float64
    assert np.max(np.abs(a - b.numpy())) < 1e-9


def test_nb_nll_accepts_per_row_dispersion_and_large_rates_without_overflow():
    y = np.array([0, 5, 2000, 30000], dtype=np.int64)
    log_mu = np.array([-3.0, 1.0, 7.5, 10.2])
    log_r = np.array([0.0, 0.5, 1.0, -0.5])
    out = C.nb_nll(y, log_mu, log_r)
    assert np.isfinite(out).all()
    r = np.exp(log_r)
    ref = -nbinom.logpmf(y, r, r / (r + np.exp(log_mu)))
    assert np.allclose(out, ref, rtol=1e-9, atol=1e-8)


# --------------------------------------------------------------------------- C3 legacy mapping
def test_legacy_v032_model_and_eval_formulas_map_onto_canonical():
    from src.topic5_group_event_state.v032_eval.nb_glm import nb_nll as eval_nll
    from src.topic5_group_event_state.v032_model.readout import _nb_log_prob_np as model_logp

    y, log_mu = _rows(seed=5)
    alpha = 0.55
    log_r = C.alpha_to_log_r(alpha)
    assert np.isclose(np.exp(-log_r), alpha)
    canonical = C.nb_nll(y, log_mu, log_r)
    assert np.allclose(canonical, eval_nll(y, np.exp(log_mu), alpha), atol=1e-8)
    assert np.allclose(canonical, -model_logp(y, np.exp(log_mu), log_r), atol=1e-8)


# --------------------------------------------------------------------------- C10 schema
def test_table_has_schema_columns_with_aligned_lengths():
    n = 17
    t = _table(n=n)
    for col in C.SCHEMA_COLUMNS:
        assert col in t, col
        assert len(t[col]) == n, col
    assert t["meta"]["schema_version"] == C.SCHEMA_VERSION
    assert t["meta"]["dispersion_rule"] == "shared"
    assert t["subject"][0] == "toy" and t["split"][0] == "dev_test"
    assert t["evidence_label"][0] == "DIAGNOSTIC" and t["eligibility"][0] == "eligible"
    assert t["per_anchor_NLL_H"].dtype == np.float64


# --------------------------------------------------------------------------- C4 permutation
def test_anchor_permutation_permutes_rows_and_leaves_reduction_invariant():
    n = 33
    y, log_mu = _rows(seed=7, n=n)
    rng = np.random.default_rng(1)
    pred_s = log_mu + rng.normal(0, 0.2, n)
    w = rng.uniform(0.5, 2.0, n)
    block = rng.integers(0, 4, n)
    perm = rng.permutation(n)
    kw = dict(subject="toy", seed=1, checkpoint_hash="x", split="dev_test", dispersion=0.3,
              eligibility="eligible", evidence_label="DIAGNOSTIC")
    a = C.build_per_anchor_table(anchor_time=np.arange(n) * 300.0, target=y, prediction_H=log_mu,
                                 prediction_H_plus_state=pred_s, weight=w, mask=None, **kw)
    b = C.build_per_anchor_table(anchor_time=(np.arange(n) * 300.0)[perm], target=y[perm],
                                 prediction_H=log_mu[perm], prediction_H_plus_state=pred_s[perm],
                                 weight=w[perm], mask=None, **kw)
    assert np.allclose(a["per_anchor_NLL_H"][perm], b["per_anchor_NLL_H"])
    for reduction, extra in (("mean", {}), ("sum", {}), ("block_mean", {"block": block})):
        ga = C.paired_gain(a, reduction=reduction, **extra)
        gb = C.paired_gain(b, reduction=reduction, **({"block": block[perm]} if extra else {}))
        assert np.isclose(ga["gain"], gb["gain"]), reduction


# --------------------------------------------------------------------------- C5 mask
def test_masked_rows_are_nan_in_table_and_excluded_from_reduction():
    n = 20
    mask = np.ones(n, bool)
    mask[[2, 9, 15]] = False
    t = _table(n=n, mask=mask)
    assert np.isnan(t["per_anchor_NLL_H"][~mask]).all()
    assert np.isnan(t["per_anchor_NLL_H_plus_state"][~mask]).all()
    assert np.isfinite(t["per_anchor_NLL_H"][mask]).all()
    g = C.paired_gain(t)
    assert g["n_rows_used"] == n - 3 and g["n_rows_masked"] == 3 and g["n_rows_total"] == n
    manual = (t["per_anchor_NLL_H"][mask] - t["per_anchor_NLL_H_plus_state"][mask]).mean()
    assert np.isclose(g["gain"], manual)


# --------------------------------------------------------------------------- C6 dispersion
def test_shared_rule_rejects_unequal_arm_dispersions_and_per_arm_requires_every_arm():
    with pytest.raises(ValueError):
        _table(dispersion={"H": 0.3, "H_plus_state": 0.4}, dispersion_rule="shared")
    with pytest.raises(ValueError):
        _table(dispersion={"H": 0.3}, dispersion_rule="per_arm")
    with pytest.raises(ValueError):
        _table(dispersion=0.3, dispersion_rule="per_arm")
    with pytest.raises(ValueError):
        _table(dispersion=0.3, dispersion_rule="whatever")


def test_per_arm_dispersion_scores_each_arm_with_its_own_log_r_through_the_canonical_formula():
    n = 25
    y, log_mu = _rows(seed=9, n=n)
    pred_s = log_mu + 0.05
    t = C.build_per_anchor_table(
        subject="toy", seed=1, checkpoint_hash="x", split="dev_val", anchor_time=np.arange(n) * 300.0,
        target=y, prediction_H=log_mu, prediction_H_plus_state=pred_s,
        dispersion={"H": 0.6, "H_plus_state": 0.45}, dispersion_rule="per_arm",
        mask=None, weight=None, eligibility="eligible", evidence_label="DIAGNOSTIC")
    assert np.allclose(t["per_anchor_NLL_H"], C.nb_nll(y, log_mu, 0.6))
    assert np.allclose(t["per_anchor_NLL_H_plus_state"], C.nb_nll(y, pred_s, 0.45))
    assert t["meta"]["dispersion_rule"] == "per_arm"
    assert t["dispersion"].shape == (n, 2)


def test_shared_dispersion_table_records_one_log_r_per_row():
    t = _table(n=10, dispersion=0.25)
    assert t["dispersion"].shape == (10,) and np.allclose(t["dispersion"], 0.25)


# --------------------------------------------------------------------------- C1 intercept / purity
def test_intercept_shift_is_a_declared_extra_arm_and_scores_analytically():
    n = 30
    y, log_mu = _rows(seed=11, n=n)
    c = 0.2
    t = C.build_per_anchor_table(
        subject="toy", seed=1, checkpoint_hash="x", split="dev_test", anchor_time=np.arange(n) * 300.0,
        target=y, prediction_H=log_mu, prediction_H_plus_state=log_mu, dispersion=0.4,
        mask=None, weight=None, eligibility="eligible", evidence_label="DIAGNOSTIC",
        extra_arms={"H_plus_intercept": log_mu + c})
    assert "per_anchor_NLL_H_plus_intercept" in t
    g = C.paired_gain(t, control="H", treated="H_plus_intercept")
    assert np.isclose(g["gain"], np.mean(C.nb_nll(y, log_mu, 0.4) - C.nb_nll(y, log_mu + c, 0.4)))
    # the evaluator has no fitting entry point: no intercept / calibration / ridge argument exists
    params = set(inspect.signature(C.build_per_anchor_table).parameters)
    assert not params & {"intercept", "calibration", "ridge", "fit_rows", "refit"}


def test_score_of_one_row_never_depends_on_other_rows():
    n = 12
    y, log_mu = _rows(seed=13, n=n)
    base = C.build_per_anchor_table(
        subject="toy", seed=1, checkpoint_hash="x", split="dev_test", anchor_time=np.arange(n) * 300.0,
        target=y, prediction_H=log_mu, prediction_H_plus_state=log_mu + 0.1, dispersion=0.3,
        mask=None, weight=None, eligibility="eligible", evidence_label="DIAGNOSTIC")
    y2, log_mu2 = y.copy(), log_mu.copy()
    y2[1:] = y2[1:] * 3
    log_mu2[1:] += 2.0
    other = C.build_per_anchor_table(
        subject="toy", seed=1, checkpoint_hash="x", split="dev_test", anchor_time=np.arange(n) * 300.0,
        target=y2, prediction_H=log_mu2, prediction_H_plus_state=log_mu2 + 0.1, dispersion=0.3,
        mask=None, weight=None, eligibility="eligible", evidence_label="DIAGNOSTIC")
    assert base["per_anchor_NLL_H"][0] == other["per_anchor_NLL_H"][0]


# --------------------------------------------------------------------------- C7 weight
def test_weights_enter_reduction_as_weighted_mean():
    n = 16
    rng = np.random.default_rng(21)
    w = rng.uniform(0.1, 3.0, n)
    t = _table(seed=21, n=n, weight=w)
    gain_rows = t["per_anchor_NLL_H"] - t["per_anchor_NLL_H_plus_state"]
    g = C.paired_gain(t)
    assert np.isclose(g["gain"], np.sum(w * gain_rows) / np.sum(w))
    assert g["weights_used"] is True
    assert C.paired_gain(_table(seed=21, n=n))["weights_used"] is False
    with pytest.raises(ValueError):
        _table(seed=21, n=n, weight=-np.ones(n))


# --------------------------------------------------------------------------- C8 sign
def test_sign_convention_positive_gain_favours_treated():
    n = 30
    y, log_mu = _rows(seed=31, n=n)
    truth = log_mu.copy()
    worse = log_mu + 1.5
    t = C.build_per_anchor_table(
        subject="toy", seed=1, checkpoint_hash="x", split="dev_test", anchor_time=np.arange(n) * 300.0,
        target=y, prediction_H=worse, prediction_H_plus_state=truth, dispersion=np.log(2.0),
        mask=None, weight=None, eligibility="eligible", evidence_label="DIAGNOSTIC")
    g = C.paired_gain(t)
    assert g["gain"] > 0 and g["direction"] == "favours_treated"
    g_rev = C.paired_gain(t, control="H_plus_state", treated="H")
    assert np.isclose(g_rev["gain"], -g["gain"]) and g_rev["direction"] == "favours_control"


# --------------------------------------------------------------------------- C9 reduction
def test_reductions_mean_sum_and_block_mean():
    n = 24
    t = _table(seed=41, n=n)
    rows = t["per_anchor_NLL_H"] - t["per_anchor_NLL_H_plus_state"]
    block = np.repeat(np.arange(4), 6)
    assert np.isclose(C.paired_gain(t, reduction="mean")["gain"], rows.mean())
    assert np.isclose(C.paired_gain(t, reduction="sum")["gain"], rows.sum())
    per_block = np.array([rows[block == b].mean() for b in range(4)])
    g = C.paired_gain(t, reduction="block_mean", block=block)
    assert np.isclose(g["gain"], per_block.mean()) and g["n_blocks"] == 4
    with pytest.raises(ValueError):
        C.paired_gain(t, reduction="block_mean")
    with pytest.raises(ValueError):
        C.paired_gain(t, reduction="median")


# --------------------------------------------------------------------------- C11 hard stop
def test_assert_tables_agree_raises_on_first_discrepant_row_above_tolerance():
    a = _table(seed=51, n=20)
    b = _table(seed=51, n=20)
    C.assert_tables_agree(a, b)
    b["per_anchor_NLL_H_plus_state"][7] += 5e-6
    with pytest.raises(C.EvaluatorDisagreement) as info:
        C.assert_tables_agree(a, b)
    assert "row 7" in str(info.value) and "H_plus_state" in str(info.value)
    b["per_anchor_NLL_H_plus_state"][7] -= 5e-6
    b["per_anchor_NLL_H_plus_state"][7] += 1e-8
    C.assert_tables_agree(a, b)
    c = _table(seed=52, n=20)
    with pytest.raises(C.EvaluatorDisagreement):
        C.assert_tables_agree(a, c)  # different targets are a different object, not a tolerance question
