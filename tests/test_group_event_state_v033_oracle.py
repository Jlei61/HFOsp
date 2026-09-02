"""Oracle Level 0-2 estimators (v0.3.3 plan Task 5, clauses O1-O5)."""
from __future__ import annotations

import numpy as np
import torch

from src.topic5_group_event_state.v033_evaluator import canonical as C
from src.topic5_group_event_state.v033_evaluator import dgp as D
from src.topic5_group_event_state.v033_evaluator import oracle as O
from tests.test_group_event_state_v033_toyutil import toy_scaffold


def _data(kind, *, beta_count=0.7, beta_grammar=2.0, seed=0):
    sc = toy_scaffold(seed=seed, rate_per_second=0.03, n_contacts=6)
    data = D.generate(sc, kind, beta_count=beta_count, beta_grammar=beta_grammar,
                      generator_seed=100 + seed, noise_seed=200 + seed)
    return sc, data


# --------------------------------------------------------------------------- heads
def test_count_head_recovers_planted_offset_slope_and_dispersion():
    rng = np.random.default_rng(0)
    n = 4000
    log_mu_h = rng.normal(3.0, 0.3, n)
    z = rng.normal(size=(n, 1))
    c_true, beta_true, log_r_true = 0.3, 0.5, np.log(4.0)
    mu = np.exp(log_mu_h + c_true + z[:, 0] * beta_true)
    r = np.exp(log_r_true)
    y = rng.negative_binomial(r, r / (r + mu))
    head = O.fit_count_head(y, log_mu_h, z)
    assert abs(head["c"] - c_true) < 0.05 and abs(head["beta"][0] - beta_true) < 0.05
    assert abs(head["log_r"] - log_r_true) < 0.1
    h_only = O.fit_count_head(y, log_mu_h, None)
    assert h_only["beta"].size == 0 and np.isfinite(h_only["log_r"])
    pred = O.predict_count_head(head, log_mu_h, z)
    assert np.allclose(pred, log_mu_h + head["c"] + z[:, 0] * head["beta"][0])


def test_count_head_can_freeze_H_dispersion_while_fitting_state_slope():
    rng = np.random.default_rng(9)
    n = 3000
    off = rng.normal(2.0, 0.2, n)
    z = rng.normal(size=(n, 1))
    frozen = np.log(3.0)
    mu = np.exp(off + 0.6 * z[:, 0])
    y = rng.negative_binomial(np.exp(frozen), np.exp(frozen) / (np.exp(frozen) + mu))
    head = O.fit_count_head(y, off, z, fixed_log_r=frozen)
    assert head["dispersion_frozen"] is True
    assert head["log_r"] == frozen
    assert abs(head["beta"][0] - 0.6) < 0.06


def test_grammar_logpmf_torch_matches_numpy():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(50, 7))
    subset = rng.uniform(size=(50, 7)) < 0.4
    ours = O.conditional_bernoulli_logpmf_torch(torch.from_numpy(logits), torch.from_numpy(subset)).numpy()
    assert np.allclose(ours, D.conditional_bernoulli_logpmf(logits, subset), atol=1e-9)


def test_grammar_head_recovers_loading_direction_with_true_state():
    sc, data = _data("D2", beta_grammar=2.5)
    train_events = sc.event_rows("base_fit")
    s = data.z_grammar_event[train_events][:, None]
    head = O.fit_grammar_head(data.participation[train_events], s, n_steps=150)
    corr = np.corrcoef(head["W"][:, 0], data.loadings)[0, 1]
    assert corr > 0.8
    h_only = O.fit_grammar_head(data.participation[train_events], None, n_steps=50)
    assert h_only["W"].shape == (sc.n_contacts, 0)


# --------------------------------------------------------------------------- levels, count view
def test_level0_count_detects_planted_state_and_reports_no_gain_under_d0():
    sc, d1 = _data("D1", beta_count=0.7)
    res = O.run_level(sc, d1, view="count", level=0, horizon=1800.0, seed=0)
    assert res["view"] == "count" and res["level"] == 0
    assert res["gain"] > 0 and res["ci_lower"] > 0
    assert res["table_meta"]["schema_version"] == C.SCHEMA_VERSION
    recomputed = C.paired_gain(res["table"])["gain"]
    assert np.isclose(recomputed, res["gain"])
    sc0, d0 = _data("D0")
    res0 = O.run_level(sc0, d0, view="count", level=0, horizon=1800.0, seed=0)
    assert res0["ci_lower"] <= 0 <= res0["ci_upper"]
    assert abs(res0["gain"]) < 0.05


def test_primary_count_profile_is_three_disjoint_bins_with_shared_H_dispersions():
    sc, d1 = _data("D1", beta_count=0.7)
    assert d1.count_profile.shape == (sc.n_anchors, 3)
    assert d1.as_meta()["count_profile_edges_seconds"] == [0.0, 300.0, 900.0, 1800.0]
    res = O.run_level(sc, d1, view="count_profile", level=0, horizon=1800.0, seed=0)
    assert res["gain"] > 0 and res["ci_lower"] > 0
    assert res["table_meta"]["score_family"] == "nb_disjoint_count_profile"
    assert res["table_meta"]["dispersion_rule"] == "shared_H_per_bin"
    assert res["table"]["prediction_H"].shape[1] == 3
    assert res["head"]["dispersion_frozen"] is True
    assert np.allclose(res["head"]["log_r"], res["head_H"]["log_r"])
    sc0, d0 = _data("D0")
    null = O.run_level(sc0, d0, view="count_profile", level=0, horizon=1800.0, seed=0)
    assert null["ci_lower"] <= 0 <= null["ci_upper"]


def test_primary_count_profile_recovery_cascades_from_truth_to_trainable_encoder():
    sc, data = _data("D1", beta_count=0.7)
    cascade = O.run_cascade(sc, data, view="count_profile", horizon=1800.0, seed=0, n_steps=120)
    assert cascade["truth_has_state"] is True
    assert [r["detected"] for r in cascade["levels"]] == [True, True, True]
    assert cascade["failure_location"] == "none"


def test_level1_matches_level0_when_the_tau_grid_contains_the_truth():
    sc, d1 = _data("D1", beta_count=0.7)
    l0 = O.run_level(sc, d1, view="count", level=0, horizon=1800.0, seed=0)
    l1 = O.run_level(sc, d1, view="count", level=1, horizon=1800.0, seed=0)
    assert l1["gain"] > 0.8 * l0["gain"]
    assert l1["state_dim"] == 3 * D.MARK_WIDTH


def test_level2_recovers_a_visible_mark_channel_and_fails_when_it_is_hidden():
    sc, d1 = _data("D1", beta_count=0.7)
    l2 = O.run_level(sc, d1, view="count", level=2, horizon=1800.0, seed=0, n_steps=200)
    assert l2["gain"] > 0 and l2["ci_lower"] > 0
    sc5, d5 = _data("D5", beta_count=0.7)
    l2h = O.run_level(sc5, d5, view="count", level=2, horizon=1800.0, seed=0, n_steps=200)
    assert l2h["ci_lower"] <= 0
    assert l2h["inputs"] == "real_tokens_only_mark_channel_hidden"


def test_cascade_reports_failure_location_by_first_lost_level():
    sc, d5 = _data("D5", beta_count=0.7)
    cascade = O.run_cascade(sc, d5, view="count", horizon=1800.0, seed=0, n_steps=200)
    assert [r["level"] for r in cascade["levels"]] == [0, 1, 2]
    assert cascade["failure_location"] == "encoder_optimizer"
    sc0, d0 = _data("D0")
    cascade0 = O.run_cascade(sc0, d0, view="count", horizon=1800.0, seed=0, n_steps=60)
    assert cascade0["truth_has_state"] is False
    assert "false_positive_by_level" in cascade0


# --------------------------------------------------------------------------- levels, grammar view
def test_level0_grammar_detects_grammar_state_and_ignores_a_count_only_state():
    sc, d2 = _data("D2", beta_grammar=2.5)
    res = O.run_level(sc, d2, view="grammar", level=0, horizon=1800.0, seed=0, n_steps=150)
    assert res["gain"] > 0 and res["ci_lower"] > 0
    assert res["n_rows_used"] > 0 and res["scoring"] == "block_average_conditional_subset_nll"
    sc1, d1 = _data("D1", beta_count=0.7)
    res1 = O.run_level(sc1, d1, view="grammar", level=0, horizon=1800.0, seed=0, n_steps=150)
    assert res1["detected"] is False and abs(res1["gain"]) < 1e-6
