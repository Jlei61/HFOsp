"""Scaffold + synthetic DGPs D0-D5 (v0.3.3 plan Task 4, clauses D-1..D-6)."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from src.topic5_group_event_state.v033_evaluator import dgp as D
from src.topic5_group_event_state.v033_evaluator import scaffold as S
from tests.test_group_event_state_v033_toyutil import HORIZONS, toy_scaffold

V032_ROOT = Path("/data/hfosp_group_event_state_v0_3_2")


# --------------------------------------------------------------------------- scaffold
def test_toy_scaffold_rows_are_phase_and_window_eligible_and_blocks_come_from_coverage():
    sc = toy_scaffold()
    rows = sc.anchor_rows("dev_test", 1800.0)
    assert rows.size > 0
    assert (sc.anchor_phase[rows] == 3).all() and sc.eligible[rows, 1].all()
    lo, hi = sc.phase_bounds[3]
    manual = sum(int(np.floor((min(b, hi) - max(a, lo)) / 1800.0)) for a, b in sc.segment_bounds if min(b, hi) > max(a, lo))
    assert sc.independent_blocks("dev_test", 1800.0) == manual
    assert sc.event_size().shape == (sc.event_times.size,)
    assert sc.n_contacts == sc.participation.shape[1]


@pytest.mark.skipif(not (V032_ROOT / "shared/history_baseline_registry.json").exists(),
                    reason="v0.3.2 registry not mounted")
def test_real_scaffold_e1146_is_the_registry_grid_and_carry_variants_differ_on_post_seizure_anchors():
    from src.topic5_group_event_state.v032_eval.contract import load_eval_config

    cfg = load_eval_config()
    seg = S.load_real_scaffold("epilepsiae_1146", cfg, carry="segment")
    assert seg.t_anchor.size == 673 and seg.anchor_rows("dev_test", 1800.0).size == 113
    assert np.isfinite(seg.log_mu_h[1800][seg.eligible[:, 1]]).all()
    assert abs(seg.log_r_h[1800] - 0.5971829965024117) < 1e-9
    ses = S.load_real_scaffold("epilepsiae_1146", cfg, carry="session")
    assert int((seg.last_event_pos != ses.last_event_pos).sum()) == 12
    assert (ses.last_event_pos >= seg.last_event_pos).all()   # session carry only adds history


# --------------------------------------------------------------------------- conditional Bernoulli
def test_conditional_bernoulli_logpmf_normalises_over_subsets_of_size_k_and_matches_conditioning():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=5)
    p = 1.0 / (1.0 + np.exp(-logits))
    for k in (0, 1, 2, 3, 5):
        subsets = [np.array([i in s for i in range(5)]) for s in itertools.combinations(range(5), k)]
        logp = D.conditional_bernoulli_logpmf(np.tile(logits, (len(subsets), 1)), np.array(subsets))
        assert np.isclose(np.exp(logp).sum(), 1.0)
        # explicit conditioning: P(S) / sum_{|S'|=k} P(S')
        joint = np.array([np.prod(np.where(s, p, 1 - p)) for s in subsets])
        assert np.allclose(np.exp(logp), joint / joint.sum())


def test_sample_conditional_bernoulli_has_exact_size_and_matches_the_pmf():
    rng = np.random.default_rng(1)
    logits = np.array([1.2, -0.4, 0.3, -1.5])
    draws = np.array([D.sample_conditional_bernoulli(rng, logits, 2) for _ in range(20_000)])
    assert (draws.sum(axis=1) == 2).all()
    subsets = [np.array([i in s for i in range(4)]) for s in itertools.combinations(range(4), 2)]
    exact = np.exp(D.conditional_bernoulli_logpmf(np.tile(logits, (6, 1)), np.array(subsets)))
    freq = np.array([np.mean((draws == s).all(axis=1)) for s in subsets])
    assert np.max(np.abs(freq - exact)) < 0.015
    assert D.sample_conditional_bernoulli(rng, logits, 0).sum() == 0
    assert D.sample_conditional_bernoulli(rng, logits, 4).all()


# --------------------------------------------------------------------------- hidden state
def test_hidden_leaky_state_is_the_declared_leaky_integral_with_carry_reset():
    sc = toy_scaffold(seed=2)
    rng = np.random.default_rng(3)
    marks = rng.normal(size=(sc.event_times.size, 2))
    anchor_state, pre_state = D.hidden_leaky_state(marks, sc.event_times, sc.event_carry, sc.t_anchor,
                                                   sc.anchor_carry, sc.last_event_pos, tau=1800.0)
    a = 40
    j = sc.last_event_pos[a]
    assert j >= 0
    same = np.flatnonzero((sc.event_carry == sc.event_carry[j]) & (np.arange(sc.event_times.size) <= j))
    brute = (marks[same] * np.exp(-(sc.t_anchor[a] - sc.event_times[same])[:, None] / 1800.0)).sum(axis=0)
    assert np.allclose(anchor_state[a], brute, atol=1e-6)
    first_of_unit = np.flatnonzero(np.r_[True, sc.event_carry[1:] != sc.event_carry[:-1]])
    assert np.allclose(pre_state[first_of_unit], 0.0)
    no_history = sc.last_event_pos < 0
    assert np.allclose(anchor_state[no_history], 0.0)


# --------------------------------------------------------------------------- DGP kinds
def _gen(kind, seed=0, **kw):
    sc = toy_scaffold(seed=seed)
    defaults = dict(beta_count=0.5, beta_grammar=1.0, generator_seed=10 + seed, noise_seed=20 + seed)
    defaults.update(kw)
    return sc, D.generate(sc, kind, **defaults)


def test_generate_never_alters_the_real_scaffold_and_records_seeds():
    sc = toy_scaffold(seed=5)
    before = {k: np.array(getattr(sc, k), copy=True) for k in ("t_anchor", "event_times", "eligible", "participation")}
    data = D.generate(sc, "D3", beta_count=0.5, beta_grammar=1.0, generator_seed=1, noise_seed=2)
    for k, v in before.items():
        assert np.array_equal(getattr(sc, k), v), k
    assert data.generator_seed == 1 and data.noise_seed == 2 and data.kind == "D3"
    assert set(data.counts) == {int(h) for h in HORIZONS}
    with pytest.raises(ValueError):
        D.generate(sc, "D9", beta_count=0.5, beta_grammar=1.0, generator_seed=1, noise_seed=2)


def test_d0_is_h_only_for_both_views_with_registry_log_mu_as_base_rate():
    sc, data = _gen("D0")
    assert data.has_state == {"count": False, "grammar": False}
    assert np.allclose(data.z_count, 0.0) and np.allclose(data.z_grammar_anchor, 0.0)
    for h in HORIZONS:
        rows = sc.anchor_rows("base_fit", h)
        assert np.allclose(data.log_mu_true[int(h)][rows], sc.log_mu_h[int(h)][rows])
        assert data.counts[int(h)].dtype.kind == "i" and (data.counts[int(h)] >= 0).all()
    assert data.marks is not None and data.marks.shape == (sc.event_times.size, D.MARK_WIDTH)


def test_d1_count_only_and_d2_grammar_only():
    sc, d1 = _gen("D1")
    assert d1.has_state == {"count": True, "grammar": False}
    train = sc.anchor_rows("base_fit", 1800.0)
    assert abs(d1.z_count[train].mean()) < 1e-6 and abs(d1.z_count[train].std() - 1.0) < 1e-6
    assert np.allclose(d1.log_mu_true[1800][train], sc.log_mu_h[1800][train] + 0.5 * d1.z_count[train])
    assert np.allclose(d1.z_grammar_anchor, 0.0)
    sc, d2 = _gen("D2")
    assert d2.has_state == {"count": False, "grammar": True}
    assert np.allclose(d2.log_mu_true[1800], sc.log_mu_h[1800]) and not np.allclose(d2.z_grammar_anchor, 0.0)


def test_d3_shares_one_state_and_d4_uses_two_independent_states():
    _sc, d3 = _gen("D3")
    assert d3.has_state == {"count": True, "grammar": True}
    assert np.allclose(d3.z_count, d3.z_grammar_anchor)
    assert d3.marks.shape[1] == D.MARK_WIDTH
    _sc, d4 = _gen("D4")
    assert d4.has_state == {"count": True, "grammar": True}
    assert abs(np.corrcoef(d4.z_count, d4.z_grammar_anchor)[0, 1]) < 0.3
    assert d4.marks.shape[1] == 2 * D.MARK_WIDTH
    assert d4.innovations["count"].shape[1] == D.MARK_WIDTH and d4.innovations["grammar"].shape[1] == D.MARK_WIDTH


def test_d5_state_exists_but_mark_channel_is_invisible():
    _sc, d5 = _gen("D5")
    assert d5.has_state == {"count": True, "grammar": True}
    assert d5.marks is None
    assert not np.allclose(d5.z_count, 0.0)


def test_grammar_subsets_keep_the_real_event_size_and_respond_to_the_state():
    sc, d3 = _gen("D3", beta_grammar=3.0)
    assert np.array_equal(d3.participation.sum(axis=1), sc.event_size())
    # contacts with positive loading should be more often chosen when the state is high
    z = d3.z_grammar_event
    hi, lo = z > np.quantile(z, 0.8), z < np.quantile(z, 0.2)
    pos = d3.loadings > 0
    rate_hi = d3.participation[hi][:, pos].mean()
    rate_lo = d3.participation[lo][:, pos].mean()
    assert rate_hi > rate_lo
    _sc, d0 = _gen("D0")
    assert np.array_equal(d0.participation.sum(axis=1), sc.event_size())


def test_generator_seed_fixes_the_state_and_noise_seed_fixes_the_draws():
    sc = toy_scaffold(seed=7)
    a = D.generate(sc, "D1", beta_count=0.5, beta_grammar=1.0, generator_seed=1, noise_seed=2)
    b = D.generate(sc, "D1", beta_count=0.5, beta_grammar=1.0, generator_seed=1, noise_seed=3)
    c = D.generate(sc, "D1", beta_count=0.5, beta_grammar=1.0, generator_seed=1, noise_seed=2)
    assert np.array_equal(a.z_count, b.z_count) and not np.array_equal(a.counts[1800], b.counts[1800])
    assert np.array_equal(a.counts[1800], c.counts[1800])
