"""Pairing discipline of the H1 arm scorer on synthetic designs."""
from __future__ import annotations

import numpy as np

from src.topic5_group_event_state.v032_eval.eligibility import contact_entropy_bits, seizure_clusters
from src.topic5_group_event_state.v032_eval.h1_eval import score_arms, paired_summaries
from src.topic5_group_event_state.v032_eval.blocks import block_ids_for_times


def _cfg():
    return {
        "nb_glm": {"ridge_grid": [0.01, 1.0, 100.0], "alpha_log_bounds": [-9.0, 6.0], "max_irls_iter": 50},
        "inference": {"bootstrap_replicates": 200, "bootstrap_seed": 1, "block_seconds_min": 1800.0},
    }


def _synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 900
    t = np.arange(n) * 300.0
    seg = np.zeros(n, int)
    h = rng.normal(size=(n, 3))                       # explicit history
    s_true = np.sin(t / 5000.0) + rng.normal(scale=0.2, size=n)   # a slow state with real information
    eta = 2.5 + 0.4 * h[:, 0] - 0.3 * h[:, 1] + 0.8 * s_true
    mu = np.exp(eta)
    y = rng.poisson(rng.gamma(shape=4.0, scale=mu / 4.0))
    rows = {"base_fit": np.arange(0, 540), "inner_val": np.arange(540, 630), "base_refit": np.arange(0, 630),
            "dev_val": np.arange(630, 720), "dev_test": np.arange(720, 900)}
    shifted = np.roll(s_true, 300)
    designs = {"H": h, "H+S_correct": np.column_stack([h, s_true]),
               "H+S_shifted:1": np.column_stack([h, shifted]),
               "H+S_mean": np.column_stack([h, np.full(n, s_true[:540].mean())])}
    blocks = {p: block_ids_for_times(t[idx], seg[idx], {0: 0.0}, 1800.0) for p, idx in rows.items() if p in ("dev_val", "dev_test")}
    return y, rows, designs, blocks


def test_score_arms_scores_identical_rows_and_never_fits_on_them():
    y, rows, designs, blocks = _synthetic()
    scored = score_arms(y, rows, designs, _cfg())
    arms = scored["arms"]
    assert set(arms) == set(designs)
    for arm in arms.values():
        assert arm["status"] == "ok"
        assert arm["n_fit_rows"] == 540 and arm["n_select_rows"] == 90 and arm["n_refit_rows"] == 630
        assert arm["scores"]["dev_test"]["nll"].shape == (180,)
    # correct-time state must beat both H and the shifted state on this synthetic truth
    summ = paired_summaries(arms, rows, blocks, _cfg())["dev_test"]["pairs"]
    assert summ["H+S_correct_vs_H"]["mean_gain"] > 0
    assert summ["H+S_correct_vs_H+S_shifted_mean"]["mean_gain"] > 0
    assert summ["H+S_correct_vs_H+S_mean"]["mean_gain"] > 0
    assert summ["H+S_correct_vs_H"]["n_pairs"] == 180


def test_missing_donor_rows_are_dropped_for_that_arm_only_and_reported():
    y, rows, designs, blocks = _synthetic(1)
    shifted = designs["H+S_shifted:1"].copy()
    shifted[rows["dev_test"][:20], -1] = np.nan     # no donor for 20 test anchors
    designs["H+S_shifted:1"] = shifted
    scored = score_arms(y, rows, designs, _cfg())
    arm = scored["arms"]["H+S_shifted:1"]
    assert arm["n_rows_dropped_nonfinite"]["dev_test"] == 20
    assert np.isnan(arm["scores"]["dev_test"]["nll"][:20]).all()
    assert np.isfinite(scored["arms"]["H"]["scores"]["dev_test"]["nll"]).all()
    summ = paired_summaries(scored["arms"], rows, blocks, _cfg())["dev_test"]["pairs"]
    assert summ["H+S_correct_vs_H+S_shifted_mean"]["n_pairs"] == 160
    assert summ["H+S_correct_vs_H+S_shifted_mean"]["n_missing_pairs"] == 20


def test_shared_alpha_rule_freezes_reference_dispersion_for_state_arms():
    y, rows, designs, blocks = _synthetic(2)
    scored = score_arms(y, rows, designs, _cfg(), dispersion_rule="shared_H_alpha")
    alpha_h = scored["arms"]["H"]["alpha"]
    for arm in ("H+S_correct", "H+S_shifted:1", "H+S_mean"):
        assert scored["arms"][arm]["alpha"] == alpha_h


def test_seizure_clusters_and_entropy_helpers():
    seizures = [{"onset_epoch": 0.0}, {"onset_epoch": 3600.0}, {"onset_epoch": 40000.0}]
    assert seizure_clusters(seizures, 14400.0) == [[0, 1], [2]]
    part = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], bool)
    assert abs(contact_entropy_bits(part) - 1.5) < 1e-9
    assert contact_entropy_bits(np.zeros((3, 2), bool)) is None
