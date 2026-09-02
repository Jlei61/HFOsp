"""E1146 row-level discrepancy audit (v0.3.3 plan Task 2, clauses E1-E4).

The audit compares the two v0.3.2 scoring paths step by step -- checkpoint,
anchor set, target, prediction_H, prediction_H_plus_state, dispersion /
intercept, weight, seed aggregation, sign / reduction -- on per-anchor rows,
reproduces both published numbers first, and names the first diverging step
and the step at which the sign flips.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.topic5_group_event_state.v033_evaluator import canonical as C
from src.topic5_group_event_state.v033_evaluator import e1146_audit as A

V032_ROOT = Path("/data/hfosp_group_event_state_v0_3_2")


def _synthetic_seed(seed: int = 0, n: int = 60, *, break_anchors: bool = False,
                    h_shift: float = 0.02, state_noise: float = 0.6) -> A.SeedArtifacts:
    """Two paths on the same anchors: model side gains through dispersion, eval side loses through S."""

    rng = np.random.default_rng(seed)
    idx = np.arange(100, 100 + n)
    t = idx * 300.0
    log_mu_h = rng.normal(5.0, 0.4, n)
    log_r_h = 0.6
    r = np.exp(log_r_h)
    count = rng.negative_binomial(r * 0.8, (r * 0.8) / (r * 0.8 + np.exp(log_mu_h))).astype(np.int64)
    modulation = rng.normal(0.0, 0.05, n)                 # tiny dynamic contribution
    log_r_adapter = 0.45                                   # re-estimated dispersion (lower -> fits overdispersed counts)
    log_mu_hs_model = log_mu_h + modulation
    log_mu_h_eval = log_mu_h + h_shift                    # refit H with a different feature set
    log_mu_hs_eval = log_mu_h_eval + rng.normal(0.0, state_noise, n)  # GLM with 12 raw S columns extrapolating
    alpha_h_eval = np.exp(-0.58)
    alpha_s_eval = np.exp(-0.65)
    eval_idx = idx.copy()
    eval_t = t.copy()
    if break_anchors:
        eval_idx = idx + 1
        eval_t = eval_t + 300.0
    nll = C.nb_nll
    art = A.SeedArtifacts(
        subject="toy", seed=seed,
        checkpoint_sha256_model="abc", checkpoint_sha256_eval="abc",
        arrays_sha256_model="arr", arrays_sha256_eval="arr",
        model_commit="m", eval_commit="e", registry_commit="m",
        anchor_index_model=idx, anchor_time_model=t,
        anchor_index_eval=eval_idx, anchor_time_eval=eval_t,
        count_model=count, count_eval=count.copy(),
        log_mu_h_model=log_mu_h, log_mu_h_eval=log_mu_h_eval,
        n_features_h_model=125, n_features_h_eval=126,
        modulation_model=modulation,
        log_mu_hs_model=log_mu_hs_model,
        log_mu_hs_eval_shared=log_mu_hs_eval, log_mu_hs_eval_per_arm=log_mu_hs_eval,
        log_r_h_model=log_r_h, log_r_hs_model=log_r_adapter,
        log_r_h_eval=-np.log(alpha_h_eval), log_r_hs_eval_per_arm=-np.log(alpha_s_eval),
        intercept_h_eval=4.9, intercept_hs_eval_shared=4.93, intercept_hs_eval_per_arm=4.93,
        nll_h_model=nll(count, log_mu_h, log_r_h), nll_hs_model=nll(count, log_mu_hs_model, log_r_adapter),
        nll_mean_model=nll(count, log_mu_h, log_r_adapter),
        nll_h_eval=nll(count, log_mu_h_eval, -np.log(alpha_h_eval)),
        nll_hs_eval_shared=nll(count, log_mu_hs_eval, -np.log(alpha_h_eval)),
        nll_hs_eval_per_arm=nll(count, log_mu_hs_eval, -np.log(alpha_s_eval)),
        published_model_h_minus_correct=float(np.mean(nll(count, log_mu_h, log_r_h) - nll(count, log_mu_hs_model, log_r_adapter))),
        published_eval_shared_gain=float(np.mean(nll(count, log_mu_h_eval, -np.log(alpha_h_eval)) - nll(count, log_mu_hs_eval, -np.log(alpha_h_eval)))),
        published_eval_per_arm_gain=float(np.mean(nll(count, log_mu_h_eval, -np.log(alpha_h_eval)) - nll(count, log_mu_hs_eval, -np.log(alpha_s_eval)))),
        block_model="6 anchors", block_eval="1800 s bins",
    )
    return art


def test_steps_are_ordered_and_first_divergence_is_prediction_H_on_matching_anchors():
    report = A.audit_seed(_synthetic_seed())
    names = [s["step"] for s in report["steps"]]
    assert names == list(A.STEP_ORDER)
    by = {s["step"]: s for s in report["steps"]}
    assert by["checkpoint"]["diverges"] is False
    assert by["anchor_set"]["diverges"] is False
    assert by["target"]["diverges"] is False
    assert by["prediction_H"]["diverges"] is True
    assert np.isclose(by["prediction_H"]["max_abs_delta_log_mu"], 0.02)
    assert by["prediction_H_plus_state"]["diverges"] is True
    assert by["dispersion_intercept"]["diverges"] is True
    assert by["weight"]["diverges"] is False and by["weight"]["uncertainty_definition_differs"] is True
    assert by["seed_aggregation"]["diverges"] is False
    assert by["score_sign_reduction"]["diverges"] is False
    assert report["first_divergence"] == "prediction_H"


def test_anchor_set_divergence_is_reported_before_any_prediction_step():
    report = A.audit_seed(_synthetic_seed(break_anchors=True))
    assert report["first_divergence"] == "anchor_set"
    by = {s["step"]: s for s in report["steps"]}
    assert by["anchor_set"]["n_common"] < by["anchor_set"]["n_model"]


def test_canonical_rescore_reproduces_both_published_numbers_row_by_row():
    art = _synthetic_seed()
    report = A.audit_seed(art)
    rep = report["reproduction"]
    assert rep["model_h_minus_correct"]["reproduced"] is True
    assert rep["eval_shared_gain"]["reproduced"] is True
    assert rep["eval_per_arm_gain"]["reproduced"] is True
    assert abs(rep["model_h_minus_correct"]["canonical"] - art.published_model_h_minus_correct) < 1e-9


def test_counterfactual_chain_locates_sign_flip_at_prediction_H_plus_state_and_model_gain_in_dispersion():
    art = _synthetic_seed(seed=4)
    report = A.audit_seed(art)
    chain = report["counterfactual_chain"]
    assert chain[0]["swap"] == "none" and chain[0]["gain"] > 0
    steps = [c["swap"] for c in chain]
    assert steps == ["none", "prediction_H", "prediction_H_plus_state", "dispersion"]
    assert report["sign_flip_origin"] == "prediction_H_plus_state"
    decomposition = report["model_side_decomposition"]
    # dispersion-only component (H vs mean-state arm) carries the model-side gain; dynamic-only is tiny
    assert decomposition["dispersion_component_h_minus_mean"] > decomposition["dynamic_component_mean_minus_correct"]
    assert np.isclose(decomposition["dispersion_component_h_minus_mean"] + decomposition["dynamic_component_mean_minus_correct"],
                      art.published_model_h_minus_correct)
    canon = report["canonical_rescore"]
    assert "model_predictions_shared_H_dispersion" in canon and "eval_predictions_shared_H_dispersion" in canon


def test_aggregate_over_seeds_reproduces_published_means_and_keeps_first_divergence():
    arts = [_synthetic_seed(seed=s) for s in range(3)]
    report = A.aggregate_seeds(arts, subject="toy")
    assert report["first_divergence"] == "prediction_H"
    assert np.isclose(report["published_means"]["model_h_minus_correct"],
                      np.mean([a.published_model_h_minus_correct for a in arts]))
    assert np.isclose(report["published_means"]["eval_shared_gain"],
                      np.mean([a.published_eval_shared_gain for a in arts]))
    assert report["n_seeds"] == 3


@pytest.mark.skipif(not (V032_ROOT / "model/runs/leaky_bank/epilepsiae_1146").exists(),
                    reason="v0.3.2 artefacts not mounted")
def test_real_e1146_artifacts_reproduce_published_numbers_and_first_divergence():
    art = A.load_seed_artifacts(V032_ROOT, "epilepsiae_1146", 20260902)
    report = A.audit_seed(art, model_row_tolerance=1e-5)
    rep = report["reproduction"]
    assert rep["model_h_minus_correct"]["reproduced"] and rep["eval_shared_gain"]["reproduced"]
    assert report["first_divergence"] == "prediction_H"
    by = {s["step"]: s for s in report["steps"]}
    assert by["anchor_set"]["diverges"] is False and by["target"]["diverges"] is False
    assert by["prediction_H"]["n_features_model"] == 125 and by["prediction_H"]["n_features_eval"] == 126
