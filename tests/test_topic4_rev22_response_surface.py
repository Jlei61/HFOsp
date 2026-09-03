import numpy as np
import pytest
from scipy.stats import qmc

from src.topic4_rev22_response_surface import (
    COMPONENTS, PARAMETER_ORDER, REFERENCE, conditional_minimax_proposal, fit_all,
    fit_component_gp, fit_feasibility, fit_tree_ensemble, from_unit_cube, loo_diagnostics,
    pooled_shrinkage_noise, proposals_disagree, tree_proposal, unit_cube,
)

DOMAIN = {"g_LEE": [0.0, 1.0], "g_LEI": [0.0, 1.5], "theta_FT_deg": [22.5, 67.5], "AR_FT": [1.0, 3.0]}
CENTER = np.array([0.5, 0.5, 0.5, 0.5])
DELTA = 0.35
CENTERS = {
    "D_support": CENTER + DELTA * np.eye(4)[0],
    "D_order": CENTER - DELTA * np.eye(4)[0],
    "D_lag": CENTER + DELTA * np.eye(4)[3],   # AR_FT unit 0.85 -> 2.7, inside the infeasible band
    "D_cover": CENTER - DELTA * np.eye(4)[3],
}
CURVATURE = 2.0
UNIT_SD = 0.10
N_UNITS = 4
AR_FEASIBLE_MAX = 2.6


def _true_component(k, u):
    u = np.atleast_2d(u)
    return CURVATURE * np.sum((u - CENTERS[k]) ** 2, axis=1)


def _design(n=80, seed=3):
    u = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(7)[:n]
    return u, from_unit_cube(u, DOMAIN)


def _rows(seed=5):
    rng = np.random.default_rng(seed)
    u, X = _design()
    rows = []
    mean_sd = UNIT_SD / np.sqrt(N_UNITS)
    for i in range(len(X)):
        row = {"candidate_id": f"c{i:03d}", "x": X[i].tolist(), "Z": {}, "jackknife_sd": {},
               "n_units": N_UNITS, "feasible": bool(X[i, 3] <= AR_FEASIBLE_MAX)}
        for k in COMPONENTS:
            row["Z"][k] = float(_true_component(k, u[i])[0] + rng.normal(0.0, mean_sd))
            row["jackknife_sd"][k] = float(mean_sd * rng.uniform(0.7, 1.3))
        rows.append(row)
    return rows, u, X


@pytest.fixture(scope="module")
def fitted():
    rows, u, X = _rows()
    return fit_all(rows, list(COMPONENTS), DOMAIN, seed=11, n_restarts=2, n_estimators=200), rows, u, X


def test_shrinkage_formula_hand_example():
    out = pooled_shrinkage_noise({"a": 0.05, "b": 0.10, "c": None}, 4)
    # unit-level variances: a = 4*0.0025 = 0.01, b = 4*0.01 = 0.04 -> pooled 0.025
    assert out["n_finite"] == 2
    assert out["s_pool_sq"] == pytest.approx(0.025)
    expected_a = (8 * 0.025 + 3 * 0.01) / 11 / 4
    expected_b = (8 * 0.025 + 3 * 0.04) / 11 / 4
    assert out["variance"]["a"] == pytest.approx(expected_a)
    assert out["variance"]["b"] == pytest.approx(expected_b)
    assert "c" not in out["variance"]
    assert pooled_shrinkage_noise({"a": None}, 4)["variance"] == {}


def test_unit_cube_round_trip_and_reference_position():
    u = unit_cube(REFERENCE, DOMAIN)
    assert np.allclose(from_unit_cube(u, DOMAIN), REFERENCE)
    assert np.allclose(u, [0.5, 1.0 / 1.5, 0.5, 0.5])
    assert tuple(PARAMETER_ORDER) == ("g_LEE", "g_LEI", "theta_FT_deg", "AR_FT")


def test_dose_only_zero_width_geometry_domain_is_supported():
    rows, _, _ = _rows()
    fallback = {**DOMAIN, "theta_FT_deg": [45.0, 45.0], "AR_FT": [2.0, 2.0]}
    for row in rows[:24]:
        row["x"][2:] = [45.0, 2.0]
        row["feasible"] = True
    fit = fit_all(rows[:24], ["D_support"], fallback, seed=3, n_restarts=0, n_estimators=20)
    proposal = conditional_minimax_proposal(
        fit["gps"], ["D_support"], "1100", REFERENCE, fallback, fit["feasibility"], seed=3,
    )
    assert proposal["status"] == "OK"
    assert proposal["x"][2:] == [45.0, 2.0]


def test_gp_loo_is_accurate_and_calibrated(fitted):
    fit, rows, u, X = fitted
    for k in COMPONENTS:
        loo = fit["loo"][k]
        assert loo["status"] == "OK"
        assert loo["rmse"] < 0.15 * loo["observed_range"]
        assert 0.75 <= loo["coverage_90"] <= 1.0
        assert loo["spearman"] > 0.9
        assert "optimizer_failed_fold_count" in loo
    assert fit["surrogate_adequacy"]["status"] == "SURROGATE_ADEQUATE"


def test_surface_predictions_track_truth(fitted):
    fit, rows, u, X = fitted
    probe_u = np.array([[0.5, 0.5, 0.5, 0.5], [0.2, 0.7, 0.4, 0.3]])
    probe_x = from_unit_cube(probe_u, DOMAIN)
    for k in COMPONENTS:
        mean, sd = fit["gps"][k].predict(probe_x)
        truth = _true_component(k, probe_u)
        assert np.allclose(mean, truth, atol=0.15)
        assert np.all(sd > 0)
        tree_mean, tree_spread = fit["trees"][k].predict(probe_x)
        assert np.allclose(tree_mean, truth, atol=0.4)
        assert np.all(tree_spread >= 0)


def test_full_mask_minimax_lands_at_symmetric_center_and_respects_feasibility(fitted):
    fit, rows, u, X = fitted
    proposal = conditional_minimax_proposal(
        fit["gps"], list(COMPONENTS), "1111", REFERENCE, DOMAIN, fit["feasibility"], seed=1,
    )
    assert proposal["status"] == "OK"
    assert np.linalg.norm(np.asarray(proposal["unit"]) - CENTER) < 0.1
    # regret on the true minimax objective is small even where the objective is shallow
    true_j = max(_true_component(k, np.asarray(proposal["unit"]))[0] for k in COMPONENTS)
    true_j_star = max(_true_component(k, CENTER)[0] for k in COMPONENTS)
    assert true_j - true_j_star < 0.1
    assert proposal["feasibility"] >= 0.80
    assert proposal["x"][3] <= AR_FEASIBLE_MAX + 0.05
    assert proposal["argmax_component"] in COMPONENTS
    assert proposal["J"] == pytest.approx(max(proposal["predicted_excess"].values()))
    assert proposal["n_feasible_candidates"] > 0


def test_infeasible_minimum_is_not_proposed_for_single_component(fitted):
    fit, rows, u, X = fitted
    # D_lag alone has its minimum at AR unit 0.85 (physical 2.7 > 2.6): must be held back
    proposal = conditional_minimax_proposal(
        fit["gps"], ["D_lag"], "0001", REFERENCE, DOMAIN, fit["feasibility"], seed=2,
    )
    assert proposal["status"] == "OK"
    assert proposal["x"][3] <= AR_FEASIBLE_MAX + 0.08
    assert proposal["feasibility"] >= 0.80
    # without the feasibility model the unconstrained minimum is inside the band
    unconstrained = conditional_minimax_proposal(
        fit["gps"], ["D_lag"], "0001", REFERENCE, DOMAIN, None, seed=2,
    )
    assert unconstrained["x"][3] > AR_FEASIBLE_MAX


def test_locked_mask_keeps_locked_coordinates_exactly_at_reference(fitted):
    fit, rows, u, X = fitted
    proposal = conditional_minimax_proposal(
        fit["gps"], list(COMPONENTS), "1100", REFERENCE, DOMAIN, fit["feasibility"], seed=3,
    )
    assert proposal["x"][2] == REFERENCE[2] and proposal["x"][3] == REFERENCE[3]
    assert abs(proposal["unit"][0] - CENTER[0]) < 0.1
    for point in proposal["pareto_set"]:
        assert point["x"][2] == REFERENCE[2] and point["x"][3] == REFERENCE[3]


def test_reference_mask_returns_reference(fitted):
    fit, rows, u, X = fitted
    proposal = conditional_minimax_proposal(
        fit["gps"], list(COMPONENTS), "0000", REFERENCE, DOMAIN, fit["feasibility"], seed=4,
    )
    assert np.allclose(proposal["x"], REFERENCE)
    assert proposal["pareto_set"] == [] and proposal["n_feasible_candidates"] == 1


def test_pareto_set_is_non_dominated(fitted):
    fit, rows, u, X = fitted
    proposal = conditional_minimax_proposal(
        fit["gps"], list(COMPONENTS), "1111", REFERENCE, DOMAIN, fit["feasibility"], seed=5,
    )
    pareto = proposal["pareto_set"]
    assert 1 <= len(pareto) <= 64
    values = np.asarray([[p["predicted_excess"][k] for k in COMPONENTS] for p in pareto])
    for i in range(len(values)):
        for j in range(len(values)):
            if i != j:
                assert not (np.all(values[j] <= values[i]) and np.any(values[j] < values[i]))


def test_tree_proposal_and_disagreement_rule(fitted):
    fit, rows, u, X = fitted
    gp = conditional_minimax_proposal(fit["gps"], list(COMPONENTS), "1111", REFERENCE, DOMAIN,
                                      fit["feasibility"], seed=6)
    tree = tree_proposal(fit["trees"], list(COMPONENTS), "1111", REFERENCE, DOMAIN,
                         fit["feasibility"], seed=6)
    assert tree["status"] == "OK" and tree["feasibility"] >= 0.80
    verdict = proposals_disagree(gp["x"], tree["x"], DOMAIN)
    assert verdict["unit_distance"] >= 0.0 and isinstance(verdict["disagree"], bool)
    far = from_unit_cube(np.array([0.05, 0.05, 0.95, 0.05]), DOMAIN)
    assert proposals_disagree(gp["x"], far, DOMAIN)["disagree"] is True
    assert proposals_disagree(gp["x"], gp["x"], DOMAIN)["disagree"] is False
    assert proposals_disagree(None, gp["x"], DOMAIN)["disagree"] is None


def test_constant_feasibility_models_and_no_feasible_region():
    rows, u, X = _rows()
    all_ok = fit_feasibility(X, np.ones(len(X), bool), DOMAIN, seed=0)
    assert all_ok.is_constant and np.all(all_ok.predict_proba(X) == 1.0)
    none_ok = fit_feasibility(X, np.zeros(len(X), bool), DOMAIN, seed=0)
    assert none_ok.is_constant and np.all(none_ok.predict_proba(X) == 0.0)
    z = np.asarray([r["Z"]["D_support"] for r in rows])
    noise = np.full(len(z), (UNIT_SD ** 2) / N_UNITS)
    gp = fit_component_gp(X, z, noise, DOMAIN, seed=0, n_restarts=1)
    proposal = conditional_minimax_proposal({"D_support": gp}, ["D_support"], "1111", REFERENCE,
                                            DOMAIN, none_ok, seed=0)
    assert proposal["status"] == "NO_FEASIBLE_REGION" and proposal["x"] is None


def test_non_finite_rows_are_dropped_and_loo_reports_size():
    rows, u, X = _rows()
    z = np.asarray([r["Z"]["D_order"] for r in rows])
    z[:5] = np.nan
    noise = np.full(len(z), (UNIT_SD ** 2) / N_UNITS)
    gp = fit_component_gp(X, z, noise, DOMAIN, seed=0, n_restarts=1)
    assert gp.n_train == len(z) - 5
    tree = fit_tree_ensemble(X, z, DOMAIN, seed=0, n_estimators=50)
    assert tree.n_train == len(z) - 5
    loo = loo_diagnostics(X, z, noise, DOMAIN, seed=0, n_restarts=1)
    assert loo["n"] == len(z) - 5
    with pytest.raises(ValueError):
        conditional_minimax_proposal({"D_order": gp}, [], "1111", REFERENCE, DOMAIN, None)
    with pytest.raises(ValueError):
        conditional_minimax_proposal({"D_order": gp}, ["D_order"], "111", REFERENCE, DOMAIN, None)
