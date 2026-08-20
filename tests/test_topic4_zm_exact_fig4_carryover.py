import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_topic4_zm_exact_fig4_carryover import (
    _calibration_parameters, _log_parameter_distance, classify_candidate)
from scripts.freeze_topic4_zm_discovery_boundary import (
    ForbiddenInputError, guard_forbidden, load_audit_config)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit_config():
    return load_audit_config()


@pytest.fixture(scope="module")
def exact(audit_config):
    return audit_config["exact_fig4_carryover"]


def _point(exact, **overrides):
    point = {name: exact[name]
             for name in exact["log_parameter_distance_coordinates"]}
    point["candidate_id"] = exact["candidate_id"]
    point.update(overrides)
    return point


def test_reference_point_is_the_only_exact_carryover(exact):
    row = classify_candidate(exact, _point(exact))
    assert row["arm"] == "exact_fig4_carryover"
    assert row["changed_parameters"] == {}
    assert row["may_be_described_as_fig4_substrate_plus_zm_only"] is True
    assert row["distance_from_exact_carryover"]["log_distance"] == 0.0


def test_scaled_threshold_is_calibrated_and_records_its_delta(exact):
    row = classify_candidate(exact, _point(exact, I_th_EI=0.8 * exact["I_th_EI"]))
    assert row["arm"] == "calibrated_transition"
    assert row["may_be_described_as_fig4_substrate_plus_zm_only"] is False
    assert row["changed_parameters"]["I_th_EI"]["ratio"] == pytest.approx(0.8)
    assert row["distance_from_exact_carryover"]["log_distance"] > 0.0


def test_current_visual_candidate_lands_in_the_calibrated_arm(exact):
    """The 0.8 x I_th_EI / 5% E-to-I point must never read as Fig.4 + Z/M."""
    row = classify_candidate(exact, _point(
        exact, I_th_EI=0.8 * exact["I_th_EI"], E_to_I_dose=0.05))
    assert row["arm"] == "calibrated_transition"
    assert set(row["changed_parameters"]) == {"I_th_EI", "E_to_I_dose"}
    assert row["may_be_described_as_fig4_substrate_plus_zm_only"] is False


def test_spatially_reregistered_run_is_never_exact_carryover(exact):
    """Regression: identical Z/M parameters, different realised node field."""
    for transform in ("r90", "r180", "mx"):
        row = classify_candidate(exact, _point(exact),
                                 field_transform=transform)
        assert row["arm"] == "spatial_reregistration_control"
        assert row["may_be_described_as_fig4_substrate_plus_zm_only"] is False
        assert row["field_transform"] == transform


def test_zeroed_pathway_arm_is_decomposition_not_carryover(exact):
    row = classify_candidate(exact, _point(
        exact, candidate_id="joint_04_ee_only", E_to_I_dose=0.0))
    assert row["arm"] == "pathway_decomposition"
    assert row["may_be_described_as_fig4_substrate_plus_zm_only"] is False


def test_zero_dose_distance_is_finite_through_the_epsilon_floor(exact):
    distance = _log_parameter_distance(_point(exact, E_to_I_dose=0.0), exact)
    assert np.isfinite(distance["log_distance"])
    assert distance["log_distance"] > 0.0
    assert distance["per_axis"]["E_to_I_dose"] == pytest.approx(
        float(np.log(exact["zero_dose_epsilon"])))


def test_missing_dose_keys_default_to_one_and_are_flagged():
    with_doses = _calibration_parameters(
        {"parameters": {"I_th_EI": 1.0, "E_to_E_dose": 1.0, "E_to_I_dose": 0.05}})
    assert with_doses["dose_source"] == "explicit"
    legacy = _calibration_parameters({"parameters": {"I_th_EI": 1.0}})
    assert legacy["E_to_E_dose"] == 1.0 and legacy["E_to_I_dose"] == 1.0
    assert legacy["dose_source"] == "runner_default_1.0_predates_dose_flag"


def test_forbidden_clinical_paths_raise_rather_than_warn(audit_config):
    for path in (
            "results/paper-ready-figure/fig3b_interictal_ictal_shared_field/x.json",
            "scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py",
            "results/anything/clinical_bridge_postfreeze.json",
            "results/anything/seizure_2_early_ictal_field.npz"):
        with pytest.raises(ForbiddenInputError):
            guard_forbidden(audit_config, path)


def test_interictal_training_inputs_stay_allowed(audit_config):
    for path in (
            "results/interictal_propagation_masked/rank_displacement/"
            "per_subject/epilepsiae_1146.json",
            "results/spatial_modulation/propagation_geometry/observation_readout/"
            "real_subjects/epilepsiae_1146_t_a.json",
            "results/topic4_sef_hfo/data_driven_zm_ictal_transition/workers/x.npz"):
        assert guard_forbidden(audit_config, path) == path


def test_audit_config_declares_the_non_blind_status(audit_config):
    assert audit_config["status"] == "DEVELOPMENT_ONLY_RETROSPECTIVE_DISCOVERY_AUDIT"
    assert audit_config["analyst_exposure"]["status"] == "NOT_BLIND"
    assert "BLIND" not in audit_config["scientific_role"].upper().replace(
        "NOT_BLIND", "")


@pytest.mark.skipif(
    not (ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
         "discovery_audit_v1/exact_carryover_audit.json").exists(),
    reason="audit artifact not built in this checkout")
def test_written_audit_has_no_label_violations_and_one_carryover_arm():
    audit = json.loads((
        ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
        "discovery_audit_v1/exact_carryover_audit.json").read_text())
    assert audit["label_guard"]["violations"] == []
    assert audit["trajectories_regenerated"] is False
    assert audit["simulation_launched"] is False
    exact_rows = [row for row in audit["candidates"]
                  if row.get("arm") == "exact_fig4_carryover"]
    assert exact_rows, "the exact carry-over arm must have completed trajectories"
    assert {row["field_transform"] for row in exact_rows} == {"none"}
    assert {row["parameters"]["candidate_id"] for row in exact_rows} == {
        "joint_04_control"}
