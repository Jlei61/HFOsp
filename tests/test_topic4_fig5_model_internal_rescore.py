import json
from pathlib import Path

import numpy as np
import pytest

from scripts.freeze_topic4_zm_discovery_boundary import (
    ForbiddenInputError, load_audit_config)
from scripts.rescore_topic4_fig5_model_internal_candidates import (
    _open_npz, build_shortlist, load_calibration_run, load_worker_run)
from src.topic4_fig5_ictal_bridge import NOT_EVALUABLE

ROOT = Path(__file__).resolve().parents[1]
AUDIT = (ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition"
         / "discovery_audit_v1")


@pytest.fixture(scope="module")
def config():
    return load_audit_config()


def _summary(key, *, arm="calibrated_transition", eligible=1.0, retained=None,
             cross_state=NOT_EVALUABLE, q05=None, distance=1.0, n_reuse=0):
    aggregate = ({"status": "OK", "bootstrap_q05": q05, "median": q05}
                 if q05 is not None else {"status": NOT_EVALUABLE})
    return {
        "candidate_key": key, "arm": arm, "field_transform": "none",
        "candidate_id": "joint_04_control",
        "parameters": {"I_th_EI": 1.0, "tau_z": 1.0, "tau_adp": 1.0,
                       "eta_m": 1.0, "E_to_E_dose": 1.0, "E_to_I_dose": 1.0},
        "changed_parameters": {},
        "log_distance_from_exact_carryover": distance,
        "n_runs": 1, "seeds": [1801], "runs": [key],
        "model_ictal": {"n_evaluable": 1, "eligible_proportion": eligible,
                        "n_eligible": 1, "failing_clauses": []},
        "repertoire": {"n_evaluable": 1, "retained_proportion": retained,
                       "n_retained": 0, "failing_clauses": []},
        "motif_reuse": {"n_evaluable": n_reuse, "n_exceeding_null_q95": n_reuse,
                        "network_aggregate": aggregate,
                        "edge_flow": NOT_EVALUABLE},
        "cross_state_discovery_eligible": cross_state,
        "missing_evidence": [],
    }


def test_forbidden_clinical_artifact_fails_hard_in_the_loader(config, tmp_path):
    forbidden = tmp_path / "clinical_bridge_postfreeze.json"
    forbidden.write_text("{}")
    with pytest.raises(ForbiddenInputError):
        _open_npz(config, forbidden)
    with pytest.raises(ForbiddenInputError):
        load_worker_run(config, forbidden)
    with pytest.raises(ForbiddenInputError):
        load_calibration_run(config, forbidden)


def test_shortlist_never_exceeds_three_and_reads_no_clinical_input():
    summaries = [_summary(f"c{i}", distance=float(i)) for i in range(8)]
    shortlist = build_shortlist(summaries)
    assert len(shortlist["shortlist"]) == 3
    assert shortlist["maximum_shortlist_size"] == 3
    assert shortlist["clinical_inputs_used"] is False


def test_without_a_layer2_candidate_the_status_is_model_ictal_only():
    shortlist = build_shortlist([_summary("a"), _summary("b")])
    assert shortlist["status"] == "MODEL_ICTAL_ONLY_SHORTLIST"


def test_a_layer2_candidate_flips_the_status_and_outranks_a_closer_one():
    summaries = [
        _summary("near_but_layer1_only", distance=0.1),
        _summary("far_but_cross_state", distance=9.0, retained=1.0,
                 cross_state=True, q05=0.4, n_reuse=1),
    ]
    shortlist = build_shortlist(summaries)
    assert shortlist["status"] == "CROSS_STATE_SHORTLIST"
    assert shortlist["shortlist"][0] == "far_but_cross_state"


def test_eligible_proportion_dominates_parameter_distance():
    summaries = [_summary("partial", eligible=0.5, distance=0.0),
                 _summary("full", eligible=1.0, distance=9.0)]
    assert build_shortlist(summaries)["shortlist"][0] == "full"


def test_not_evaluable_and_ineligible_are_excluded_with_distinct_reasons():
    summaries = [_summary("unknown", eligible=None),
                 _summary("failed", eligible=0.0),
                 _summary("good", eligible=1.0)]
    shortlist = build_shortlist(summaries)
    assert shortlist["shortlist"] == ["good"]
    reasons = {row["candidate_key"]: row["reason"] for row in shortlist["excluded"]}
    assert "not evaluable" in reasons["unknown"]
    assert reasons["failed"] == "not model-ictal eligible"


def test_spatial_reregistration_controls_never_enter_the_pool():
    summaries = [_summary("ctl", arm="spatial_reregistration_control"),
                 _summary("real")]
    shortlist = build_shortlist(summaries)
    assert shortlist["shortlist"] == ["real"]
    assert "ctl" not in {row["candidate_key"] for row in shortlist["excluded"]}


@pytest.mark.skipif(not (AUDIT / "model_internal_shortlist.json").exists(),
                    reason="rescore artifact not built in this checkout")
def test_written_shortlist_declares_no_clinical_input_and_respects_the_cap():
    shortlist = json.loads((AUDIT / "model_internal_shortlist.json").read_text())
    assert shortlist["clinical_inputs_used"] is False
    assert len(shortlist["shortlist"]) <= 3
    rescore = json.loads((AUDIT / "model_internal_candidate_rescore.json").read_text())
    assert rescore["simulation_launched"] is False
    assert rescore["clinical_ictal_target_read"] is False
    for row in rescore["candidates"]:
        assert row["motif_reuse"]["edge_flow"] == NOT_EVALUABLE
