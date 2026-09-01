import json
from pathlib import Path

import numpy as np
import pytest

from scripts.freeze_topic4_zm_discovery_boundary import (
    ForbiddenInputError, load_audit_config)
from scripts.rescore_topic4_fig5_model_internal_candidates import (
    _aggregate_over_runs, _open_npz, build_shortlist, load_calibration_run,
    load_worker_run, run_contact_layer_joint, run_cross_state,
    run_layer_verdicts)
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
        "motif_reuse": {"n_evaluable": n_reuse, "n_supporting_reuse": n_reuse,
                        "n_exceeding_null_q95_including_negative": n_reuse,
                        "network_aggregate": aggregate,
                        "edge_flow": NOT_EVALUABLE},
        "cross_state_discovery_eligible": cross_state,
        "qualification_sensitivity": [],
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
    assert shortlist["is_frozen_workpoint"] is False


def test_without_a_layer2_candidate_it_is_a_replication_design_set():
    shortlist = build_shortlist([_summary("a"), _summary("b")])
    assert shortlist["status"] == "REPLICATION_DESIGN_SET"
    assert shortlist["is_frozen_workpoint"] is False
    assert "not a frozen shortlist" in shortlist["why_not_frozen"]


def test_a_degenerate_ordering_is_declared_as_such():
    """All keys above the distance tie-break equal means no science ranked it."""
    tied = [_summary(f"c{i}", distance=float(i)) for i in range(4)]
    assert build_shortlist(tied)["ordering_is_degenerate"] is True
    separated = tied + [_summary("better", eligible=0.5, distance=0.0)]
    assert build_shortlist(separated)["ordering_is_degenerate"] is False


def test_the_canary_and_its_stop_rule_are_recorded():
    requirements = build_shortlist([_summary("a")])["next_round_requirements"]
    canary = requirements["canary_before_any_replication"]
    assert canary["post_onset_record_ms_minimum"] >= 1200.0
    assert canary["combined_recorder"] is True
    assert "STOP" in canary["stop_rule"]
    assert requirements["if_the_canary_stops"]["name"] == (
        "staged_release_continuity_assay")


def _verdict_row(eligible, repertoire_verdict, rank_ok, precedence_ok):
    layer1 = ({"status": NOT_EVALUABLE} if eligible is None
              else {"status": "x", "eligible": eligible})
    repertoire = ({"status": NOT_EVALUABLE} if repertoire_verdict is None
                  else {"status": "OK", "verdict": repertoire_verdict})
    if rank_ok is None or precedence_ok is None:
        motif = {"status": NOT_EVALUABLE}
    else:
        motif = {"status": "OK",
                 "rank_reuse": {"status": "OK",
                                "null": {"reuse_supported": rank_ok}},
                 "precedence_reuse": {"status": "OK",
                                      "null": {"reuse_supported": precedence_ok}}}
    return {"run": "r", "seed": 1, "layer1_model_ictal": layer1,
            "layer2_repertoire": repertoire, "layer2_motif": motif}


def test_layers_cannot_be_assembled_from_different_runs():
    """One run supplying each layer must not add up to a cross-state candidate."""
    runs = [_verdict_row(True, None, None, None),
            _verdict_row(None, "REPERTOIRE_RETAINED", None, None),
            _verdict_row(None, None, True, True)]
    per_run = [run_cross_state(run_layer_verdicts(row)) for row in runs]
    assert per_run == [None, None, None]
    assert _aggregate_over_runs(per_run) == NOT_EVALUABLE
    contact = [run_contact_layer_joint(run_layer_verdicts(row)) for row in runs]
    assert contact == [None, None, None]
    assert _aggregate_over_runs(contact) == NOT_EVALUABLE


def test_insufficient_preictal_evidence_is_none_not_false():
    verdict = run_layer_verdicts(
        _verdict_row(True, "PREICTAL_EVIDENCE_INSUFFICIENT", True, True))
    assert verdict["repertoire_retained"] is None
    assert run_cross_state(verdict) is None
    assert run_contact_layer_joint(verdict) is None


def test_one_contact_family_alone_does_not_support_the_contact_layer():
    verdict = run_layer_verdicts(
        _verdict_row(True, "REPERTOIRE_RETAINED", True, False))
    assert verdict["contact_order_supported"] is False
    both = run_layer_verdicts(
        _verdict_row(True, "REPERTOIRE_RETAINED", True, True))
    assert both["contact_order_supported"] is True
    assert run_contact_layer_joint(both) is True


def test_the_motif_gate_stays_unanswerable_while_edge_flow_is_missing():
    verdict = run_layer_verdicts(
        _verdict_row(True, "REPERTOIRE_RETAINED", True, True))
    assert verdict["motif_gate"] is None
    assert "edge-flow" in verdict["motif_gate_reason"]
    assert run_cross_state(verdict) is None


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


@pytest.mark.skipif(
    not (AUDIT / "model_internal_replication_design_set.json").exists(),
    reason="rescore artifact not built in this checkout")
def test_written_design_set_declares_no_clinical_input_and_respects_the_cap():
    shortlist = json.loads(
        (AUDIT / "model_internal_replication_design_set.json").read_text())
    assert shortlist["clinical_inputs_used"] is False
    assert shortlist["is_frozen_workpoint"] is False
    assert len(shortlist["shortlist"]) <= 3
    rescore = json.loads((AUDIT / "model_internal_candidate_rescore.json").read_text())
    assert rescore["simulation_launched"] is False
    assert rescore["clinical_ictal_target_read"] is False
    for row in rescore["candidates"]:
        assert row["motif_reuse"]["edge_flow"] == NOT_EVALUABLE
        assert row["motif_reuse"]["gate_status"] == NOT_EVALUABLE
        assert row["cross_state_discovery_eligible"] == NOT_EVALUABLE
