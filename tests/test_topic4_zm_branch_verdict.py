"""Task 13 (spec rev3.1 §13): the adjudicator is pure and fails closed.

The adversarial cases matter more than the happy path: missing seeds, missing noise replicas, a
single lucky seed, a source-only carrier and an un-run neighbourhood must all degrade the verdict --
none of them may become Branch F or an actuator authorization.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_branch_verdict as BV  # noqa: E402
import src.topic4_zm_minimal_carrier as MC  # noqa: E402
import src.topic4_zm_neighbourhood as NBH  # noqa: E402

BINS = ("bounded_early", "bounded_mid", "bounded_late")
PHASES = ("trough", "peak")
ARMS = ("freeze_all", "freeze_zm", "dynamic_replay")


def _rows(seeds, klass_by_arm, bins=BINS, phases=PHASES, n_rep=3):
    out = []
    for s in seeds:
        for b in bins:
            for p in phases:
                for a, survived in klass_by_arm.items():
                    for i in range(n_rep):
                        out.append(dict(seed=s, bin_name=b, fast_phase=p, arm=a,
                                        replicate=f"noise_{i}", survived=survived,
                                        lifetime_ms=8000.0 if survived else 150.0,
                                        end_reason=None if survived else "rest_return",
                                        rest_returns=0 if survived else 1,
                                        stationarity_ok=True,
                                        is_control_arm=(a == "dynamic_replay")))
    return out


def _cells(rows):
    return BV.classify_matrix(rows, {1: 100.0, 3: 100.0, 4: 100.0}, MC.classify_replicas)


def _adj(rows, **kw):
    cells = _cells(rows)
    per_arm = BV.carrier_window(cells)
    base = dict(state_inventory_ok=True, exact_resume_ok=True,
                eligible_seeds=kw.pop("eligible_seeds", [1, 3, 4]), cells=cells, per_arm=per_arm)
    base.update(kw)
    return BV.adjudicate(**base), cells, per_arm


def test_verdict_vocabulary_is_exactly_the_spec_list():
    assert set(BV.VERDICTS) == {
        "blocked_state_inventory", "blocked_exact_resume", "blocked_reference_artifacts",
        "insufficient_bounded_anchors", "representation_sensitive_no_branch",
        "carrier_at_visited_states", "branch_T_slow_trajectory_repair",
        "branch_F_fast_carrier_repair", "branch_M_calibration", "existing_M_lifecycle_candidate",
        "phase3_driver_selection_required", "observation_layer_blocked", "no_evidence"}


def test_state_and_resume_gates_block_first():
    rows = _rows([1, 3, 4], {"freeze_all": True})
    cells = _cells(rows)
    out = BV.adjudicate(state_inventory_ok=False, exact_resume_ok=True, eligible_seeds=[1, 3, 4],
                        cells=cells, per_arm=BV.carrier_window(cells))
    assert out["verdict"] == "blocked_state_inventory"
    out = BV.adjudicate(state_inventory_ok=True, exact_resume_ok=False, eligible_seeds=[1, 3, 4],
                        cells=cells, per_arm=BV.carrier_window(cells))
    assert out["verdict"] == "blocked_exact_resume"


def test_replicated_positive_is_a_carrier_window():
    out, cells, per_arm = _adj(_rows([1, 3, 4], {"freeze_all": True}))
    assert out["verdict"] == "carrier_at_visited_states"
    assert per_arm["freeze_all"]["status"] == "carrier_window"
    assert out["layers"]["source_space_carrier"] == "carrier_window"


def test_one_lucky_seed_is_isolated_not_a_window():
    rows = _rows([1], {"freeze_all": True}) + _rows([3, 4], {"freeze_all": False})
    out, cells, per_arm = _adj(rows)
    assert per_arm["freeze_all"]["status"] == "isolated_carrier_candidate"
    assert out["verdict"] != "carrier_at_visited_states"
    assert "needs>=2 seeds" in per_arm["freeze_all"]["unmet"]


def test_positive_in_one_fast_phase_only_is_not_a_window():
    rows = _rows([1, 3, 4], {"freeze_all": True}, phases=("peak",)) + \
        _rows([1, 3, 4], {"freeze_all": False}, phases=("trough",))
    _, _, per_arm = _adj(rows)
    assert per_arm["freeze_all"]["status"] == "isolated_carrier_candidate"
    assert "needs>=2 fast phases" in per_arm["freeze_all"]["unmet"]


def test_positive_in_one_slow_bin_only_is_not_a_window():
    rows = _rows([1, 3, 4], {"freeze_all": True}, bins=("bounded_mid",)) + \
        _rows([1, 3, 4], {"freeze_all": False}, bins=("bounded_early", "bounded_late"))
    _, _, per_arm = _adj(rows)
    assert not per_arm["freeze_all"]["adjacent_bins"]
    assert per_arm["freeze_all"]["status"] == "isolated_carrier_candidate"


def test_fewer_than_three_eligible_anchors_can_never_be_branch_F():
    out, _, _ = _adj(_rows([1, 3], {"freeze_all": False}), eligible_seeds=[1, 3],
                     neighbourhood=NBH.branch_verdict(False, [], [1, 3], True))
    assert out["verdict"] == "insufficient_bounded_anchors"


def test_no_carrier_without_a_neighbourhood_audit_is_no_evidence_not_branch_F():
    out, _, _ = _adj(_rows([1, 3, 4], {"freeze_all": False}))
    assert out["verdict"] == "no_evidence"
    assert "neighbourhood" in out["reason"]
    assert out["layers"]["source_space_carrier"] == "no_carrier_in_completed_cells"


def test_branch_F_requires_three_seeds_neighbourhood_and_agreeing_representations():
    nb = NBH.branch_verdict(False, [], [1, 3, 4], representations_agree=True,
                            local_negative_seeds=[1, 3, 4], evidence_complete=True)
    out, _, _ = _adj(_rows([1, 3, 4], {"freeze_all": False}), neighbourhood=nb)
    assert out["verdict"] == "branch_F_fast_carrier_repair"
    nb_dis = NBH.branch_verdict(False, [], [1, 3, 4], representations_agree=False,
                                local_negative_seeds=[1, 3, 4], evidence_complete=True)
    out2, _, _ = _adj(_rows([1, 3, 4], {"freeze_all": False}), neighbourhood=nb_dis)
    assert out2["verdict"] == "representation_sensitive_no_branch"


def test_local_positive_in_two_seeds_is_branch_T():
    nb = NBH.branch_verdict(False, [1, 3], [1, 3, 4], representations_agree=True)
    out, _, _ = _adj(_rows([1, 3, 4], {"freeze_all": False}), neighbourhood=nb)
    assert out["verdict"] == "branch_T_slow_trajectory_repair"


def test_source_only_carrier_never_authorizes_an_actuator():
    out, _, _ = _adj(_rows([1, 3, 4], {"freeze_all": True}))
    tagged = BV.apply_observation_status(out, reference_lock=dict(sufficient_reference_sample=False))
    assert tagged["observation_layer_blocked"] is True
    assert tagged["actuator_authorized"] is False


def test_coverage_report_names_what_was_not_run():
    rows = _rows([1], {"freeze_all": False}, bins=("bounded_mid",), phases=("peak",))
    cov = BV.coverage_report(_cells(rows), dict(seeds=[1, 3, 4], bins=list(BINS),
                                                phases=list(PHASES), arms=["freeze_all"]))
    assert cov["n_cells_run"] == 1 and cov["n_cells_planned"] == 18
    assert cov["n_not_run"] == 17 and cov["not_run"]


def test_control_arms_are_flagged_and_do_not_create_a_carrier_claim():
    rows = _rows([1, 3, 4], {"dynamic_replay": True, "freeze_all": False})
    out, cells, per_arm = _adj(rows)
    assert all(v["is_control_arm"] for k, v in cells.items() if k[3] == "dynamic_replay")
    assert per_arm["freeze_all"]["status"] == "no_carrier"
    assert per_arm["dynamic_replay"]["status"] == "control_window"
    assert out["verdict"] != "carrier_at_visited_states"


def test_disjoint_seed_bin_phase_positives_cannot_fake_compatible_window():
    rows = _rows([1, 3, 4], {"freeze_all": False})
    for r in rows:
        r["survived"] = bool(
            (r["seed"] == 1 and r["bin_name"] == "bounded_early" and r["fast_phase"] == "trough")
            or (r["seed"] == 3 and r["bin_name"] == "bounded_mid" and r["fast_phase"] == "peak")
            or (r["seed"] == 4 and r["bin_name"] == "bounded_late" and
                r["fast_phase"] == "trough"))
        r["lifetime_ms"] = 8000.0 if r["survived"] else 100.0
    out, _, per_arm = _adj(rows)
    assert per_arm["freeze_all"]["status"] == "isolated_carrier_candidate"
    assert not per_arm["freeze_all"]["compatible_witnesses"]
    assert out["verdict"] != "carrier_at_visited_states"


def test_partial_neighbourhood_negative_cannot_default_to_branch_F():
    nb = NBH.branch_verdict(False, [], [1, 3, 4], representations_agree=True,
                            local_negative_seeds=[1], evidence_complete=False)
    out, _, _ = _adj(_rows([1, 3, 4], {"freeze_all": False}), neighbourhood=nb)
    assert out["verdict"] == "no_evidence"


def test_incomplete_fork_ladder_is_partial_evidence_not_no_carrier_in_visited_states():
    rows = _rows([1], {"freeze_all": False}, bins=("bounded_mid",), phases=("peak",))
    cells = _cells(rows)
    coverage = BV.coverage_report(
        cells, dict(seeds=[1, 3, 4], bins=["bounded_mid"], phases=["trough", "peak"],
                    arms=["freeze_all"]))
    out = BV.adjudicate(
        state_inventory_ok=True, exact_resume_ok=True, eligible_seeds=[1, 3, 4],
        cells=cells, per_arm=BV.carrier_window(cells), coverage=coverage)
    assert out["verdict"] == "no_evidence"
    assert out["layers"]["source_space_carrier"] == "partial_no_carrier_evidence"
    assert "incomplete" in out["reason"]


def test_mean_input_only_replicates_are_excluded_from_the_posterior():
    rows = _rows([1, 3, 4], {"freeze_all": False})
    for r in list(rows):
        if r["bin_name"] == "bounded_mid":
            rows.append(dict(r, replicate="mean_input_only", survived=True, lifetime_ms=8000.0))
    cells = _cells(rows)
    for k, v in cells.items():
        assert v["n_replicates"] == 3, "the diagnostic replicate must not enter the paired posterior"
