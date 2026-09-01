import math

import numpy as np
import pytest

from src.topic4_component_pair_search import DESCRIPTOR_NAMES
from src.topic4_rev9l_capacity_audit import (
    audit_finite_library_capacity,
    descriptor_event_count_consistency,
    descriptor_support_audit,
    library_descriptor_extremes,
    patient_equivalent_objective,
    per_network_mode_descriptor_ratios,
    recruitment_reachability,
)


OBJECTIVE = {
    "readable_fraction_penalty_weight": 2.0,
    "weakest_mode_lse_tau": 0.25,
    "ood_weight": 0.1,
}


def _floor(count=3, *, median=1.0, q95=2.0, scale=0.5):
    modes = {
        mode: {
            name: {"median": median, "q95": q95, "scale_iqr": scale}
            for name in DESCRIPTOR_NAMES
        }
        for mode in ("A", "B")
    }
    return {"n_events_per_mode_per_draw": count, "floor": {"modes": modes}}


def _mode_block(recruitment, patient_recruitment, *, raws, profile, patient_profile,
                pairs, patient_pairs, n_events=3):
    block = {
        "n_model_events": n_events,
        "model_recruitment_probability": list(recruitment),
        "patient_recruitment_probability": list(patient_recruitment),
        "model_mean_normalized_rank": list(profile),
        "patient_mean_normalized_rank": list(patient_profile),
        "model_precedence_pairs_with_support": pairs,
        "patient_precedence_pairs_with_support": patient_pairs,
    }
    block.update(raws)
    return block


def _row(candidate_id, seed, *, recruitment_a, raw_a, n_events_a=3,
         floor_count_a=3):
    patient = [0.8, 0.6, 0.9]
    raws_a = {name: 0.4 for name in DESCRIPTOR_NAMES}
    raws_a["recruitment_mean_absolute_error"] = raw_a
    raws_b = {name: 0.3 for name in DESCRIPTOR_NAMES}
    return {
        "candidate_id": candidate_id,
        "network_seed": seed,
        "score": {
            "objective": raw_a,
            "matched_floor_event_count_by_mode": {"A": floor_count_a, "B": 3},
        },
        "mode_descriptors": {"modes": {
            "A": _mode_block(
                recruitment_a, patient, raws=raws_a,
                profile=[0.1, np.nan, 0.5], patient_profile=[0.1, 0.2, 0.5],
                pairs=2, patient_pairs=6, n_events=n_events_a),
            "B": _mode_block(
                [1.0, 1.0, 1.0], patient, raws=raws_b,
                profile=[0.1, 0.2, 0.5], patient_profile=[0.1, 0.2, 0.5],
                pairs=6, patient_pairs=6),
        }},
    }


def _payload(rows):
    return {
        "network_seeds": sorted({row["network_seed"] for row in rows}),
        "candidate_network_rows": rows,
        "oracle": {"per_network": [
            {"network_seed": seed, "minimum_objective": 0.4,
             "representative_candidate_id": "sobol_000"}
            for seed in sorted({row["network_seed"] for row in rows})
        ]},
    }


def test_patient_equivalent_objective_matches_softplus_of_zero_at_the_floor():
    reference = patient_equivalent_objective(_floor()["floor"], **{
        "readable_weight": OBJECTIVE["readable_fraction_penalty_weight"],
        "tau": OBJECTIVE["weakest_mode_lse_tau"],
        "ood_weight": OBJECTIVE["ood_weight"]})
    assert np.isclose(reference["floor_median_objective"], math.log(2.0))
    assert reference["floor_q95_objective"] > reference["floor_median_objective"]


def test_extremes_separate_never_improved_from_inert_descriptor():
    rows = [
        _row("sobol_000", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35),
        _row("sobol_000", 2, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35),
        _row("sobol_007", 1, recruitment_a=[1.0, 0.0, 0.0], raw_a=0.50),
        _row("sobol_007", 2, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35),
    ]
    summary = library_descriptor_extremes(_payload(rows))
    record = summary["A"]["recruitment_mean_absolute_error"]
    assert record["baseline_equals_library_minimum_in_every_network"] is True
    assert record["n_rows_worse_than_baseline_minimum"] == 1
    # A descriptor that takes more than one value is not "fixed"; claiming so
    # would misreport a capacity failure as an inert instrument.
    assert record["n_distinct_values"] == 2
    assert record["minimum"] < record["maximum"]


def test_extremes_detect_a_descriptor_the_library_actually_improves():
    rows = [
        _row("sobol_000", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35),
        _row("sobol_009", 1, recruitment_a=[1.0, 1.0, 1.0], raw_a=0.20),
    ]
    record = library_descriptor_extremes(
        _payload(rows))["A"]["recruitment_mean_absolute_error"]
    assert record["baseline_equals_library_minimum_in_every_network"] is False


def test_reachability_attributes_error_to_contacts_no_parameter_reaches():
    rows = [
        _row("sobol_000", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35),
        _row("sobol_007", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35),
    ]
    result = recruitment_reachability(_payload(rows))
    mode_a = result["modes"]["A"]
    assert mode_a["never_recruited_contact_indices"] == [1]
    assert np.isclose(
        mode_a["share_of_best_error_from_never_recruited"],
        0.6 / (0.2 + 0.6 + 0.1))
    assert mode_a["best_recruitment_error_if_never_recruited_were_matched"] < (
        mode_a["best_recruitment_mean_absolute_error"])
    assert result["never_recruited_in_both_modes"] == []


def test_reachability_rejects_a_drifting_patient_reference():
    rows = [_row("sobol_000", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35)]
    rows.append(_row("sobol_007", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35))
    rows[1]["mode_descriptors"]["modes"]["A"][
        "patient_recruitment_probability"] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        recruitment_reachability(_payload(rows))


def test_support_audit_reports_the_smaller_model_support():
    rows = [_row("sobol_000", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35)]
    summary = descriptor_support_audit(_payload(rows))
    assert summary["A"]["model_profile_contacts"] == [2, 2]
    assert summary["A"]["patient_profile_contacts"] == [3, 3]
    assert summary["A"]["model_support_is_smaller"] is True
    assert summary["B"]["model_support_is_smaller"] is False


def test_event_count_consistency_catches_a_floor_count_mismatch():
    rows = [_row("sobol_000", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35)]
    assert descriptor_event_count_consistency(_payload(rows))["consistent"]
    rows.append(_row("sobol_007", 1, recruitment_a=[1.0, 0.0, 1.0],
                     raw_a=0.35, n_events_a=2, floor_count_a=3))
    broken = descriptor_event_count_consistency(_payload(rows))
    assert broken["consistent"] is False
    assert broken["mismatches"][0]["mode"] == "A"


def test_per_network_ratios_report_both_modes_not_only_the_weak_one():
    rows = [_row("sobol_000", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=3.0)]
    ratios = per_network_mode_descriptor_ratios(_payload(rows), {3: _floor()})
    assert set(ratios[0]["modes"]) == {"A", "B"}
    assert ratios[0]["modes"]["A"]["recruitment_above_q95"] is True
    assert ratios[0]["modes"]["B"]["recruitment_above_q95"] is False
    assert ratios[0]["modes"]["B"]["n_shape_descriptors_above_q95"] == 0


def test_audit_bundles_the_facts_the_negative_conclusion_depends_on():
    rows = [
        _row("sobol_000", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.35),
        _row("sobol_007", 1, recruitment_a=[1.0, 0.0, 1.0], raw_a=0.50),
    ]
    audit = audit_finite_library_capacity(
        _payload(rows), {3: _floor()}, objective=OBJECTIVE)
    assert audit["delta_network_is_non_negative_by_construction"] is True
    assert audit["delta_network_noise_null_tested"] is False
    assert audit["descriptor_event_count_consistency"]["consistent"] is True
    assert "n3" in audit["patient_equivalent_objective"]
    assert audit["median_gap_above_patient_q95_objective"] == pytest.approx(
        audit["library_objective_median"]
        - audit["patient_equivalent_objective"]["n3"]["floor_q95_objective"])
