from scripts.analyze_topic4_rev12_local_curvature_results import (
    analyze_direct_results,
)
from tests.test_topic4_rev12_orthogonal_anchor_relative import _row


def test_direct_result_requires_all_aggregate_and_network_endpoints():
    rows = {
        "stage_w_anchor": _row(
            objective=1.0, mode0=1.0, mode1=1.0, kmeans=0.5, direction=0.4,
        ),
        "stage_w_good": _row(
            objective=0.9, mode0=0.9, mode1=0.9, kmeans=0.6, direction=0.5,
        ),
        "stage_w_mode_fail": _row(
            objective=0.9, mode0=0.9, mode1=1.1, kmeans=0.6, direction=0.5,
        ),
    }
    result = analyze_direct_results(
        rows, minimum_same_sign=2,
        required_aggregate=[
            "objective_utility", "patient_mode_0_utility",
            "patient_mode_1_utility", "kmeans_utility", "direction_utility",
        ],
        required_network=[
            "objective_utility", "patient_mode_0_utility",
            "patient_mode_1_utility",
        ],
    )
    assert result["fit_advancement_candidates"] == ["stage_w_good"]
    assert result["decision"] == "OPEN_FRESH_SELECTION_REVIEW"
    failed = next(row for row in result["candidate_results"]
                  if row["candidate_id"] == "stage_w_mode_fail")
    assert "patient_mode_1_utility" in failed["failed_aggregate_endpoints"]
