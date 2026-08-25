import numpy as np

from scripts.freeze_topic4_rev12_local_curvature_replication import build_candidates


def _field(value):
    return {"field_sha256": str(value), "coefficients": [[float(value)]]}


def test_replication_freezes_only_anchor_and_declared_near_miss():
    manifest = {"candidates": [
        {"candidate_id": "stage_w_anchor", "node_field": _field(0)},
        {"candidate_id": "stage_w_f14p03", "node_field": _field(1)},
        {"candidate_id": "stage_w_other", "node_field": _field(2)},
    ]}
    result = {
        "status": "REV12ND_LOCAL_CURVATURE_DIRECT_ANALYSIS_COMPLETE",
        "decision": "STOP_NO_LOCAL_CANARY_JOINTLY_IMPROVED",
        "candidate_results": [{
            "candidate_id": "stage_w_f14p03",
            "aggregate_delta_from_anchor": {
                "objective_utility": 0.01,
                "patient_mode_0_utility": 0.02,
                "patient_mode_1_utility": 0.01,
                "direction_utility": 0.03,
                "kmeans_utility": -0.01,
            },
            "fit_advancement_eligible": False,
            "failed_aggregate_endpoints": ["kmeans_utility"],
            "failed_network_support_endpoints": ["objective_utility"],
        }],
    }
    candidates, audit = build_candidates(manifest, result, {
        "candidate_id": "stage_w_f14p03",
        "near_miss_positive_endpoints": [
            "objective_utility", "patient_mode_0_utility",
            "patient_mode_1_utility", "direction_utility",
        ],
    })
    assert [row["candidate_id"] for row in candidates] == [
        "stage_x_anchor", "stage_x_f14p03",
    ]
    assert all(not row["selection_eligible"] for row in candidates)
    assert audit["patient_heldout_used_for_replication_choice"] is False
    assert not np.array_equal(
        candidates[0]["node_field"]["coefficients"],
        candidates[1]["node_field"]["coefficients"],
    )
