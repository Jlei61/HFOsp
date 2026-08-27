import json

from scripts.paper_figures.plot_topic4_rev12_node_local_bridge_audit import (
    ENDPOINTS,
    build_audit,
)


def _metadata(k2, matrix):
    return {"natural_kmeans": {
        "heldout_gmm_k2_minus_k1_loglik_per_event": k2,
        "similarity_matrix": matrix,
        "direction_balanced_alignment": 0.8,
        "n_events": 20,
    }}


def test_bridge_audit_requires_k2_and_both_positive_diagonals(tmp_path):
    candidates = {
        "k2_wrong_mode": _metadata(2.0, [[0.8, -0.2], [-0.3, -0.1]]),
        "right_modes_k1": _metadata(-1.0, [[0.8, -0.2], [-0.3, 0.4]]),
    }
    for candidate_id, metadata in candidates.items():
        path = tmp_path / "figures" / candidate_id
        path.mkdir(parents=True)
        (path / "metadata.json").write_text(json.dumps(metadata))
    utilities = {
        endpoint: {"mean_utility": 0.1} for endpoint in ENDPOINTS
    }
    result = {"candidates": [
        {"candidate_id": candidate_id,
         "relative_to_historical_anchor": utilities}
        for candidate_id in candidates
    ]}
    audit = build_audit(
        tmp_path, result, _metadata(3.0, [[0.6, -0.4], [-0.5, 0.5]]),
        _metadata(1.0, [[0.7, -0.4], [-0.3, 0.2]]),
    )
    assert audit["hidden_pass_candidate_ids"] == []
    assert audit["status"] == (
        "NO_LOCAL_BRIDGE_CANDIDATE_JOINTLY_SUPPORTS_K2_AND_BOTH_PATIENT_MODES"
    )


def test_bridge_audit_is_invariant_to_kmeans_row_labels(tmp_path):
    candidate_id = "swapped_labels"
    path = tmp_path / "figures" / candidate_id
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps(
        _metadata(2.0, [[-0.6, 0.7], [0.8, -0.5]])
    ))
    utilities = {
        endpoint: {"mean_utility": 0.1} for endpoint in ENDPOINTS
    }
    audit = build_audit(
        tmp_path,
        {"candidates": [{
            "candidate_id": candidate_id,
            "relative_to_historical_anchor": utilities,
        }]},
        _metadata(3.0, [[0.6, -0.4], [-0.5, 0.5]]),
        _metadata(1.0, [[0.7, -0.4], [-0.3, 0.2]]),
    )
    row = audit["rows"][0]
    assert row["cluster_row_permutation"] == [1, 0]
    assert row["patient_sign_structure"] is True
    assert row["weakest_patient_diagonal"] == 0.7
    assert audit["hidden_pass_candidate_ids"] == [candidate_id]
