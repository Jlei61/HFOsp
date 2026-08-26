from scripts.analyze_topic4_rev12_signed_depth_mapping_audit import mapping_audit


def _record(seed, objective, mode_0, mode_1, direction_0, direction_1,
            monotonicity_0=0.2, monotonicity_1=0.2):
    return {
        "seed": seed,
        "soft_objective": {
            "objective": objective,
            "modes": {"0": {"mean": mode_0}, "1": {"mean": mode_1}},
        },
        "soft_causal_direction": {"modes": {
            "0": {"alignment_score": direction_0},
            "1": {"alignment_score": direction_1},
        }},
        "soft_causal_monotonicity": {"modes": {
            "0": {"alignment_score": monotonicity_0},
            "1": {"alignment_score": monotonicity_1},
        }},
    }


def test_mapping_audit_requires_absolute_balanced_improvement():
    anchor = [_record(s, 2, 2, 2, 0.2, 0.2) for s in range(6)]
    source = [_record(s, 1.8, 1.8, 1.8, 0.3, 0.1) for s in range(6)]
    resolved = [_record(s, 1.7, 1.7, 1.7, 0.4, 0.4) for s in range(6)]
    unresolved = [_record(s, 1.6, 1.6, 1.6, 0.5, 0.1) for s in range(6)]
    stage_ag = {"rows": [
        {"candidate_id": "stage_ag_anchor", "per_network": anchor},
        {"candidate_id": "stage_ag_g10_p", "per_network": source},
    ]}
    stage_ah = {"rows": [
        {"candidate_id": "resolved", "per_network": resolved},
        {"candidate_id": "unresolved", "per_network": unresolved},
    ]}
    manifest = {"candidates": [
        {"candidate_id": "resolved", "source_candidate_ids": ["stage_ag_g10_p"],
         "node_field": {"field_sha256": "f"},
         "node_mapping": {"signed_depth_shrinkage": 0.0, "mapping_sha256": "a"}},
        {"candidate_id": "unresolved", "source_candidate_ids": ["stage_ag_g10_p"],
         "node_field": {"field_sha256": "f"},
         "node_mapping": {"signed_depth_shrinkage": 0.5, "mapping_sha256": "b"}},
    ]}
    result = mapping_audit(
        stage_ah, stage_ag, manifest,
        primary_endpoints=[
            "soft_objective", "mode_0", "mode_1",
            "mode_0_direction", "mode_1_direction",
        ],
        diagnostic_endpoints=["mode_0_monotonicity", "mode_1_monotonicity"],
        minimum_positive_networks=4,
        bootstrap_draws=1000, bootstrap_confidence=0.90, bootstrap_seed=3,
    )
    assert result["status"] == "SIGNED_DEPTH_SHRINKAGE_BALANCED_NODE_MAPPING_CANDIDATE_FOUND"
    assert result["balanced_candidate_ids"] == ["resolved"]
    by_id = {row["candidate_id"]: row for row in result["candidates"]}
    assert by_id["unresolved"]["balanced_absolute_primary_improvement"] is False
