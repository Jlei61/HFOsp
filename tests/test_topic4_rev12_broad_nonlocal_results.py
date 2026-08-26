import copy

from scripts.analyze_topic4_rev12_broad_nonlocal_results import paired_candidate_audit


def _record(seed, objective, mode_0, mode_1, direction_0, direction_1,
            monotonicity_0, monotonicity_1):
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


def _summary():
    anchor = [_record(seed, 2, 2, 2, 0.2, 0.2, 0.2, 0.2) for seed in range(6)]
    balanced = [_record(seed, 1, 1, 1, 0.4, 0.4, 0.4, 0.4) for seed in range(6)]
    tradeoff = [_record(seed, 1, 1, 1, 0.4, 0.1, 0.4, 0.1) for seed in range(6)]
    return {"rows": [
        {"candidate_id": "anchor", "role": "anchor", "per_network": anchor},
        {"candidate_id": "balanced", "role": "candidate", "per_network": balanced},
        {"candidate_id": "tradeoff", "role": "candidate", "per_network": tradeoff},
    ]}


def _audit(summary):
    return paired_candidate_audit(
        summary,
        anchor_candidate_id="anchor",
        primary_endpoints=[
            "soft_objective", "mode_0", "mode_1",
            "mode_0_direction", "mode_1_direction",
        ],
        diagnostic_endpoints=["mode_0_monotonicity", "mode_1_monotonicity"],
        bootstrap_draws=1000,
        bootstrap_confidence=0.90,
        bootstrap_seed=7,
    )


def test_paired_audit_finds_only_jointly_balanced_candidate():
    result = _audit(_summary())
    assert result["status"] == "BROAD_FIELD_BALANCED_FIT_DIRECTION_CANDIDATE_FOUND"
    assert result["balanced_candidate_ids"] == ["balanced"]
    rows = {row["candidate_id"]: row for row in result["candidates"]}
    assert rows["balanced"]["joint_all_endpoint_positive_networks"] == 6
    assert rows["tradeoff"]["joint_primary_positive_networks"] == 0


def test_paired_audit_rejects_candidate_that_sacrifices_one_mode_direction():
    summary = copy.deepcopy(_summary())
    summary["rows"] = [row for row in summary["rows"] if row["candidate_id"] != "balanced"]
    result = _audit(summary)
    assert result["status"].endswith("NO_BALANCED_NODE_CANDIDATE")
    assert result["balanced_candidate_ids"] == []
    row = result["candidates"][0]
    assert row["endpoints"]["mode_1_direction"]["positive_networks"] == 0
