from __future__ import annotations

from scripts import aggregate_topic4_rev16_joint_candidates as aggregate
from scripts import wait_topic4_rev16_m4_shell_then_prepare_joint as waiter


def test_fresh_summary_requires_same_candidate_to_pass_all_clauses():
    manifest = {
        "selection": {
            "fresh_J14_improvement_required_networks": 3,
            "fresh_A_improvement_required_networks": 3,
            "fresh_B_protection_required_networks": 3,
            "B_protection_ratio": 1.10,
            "equal_network_effective_support_minimum_per_mode": 6.0,
        },
        "candidates": [
            {"candidate_id": "exact_off"},
            {"candidate_id": "joint"},
        ],
    }
    rows = []
    for seed in (2351, 2352, 2353):
        rows.extend([
            {
                "candidate_id": "exact_off", "seed": seed,
                "mode_0_mean": 1.0, "mode_1_mean": 1.0, "j14": 2.0,
                "mode_0_effective_events": 8.0,
                "mode_1_effective_events": 8.0,
            },
            {
                "candidate_id": "joint", "seed": seed,
                "mode_0_mean": 0.8, "mode_1_mean": 1.05, "j14": 1.8,
                "mode_0_effective_events": 7.0,
                "mode_1_effective_events": 7.0,
                "family": "mean_a", "target_rms": 0.6,
                "m3_l2_fraction": 0.8, "m4_shell_l2_fraction": 0.6,
            },
        ])
    result = aggregate.summaries(rows, manifest)
    assert len(result) == 1
    assert result[0]["usable_two_mode_anchor"] is True
    assert result[0]["fresh_J14_improvement_count"] == 3
    assert result[0]["fresh_A_improvement_count"] == 3
    assert result[0]["fresh_B_protection_count"] == 3


def test_fresh_summary_rejects_one_network_b_failure():
    manifest = {
        "selection": {
            "fresh_J14_improvement_required_networks": 3,
            "fresh_A_improvement_required_networks": 3,
            "fresh_B_protection_required_networks": 3,
            "B_protection_ratio": 1.10,
            "equal_network_effective_support_minimum_per_mode": 6.0,
        },
        "candidates": [
            {"candidate_id": "exact_off"}, {"candidate_id": "joint"},
        ],
    }
    rows = []
    for seed in (2351, 2352, 2353):
        rows.append({
            "candidate_id": "exact_off", "seed": seed,
            "mode_0_mean": 1.0, "mode_1_mean": 1.0, "j14": 2.0,
            "mode_0_effective_events": 8.0, "mode_1_effective_events": 8.0,
        })
        rows.append({
            "candidate_id": "joint", "seed": seed,
            "mode_0_mean": 0.8,
            "mode_1_mean": 1.2 if seed == 2353 else 1.05,
            "j14": 1.8, "mode_0_effective_events": 7.0,
            "mode_1_effective_events": 7.0, "family": "mean_a",
            "target_rms": 0.6, "m3_l2_fraction": 0.8,
            "m4_shell_l2_fraction": 0.6,
        })
    assert aggregate.summaries(rows, manifest)[0]["usable_two_mode_anchor"] is False


def test_fresh_summary_rejects_j14_worsening_despite_a_b_passing():
    manifest = {
        "selection": {
            "fresh_J14_improvement_required_networks": 3,
            "fresh_A_improvement_required_networks": 3,
            "fresh_B_protection_required_networks": 3,
            "B_protection_ratio": 1.10,
            "equal_network_effective_support_minimum_per_mode": 6.0,
        },
        "candidates": [
            {"candidate_id": "exact_off"}, {"candidate_id": "joint"},
        ],
    }
    rows = []
    for seed in (2351, 2352, 2353):
        rows.extend([
            {
                "candidate_id": "exact_off", "seed": seed,
                "mode_0_mean": 1.0, "mode_1_mean": 1.0, "j14": 2.0,
                "mode_0_effective_events": 8.0,
                "mode_1_effective_events": 8.0,
            },
            {
                "candidate_id": "joint", "seed": seed,
                "mode_0_mean": 0.8, "mode_1_mean": 1.05, "j14": 2.1,
                "mode_0_effective_events": 7.0,
                "mode_1_effective_events": 7.0,
                "family": "mean_a", "target_rms": 0.6,
                "m3_l2_fraction": 0.8, "m4_shell_l2_fraction": 0.6,
            },
        ])
    result = aggregate.summaries(rows, manifest)[0]
    assert result["fresh_A_improvement_count"] == 3
    assert result["fresh_B_protection_count"] == 3
    assert result["fresh_J14_improvement_count"] == 0
    assert result["usable_two_mode_anchor"] is False


def test_waiter_requires_exact_complete_inventory():
    complete = {
        "status": waiter.COMPLETE, "n_jobs": 120, "n_complete": 120,
        "n_failed": 0, "n_invalid_artifact": 0,
    }
    assert waiter.classify(complete) == "complete"
    assert waiter.classify({**complete, "n_complete": 119}) == "failed"
    assert waiter.classify({**complete, "n_invalid_artifact": 1}) == "failed"
    assert waiter.classify({"status": "REV16_M4_SHELL_QUEUE_RUNNING"}) == "wait"
