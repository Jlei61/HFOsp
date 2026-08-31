from __future__ import annotations

from scripts.aggregate_topic4_rev17_dual_field_selection import evaluate_candidates


def _row(candidate: str, seed: int, *, j14: float, a: float, b: float,
         support_a: float = 8.0, support_b: float = 8.0) -> dict:
    return {
        "candidate_id": candidate, "seed": seed, "j14": j14,
        "mode_0_mean": a, "mode_1_mean": b,
        "mode_0_effective_events": support_a,
        "mode_1_effective_events": support_b,
        "run_status": "VALID",
    }


def _selection() -> dict:
    return {
        "B_protection_ratio": 1.1,
        "minimum_effective_support_per_mode_per_network": 6.0,
    }


def test_fresh_selection_requires_every_network_not_pooled_average():
    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("exact_dual_anchor", seed, j14=3.0, a=2.0, b=1.0))
        rows.append(_row(
            "candidate", seed,
            j14=2.5 if seed < 3 else 3.1,
            a=1.5 if seed < 3 else 2.1, b=1.0,
        ))
    candidates = {
        "exact_dual_anchor": {"selection_eligible": False},
        "candidate": {
            "selection_eligible": True,
            "residual_coordinates": {"family": "mean_a", "joint_dual_field_radius": 0.1},
        },
    }
    evaluations, eligible = evaluate_candidates(
        rows, candidates, [1, 2, 3], _selection(),
    )
    assert len(evaluations) == 1
    assert evaluations[0]["J14_improves_all_networks"] is False
    assert evaluations[0]["A_improves_all_networks"] is False
    assert eligible == []


def test_fresh_selection_accepts_balanced_candidate_and_ranks_worst_network():
    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("exact_dual_anchor", seed, j14=3.0, a=2.0, b=1.0))
        rows.append(_row("slow", seed, j14=2.8, a=1.8, b=1.05))
        rows.append(_row("strong", seed, j14=2.5, a=1.6, b=1.08))
    candidates = {
        "exact_dual_anchor": {"selection_eligible": False},
        "slow": {
            "selection_eligible": True,
            "residual_coordinates": {"family": "mean_a", "joint_dual_field_radius": 0.1},
        },
        "strong": {
            "selection_eligible": True,
            "residual_coordinates": {"family": "maximin", "joint_dual_field_radius": 0.2},
        },
    }
    _, eligible = evaluate_candidates(rows, candidates, [1, 2, 3], _selection())
    assert [row["candidate_id"] for row in eligible] == ["strong", "slow"]
    assert eligible[0]["fresh_selection_rank"] == 1
