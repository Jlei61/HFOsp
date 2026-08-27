from __future__ import annotations

import numpy as np

from scripts import aggregate_topic4_rev15_m3_coordinate_atlas as aggregate


def _row(candidate_id, coordinate, sign, a, b, j14, selectable=True):
    return {
        "candidate_id": candidate_id,
        "selection_eligible": selectable,
        "coordinate_index": coordinate,
        "sign": sign,
        "mode_nx": coordinate,
        "mode_ny": 0,
        "phase": "cos",
        "mode_0_mean": a,
        "mode_1_mean": b,
        "j14": j14,
    }


def test_ranking_requires_A_improvement_before_B_tolerance_and_j14():
    rows = [
        _row("exact_off", None, None, 1.0, 1.0, 2.0, False),
        _row("uniform_node", None, None, 1.2, 1.2, 1.0, False),
    ]
    for coordinate in range(28):
        if coordinate == 0:
            rows.extend([
                _row("m3_c00_m_r08", 0, -1, 0.8, 1.05, 1.8),
                _row("m3_c00_p_r08", 0, 1, 0.7, 1.20, 1.4),
            ])
        elif coordinate == 1:
            rows.extend([
                _row("m3_c01_m_r08", 1, -1, 1.1, 0.8, 1.0),
                _row("m3_c01_p_r08", 1, 1, 0.9, 1.00, 1.9),
            ])
        else:
            rows.extend([
                _row(f"m3_c{coordinate:02d}_m_r08", coordinate, -1, 1.2, 1.0, 1.0),
                _row(f"m3_c{coordinate:02d}_p_r08", coordinate, 1, 1.3, 1.0, 1.0),
            ])
    ranked, pairs = aggregate.rank_coordinate_responses(rows)
    assert ranked[0]["candidate_id"] == "m3_c00_m_r08"
    assert ranked[1]["candidate_id"] == "m3_c01_p_r08"
    assert ranked[0]["mode_A_improves"] is True
    assert ranked[0]["mode_B_within_10pct"] is True
    assert len(pairs) == 28
    pair0 = pairs[0]
    assert np.isclose(pair0["signed_A_response_half_difference"], -0.05)
    assert np.isclose(pair0["signed_B_response_half_difference"], 0.075)


def test_analysis_descendant_allowlist_cannot_change_worker_or_objective():
    assert set(aggregate.ANALYSIS_ONLY_ALLOWED_PATHS) == {
        "scripts/aggregate_topic4_rev15_m3_coordinate_atlas.py",
        "tests/test_topic4_rev15_m3_coordinate_atlas_aggregate.py",
    }
    assert "scripts/run_topic4_rev15_m3_coordinate_atlas_worker.py" not in (
        aggregate.ANALYSIS_ONLY_ALLOWED_PATHS
    )
    assert "src/topic4_rev14_static_node_objective.py" not in (
        aggregate.ANALYSIS_ONLY_ALLOWED_PATHS
    )
