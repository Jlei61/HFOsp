import pytest

from scripts.aggregate_topic4_rev21_zm_screen import (
    _neighbor_scores, reference_support, summarize_candidate,
)


def _cell(topology, dynamics, d, alignment, ood, eligible):
    return {
        "topology_seed": topology, "dynamics_seed": dynamics,
        "operational_onset_ms": 3000.0 if eligible else None,
        "model_ictal": {"status": (
            "MODEL_ICTAL_ELIGIBLE_REV21" if eligible
            else "MODEL_ICTAL_NOT_ELIGIBLE_REV21")},
        "selection": {"complete_distribution_distance_training": d,
                      "n_returned_families": 5},
        "validation": {"direction_balanced_alignment": alignment,
                       "ood_all_returned": ood},
    }


def test_reference_support_is_computed_from_all_orthogonal_cells():
    matrices = {
        "training_complete_distribution": [[0.2, 0.3], [0.4, 0.5]],
        "two_template_alignment": [[0.6, 0.7], [0.8, 0.9]],
        "ood_all_returned": [[0.1, 0.2], [0.3, 0.4]],
    }
    support = reference_support(matrices)
    assert support["complete_distribution"]["n"] == 4
    assert support["two_template_alignment"]["q05"] < 0.7


def test_candidate_retention_uses_cell_metrics_and_pooled_cluster_presence():
    support = {
        "complete_distribution": {"q05": 0.2, "q50": 0.3, "q95": 0.5},
        "two_template_alignment": {"q05": 0.5, "q50": 0.7, "q95": 0.9},
        "ood": {"q05": 0.1, "q50": 0.2, "q95": 0.4},
    }
    off = _cell(1, 2, 0.3, 0.7, 0.2, False)
    active = _cell(1, 2, 0.4, 0.6, 0.3, True)
    pooled = {"selection": {}, "validation": {"two_clusters_present": True}}
    row = summarize_candidate(
        "active", [active], {(1, 2): off}, support, pooled,
        {"I_th_EI_scale": 0.9, "integrated_M_scale": 1.0},
    )
    assert row["interictal_substrate_retained"] is True
    assert row["model_ictal_eligible_fraction"] == 1.0
    assert row["paired_candidate_minus_off"]["ood"][
        "median_candidate_minus_off"] == pytest.approx(0.1)


def test_pooled_two_cluster_collapse_cannot_be_hidden_by_good_marginals():
    support = {
        "complete_distribution": {"q05": 0.2, "q50": 0.3, "q95": 0.5},
        "two_template_alignment": {"q05": 0.5, "q50": 0.7, "q95": 0.9},
        "ood": {"q05": 0.1, "q50": 0.2, "q95": 0.4},
    }
    active = _cell(1, 2, 0.3, 0.7, 0.2, True)
    row = summarize_candidate(
        "active", [active], {}, support,
        {"selection": {}, "validation": {"two_clusters_present": False}},
        {"I_th_EI_scale": 1.0, "integrated_M_scale": 1.0},
    )
    assert row["interictal_substrate_retained"] is False


def test_neighbor_stability_uses_only_adjacent_grid_levels():
    rows = []
    for i, s_i in enumerate((0.7, 0.8, 0.9)):
        for j, s_m in enumerate((0.5, 1.0, 2.0)):
            rows.append({
                "level": {"I_th_EI_scale": s_i,
                          "integrated_M_scale": s_m},
                "model_ictal_eligible_fraction": float(i + j),
            })
    _neighbor_scores(rows)
    center = next(row for row in rows if row["level"] == {
        "I_th_EI_scale": 0.8, "integrated_M_scale": 1.0,
    })
    assert center["neighbor_eligible_fraction"] == pytest.approx(2.0)
