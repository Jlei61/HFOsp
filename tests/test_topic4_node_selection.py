import numpy as np

from src.topic4_node_selection import (
    normalized_coordinates,
    pareto_mask,
    select_pareto_knee,
)


def _row(candidate, weak, r2, support, reproducibility, separation, roughness):
    return {
        "candidate_id": candidate,
        "score": {
            "mean_weakest_mode_lse": weak,
            "model_prototype_r2_on_heldout": r2,
            "same_network_both_fraction": support,
            "roughness": roughness,
        },
        "source_topology": {
            "mean_across_network_template_cosine": reproducibility,
            "equal_network_between_mode_distance": separation,
        },
    }


def test_pareto_mask_rejects_strictly_dominated_row():
    values = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.5, -1.0]])
    assert pareto_mask(values).tolist() == [True, False, True]


def test_degenerate_coordinate_has_zero_normalized_weight():
    values = np.asarray([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])
    normalized = normalized_coordinates(values)
    assert np.allclose(normalized[:, 1], 0.0)
    assert np.allclose(normalized[:, 0], [0.0, 0.5, 1.0])


def test_knee_protects_topology_separation_and_complete_distribution():
    rows = [
        _row("weak_only", 0.1, -0.8, 1.0, 0.98, 0.02, 0.1),
        _row("balanced", 0.2, 0.2, 1.0, 0.95, 0.20, 0.2),
        _row("topology_only", 0.8, -0.4, 1.0, 0.95, 0.40, 0.1),
        _row("dominated", 0.9, -0.9, 0.8, 0.90, 0.01, 0.0),
    ]
    result = select_pareto_knee(rows)
    assert result["selected_candidate_id"] == "balanced"
    detail = {row["candidate_id"]: row for row in result["details"]}
    assert not detail["dominated"]["pareto"]


def test_roughness_only_breaks_exact_metric_tie():
    rows = [
        _row("rough", 0.2, 0.1, 1.0, 0.9, 0.2, 0.5),
        _row("smooth", 0.2, 0.1, 1.0, 0.9, 0.2, 0.2),
    ]
    assert select_pareto_knee(rows)["selected_candidate_id"] == "smooth"
