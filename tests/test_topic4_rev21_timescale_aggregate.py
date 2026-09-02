from scripts.aggregate_topic4_rev21_timescale import (
    annotate_timescale_neighbors, rank_timescale_candidates,
    timescale_log_distance,
)


def _row(candidate, tz, ta, eligible, retained, deterioration, distance=None):
    level = {"tau_z_ms": tz, "tau_adp_ms": ta}
    return {
        "candidate_id": candidate,
        "level": level,
        "model_ictal_eligible_fraction": eligible,
        "interictal_substrate_retained": retained,
        "worst_standardized_deterioration": deterioration,
        "log_distance_from_reference": (
            timescale_log_distance(level, {"tau_z_ms": 5000, "tau_adp_ms": 500})
            if distance is None else distance
        ),
    }


def test_timescale_distance_is_zero_only_at_reference():
    reference = {"tau_z_ms": 5000, "tau_adp_ms": 500}
    assert timescale_log_distance(reference, reference) == 0.0
    assert timescale_log_distance(
        {"tau_z_ms": 3000, "tau_adp_ms": 500}, reference,
    ) > 0.0


def test_timescale_ranking_prioritizes_eligibility_then_retention():
    rows = [
        _row("robust", 5000, 500, 0.75, True, 0.2),
        _row("fragile", 3000, 500, 0.75, False, -1.0),
        _row("weak", 5000, 1000, 0.50, True, -2.0),
    ]
    annotate_timescale_neighbors(rows)
    assert rank_timescale_candidates(rows)[0]["candidate_id"] == "robust"


def test_timescale_neighbors_are_only_one_grid_step_away():
    rows = [
        _row("center", 5000, 500, 1.0, True, 0.0),
        _row("z_neighbor", 3000, 500, 0.5, True, 0.0),
        _row("a_neighbor", 5000, 250, 0.0, True, 0.0),
        _row("diagonal", 3000, 250, 1.0, True, 0.0),
    ]
    annotate_timescale_neighbors(rows)
    center = next(row for row in rows if row["candidate_id"] == "center")
    assert center["neighbor_eligible_fraction"] == 0.25
