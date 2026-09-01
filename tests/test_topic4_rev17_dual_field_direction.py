from __future__ import annotations

import numpy as np

from src.topic4_rev17_dual_field_direction import (
    construct_directions,
    maximin_direction,
    response_tensor,
)


def _finite_rows() -> list[dict]:
    rows = []
    for channel in ("mean", "dispersion"):
        for mode in range(15):
            for seed in (2361, 2362, 2363):
                scale = 1.0 + 0.1 * (seed - 2362)
                row = {"channel": channel, "mode_index": mode, "seed": seed}
                for prefix, derivative in (
                    ("mode_0_mean", scale * (mode + 1)),
                    ("mode_1_mean", 0.0),
                    ("j14", scale * (mode + 1)),
                    ("mode_0_effective_events", 1.0),
                    ("mode_1_effective_events", 1.0),
                ):
                    row[f"{prefix}_derivative"] = derivative
                    row[f"{prefix}_curvature"] = 0.0
                rows.append(row)
    return rows


def test_response_tensor_keeps_dual_channels_and_equal_network_rows():
    tensor = response_tensor(_finite_rows(), amplitude=0.15)
    assert tensor["network_seeds"] == [2361, 2362, 2363]
    assert np.asarray(tensor["gradients"]["A"]).shape == (3, 30)
    assert len(tensor["linear_eligible_coordinates"]) == 30
    assert tensor["coordinates"][0]["channel"] == "mean"
    assert tensor["coordinates"][15]["channel"] == "dispersion"


def test_response_tensor_rejects_coordinate_with_dominant_curvature():
    rows = _finite_rows()
    target = next(
        row for row in rows
        if row["channel"] == "mean" and row["mode_index"] == 0
        and row["seed"] == 2361
    )
    target["mode_0_mean_curvature"] = 1000.0
    tensor = response_tensor(rows, amplitude=0.15)
    assert 0 not in tensor["linear_eligible_coordinates"]


def test_generic_maximin_improves_primary_and_protects_loss_and_support():
    primary = np.zeros((3, 30), dtype=float)
    protected = np.zeros((3, 30), dtype=float)
    support = np.zeros((3, 30), dtype=float)
    primary[:, 0] = [1.0, 1.1, 0.9]
    protected[:, 1] = [1.0, 0.8, 1.2]
    support[:, 0] = [-0.2, -0.2, -0.2]
    result = maximin_direction(
        primary, protected_losses=(protected,), protected_supports=(support,),
    )
    assert result["feasible_positive_margin"] is True
    direction = np.asarray(result["direction"])
    assert np.all(primary @ direction < 0.0)
    assert np.all(protected @ direction <= 1e-7)
    assert np.all(support @ direction >= -1e-7)


def test_direction_construction_never_reactivates_nonlinear_coordinate():
    tensor = response_tensor(_finite_rows(), amplitude=0.15)
    tensor["linear_eligible_coordinates"] = list(range(1, 30))
    directions = construct_directions(tensor)
    for record in directions.values():
        if not isinstance(record, dict):
            continue
        if record.get("direction") is not None:
            assert np.asarray(record["direction"])[0] == 0.0


def test_no_linear_coordinate_is_a_completed_response_not_an_exception():
    tensor = response_tensor(_finite_rows(), amplitude=0.15)
    tensor["linear_eligible_coordinates"] = []
    directions = construct_directions(tensor)
    assert directions["analysis_status"] == "NO_LOCALLY_LINEAR_COORDINATE"
    assert directions["linear_eligible_coordinate_count"] == 0
    assert not any(
        isinstance(record, dict) and record.get("direction") is not None
        for record in directions.values()
    )
