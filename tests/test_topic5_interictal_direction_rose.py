import numpy as np

from src.topic5_interictal_direction_rose import (
    assess_event_direction_qc,
    axis_pair_display_basis,
    fit_endpoint_direction_3d,
    fit_event_endpoint_directions_3d,
    fit_event_directions_3d,
    project_directions_to_angles,
)


def test_event_rank_gradient_points_early_to_late():
    rng = np.random.default_rng(4)
    coords = rng.normal(size=(20, 3))
    true_direction = np.array([0.8, -0.4, 0.25])
    true_direction /= np.linalg.norm(true_direction)
    rank = coords @ true_direction
    events = np.column_stack([rank, rank + 0.02 * rng.normal(size=rank.size)])
    events[:3, 1] = np.nan
    result = fit_event_directions_3d(events, coords)
    assert float(result["directions"][0] @ true_direction) > 0.999
    assert float(result["directions"][1] @ true_direction) > 0.99


def test_display_basis_sets_frozen_ta_to_zero_and_preserves_tb_angle():
    axis_a = np.array([1.0, 0.0, 0.0])
    axis_b = np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0])
    basis = axis_pair_display_basis(axis_a, axis_b)
    projected = project_directions_to_angles(
        np.vstack([axis_a, axis_b]), basis["axis_a"], basis["transverse"]
    )
    np.testing.assert_allclose(np.degrees(projected["angles"]), [0.0, 120.0], atol=1e-8)
    assert np.isclose(basis["theta_b_deg"], 120.0)


def test_collinear_pair_uses_frozen_transverse_fallback():
    basis = axis_pair_display_basis(
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], fallback_transverse=[0.0, 1.0, 0.0]
    )
    assert basis["basis_source"] == "frozen_ta_transverse_fallback"
    assert np.isclose(basis["theta_b_deg"], 180.0)


def test_event_direction_qc_passes_stable_multishaft_gradient():
    rng = np.random.default_rng(22)
    coords = rng.normal(size=(12, 3))
    shafts = np.repeat(["A", "B", "C"], 4)
    true_direction = np.array([0.7, -0.2, 0.5])
    rank = coords @ true_direction
    result = assess_event_direction_qc(rank[:, None], coords, shafts)
    assert bool(result["passes"][0])
    assert result["n_valid_contacts"][0] == 12
    assert result["n_shafts"][0] == 3
    assert result["loco_valid_fraction"][0] == 1.0
    assert result["loco_median_signed_cosine"][0] > 0.999


def test_event_direction_qc_rejects_too_few_contacts():
    rng = np.random.default_rng(23)
    coords = rng.normal(size=(8, 3))
    rank = coords @ np.array([0.3, 0.6, -0.2])
    rank[5:] = np.nan
    result = assess_event_direction_qc(rank[:, None], coords, ["A"] * 4 + ["B"] * 4)
    assert not bool(result["passes"][0])
    assert result["n_valid_contacts"][0] == 5
    assert result["loco_n_attempted"][0] == 0


def test_event_direction_qc_rejects_single_shaft_geometry():
    rng = np.random.default_rng(24)
    coords = rng.normal(size=(10, 3))
    rank = coords @ np.array([0.2, -0.8, 0.4])
    result = assess_event_direction_qc(rank[:, None], coords, ["A"] * 10)
    assert not bool(result["passes"][0])
    assert result["n_shafts"][0] == 1
    assert result["loco_n_attempted"][0] == 0


def test_endpoint_direction_uses_top_bottom_three_for_seven_contacts():
    coords = np.column_stack([np.arange(7.0), np.zeros(7), np.zeros(7)])
    result = fit_endpoint_direction_3d(np.arange(7.0), coords)
    np.testing.assert_allclose(result["direction"], [1.0, 0.0, 0.0])
    assert result["tier"] == "primary"
    assert result["k_used"] == 3
    assert result["source_idx"] == [0, 1, 2]
    assert result["sink_idx"] == [4, 5, 6]


def test_endpoint_direction_uses_k2_fallback_for_six_contacts():
    coords = np.column_stack([np.arange(6.0), np.zeros(6), np.zeros(6)])
    result = fit_endpoint_direction_3d(np.arange(5.0, -1.0, -1.0), coords)
    np.testing.assert_allclose(result["direction"], [-1.0, 0.0, 0.0])
    assert result["tier"] == "fallback"
    assert result["k_used"] == 2


def test_endpoint_event_direction_is_missing_below_old_contact_floor():
    coords = np.column_stack([np.arange(6.0), np.zeros(6), np.zeros(6)])
    ranks = np.arange(6.0)[:, None]
    ranks[:2] = np.nan
    result = fit_event_endpoint_directions_3d(ranks, coords)
    assert not np.isfinite(result["directions"][0]).any()
    assert result["n_valid_contacts"][0] == 4
    assert result["k_used"][0] == 0
