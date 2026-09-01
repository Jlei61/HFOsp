import numpy as np
import pytest
from pathlib import Path
import json

from src.topic4_continuous_field import (
    background_anchors,
    build_continuous_field_candidates,
    build_continuous_support_candidates,
    continuous_corridor_field_h,
    continuous_field_h,
    continuous_surface,
    distance_to_segments,
    fit_contact_target,
    patient_contact_targets,
    shaft_balanced_contact_weights,
    spline_basis_1d,
    tensor_basis,
)
from src.topic4_core_field_rev9 import (
    reconstruct_frozen_node,
    reconstruct_node_from_h,
)
from scripts.run_topic4_rev10_sa_dual_shaft_worker import _candidate_node
from scripts.freeze_topic4_rev10_sa_continuous_field_candidates import build_manifest
from scripts.freeze_topic4_rev10_sa_continuous_support_candidates import (
    build_manifest as build_support_manifest,
)
from scripts.aggregate_topic4_rev10_sa_continuous_field_capacity import _relative_field


ROOT = Path(__file__).resolve().parents[1]


def _contacts():
    return np.asarray([
        [2.0, 5.0], [5.0, 5.0], [8.0, 5.0], [11.0, 5.0],
        [6.0, 15.0], [9.0, 15.0],
    ]), np.asarray(["ICL"] * 4 + ["SCL"] * 2)


def test_bspline_basis_is_continuous_partition_of_unity():
    values = np.linspace(0.0, 20.0, 101)
    basis = spline_basis_1d(values, n_basis=6, degree=3, L=20.0)
    assert basis.shape == (101, 6)
    assert np.all(basis >= -1e-12)
    assert np.allclose(basis.sum(axis=1), 1.0)
    assert np.max(np.abs(np.diff(basis, axis=0))) < 0.2


def test_tensor_surface_has_no_component_or_peak_count_contract():
    positions = np.asarray([[1.0, 2.0], [10.0, 10.0], [19.0, 18.0]])
    assert tensor_basis(positions, 4).shape == (3, 16)
    coefficients = np.arange(16, dtype=float)
    values = continuous_surface(coefficients, positions, n_basis=4)
    assert values.shape == (3,)
    shifted = continuous_surface(coefficients + 19.0, positions, n_basis=4)
    assert np.allclose(values, shifted)


def test_continuous_field_projection_has_exact_mass_without_k():
    rng = np.random.default_rng(4)
    positions = rng.uniform(0.0, 20.0, size=(1000, 2))
    coefficients = rng.normal(size=16)
    h, diagnostics = continuous_field_h(
        coefficients, positions, n_basis=4, target_count=123.0,
    )
    assert h.sum() == pytest.approx(123.0, abs=1e-9)
    assert np.all((h > 0.0) & (h < 1.0))
    assert diagnostics["surface_max"] > diagnostics["surface_min"]


def test_continuous_corridor_is_smooth_exact_mass_and_not_components():
    positions = np.asarray([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, 2.0],
    ])
    segments = np.asarray([[[0.0, 0.0], [2.0, 0.0]]])
    distance = distance_to_segments(positions, segments)
    assert np.allclose(distance, [0.0, 0.0, 0.0, 1.0, 2.0])
    h, _ = continuous_corridor_field_h(
        segments, positions, width_mm=0.5, target_count=2.0,
    )
    assert h.sum() == pytest.approx(2.0, abs=1e-9)
    assert h[1] > h[3] > h[4]


def test_direct_h_node_reconstruction_matches_legacy_theta_path():
    positions = np.asarray([
        [2.0, 3.0], [4.0, 5.0], [8.0, 9.0], [15.0, 14.0],
    ])
    theta = np.asarray([
        3.0, 4.0, np.log(1.0), np.log(1.5), 0.2,
        9.0, 8.0, np.log(2.0), np.log(0.8), 1.0,
        15.0, 14.0, np.log(1.2), np.log(1.3), 2.0,
        0.3, -0.4,
    ])
    legacy = reconstruct_frozen_node(
        theta, positions, n_total=6, target_count=1.5, quantile_seed=7,
        core_mean=17.5, core_std=1.0, v_base=18.0,
    )
    direct = reconstruct_node_from_h(
        legacy["h"], n_total=6, quantile_seed=7, core_mean=17.5,
        core_std=1.0, v_base=18.0,
    )
    for key in ("h", "d", "vtheta", "delta_vtheta"):
        assert np.array_equal(legacy[key], direct[key])
    assert legacy["hashes"] == direct["hashes"]


def test_worker_accepts_continuous_candidate_without_k_or_components():
    rng = np.random.default_rng(8)
    positions = rng.uniform(0.0, 20.0, size=(100, 2))
    stage = {
        "N_core_manual": 20.0,
        "quantile_seed": 3,
        "engine": {
            "L": 20.0, "core_mean": 17.5, "core_std": 1.0, "v_base": 18.0,
        },
    }
    candidate = {
        "field_type": "continuous_bspline", "n_basis": 4, "degree": 3,
        "coefficients": np.linspace(-1.0, 1.0, 16).tolist(),
    }
    node = _candidate_node(candidate, positions, n_total=125, stage=stage)
    assert node["h"].sum() == pytest.approx(20.0, abs=1e-9)
    assert node["vtheta"].shape == (125,)


def test_shaft_weights_do_not_let_more_icl_contacts_dominate():
    _, shafts = _contacts()
    weights = shaft_balanced_contact_weights(shafts)
    assert weights[shafts == "ICL"].sum() == pytest.approx(0.5)
    assert weights[shafts == "SCL"].sum() == pytest.approx(0.5)


def test_background_anchors_are_sheet_wide_and_contact_excluded():
    contacts, _ = _contacts()
    anchors = background_anchors(
        contacts, L=20.0, spacing_mm=2.5, exclusion_radius_mm=2.0,
    )
    assert np.all((anchors >= 0.0) & (anchors <= 20.0))
    distance = np.min(
        np.linalg.norm(anchors[:, None] - contacts[None], axis=2), axis=1,
    )
    assert np.all(distance >= 2.0)
    assert len(anchors) > len(contacts)


def test_contact_fit_is_smooth_and_not_component_assigned():
    contacts, shafts = _contacts()
    target = np.asarray([0.9, 0.8, 0.7, 0.6, 0.85, 0.75])
    coefficients, diagnostics = fit_contact_target(
        contacts, target, shafts, n_basis=4, roughness=0.1,
    )
    assert coefficients.shape == (16,)
    assert coefficients.mean() == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["weighted_contact_rmse"] < 2.0
    assert np.isfinite(diagnostics["background_rmse"])
    assert diagnostics["curvature_energy"] >= 0.0


def test_patient_targets_keep_old_direction_modes_separate():
    onsets = np.asarray([
        [0.0, 1.0, np.nan],
        [0.0, 2.0, 1.0],
        [2.0, 1.0, 0.0],
        [np.nan, 1.0, 0.0],
    ])
    targets = patient_contact_targets(onsets, np.asarray([0, 0, 1, 1]))
    assert targets["mode_A_recruitment"][0] == 1.0
    assert targets["mode_B_recruitment"][0] == 0.5
    assert targets["weakest_mode_recruitment"][0] == 0.5
    assert targets["either_mode_early_support"].shape == (3,)


def test_candidate_builder_matches_dof_without_interpreting_coefficients_as_cores():
    contacts, shafts = _contacts()
    onsets = np.asarray([
        [0.0, 1.0, 2.0, 3.0, 0.5, 1.5],
        [3.0, 2.0, 1.0, 0.0, 1.5, 0.5],
    ] * 4)
    labels = np.asarray([0, 1] * 4)
    result = build_continuous_field_candidates(
        contacts, shafts, onsets, labels,
        designs=[{"n_basis": 4, "roughness": [0.1], "contrast": [1.0, 2.0]}],
    )
    assert len(result["candidates"]) == 7
    assert all(row["n_basis"] == 4 for row in result["candidates"])
    assert all(row["no_component_or_peak_assignment"] for row in result["candidates"])
    assert len({row["field_sha256"] for row in result["candidates"]}) == 7
    recruitment = next(row for row in result["candidates"]
                       if "mode_A_recruitment" in row["target_aliases"])
    assert "mode_B_recruitment" in recruitment["target_aliases"]
    assert "weakest_mode_recruitment" in recruitment["target_aliases"]


def test_real_continuous_manifest_keeps_k3_as_benchmark_only():
    config_path = ROOT / "config/topic4_rev10_sa_continuous_field_canary.json"
    config = json.loads(config_path.read_text())
    assay = config["sa6f_continuous_field"]
    assert assay["component_count"] is None
    assert assay["peak_count_constraint"] is None
    manifest = build_manifest(config_path)
    candidates = manifest["candidate_set"]["candidates"]
    assert len(candidates) == 37
    assert sum(row["field_type"] == "gaussian_k3_benchmark"
               for row in candidates) == 1
    continuous = [row for row in candidates
                  if row["field_type"] == "continuous_bspline"]
    assert {row["n_basis"] for row in continuous} == {4, 6}
    assert all(row["component_count"] is None for row in continuous)
    assert not np.allclose(
        manifest["forced_sources"][0]["xy_mm"],
        manifest["forced_sources"][1]["xy_mm"],
    )


def test_continuous_support_manifest_has_no_k_component_or_peak_contract():
    config_path = ROOT / "config/topic4_rev10_sa_continuous_support_canary.json"
    manifest = build_support_manifest(config_path)
    candidates = manifest["candidate_set"]["candidates"]
    assert len(candidates) == 8
    assert {row["support_id"] for row in candidates} == {
        "dual_shaft_disconnected", "dual_shaft_connected",
    }
    assert all(row["field_type"] == "continuous_corridor" for row in candidates)
    assert all(row["component_count"] is None for row in candidates)
    assert all(row["peak_count_constraint"] is None for row in candidates)
    assert max(row["mean_h_within_path_radius"]
               for row in manifest["candidate_preflight"]) > 0.5


def test_worker_accepts_continuous_support_candidate_without_k():
    contacts = [
        {"shaft_id": "ICL", "within_shaft_order_by_shared_axis": 0,
         "sheet_xy_mm": [1.0, 1.0]},
        {"shaft_id": "ICL", "within_shaft_order_by_shared_axis": 1,
         "sheet_xy_mm": [8.0, 1.0]},
        {"shaft_id": "SCL", "within_shaft_order_by_shared_axis": 0,
         "sheet_xy_mm": [2.0, 8.0]},
        {"shaft_id": "SCL", "within_shaft_order_by_shared_axis": 1,
         "sheet_xy_mm": [7.0, 8.0]},
    ]
    candidate = build_continuous_support_candidates(
        contacts, widths_mm=[0.5],
    )["candidates"][1]
    rng = np.random.default_rng(22)
    positions = rng.uniform(0.0, 20.0, size=(100, 2))
    stage = {
        "N_core_manual": 20.0,
        "quantile_seed": 3,
        "engine": {
            "L": 20.0, "core_mean": 17.5, "core_std": 1.0, "v_base": 18.0,
        },
    }
    node = _candidate_node(candidate, positions, n_total=125, stage=stage)
    assert node["h"].sum() == pytest.approx(20.0, abs=1e-9)
    assert node["vtheta"].shape == (125,)


def test_continuous_plot_field_is_finite_and_normalized():
    candidate = {
        "field_type": "continuous_bspline", "n_basis": 4, "degree": 3,
        "coefficients": np.linspace(-2.0, 2.0, 16).tolist(),
    }
    grid = np.asarray([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    values = _relative_field(candidate, grid)
    assert np.isfinite(values).all()
    assert values.max() == pytest.approx(1.0)
    assert np.all(values > 0.0)
