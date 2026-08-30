from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import freeze_topic4_rev16_joint_m3_m4_candidates as freezer
from scripts import prepare_topic4_rev16_joint_candidate_config as prepare


def synthetic_aggregate() -> dict:
    directions = {}
    for index, family in enumerate(prepare.FAMILY_ORDER):
        vector = np.zeros(48, dtype=float)
        vector[index] = 0.8
        vector[28 + index] = 0.6
        row = {"direction": vector.tolist()}
        if family.startswith("maximin"):
            row["feasible_positive_margin"] = True
        else:
            row["feasible"] = True
        directions[family] = row
    return {
        "status": "COMPLETE",
        "inventory": {"complete_cartesian_product": True},
        "ranking_contract": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "figure_used": False, "EE_EtoI_ZM": "off",
        },
        "robust_directions": directions,
    }


def test_blueprint_contains_joint_m3_and_m4_coefficients():
    rows, audit = prepare.candidate_blueprint(synthetic_aggregate())
    assert audit["selectable_candidate_count"] == 12
    assert audit["candidate_count_including_exact"] == 13
    assert len(rows) == 12
    for row in rows:
        coefficients = np.asarray(row["coefficients"])
        assert coefficients.shape == (24, 2)
        assert np.count_nonzero(coefficients[:14]) > 0
        assert np.count_nonzero(coefficients[14:]) > 0
        assert row["m3_l2_fraction"] > 0.0
        assert row["m4_shell_l2_fraction"] > 0.0


def test_incomplete_aggregate_cannot_prepare_candidates():
    aggregate = synthetic_aggregate()
    aggregate["status"] = "INCOMPLETE"
    with pytest.raises(RuntimeError, match="incomplete"):
        prepare.candidate_blueprint(aggregate)


def test_kmeans_or_heldout_contamination_is_rejected():
    aggregate = synthetic_aggregate()
    aggregate["ranking_contract"]["natural_kmeans_used"] = True
    with pytest.raises(RuntimeError, match="forbidden"):
        prepare.candidate_blueprint(aggregate)


def test_freezer_accepts_synthetic_dynamic_config(monkeypatch, tmp_path):
    aggregate = synthetic_aggregate()
    rows, audit = prepare.candidate_blueprint(aggregate)
    config = {
        "schema_id": freezer.EXPECTED_SCHEMA,
        "inputs": {
            name: {"path": "x", "sha256": "0" * 64}
            for name in (
                "rev13_config", "rev13_exact_off_manifest", "source_shell_config",
                "source_shell_manifest", "joint_response_analysis_config",
                "joint_response_aggregate", "j14_config", "patient_support_config",
            )
        },
        "field_design": {
            "basis_family": "absolute_paired_phase_whole_sheet_fourier",
            "maximum_order": 4, "expected_modes": 24,
            "expected_real_coefficients": 48,
            "sheet_length_mm": 20.0, "quadrature_per_axis": 128,
            "coordinate_decimal_places": 13,
            "robust_direction_ids": audit["feasible_direction_ids"],
            "candidate_rms_levels": [0.4, 0.6, 0.8],
            "candidate_ids": audit["candidate_ids"],
            "candidate_count": 13, "selectable_candidate_count": 12,
            "deduplication_audit": [], "candidate_blueprint": rows,
            "basis_uses_observation_geometry": False,
            "basis_uses_predeclared_objects": False,
        },
        "search": {
            "construction_network_seeds": [2331, 2332, 2333],
            "canary_network_seeds": [2351, 2352, 2353],
            "active_network_seeds": [2351, 2352, 2353],
            "simulation": {"duration_ms": 20000.0},
        },
        "selection": {
            "fresh_A_improvement_required_networks": 3,
            "fresh_B_protection_required_networks": 3,
            "B_protection_ratio": 1.10,
            "equal_network_effective_support_minimum_per_mode": 6.0,
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_used": False,
        },
        "pathways": freezer.EXPECTED_PATHWAYS,
    }
    freezer._validate_config(config)
