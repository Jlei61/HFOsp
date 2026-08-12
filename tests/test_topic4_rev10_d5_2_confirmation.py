import numpy as np
import subprocess
from pathlib import Path

from scripts.audit_topic4_rev10_d5_2_spatial_ou_confirmation import (
    _purity,
    adjudicate,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(both, occupancy=0.1, runaway=0):
    return {
        "n_runaway_networks": runaway,
        "networks_with_both_clean_modes": both,
        "mean_network_detected_events_descriptive": 20,
        "mean_network_returned_events_scored": 15,
        "mean_network_fraction_time_above_detector": occupancy,
        "max_network_fraction_time_above_detector": occupancy,
        "mean_network_peak_active_fraction": 0.1,
        "mean_network_ood_fraction": 0.2,
    }


def test_direction_purity_is_label_swap_invariant():
    value, table = _purity(
        np.asarray([1, 1, 0, 0]), np.asarray([0, 0, 1, 1]),
    )
    assert value == 1.0
    assert table.tolist() == [[0, 2], [2, 0]]


def test_confirmation_auditor_imports_when_executed_as_a_script():
    completed = subprocess.run(
        [
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python",
            str(ROOT / "scripts/audit_topic4_rev10_d5_2_spatial_ou_confirmation.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_d5_2_positive_status_requires_support_kmeans_and_patient_geometry():
    verdict = adjudicate(
        minimum_same_networks=4, minimum_events_per_mode=6,
        local_row=_row(4, 0.2),
        permuted_row=_row(3, 0.15), off_row=_row(0, 0.05),
        clean_counts=[6, 7], kmeans_purity=0.9,
        kmeans_permutation_p=0.01, patient_purity_q05=0.85,
        patient_matrix=[[0.5, -0.2], [-0.4, 0.8]],
    )
    assert verdict["status"].endswith("PATIENT_MODE_CONSISTENCY_OBSERVED")
    assert verdict["fresh_network_dual_mode_support"] is True
    assert verdict["activity_burden"]["local"]["occupancy_ratio_to_off"] == 4.0


def test_d5_2_fails_closed_on_cross_network_pooling_or_kmeans_mismatch():
    support_fail = adjudicate(
        minimum_same_networks=4, minimum_events_per_mode=6,
        local_row=_row(3),
        permuted_row=_row(3), off_row=_row(0), clean_counts=[20, 20],
        kmeans_purity=0.95, kmeans_permutation_p=0.01,
        patient_purity_q05=0.85,
        patient_matrix=[[0.8, -0.5], [-0.5, 0.8]],
    )
    assert support_fail["status"].endswith("DUAL_MODE_SUPPORT_NOT_CONFIRMED")
    kmeans_fail = adjudicate(
        minimum_same_networks=4, minimum_events_per_mode=6,
        local_row=_row(5),
        permuted_row=_row(3), off_row=_row(0), clean_counts=[20, 20],
        kmeans_purity=0.7, kmeans_permutation_p=0.2,
        patient_purity_q05=0.85,
        patient_matrix=[[0.8, -0.5], [-0.5, 0.8]],
    )
    assert kmeans_fail["status"].endswith("KMEANS_DIRECTION_ASSOCIATION_NOT_OBSERVED")


def test_d5_2_reports_partial_when_kmeans_is_real_but_below_patient_floor():
    verdict = adjudicate(
        minimum_same_networks=4, minimum_events_per_mode=6,
        local_row=_row(5), permuted_row=_row(3), off_row=_row(0),
        clean_counts=[20, 20], kmeans_purity=0.7,
        kmeans_permutation_p=0.01, patient_purity_q05=0.85,
        patient_matrix=[[0.8, -0.5], [-0.5, 0.8]],
    )
    assert verdict["kmeans_direction_association"] is True
    assert verdict["kmeans_reaches_patient_matched_q05"] is False
    assert verdict["status"].endswith("KMEANS_BELOW_PATIENT_BENCHMARK")


def test_d5_2_separates_evaluable_support_from_patient_geometry():
    verdict = adjudicate(
        minimum_same_networks=4, minimum_events_per_mode=6,
        local_row=_row(5),
        permuted_row=_row(3), off_row=_row(0), clean_counts=[20, 20],
        kmeans_purity=0.95, kmeans_permutation_p=0.01,
        patient_purity_q05=0.85,
        patient_matrix=[[0.8, 0.1], [-0.5, 0.8]],
    )
    assert verdict["fresh_network_dual_mode_support"] is True
    assert verdict["supervised_patient_prototype_geometry"] is False
    assert verdict["status"].endswith("DIRECTION_PROTOTYPE_GEOMETRY_NOT_CONFIRMED")
