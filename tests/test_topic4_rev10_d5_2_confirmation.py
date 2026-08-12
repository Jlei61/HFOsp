import numpy as np

from scripts.audit_topic4_rev10_d5_2_spatial_ou_confirmation import (
    adjudicate,
    map_kmeans_by_supervised_direction,
)


ACCEPTANCE = {
    "minimum_same_networks_with_both_modes": 4,
    "minimum_pooled_clean_events_per_mode": 6,
    "minimum_kmeans_ami_with_supervised_direction": 0.8,
}


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


def test_cluster_mapping_uses_supervised_contingency_not_patient_correlation():
    mapping, table = map_kmeans_by_supervised_direction(
        np.asarray([1, 1, 0, 0]), np.asarray([0, 0, 1, 1]),
    )
    assert mapping == (1, 0)
    assert table.tolist() == [[0, 2], [2, 0]]


def test_d5_2_positive_status_requires_support_kmeans_and_patient_geometry():
    verdict = adjudicate(
        acceptance=ACCEPTANCE, local_row=_row(4, 0.2),
        permuted_row=_row(3, 0.15), off_row=_row(0, 0.05),
        clean_counts=[6, 7],
        kmeans_audit={"status": "OK", "ami_with_supervised_direction": 0.9},
        patient_matrix=[[0.5, -0.2], [-0.4, 0.8]],
    )
    assert verdict["status"].endswith("PATIENT_MODE_CONSISTENCY_OBSERVED")
    assert verdict["fig4_support_evaluable"] is True
    assert verdict["activity_burden"]["local"]["occupancy_ratio_to_off"] == 4.0


def test_d5_2_fails_closed_on_cross_network_pooling_or_kmeans_mismatch():
    support_fail = adjudicate(
        acceptance=ACCEPTANCE, local_row=_row(3),
        permuted_row=_row(3), off_row=_row(0), clean_counts=[20, 20],
        kmeans_audit={"status": "OK", "ami_with_supervised_direction": 1.0},
        patient_matrix=[[0.8, -0.5], [-0.5, 0.8]],
    )
    assert support_fail["status"].endswith("DUAL_MODE_SUPPORT_NOT_CONFIRMED")
    kmeans_fail = adjudicate(
        acceptance=ACCEPTANCE, local_row=_row(5),
        permuted_row=_row(3), off_row=_row(0), clean_counts=[20, 20],
        kmeans_audit={"status": "OK", "ami_with_supervised_direction": 0.2},
        patient_matrix=[[0.8, -0.5], [-0.5, 0.8]],
    )
    assert kmeans_fail["status"].endswith("KMEANS_PATIENT_IDENTITY_NOT_CONFIRMED")


def test_d5_2_separates_evaluable_support_from_patient_geometry():
    verdict = adjudicate(
        acceptance=ACCEPTANCE, local_row=_row(5),
        permuted_row=_row(3), off_row=_row(0), clean_counts=[20, 20],
        kmeans_audit={"status": "OK", "ami_with_supervised_direction": 0.95},
        patient_matrix=[[0.8, 0.1], [-0.5, 0.8]],
    )
    assert verdict["fig4_support_evaluable"] is True
    assert verdict["patient_prototype_sign_geometry_consistent"] is False
    assert verdict["status"].endswith("PROTOTYPE_GEOMETRY_NOT_CONFIRMED")
