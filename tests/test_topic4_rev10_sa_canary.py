import numpy as np
import pytest

from src.topic4_rev10_sa_canary import (
    build_dual_shaft_candidates,
    classify_contact_detectability,
    contact_response_metrics,
    equal_mode_earliest_shaft_centroid,
    field_budget_summary,
    matched_contact_packets,
    paired_shaft_ratio,
    shaft_geometry,
)


def test_matched_contact_packets_keep_fixed_radius_and_equal_count():
    positions = np.asarray([
        [0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [1.0, 0.0],
        [1.1, 0.0], [1.2, 0.0], [1.3, 0.0],
    ])
    contacts = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    result = matched_contact_packets(
        positions, contacts, radius_mm=0.31, requested_count=3,
    )
    assert result["common_count"] == 3
    assert np.array_equal(result["masks"].sum(axis=1), [3, 3])
    for mask, contact in zip(result["masks"], contacts):
        assert np.all(np.linalg.norm(positions[mask] - contact, axis=1) <= 0.31)


def test_matched_contact_packets_fail_when_common_support_is_too_small():
    with pytest.raises(RuntimeError, match="common cells"):
        matched_contact_packets(
            np.asarray([[0.0, 0.0], [2.0, 0.0]]),
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            radius_mm=0.1, requested_count=1, minimum_count=1,
        )


def test_contact_response_separates_neural_and_lfp_readouts():
    forced_lfp = np.zeros(30)
    sham_lfp = np.ones(30)
    forced_lfp[:] = sham_lfp
    forced_lfp[12:16] += [1.0, 3.0, 2.0, 1.0]
    forced = np.zeros((30, 4), bool)
    sham = np.zeros_like(forced)
    forced[10, :2] = True
    forced[12:14, :3] = True
    metrics = contact_response_metrics(
        forced_lfp, sham_lfp, forced, sham,
        np.asarray([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [2.0, 0.0]]),
        np.asarray([0.0, 0.0]), np.asarray([True, True, False, False]),
        dt_ms=1.0, forced_spike_ms=10.0, response_stop_ms=20.0,
        baseline_window_ms=[0.0, 9.0], local_radius_mm=0.5,
    )
    assert metrics["peak_lfp_excess"] == pytest.approx(3.0)
    assert metrics["peak_lfp_excess_per_packet_cell"] == pytest.approx(1.5)
    assert metrics["local_positive_spike_excess_per_cell"] == pytest.approx(2.0)
    assert metrics["packet_positive_respike_excess_per_cell"] == pytest.approx(2.0)


def test_paired_shaft_ratio_uses_equal_shaft_weighting():
    rows = [
        {"shaft_id": "ICL", "gain": 2.0},
        {"shaft_id": "ICL", "gain": 4.0},
        {"shaft_id": "SCL", "gain": 1.5},
        {"shaft_id": "SCL", "gain": 1.5},
    ]
    result = paired_shaft_ratio(rows, "gain")
    assert result["ICL_median"] == 3.0
    assert result["SCL_over_ICL"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("lfp", "neural", "expected"),
    [
        (0.8, 0.9, "SCL_READOUT_NOT_PRIMARY_LIMIT"),
        (0.2, 0.9, "VIRTUAL_CONTACT_OBSERVATION_FAIL"),
        (0.9, 0.2, "SCL_LOCAL_NETWORK_RESPONSE_LIMIT"),
    ],
)
def test_contact_detectability_branch_labels(lfp, neural, expected):
    assert classify_contact_detectability(lfp, neural) == expected


def _frozen_theta():
    return np.asarray([
        3.6, 7.1, np.log(1.4), np.log(1.2), 0.2,
        11.8, 3.6, np.log(1.1), np.log(1.3), 0.8,
        12.3, 17.2, np.log(0.7), np.log(0.6), 2.1,
        1.0, 0.9,
    ])


def test_equal_mode_earliest_centroid_does_not_overweight_larger_mode():
    onsets = np.asarray([
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 5.0, 4.0],
    ])
    labels = np.asarray([0, 0, 1])
    xy = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    result = equal_mode_earliest_shaft_centroid(onsets, labels, [1, 2], xy)
    assert np.allclose(result["mode_centroids"], [[10.0, 0.0], [20.0, 0.0]])
    assert np.allclose(result["centroid"], [15.0, 0.0])


def test_dual_shaft_candidate_contract_has_fixed_grid_and_matched_control():
    contacts = np.asarray([
        [3.0, 5.0], [8.0, 15.0], [10.0, 14.0], [18.0, 4.0],
    ])
    geometry = shaft_geometry(contacts[1:3])
    result = build_dual_shaft_candidates(
        _frozen_theta(), scl_midpoint=geometry["midpoint"],
        scl_earliest_centroid=[8.5, 14.7], scl_phi=geometry["phi"],
        contact_xy=contacts,
    )
    assert len(result["candidates"]) == 21
    assert len({row["candidate_id"] for row in result["candidates"]}) == 21
    assert len({row["theta_sha256"] for row in result["candidates"]}) == 21
    displacement = np.linalg.norm(
        np.asarray(result["geometry"]["scl_midpoint"])
        - np.asarray(result["geometry"]["frozen_component_3_center"])
    )
    control = np.linalg.norm(
        np.asarray(result["geometry"]["matched_offshaft_center"])
        - np.asarray(result["geometry"]["frozen_component_3_center"])
    )
    assert control == pytest.approx(displacement)
    weights = np.asarray([
        row["weight"] for row in result["candidates"][-1]["components"]
    ])
    assert weights.sum() == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(0.35)


def test_field_budget_is_exact_for_every_candidate():
    rng = np.random.default_rng(1)
    positions = rng.uniform(0.0, 20.0, size=(500, 2))
    result = build_dual_shaft_candidates(
        _frozen_theta(), scl_midpoint=[8.0, 15.0],
        scl_earliest_centroid=[8.5, 14.7], scl_phi=0.4,
        contact_xy=np.asarray([[3.0, 5.0], [8.0, 15.0], [18.0, 4.0]]),
    )
    for row in result["candidates"]:
        _, summary = field_budget_summary(
            row["theta"], positions, target_count=100.0,
        )
        assert summary["sum_h"] == pytest.approx(100.0, abs=1e-9)
