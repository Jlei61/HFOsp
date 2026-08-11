import numpy as np
import pytest

from src.topic4_rev10_sa_canary import (
    classify_contact_detectability,
    contact_response_metrics,
    matched_contact_packets,
    paired_shaft_ratio,
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
