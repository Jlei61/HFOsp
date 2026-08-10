import numpy as np
import pytest

from src.topic4_forced_source_capacity import (
    paired_excess_geometry,
    select_packet_fraction,
    select_source_indices,
    select_triggered_event,
    source_mode_correlation_summary,
)


def test_component_source_uses_raw_contribution_not_relative_responsibility():
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    contributions = np.asarray([
        [0.9, 0.01],
        [0.8, 0.02],
        [0.1, 0.90],
        [0.01, 0.80],
    ])
    source = {"id": "component_1", "kind": "component", "component_1based": 1}
    selected = select_source_indices(
        positions, source, n_cells=2,
        component_contribution=contributions)
    np.testing.assert_array_equal(selected, [0, 1])


def test_control_source_uses_nearest_equal_count_cells():
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    source = {"id": "control_1", "kind": "matched_off_field", "xy_mm": [2.2, 0.0]}
    selected = select_source_indices(positions, source, n_cells=2)
    np.testing.assert_array_equal(selected, [2, 3])


def test_paired_excess_geometry_excludes_forced_source_from_downstream():
    forced = np.zeros((10, 4), bool)
    sham = np.zeros_like(forced)
    forced[2, 0] = True
    forced[3, 1] = True
    forced[4, 2] = True
    sham[4, 2] = True
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    source_mask = np.asarray([True, False, False, False])
    result = paired_excess_geometry(
        forced, sham, positions, source_mask,
        dt_ms=1.0, start_ms=2.0, end_ms=8.0,
        source_center=np.asarray([0.0, 0.0]))
    assert result["source_positive_spike_mass"] == pytest.approx(1.0)
    assert result["downstream_positive_spike_mass"] == pytest.approx(1.0)
    assert result["downstream_positive_neurons"] == 1
    assert result["r50_mm"] == pytest.approx(1.0)
    assert result["r90_mm"] == pytest.approx(1.0)


def test_triggered_event_selection_rejects_late_and_nonreturning_events():
    events = [
        {"t_on": 101.0, "t_off": 130.0, "returned": False},
        {"t_on": 125.0, "t_off": 150.0, "returned": True},
        {"t_on": 170.0, "t_off": 190.0, "returned": True},
    ]
    selected = select_triggered_event(
        events, trigger_ms=100.0, max_latency_ms=40.0)
    assert selected["t_on"] == pytest.approx(125.0)
    assert select_triggered_event(
        events, trigger_ms=100.0, max_latency_ms=10.0) is None


def test_packet_selection_uses_smallest_fraction_with_both_sources_readable():
    rows = []
    for fraction, coverage in ((0.005, {"c1": 2, "c2": 1}),
                               (0.01, {"c1": 3, "c2": 2}),
                               (0.02, {"c1": 3, "c2": 3})):
        for source in ("c1", "c2"):
            for index in range(3):
                eligible = index < coverage[source]
                rows.append({
                    "packet_fraction_of_E": fraction,
                    "source_id": source,
                    "pretrigger_spikes_bit_identical": True,
                    "paired_excess_readout": {"curve_usable": eligible},
                    "paired_geometry": {"downstream_any_positive": eligible},
                    "runaway_early_stop_ms": None,
                })
    selected = select_packet_fraction(
        rows, source_ids=["c1", "c2"], min_networks_per_source=2)
    assert selected["status"] == "PACKET_FRACTION_FROZEN"
    assert selected["selected"]["packet_fraction_of_E"] == pytest.approx(0.01)


def test_source_mode_summary_keeps_source_identity_fixed():
    patient = np.asarray([
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
    ])
    curves = np.asarray([
        [3.0, 2.1, 1.0, 0.0],
        [0.0, 0.9, 2.0, 3.0],
        [0.0, 1.1, 2.0, 3.0],
    ])
    result = source_mode_correlation_summary(
        curves, ["component_1", "component_2", "component_2"], patient,
        source_order=["component_1", "component_2"])
    matrix = np.asarray(result["median_correlation_matrix"])
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[0, 0] == pytest.approx(-1.0)
    assert matrix[1, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(-1.0)
    assert result["sources"]["component_2"]["n_total"] == 2


def test_source_mode_summary_reports_unusable_curve_without_relabeling():
    result = source_mode_correlation_summary(
        np.asarray([[np.nan, np.nan, np.nan]]), ["control_1"],
        np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        source_order=["control_1"])
    assert result["sources"]["control_1"]["n_total"] == 1
    assert result["sources"]["control_1"]["n_usable"] == 0
    assert result["median_correlation_matrix"] == [[None, None]]
