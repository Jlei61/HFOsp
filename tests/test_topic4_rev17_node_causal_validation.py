from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.paper_figures import plot_topic4_rev17_node_causal_validation as figure


def _selectivity() -> dict:
    per_network = []
    for seed in (2381, 2382, 2383):
        per_network.append({
            "network_seed": seed,
            "own_mode_effect_event_abolished_then_delay": [1, 0.0],
            "opposite_mode_effect_event_abolished_then_delay": [0, 3.0],
            "matched_control_effect_event_abolished_then_delay": [0, 1.0],
            "selective": True,
        })
    return {
        "primary_ordered_effect": (
            "native supported-mode loss first, then nonnegative onset delay"
        ),
        "modes": {
            "0": {"per_network": deepcopy(per_network)},
            "1": {"per_network": deepcopy(per_network)},
        },
    }


def test_effect_records_do_not_encode_event_loss_as_zero_delay():
    rows = figure.intervention_effect_records(_selectivity())
    assert len(rows) == 18
    own = [row for row in rows if row["effect"] == "own"]
    assert all(row["event_lost"] for row in own)
    assert all(row["delay_if_retained_ms"] is None for row in own)
    other = [row for row in rows if row["effect"] == "other"]
    assert all(not row["event_lost"] for row in other)
    assert all(row["delay_if_retained_ms"] == 3.0 for row in other)
    assert {row["hotspot_name"] for row in rows} == {"MTA", "MTB"}


def test_effect_records_reject_abolished_event_with_delay():
    payload = _selectivity()
    payload["modes"]["1"]["per_network"][0][
        "own_mode_effect_event_abolished_then_delay"
    ] = [1, 2.0]
    with pytest.raises(RuntimeError, match="cannot carry"):
        figure.intervention_effect_records(payload)


def test_topology_records_keep_candidate_and_reference_nulls_separate():
    payload = {}
    for key, observed in (
        ("reference_source_topology_permutation", 0.2),
        ("source_topology_permutation", 0.6),
    ):
        payload[key] = {
            "observed_weakest_mode_quality": observed,
            "null_q05": 0.1,
            "null_q50": 0.25,
            "null_q95": 0.4,
            "upper_tail_p": 0.02,
        }
    rows = figure.topology_plot_records(payload)
    assert [row["label"] for row in rows] == ["Exact anchor", "Rev17 field"]
    assert [row["observed"] for row in rows] == [0.2, 0.6]


def test_topology_records_reject_unordered_null_quantiles():
    payload = {
        key: {
            "observed_weakest_mode_quality": 0.3,
            "null_q05": 0.4,
            "null_q50": 0.2,
            "null_q95": 0.5,
            "upper_tail_p": 0.1,
        }
        for key in (
            "reference_source_topology_permutation",
            "source_topology_permutation",
        )
    }
    with pytest.raises(RuntimeError, match="unordered"):
        figure.topology_plot_records(payload)
