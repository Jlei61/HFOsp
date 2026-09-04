from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_topic4_rev22_final_candidate_acceptance",
    ROOT / "scripts/paper_figures/plot_topic4_rev22_final_candidate_acceptance.py",
)
acceptance = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(acceptance)


def test_contact_order_groups_shafts_and_sorts_numeric_suffixes():
    names = ["SCL9", "ICL11", "ICL2", "SCL6", "ICL1"]
    shafts = ["SCL", "ICL", "ICL", "SCL", "ICL"]
    order = acceptance._contact_order(names, shafts)
    assert [names[index] for index in order] == ["ICL1", "ICL2", "ICL11", "SCL6", "SCL9"]


def test_representative_pair_is_same_network_and_one_event_per_mode():
    events = {
        "ranks": np.asarray([
            [0, 1, 2], [2, 1, 0], [0, 1, 2], [2, 1, 0],
        ], float),
        "labels": np.asarray([0, 1, 0, 1]),
        "event_map": [(0, 0), (0, 1), (1, 0), (1, 1)],
    }
    rows = [
        {"arrays": {"event_t_on_ms": np.asarray([100.0, 220.0]),
                    "event_t_off_ms": np.asarray([150.0, 270.0])}},
        {"arrays": {"event_t_on_ms": np.asarray([100.0, 1500.0]),
                    "event_t_off_ms": np.asarray([150.0, 1550.0])}},
    ]
    pair = acceptance._representative_pair(events, rows)
    assert pair["unit_index"] == 0
    assert pair["source_event_indices"] == [0, 1]
    assert pair["span_ms"] == 170.0


def test_representative_pair_rejects_cross_network_only_modes():
    events = {
        "ranks": np.asarray([[0, 1, 2], [2, 1, 0]], float),
        "labels": np.asarray([0, 1]),
        "event_map": [(0, 0), (1, 0)],
    }
    rows = [
        {"arrays": {"event_t_on_ms": np.asarray([100.0]),
                    "event_t_off_ms": np.asarray([150.0])}},
        {"arrays": {"event_t_on_ms": np.asarray([200.0]),
                    "event_t_off_ms": np.asarray([250.0])}},
    ]
    with pytest.raises(RuntimeError, match="same-network"):
        acceptance._representative_pair(events, rows)
