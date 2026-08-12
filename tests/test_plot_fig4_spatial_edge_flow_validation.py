import ast
from pathlib import Path

import numpy as np

from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (
    _same_network_pair,
    formal_clean_mask,
    normalize_event_ranks,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formal_clean_mask_requires_both_shafts_and_patient_support():
    onsets = np.asarray([
        [0.0, np.nan, 1.0],
        [0.0, 1.0, np.nan],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ])
    clean = formal_clean_mask(
        onsets, np.asarray([0, 0, 1, 1]), np.asarray([False, False, False, True]),
        {"ICL": np.asarray([0]), "SCL": np.asarray([1, 2])},
    )
    assert clean.tolist() == [True, True, True, False]


def test_formal_clean_mask_excludes_nonreturned_event():
    clean = formal_clean_mask(
        np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        np.asarray([0, 1]), np.asarray([False, False]),
        {"ICL": np.asarray([0]), "SCL": np.asarray([1])},
        event_returned=np.asarray([True, False]),
    )
    assert clean.tolist() == [True, False]


def test_rank_normalization_preserves_missing_contacts():
    values = normalize_event_ranks(np.asarray([[4.0, np.nan, 2.0, 3.0]]))
    np.testing.assert_allclose(values[0, [0, 2, 3]], [1.0, 0.0, 0.5])
    assert np.isnan(values[0, 1])


def test_same_network_pair_never_crosses_networks():
    bundle = {
        "config": {"search": {"fit_network_seeds": [1, 2]}},
        "clean": np.asarray([True, True]),
        "labels": np.asarray([0, 1]),
        "records": [
            {"seed": 1, "local_index": 0},
            {"seed": 2, "local_index": 0},
        ],
        "blocks": [{"seed": 1}, {"seed": 2}],
    }
    assert _same_network_pair(bundle) is None
    bundle["records"][1]["seed"] = 1
    pair = _same_network_pair(bundle)
    assert pair[0:2] == (0, 1)
    assert pair[2]["seed"] == 1


def test_figure_consumer_has_no_simulation_or_candidate_selection():
    path = ROOT / "scripts/paper_figures/plot_fig4_spatial_edge_flow_validation.py"
    tree = ast.parse(path.read_text())
    calls = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "simulate_kick" not in calls
    source = path.read_text()
    assert 'summary["diagnostic_best_candidate_id"]' in source
    assert "fit_network_seeds" in source
    assert "same-network" in source
    assert "a.u." in source
