import json
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate_topic4_dual_core_pathway_refit import (
    _event_spans_ms,
    nondominated_mask,
    paired_bootstrap_difference,
    pathway_objective,
)
from scripts.freeze_topic4_dual_core_pathway_refit import _candidate_id
from scripts.run_topic4_rev10_r_edge_flow_worker import active_network_seeds


ROOT = Path(__file__).resolve().parents[1]


def test_registered_pathway_surface_has_twenty_unique_cells():
    config = json.loads(
        (ROOT / "config/topic4_dual_core_pathway_refit.json").read_text()
    )
    cells = {
        _candidate_id(g_ee, g_etoi)
        for g_ee in config["pathway_refit"]["g_EE"]
        for g_etoi in config["pathway_refit"]["g_EtoI"]
    }
    assert len(cells) == 20
    assert "gee000_getoi000" in cells
    assert "gee100_getoi100" in cells
    assert set(config["search"]["fit_network_seeds"]).issubset(
        active_network_seeds(config)
    )


def test_pathway_objective_uses_registered_additive_components():
    components = {
        "ood": 0.3,
        "mode_fraction": 0.2,
        "kmeans": 0.1,
        "event_yield": 0.4,
        "absolute_timing": 0.5,
    }
    weights = {
        "mode_fraction": 0.25,
        "kmeans": 0.2,
        "event_yield": 0.15,
        "absolute_timing": 0.15,
    }
    assert pathway_objective(components, weights) == pytest.approx(0.505)


def test_nondominated_mask_keeps_tradeoffs_and_removes_dominated_row():
    matrix = np.array([
        [0.1, 0.5],
        [0.5, 0.1],
        [0.6, 0.6],
        [0.1, 0.5],
    ])
    assert np.array_equal(
        nondominated_mask(matrix), [True, True, False, True]
    )


def test_event_spans_ignore_missing_contacts_without_normalizing_time():
    onsets = np.array([
        [0.0, 12.0, np.nan],
        [np.nan, 5.0, np.nan],
        [4.0, 10.0, 19.0],
    ])
    spans = _event_spans_ms(onsets)
    assert spans[0] == 12.0
    assert np.isnan(spans[1])
    assert spans[2] == 15.0


def test_confirmation_manifest_freezes_one_candidate_plus_node_reference():
    path = (
        ROOT / "results/topic4_sef_hfo/data_driven_dual_core_ood/"
        "pathway_refit/confirmation/candidate_manifest.json"
    )
    if not path.is_file():
        pytest.skip("confirmation manifest is generated after selection")
    manifest = json.loads(path.read_text())
    selectable = [
        row for row in manifest["candidate_set"]["candidates"]
        if row["selection_role"] == "selectable_candidate"
    ]
    references = [
        row for row in manifest["candidate_set"]["candidates"]
        if row["selection_role"] == "paired_node_reference"
    ]
    assert len(selectable) == 1
    assert len(references) == 1
    assert manifest["fixed_contract"]["selection_candidate_ids"] == [
        selectable[0]["candidate_id"]
    ]


def test_paired_bootstrap_uses_network_difference_as_unit():
    result = paired_bootstrap_difference(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 1.5, 2.5]),
        draws=128, seed=7,
    )
    assert result["n_pairs"] == 3
    assert result["mean_difference"] == -0.5
    assert result["ci90"] == [-0.5, -0.5]
    assert result["candidate_lower_pairs"] == 3
