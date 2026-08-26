import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev12_broad_nonlocal_field_screen import build_candidates


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _load(relative):
    local = ROOT / relative
    path = local if local.exists() else ARTIFACT_ROOT / relative
    return json.loads(path.read_text())


def test_broad_nonlocal_candidates_are_free_continuous_and_antithetic():
    config = _load("config/topic4_rev12_nd_broad_nonlocal_field_screen.json")
    candidates, audit = build_candidates(
        _load(config["inputs"]["stage_aa_summary"]["path"]),
        _load(config["inputs"]["stage_af_crossvalidation"]["path"]),
        _load(config["inputs"]["search_span_resolution"]["path"]),
        config["broad_field_screen"],
    )
    assert len(candidates) == 25
    assert audit["basis"]["n_modes"] == 35
    assert not audit["manual_coefficients_or_geometry_used"]
    anchor = np.asarray(candidates[0]["node_field"]["coefficients"], float)
    for pair_index in range(12):
        pair = [
            row for row in candidates
            if row["node_field"].get("residual_coordinates", {}).get(
                "pair_index"
            ) == pair_index
        ]
        assert len(pair) == 2
        left, right = [np.asarray(row["node_field"]["coefficients"], float) for row in pair]
        assert np.max(np.abs((left + right) / 2.0 - anchor)) < 1e-11
        assert all(row["selection_eligible"] for row in pair)


def test_broad_nonlocal_resource_and_split_contract():
    config = _load("config/topic4_rev12_nd_broad_nonlocal_field_screen.json")
    assert config["search"]["fit_network_seeds"] == list(range(2281, 2287))
    assert config["resources"]["maximum_workers"] == 14
    assert config["resources"]["reserved_available_memory_gib"] >= 32
    assert config["pareto_selection"]["patient_heldout_used"] is False
    assert config["pareto_selection"]["natural_kmeans_used"] is False
