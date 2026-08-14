import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_topic4_fcxr_lc6a_graph_family.py"
SPEC = importlib.util.spec_from_file_location("lc6_graph_builder", SCRIPT)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_manifest_locks_five_graphs_and_private_graph_seeds():
    path, payload = BUILDER._validate_manifest(
        ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json"
    )
    assert path.name == "topic4_fcxr_lc6a_patient_axis_surround.json"
    assert [row["id"] for row in payload["graph_family"]] == [
        "C0", "C1", "Q1", "Q2", "Q3",
    ]
    assert payload["graph_generation"]["maximum_sweeps"] == 6
    assert set(payload["graph_generation"]["condition_graph_seeds"]) == {
        "C1", "Q1", "Q2", "Q3",
    }


def test_parallel_width_initialization_uses_graph_only_covariance():
    l_parallel, desired = BUILDER._target_parallel_width(
        1.25,
        {"sigma_parallel_mm": .30},
        {"sigma_parallel_mm": .20},
        {"sigma_parallel_mm": .40},
        .25,
    )
    assert np.isclose(desired, np.sqrt((1.25 * .40) ** 2 - .20 ** 2))
    assert np.isclose(l_parallel, .25 * desired / .30)


def test_graph_builder_does_not_import_or_read_trajectory_outcomes():
    text = SCRIPT.read_text()
    assert "trajectory_outcome_read" in text
    assert "run_fcxr_loop" not in text
    assert "summary.json" not in text
    manifest = json.loads(
        (ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json").read_text()
    )
    assert manifest["graph_generation"]["trajectory_outcome_used_for_graph_selection"] is False
