import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_topic4_fcxr_lc6a_functional_probe.py"
SPEC = importlib.util.spec_from_file_location("lc6a_functional_runner", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_probe_matrix_and_scientific_boundary_are_prelocked():
    prelock = MOD._validate_prelock()
    assert prelock["probe_matrix"] == {
        "C0": ["core_adjacent", "neutral_axis"],
        "C1": ["neutral_axis"],
        "Q1": ["neutral_axis"],
        "Q2": ["core_adjacent", "neutral_axis"],
        "Q3": ["core_adjacent", "neutral_axis"],
    }
    assert prelock["interpretation"]["zero_crossing_is_a_gate"] is False
    assert prelock["timing"]["registered_windows_ms"] == [[0.0, 50.0], [50.0, 150.0], [150.0, 300.0]]


def test_active_fraction_uses_distinct_cells_per_1ms():
    raster = np.zeros((20, 10), bool)
    raster[0, 0] = True
    raster[10, :2] = True
    got = MOD._active_fraction_1ms(raster, .1)
    np.testing.assert_allclose(got, [.1, .2])


def test_runner_requires_explicit_lc5_authorization_and_never_reads_q_trajectory_for_lock():
    source = SCRIPT.read_text()
    assert "lc5_to_lc6a_authorization.json" in source
    assert 'q_trajectory_outcome_read": False' in source
    assert "paired functional arms did not share exact external input" in source
