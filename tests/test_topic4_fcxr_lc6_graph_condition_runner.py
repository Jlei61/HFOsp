import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lc6_graph_condition", ROOT / "scripts/build_topic4_fcxr_lc6a_graph_condition.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_single_condition_runner_is_limited_to_the_four_independent_rewirings():
    assert MOD.ALLOWED == ("C1", "Q1", "Q2", "Q3")
    with pytest.raises(ValueError, match="condition"):
        MOD.build_condition(ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json", "C0")


def test_single_condition_runner_uses_same_locked_generator_and_not_trajectory_data():
    source = Path(MOD.__file__).read_text()
    assert "rewire_e_to_i_targetwise" in source
    assert 'trajectory_outcome_read": False' in source
    assert "condition_graph_seeds" in source
    assert "source_out_degree_relative_tolerance" in source
