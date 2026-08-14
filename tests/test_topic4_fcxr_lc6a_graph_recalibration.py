import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/recalibrate_topic4_fcxr_lc6a_graph_conditions.py"
SPEC = importlib.util.spec_from_file_location("lc6a_graph_recalibration", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_secant_width_recovers_linear_target_and_clamps():
    assert MOD.secant_width(
        l_low=.25, sigma_low=1.0, l_high=.5, sigma_high=1.5,
        sigma_target=2.0, lower=.05, upper=2.0,
    ) == pytest.approx(.75)
    assert MOD.secant_width(
        l_low=.25, sigma_low=1.0, l_high=.5, sigma_high=1.5,
        sigma_target=10.0, lower=.05, upper=2.0,
    ) == 2.0


def test_secant_rejects_uninformative_anchors():
    with pytest.raises(RuntimeError, match="do not separate"):
        MOD.secant_width(
            l_low=.25, sigma_low=1.0, l_high=.5, sigma_high=1.0,
            sigma_target=2.0, lower=.05, upper=2.0,
        )


def test_recalibration_is_graph_only_and_does_not_relax_contracts():
    source = SCRIPT.read_text()
    assert MOD.Q_IDS == ("Q1", "Q2", "Q3")
    assert "trajectory_outcome_read" in source
    assert "run_fcxr_loop" not in source
    assert "graph_legality" in source
    assert "q_tolerance" not in source
