"""Small contract tests for the Phase-C futility diagnostic plotter."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plot_topic4_zm_phasec_futility.py"
SPEC = importlib.util.spec_from_file_location(
    "phasec_futility_plot", SCRIPT
)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def test_short_cell_labels_preserve_path_and_stage():
    assert P._short_cell("primary__rising__bounded_early") == "R: early"
    assert P._short_cell(
        "primary__peak__mid_late_midpoint"
    ) == "P: mid–late"


def test_representative_row_uses_median_modulation():
    rows = [
        {"modulation_depth": 0.01},
        {"modulation_depth": 0.03},
        {"modulation_depth": 0.08},
    ]
    assert P._representative_row(rows)["modulation_depth"] == 0.03
