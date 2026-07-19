"""Contract tests for the cross-seed early-field bridge diagnostic."""

import pytest

from scripts.paper_figures import plot_fig_mz_cross_seed_transfer as X


def _cells():
    matrix = {
        1: {1: 0.90, 3: 0.70, 4: 0.80},
        3: {1: 0.88, 3: 0.72, 4: 0.82},
        4: {1: 0.92, 3: 0.74, 4: 0.80},
    }
    return {
        (template, target): {
            "template_seed": template,
            "energy_seed": target,
            "same_seed": template == target,
            "winner_direction": "B_to_A",
            "rho_maxab": value,
            "within_shaft_p": 0.01,
        }
        for template, row in matrix.items()
        for target, value in row.items()
    }


def test_cross_seed_summary_uses_target_fields_as_repeated_units():
    summary = X.summarize(_cells())

    assert summary["n_independent_target_fields"] == 3
    assert summary["n_descriptive_template_target_cells"] == 9
    assert summary["scientific_tier"].startswith("exploratory descriptive")
    assert summary["winner_direction_counts"] == {"B_to_A": 9}


def test_cross_seed_summary_reports_matched_same_seed_advantage():
    summary = X.summarize(_cells())
    matched = {row["target_energy_seed"]: row for row in summary["matched_same_seed_vs_foreign_template"]}

    assert matched[1]["same_minus_foreign_median"] == pytest.approx(0.0)
    assert matched[3]["same_minus_foreign_median"] == pytest.approx(0.0)
    assert matched[4]["same_minus_foreign_median"] == pytest.approx(-0.01)
