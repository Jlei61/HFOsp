import numpy as np

from scripts.paper_figures.run_fig3e_template_expression_cohort import (
    _exact_sign_flip_mean_p,
    cohort_test,
    phase_window_mask,
)
from src.topic5_tspectral_field_concordance import make_complete_window_grid


def test_phase_masks_only_include_complete_windows():
    grid = make_complete_window_grid(-120, 20, 10, 2)
    distal = grid[phase_window_mask(grid, "distal_pre")]
    proximal = grid[phase_window_mask(grid, "proximal_pre")]
    early = grid[phase_window_mask(grid, "early_ictal")]
    assert np.all(distal[:, 0] >= -120) and np.all(distal[:, 1] <= -90)
    assert np.all(proximal[:, 0] >= -30) and np.all(proximal[:, 1] <= -10)
    assert np.all(early[:, 0] >= 0) and np.all(early[:, 1] <= 20)


def test_exact_sign_flip_uses_subjects_not_windows():
    values = np.ones(7)
    assert _exact_sign_flip_mean_p(values) == 1 / 128
    result = cohort_test(values, seed=1)
    assert result["n_subjects"] == 7
    assert result["n_positive"] == 7
    assert result["exact_sign_test_greater_p"] == 1 / 128
