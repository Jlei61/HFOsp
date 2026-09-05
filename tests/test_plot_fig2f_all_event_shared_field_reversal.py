from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.paper_figures import plot_fig2f_all_event_shared_field_reversal as F


def test_significance_thresholds_match_figure_legend() -> None:
    assert F._significance_label(0.06) == "ns"
    assert F._significance_label(0.04) == "*"
    assert F._significance_label(0.004) == "**"
    assert F._significance_label(0.0004) == "***"


def test_distribution_and_null_show_separate_significance_contracts() -> None:
    rows = [
        {"subject_id": f"s{i}", "display_id": f"E{i}", "r": value}
        for i, value in enumerate(np.linspace(-0.9, 0.2, 8), start=1)
    ]
    fig, (ax_distribution, ax_null) = plt.subplots(2, 1)
    distribution = F._draw_distribution(
        ax_distribution, rows, sign_test_p=0.0037689
    )
    F._draw_null(
        ax_null,
        np.linspace(-0.3, 0.3, 1001),
        distribution["median"],
        p_negative=9.9999e-6,
    )
    distribution_text = " ".join(text.get_text() for text in ax_distribution.texts)
    null_text = " ".join(text.get_text() for text in ax_null.texts)
    assert "negative: **" in distribution_text
    assert "median r" not in distribution_text
    upper_diamond = next(
        collection for collection in ax_distribution.collections
        if np.any(np.isclose(collection.get_sizes(), 27.0))
    )
    lower_diamond = next(
        collection for collection in ax_null.collections
        if np.any(np.isclose(collection.get_sizes(), 28.0))
    )
    assert upper_diamond.get_offsets()[0, 0] == lower_diamond.get_offsets()[0, 0]
    assert "P=" not in distribution_text
    assert "***" in null_text
    assert "P_{perm}" not in null_text
    plt.close(fig)
