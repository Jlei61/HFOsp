from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.paper_figures import plot_fig3b_interictal_ictal_shared_field as fig3c


def test_rank_colorbar_matches_fig2_normalized_rank_grammar() -> None:
    fig, cax = plt.subplots(figsize=(1.0, 3.0))
    colorbar, raw_range = fig3c._normalized_rank_colorbar(
        fig, cax, np.array([0.0, 2.0, 7.0])
    )
    try:
        assert raw_range == pytest.approx((0.0, 7.0))
        assert colorbar.get_ticks() == pytest.approx([0.0, 0.5, 1.0])
        assert [tick.get_text() for tick in colorbar.ax.get_yticklabels()] == [
            "0  early",
            "0.5",
            "1  late",
        ]
        assert colorbar.ax.get_title(loc="left") == "ranks"
    finally:
        plt.close(fig)


def test_spatial_ylabel_is_plain_y() -> None:
    source = open(fig3c.__file__, encoding="utf-8").read()
    assert 'ax.set_ylabel("Y (mm)"' in source
    assert "transverse (mm)" not in source


def test_panel_c_has_no_global_title_and_uses_compact_right_labels() -> None:
    source = open(fig3c.__file__, encoding="utf-8").read()
    assert "fig.suptitle(" not in source
    assert fig3c.ICTAL_FIELD_TITLE == "Early ictal field"
    assert fig3c.POWER_COLORBAR_TITLE == "power\nz"
