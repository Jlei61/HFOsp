from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper_figures import plot_fig3_raw_spectral_context as fig3ab


def test_compact_axis_preserves_only_baseline_and_onset_windows() -> None:
    times = np.arange(-120.0, 21.0, 1.0)
    segments = fig3ab._compressed_segments(times, fig3ab.MAIN_DISPLAY_WINDOWS)

    kept = np.concatenate([times[mask] for mask, _mapped in segments])
    assert np.all(((kept >= -110.0) & (kept <= -90.0)) | (kept >= -10.0))
    assert fig3ab._compressed_position(-90.0, fig3ab.MAIN_DISPLAY_WINDOWS) == 20.0
    assert fig3ab._compressed_position(-10.0, fig3ab.MAIN_DISPLAY_WINDOWS) == 34.0
    assert fig3ab._compressed_position(0.0, fig3ab.MAIN_DISPLAY_WINDOWS) == 44.0


def test_compact_axis_marks_the_omitted_interval_without_tick_overlap() -> None:
    fig, ax = plt.subplots()
    fig3ab._configure_compressed_time_axis(ax, fig3ab.MAIN_DISPLAY_WINDOWS)
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    break_markers = [
        line for line in ax.lines
        if line.get_gid() == "compressed-time-axis-break"
    ]

    assert labels == ["−110", "−100", "−90", "−10", "0", "10", "20"]
    assert len(break_markers) == 2
    assert all(20.0 < np.mean(line.get_xdata()) < 34.0 for line in break_markers)
    assert not any(text.get_text() == "⋯" for text in ax.texts)
    plt.close(fig)


def test_baseline_label_is_centered_in_visible_twenty_second_segment() -> None:
    fig, ax = plt.subplots()
    fig3ab._label_shaded_windows(
        ax,
        (-120.0, -90.0),
        None,
        (0.0, 10.0),
        display_windows=fig3ab.MAIN_DISPLAY_WINDOWS,
    )
    baseline_label = next(text for text in ax.texts if text.get_text() == "BASELINE")
    assert baseline_label.get_position()[0] == 10.0
    plt.close(fig)


def test_locked_examples_have_opposite_phenotypes() -> None:
    broadband = fig3ab._phenotype_row("epilepsiae_1146", 7)
    gamma = fig3ab._phenotype_row("epilepsiae_635", 7)

    assert broadband["simple_phenotype"] == "broadband_1_150"
    assert gamma["simple_phenotype"] == "gamma_nonbroadband"
    assert fig3ab.PHENOTYPE_COLORS["broadband_1_150"] == "#8D9FCD"
    assert fig3ab.PHENOTYPE_COLORS["gamma_nonbroadband"] == "#62BE9F"
    assert fig3ab.PHENOTYPE_LEGEND_LABELS == {
        "broadband_1_150": "Broadband",
        "gamma_nonbroadband": "Gamma",
    }


def test_band_axes_use_compact_db_label_and_black_onset_marker() -> None:
    fig, ax = plt.subplots()
    fig3ab._draw_clinical_onset_marker(ax)
    marker = next(
        line for line in ax.lines if line.get_gid() == "clinical-onset-marker"
    )

    assert fig3ab.BAND_YLABEL == "dB"
    assert list(marker.get_xdata()) == [0.0, 0.0]
    assert marker.get_color() == "black"
    assert marker.get_linestyle() == "--"
    plt.close(fig)


def test_panel_b_uses_continuous_physical_time_without_ellipsis() -> None:
    fig, ax = plt.subplots()
    fig3ab._configure_continuous_band_time_axis(ax, (-120.0, 20.0))

    assert ax.get_xlim() == (-120.0, 20.0)
    assert list(ax.get_xticks()) == [-120.0, -80.0, -40.0, 0.0, 20.0]
    assert not any(text.get_text() == "⋯" for text in ax.texts)
    plt.close(fig)


def test_identity_title_is_larger_than_interval_annotations() -> None:
    assert fig3ab.EXAMPLE_IDENTITY_TITLE_FONTSIZE > (
        fig3ab.SIGNAL_CONTEXT_TYPOGRAPHY.annotation
    )
