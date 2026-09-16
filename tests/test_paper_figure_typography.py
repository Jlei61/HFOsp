import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.paper_figure_typography import (
    COMPACT_STATISTICAL_TYPOGRAPHY,
    DENSE_COMPARISON_TYPOGRAPHY,
    DENSE_MULTIPANEL_TYPOGRAPHY,
    FINAL_MAIN_FIGURE_TYPOGRAPHY,
    FINAL_VISUAL_TYPOGRAPHY_POLICY,
    LOCKED_PANEL_TYPOGRAPHY_POLICY,
    apply_final_figure_typography,
    apply_panel_aware_figure_typography,
    resolve_typography_for_figure,
)


def test_final_main_figure_typography_contract_has_no_sub_20pt_text() -> None:
    spec = FINAL_MAIN_FIGURE_TYPOGRAPHY
    assert min(spec.as_dict().values()) == 20.0
    assert spec.tick_label == spec.legend == 24.0
    assert spec.axis_label == 26.0
    assert spec.identity_label == 28.0
    assert spec.condition_label == 24.0
    assert spec.panel_letter == 30.0


def test_apply_final_figure_typography_updates_axes_legend_and_annotations() -> None:
    spec = FINAL_MAIN_FIGURE_TYPOGRAPHY
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="signal")
    ax.set_title("Panel title")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    significance = ax.text(0.5, 0.9, "***")
    annotation = ax.text(0.5, 0.8, "n = 12")
    ax.legend()

    apply_final_figure_typography(fig)

    assert ax.title.get_fontsize() == spec.condition_label
    assert ax.xaxis.label.get_fontsize() == spec.axis_label
    assert ax.yaxis.label.get_fontsize() == spec.axis_label
    assert all(label.get_fontsize() == spec.tick_label for label in ax.get_xticklabels())
    assert all(text.get_fontsize() == spec.legend for text in ax.get_legend().get_texts())
    assert significance.get_fontsize() == spec.significance
    assert annotation.get_fontsize() == spec.annotation
    plt.close(fig)


def test_dense_axis_uses_only_the_locked_20pt_exception() -> None:
    spec = FINAL_MAIN_FIGURE_TYPOGRAPHY
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    apply_final_figure_typography(fig, dense_axes=[ax])
    assert all(label.get_fontsize() == spec.dense_tick for label in ax.get_yticklabels())
    plt.close(fig)


def test_wide_canvas_resolves_larger_producer_fonts_for_equal_final_display() -> None:
    spec = FINAL_MAIN_FIGURE_TYPOGRAPHY
    policy = FINAL_VISUAL_TYPOGRAPHY_POLICY
    fig, ax = plt.subplots(figsize=(2 * policy.final_display_width_in, 5.0))
    ax.plot([0, 1], [0, 1], label="signal")
    ax.legend()
    resolved, scale = resolve_typography_for_figure(fig)
    diagnostics = apply_panel_aware_figure_typography(fig)

    assert scale == policy.maximum_scale
    assert resolved.tick_label == policy.maximum_scale * spec.tick_label
    assert all(label.get_fontsize() == resolved.tick_label for label in ax.get_xticklabels())
    assert diagnostics["canvas_to_display_font_scale"] == policy.maximum_scale
    plt.close(fig)


def test_locked_layout_can_report_axes_without_triggering_a_reflow_gate() -> None:
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.plot([0, 1], [0, 1])
    diagnostics = apply_panel_aware_figure_typography(
        fig,
        enforce_atomic_axis_gate=False,
    )
    assert diagnostics["axis_size_gate_enforced"] is False
    assert diagnostics["all_atomic_axes_pass"] is None
    assert len(diagnostics["atomic_axes"]) == 1
    plt.close(fig)


def test_partial_figure_profiles_are_density_based_and_do_not_canvas_scale() -> None:
    assert DENSE_MULTIPANEL_TYPOGRAPHY.dense_tick < COMPACT_STATISTICAL_TYPOGRAPHY.tick_label
    assert DENSE_MULTIPANEL_TYPOGRAPHY.axis_label > DENSE_MULTIPANEL_TYPOGRAPHY.dense_tick
    assert DENSE_COMPARISON_TYPOGRAPHY.dense_tick < DENSE_COMPARISON_TYPOGRAPHY.tick_label
    fig, ax = plt.subplots(figsize=(11.6, 3.5))
    resolved, scale = resolve_typography_for_figure(
        fig,
        spec=DENSE_MULTIPANEL_TYPOGRAPHY,
        policy=LOCKED_PANEL_TYPOGRAPHY_POLICY,
    )
    assert scale == 1.0
    assert resolved == DENSE_MULTIPANEL_TYPOGRAPHY
    plt.close(fig)
