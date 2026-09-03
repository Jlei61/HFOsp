"""Panel-aware typography and layout rules for manuscript Figures 1--3.

The public sizes are visual reference sizes. Accepted panel layout remains
owned by the canonical producer; a bounded canvas-width correction keeps wider
composites from reverting to tiny source type. Actual atomic axes are measured
for QA without changing their rows, columns, or reading order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FigureTypography:
    """Reader-facing font hierarchy at the final display width."""

    panel_letter: float = 30.0
    identity_label: float = 28.0
    condition_label: float = 24.0
    axis_label: float = 26.0
    tick_label: float = 24.0
    legend: float = 24.0
    colorbar_label: float = 24.0
    colorbar_tick: float = 22.0
    annotation: float = 22.0
    significance: float = 24.0
    dense_tick: float = 20.0

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    def scaled(self, factor: float) -> "FigureTypography":
        """Return producer point sizes resolving to this hierarchy on export."""
        values = {
            key: round(float(value) * factor, 2)
            for key, value in asdict(self).items()
        }
        return FigureTypography(**values)


@dataclass(frozen=True)
class VisualTypographyPolicy:
    """Rules connecting producer geometry to final visible typography.

    ``final_display_width_in`` is the common width at which panels are judged.
    Font points on a wider producer canvas receive a bounded square-root width
    correction after the accepted layout is loaded. Atomic-axis sizes are
    measured at the common final-display width. Tick reduction, label sharing,
    and outer-panel allocation may change; rows and columns do not.
    """

    final_display_width_in: float = 7.4
    min_atomic_axis_width_in: float = 1.85
    min_atomic_axis_height_in: float = 1.35
    minimum_scale: float = 1.0
    maximum_scale: float = 1.25
    axis_label_pad_pt: float = 8.0
    tick_label_pad_pt: float = 3.0

    def as_dict(self) -> dict[str, float | int]:
        return {key: value for key, value in asdict(self).items()}


FINAL_MAIN_FIGURE_TYPOGRAPHY = FigureTypography()
FINAL_VISUAL_TYPOGRAPHY_POLICY = VisualTypographyPolicy()

# Producer-side profiles for panels that occupy only part of a complete figure.
# These values are intentionally lower than the full-width reference above:
# the compositor reduces each panel again, so applying the full-width 24--30 pt
# hierarchy to every source canvas makes dense ticks invade the data.  Profiles
# are selected by visual density, not by a figure/panel identifier.
DENSE_MULTIPANEL_TYPOGRAPHY = FigureTypography(
    panel_letter=24.0,
    identity_label=20.0,
    condition_label=18.0,
    axis_label=17.0,
    tick_label=14.0,
    legend=14.0,
    colorbar_label=15.0,
    colorbar_tick=13.0,
    annotation=14.0,
    significance=16.0,
    dense_tick=12.0,
)
COMPACT_STATISTICAL_TYPOGRAPHY = FigureTypography(
    panel_letter=26.0,
    identity_label=22.0,
    condition_label=20.0,
    axis_label=18.0,
    tick_label=16.0,
    legend=15.0,
    colorbar_label=16.0,
    colorbar_tick=14.0,
    annotation=15.0,
    significance=18.0,
    dense_tick=14.0,
)
ILLUSTRATIVE_PANEL_TYPOGRAPHY = FigureTypography(
    panel_letter=22.0,
    identity_label=16.0,
    condition_label=14.0,
    axis_label=12.0,
    tick_label=10.0,
    legend=10.0,
    colorbar_label=11.0,
    colorbar_tick=10.0,
    annotation=9.0,
    significance=12.0,
    dense_tick=9.0,
)
DENSE_COMPARISON_TYPOGRAPHY = FigureTypography(
    panel_letter=24.0,
    identity_label=18.0,
    condition_label=16.0,
    axis_label=16.0,
    tick_label=12.0,
    legend=11.0,
    colorbar_label=12.0,
    colorbar_tick=11.0,
    annotation=10.0,
    significance=15.0,
    dense_tick=9.0,
)
SHARED_FIELD_GRID_TYPOGRAPHY = FigureTypography(
    panel_letter=26.0,
    identity_label=24.0,
    condition_label=24.0,
    axis_label=20.0,
    tick_label=18.0,
    legend=18.0,
    colorbar_label=18.0,
    colorbar_tick=18.0,
    annotation=18.0,
    significance=20.0,
    dense_tick=15.0,
)
SIGNAL_CONTEXT_TYPOGRAPHY = FigureTypography(
    panel_letter=24.0,
    identity_label=18.0,
    condition_label=14.0,
    axis_label=14.0,
    tick_label=11.5,
    legend=11.0,
    colorbar_label=13.0,
    colorbar_tick=11.5,
    annotation=11.0,
    significance=14.0,
    dense_tick=9.5,
)
LOCKED_PANEL_TYPOGRAPHY_POLICY = VisualTypographyPolicy(
    minimum_scale=1.0,
    maximum_scale=1.0,
    axis_label_pad_pt=6.0,
    tick_label_pad_pt=2.5,
)


def resolve_typography_for_figure(
    fig: plt.Figure,
    *,
    spec: FigureTypography = FINAL_MAIN_FIGURE_TYPOGRAPHY,
    policy: VisualTypographyPolicy = FINAL_VISUAL_TYPOGRAPHY_POLICY,
) -> tuple[FigureTypography, float]:
    """Resolve producer point sizes from a bounded canvas-width correction."""
    # Canvas width influences source typography while subplot geometry remains
    # locked. The square-root response avoids the unusable 2x font jump of
    # direct full-canvas scaling while still compensating wider composites.
    raw_scale = sqrt(float(fig.get_figwidth()) / policy.final_display_width_in)
    scale = min(policy.maximum_scale, max(policy.minimum_scale, raw_scale))
    return spec.scaled(scale), scale


def measure_atomic_axes(
    fig: plt.Figure,
    axes: Iterable[plt.Axes],
    *,
    policy: VisualTypographyPolicy = FINAL_VISUAL_TYPOGRAPHY_POLICY,
) -> list[dict[str, float | bool]]:
    """Measure axes after reduction to the shared final display width."""
    fig.canvas.draw()
    reduction = policy.final_display_width_in / float(fig.get_figwidth())
    measurements: list[dict[str, float | bool]] = []
    for ax in axes:
        bbox = ax.get_position()
        width = float(bbox.width * fig.get_figwidth() * reduction)
        height = float(bbox.height * fig.get_figheight() * reduction)
        measurements.append(
            {
                "display_width_in": round(width, 3),
                "display_height_in": round(height, 3),
                "width_pass": width >= policy.min_atomic_axis_width_in,
                "height_pass": height >= policy.min_atomic_axis_height_in,
            }
        )
    return measurements


def _is_panel_letter(text: str) -> bool:
    value = text.strip()
    return len(value) == 1 and value in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _is_significance_text(text: str) -> bool:
    value = text.strip().lower()
    return value in {"*", "**", "***", "n.s.", "ns"}


def _apply_typography(
    fig: plt.Figure,
    *,
    spec: FigureTypography,
    dense_axes: Iterable[plt.Axes],
    colorbar_axes: Iterable[plt.Axes],
) -> None:
    dense_ids = {id(ax) for ax in dense_axes}
    colorbar_ids = {id(ax) for ax in colorbar_axes}

    for ax in fig.axes:
        is_colorbar = id(ax) in colorbar_ids or ax.get_label() == "<colorbar>"
        if is_colorbar:
            ax.tick_params(
                axis="both",
                labelsize=spec.colorbar_tick,
                width=1.2,
                length=5.0,
            )
            ax.xaxis.label.set_fontsize(spec.colorbar_label)
            ax.yaxis.label.set_fontsize(spec.colorbar_label)
            ax.title.set_fontsize(spec.colorbar_label)
        else:
            tick_size = spec.dense_tick if id(ax) in dense_ids else spec.tick_label
            ax.tick_params(
                axis="both", labelsize=tick_size, width=1.3, length=6.0
            )
            ax.xaxis.label.set_fontsize(spec.axis_label)
            ax.yaxis.label.set_fontsize(spec.axis_label)
            # Axis titles are reserved for short identity/condition headers;
            # narrative subplot titles belong in the figure legend.
            ax.title.set_fontsize(spec.condition_label)

        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(max(1.2, float(spine.get_linewidth())))

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(spec.legend)
            if legend.get_title() is not None:
                legend.get_title().set_fontsize(spec.legend)
            for line in legend.get_lines():
                line.set_linewidth(max(3.0, float(line.get_linewidth())))

        for text in ax.texts:
            value = text.get_text()
            if _is_panel_letter(value):
                text.set_fontsize(spec.panel_letter)
                text.set_fontweight("bold")
            elif _is_significance_text(value):
                text.set_fontsize(spec.significance)
                text.set_fontweight("bold")
            else:
                text.set_fontsize(max(float(text.get_fontsize()), spec.annotation))

    for text in fig.texts:
        value = text.get_text()
        if _is_panel_letter(value):
            text.set_fontsize(spec.panel_letter)
            text.set_fontweight("bold")
        else:
            text.set_fontsize(max(float(text.get_fontsize()), spec.annotation))


def apply_final_figure_typography(
    fig: plt.Figure,
    *,
    spec: FigureTypography = FINAL_MAIN_FIGURE_TYPOGRAPHY,
    dense_axes: Iterable[plt.Axes] = (),
    colorbar_axes: Iterable[plt.Axes] = (),
) -> None:
    """Apply the reference hierarchy directly (legacy compatibility helper)."""
    _apply_typography(
        fig, spec=spec, dense_axes=dense_axes, colorbar_axes=colorbar_axes
    )


def apply_panel_aware_figure_typography(
    fig: plt.Figure,
    *,
    spec: FigureTypography = FINAL_MAIN_FIGURE_TYPOGRAPHY,
    policy: VisualTypographyPolicy = FINAL_VISUAL_TYPOGRAPHY_POLICY,
    dense_axes: Iterable[plt.Axes] = (),
    colorbar_axes: Iterable[plt.Axes] = (),
    enforce_atomic_axis_gate: bool = True,
) -> dict[str, object]:
    """Apply visually normalized type and return auditable layout diagnostics."""
    dense_axes = tuple(dense_axes)
    colorbar_axes = tuple(colorbar_axes)
    resolved, scale = resolve_typography_for_figure(fig, spec=spec, policy=policy)
    _apply_typography(
        fig, spec=resolved, dense_axes=dense_axes, colorbar_axes=colorbar_axes
    )
    for ax in fig.axes:
        ax.xaxis.labelpad = max(
            float(ax.xaxis.labelpad), policy.axis_label_pad_pt * scale
        )
        ax.yaxis.labelpad = max(
            float(ax.yaxis.labelpad), policy.axis_label_pad_pt * scale
        )
        ax.tick_params(
            axis="both",
            pad=policy.tick_label_pad_pt * scale,
        )
    atomic_axes = [ax for ax in fig.axes if ax not in colorbar_axes]
    measurements = measure_atomic_axes(fig, atomic_axes, policy=policy)
    axis_gate_result = (
        all(
            bool(row["width_pass"]) and bool(row["height_pass"])
            for row in measurements
        )
        if enforce_atomic_axis_gate
        else None
    )
    return {
        "canvas_width_in": float(fig.get_figwidth()),
        "canvas_height_in": float(fig.get_figheight()),
        "canvas_to_display_font_scale": round(scale, 3),
        "reference_final_display_typography_pt": spec.as_dict(),
        "resolved_producer_typography_pt": resolved.as_dict(),
        "atomic_axes": measurements,
        "axis_size_gate_enforced": bool(enforce_atomic_axis_gate),
        "all_atomic_axes_pass": axis_gate_result,
    }


__all__ = [
    "FigureTypography",
    "VisualTypographyPolicy",
    "FINAL_MAIN_FIGURE_TYPOGRAPHY",
    "FINAL_VISUAL_TYPOGRAPHY_POLICY",
    "DENSE_MULTIPANEL_TYPOGRAPHY",
    "COMPACT_STATISTICAL_TYPOGRAPHY",
    "ILLUSTRATIVE_PANEL_TYPOGRAPHY",
    "DENSE_COMPARISON_TYPOGRAPHY",
    "SHARED_FIELD_GRID_TYPOGRAPHY",
    "SIGNAL_CONTEXT_TYPOGRAPHY",
    "LOCKED_PANEL_TYPOGRAPHY_POLICY",
    "apply_final_figure_typography",
    "apply_panel_aware_figure_typography",
    "measure_atomic_axes",
    "resolve_typography_for_figure",
]
