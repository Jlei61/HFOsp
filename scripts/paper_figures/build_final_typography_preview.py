#!/usr/bin/env python3
"""Build non-canonical typography previews for representative Fig. 1--3 panels.

The preview reuses accepted data loaders and painters.  It changes only canvas
space, text hierarchy, and legend placement so the author can approve the
final readability scale before it is propagated to every canonical panel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures import (  # noqa: E402
    plot_fig1_interictal_hfo_temporal_scaffold as fig1,
    plot_fig2_shared_field_reversal_row as fig2ef,
    plot_fig3_peri_onset_field_similarity as fig3e,
)
from scripts.plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    DEFAULT_DISPLAY_SIGMA_MM,
    TA_COLOR,
    TB_COLOR,
    _load_yuquan_crosswalk,
    build_interictal_ab_panel_payloads,
    draw_interictal_rank_field_panel,
)
from src.paper_figure_typography import (  # noqa: E402
    FINAL_MAIN_FIGURE_TYPOGRAPHY,
    FINAL_VISUAL_TYPOGRAPHY_POLICY,
    apply_panel_aware_figure_typography,
)


OUT_ROOT = ROOT / "results/paper-ready-figure/typography-preview-fig1-3"
FIGURES = OUT_ROOT / "figures"
FIG3E_SOURCE = (
    ROOT
    / "results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages"
    / "fig3_peri_onset_field_similarity/runs/20260718T071020Z_d99c96ec"
    / "artifacts/field_dynamics_signed"
    / "epilepsiae_1146_signed_broadband_1_150Hz_"
    "similarity_timecourse_m120_p20_10s_step2s_per_seizure.csv"
)
DIAGNOSTICS: dict[str, object] = {}


def _save(fig: plt.Figure, stem: str) -> list[str]:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))]


def _build_fig1d_preview() -> list[str]:
    """Statistics example: large ticks/labels and significance annotation."""
    fig1.propagation_plot._apply_masked_paths()
    records = fig1._load_temporal_records()
    if len(records) != 40:
        raise ValueError(f"expected 40 masked temporal records, found {len(records)}")
    fig1._assert_masked_mi_records(records)

    fig, ax = plt.subplots(figsize=(7.4, 6.2), facecolor="white")
    fig1._plot_mi(ax, records)
    for text in ax.texts:
        if text.get_text() in {"Yuquan", "Epilepsiae"}:
            text.set_y(-0.235)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.31, top=0.88)
    DIAGNOSTICS["fig1d"] = apply_panel_aware_figure_typography(fig)
    return _save(fig, "fig1-paneld-final-typography-preview")


def _build_fig2e_preview() -> list[str]:
    """Dense field maps with the accepted four-column/two-row layout locked."""
    labels = _load_yuquan_crosswalk(fig2ef.DEFAULT_YUQUAN_CROSSWALK)
    rows = fig2ef.load_shared_field_rows(fig2ef.INPUT_ROOT, yuquan_labels=labels)
    examples = fig2ef.select_examples(rows)
    # Same aspect, rows, columns, reading order, and shared-colorbar placement
    # as the accepted canonical Fig2E. Only the producer canvas and type are
    # enlarged for visual QA.
    fig = plt.figure(figsize=(12.3, 6.1), facecolor="white")
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 0.065),
        left=0.16,
        right=0.92,
        top=0.91,
        bottom=0.16,
        wspace=0.32,
        hspace=0.28,
    )
    field_axes: list[plt.Axes] = []
    for column, row in enumerate(examples):
        dat_a, dat_b, mode = build_interictal_ab_panel_payloads(
            row["record"], display_sigma_mm=DEFAULT_DISPLAY_SIGMA_MM,
        )
        if mode != "shared":
            raise ValueError(f"example {row['subject_id']} is not on a shared plane")
        fig2ef.apply_common_display_window(dat_a, dat_b)
        ax_a = fig.add_subplot(grid[0, column])
        ax_b = fig.add_subplot(grid[1, column], sharex=ax_a, sharey=ax_a)
        draw_interictal_rank_field_panel(
            ax_a,
            dat_a,
            "TA",
            compact=True,
            panel_title=str(row["display_id"]),
            contact_outline_lw=1.2,
            contact_size=58,
            show_template_tag=False,
        )
        draw_interictal_rank_field_panel(
            ax_b,
            dat_b,
            "TB",
            compact=True,
            contact_outline_lw=1.2,
            contact_size=58,
            show_template_tag=False,
        )
        ax_a.set_title(str(row["display_id"]), fontweight="bold", pad=6)
        ax_b.set_title("")
        fig2ef._restore_compact_axis_ticks(ax_a)
        fig2ef._restore_compact_axis_ticks(ax_b)
        if column == 0:
            ax_a.set_ylabel("y (mm)")
            ax_b.set_ylabel("y (mm)")
        else:
            # All eight maps use the same physical display window. Repeating
            # y tick labels in every column adds no information and makes the
            # enlarged labels collide with the preceding map.
            ax_a.tick_params(axis="y", labelleft=False)
            ax_b.tick_params(axis="y", labelleft=False)
        field_axes.extend((ax_a, ax_b))

    cbar_ax = fig.add_subplot(grid[:, 4])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
        cax=cbar_ax,
    )
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["0 (early)", "1 (late)"])
    fig.canvas.draw()
    field_left = min(ax.get_position().x0 for ax in field_axes)
    field_right = max(ax.get_position().x1 for ax in field_axes)
    field_bottom = min(ax.get_position().y0 for ax in field_axes)
    field_top = max(ax.get_position().y1 for ax in field_axes)
    first_ta = field_axes[0].get_position()
    first_tb = field_axes[1].get_position()
    cbar_pos = cbar_ax.get_position()
    cbar_ax.set_position(
        [cbar_pos.x0, field_bottom, cbar_pos.width, field_top - field_bottom]
    )
    fig.text(
        0.5 * (field_left + field_right),
        0.035,
        "Shared TA axis (mm)",
        ha="center",
        va="bottom",
    )
    fig.text(
        0.025,
        0.5 * (first_ta.y0 + first_ta.y1),
        "TA field",
        ha="center",
        va="center",
        rotation=90,
        fontweight="bold",
        color=TA_COLOR,
    )
    fig.text(
        0.025,
        0.5 * (first_tb.y0 + first_tb.y1),
        "TB field",
        ha="center",
        va="center",
        rotation=90,
        fontweight="bold",
        color=TB_COLOR,
    )
    diagnostics = apply_panel_aware_figure_typography(
        fig,
        dense_axes=field_axes,
        colorbar_axes=[cbar_ax],
        enforce_atomic_axis_gate=False,
    )
    for ax in field_axes:
        ax.tick_params(axis="x", pad=3.0)
    for ax in (field_axes[0], field_axes[1]):
        ax.tick_params(axis="y", pad=3.0)
    diagnostics["layout_lock"] = {
        "status": "accepted_layout_preserved",
        "subject_columns": 4,
        "field_rows": 2,
        "row_order": ["TA", "TB"],
        "shared_colorbar": "right",
    }
    diagnostics["atomic_axis_count"] = len(field_axes)
    DIAGNOSTICS["fig2e"] = diagnostics
    return _save(fig, "fig2-panele-final-typography-preview")


def _rebuild_legend_above(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    handles, labels = ax.get_legend_handles_labels()
    legend.remove()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=max(1, len(labels)),
        frameon=False,
        handlelength=1.7,
        columnspacing=1.1,
    )


def _build_fig3e_preview() -> list[str]:
    """Line/legend example: legends move into reserved top whitespace."""
    if not FIG3E_SOURCE.exists():
        raise FileNotFoundError(FIG3E_SOURCE)
    df = fig3e._load_peri_onset(
        FIG3E_SOURCE,
        "epilepsiae_1146",
        readout=fig3e.READOUT_SIMILARITY,
    )
    agg = fig3e._agg(df, readout=fig3e.READOUT_SIMILARITY)
    fig = fig3e._make_figure(
        df,
        agg,
        subject_label="E1146",
        design_variant=fig3e.DESIGN_JOURNAL_CLEAN,
        readout=fig3e.READOUT_SIMILARITY,
    )
    fig.set_size_inches(14.0, 5.6, forward=True)
    for ax in fig.axes:
        _rebuild_legend_above(ax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.24, top=0.72, wspace=0.42)
    DIAGNOSTICS["fig3e"] = apply_panel_aware_figure_typography(fig)
    return _save(fig, "fig3-panele-final-typography-preview")


def _fit(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image.copy()
    height = int(round(image.height * max_width / image.width))
    resampling = getattr(Image, "Resampling", Image)
    return image.resize((max_width, height), resampling.LANCZOS)


def _vertical_comparison(
    pairs: list[tuple[str, Path, Path]],
    *,
    filename: str,
) -> str:
    """Stack current/preview at full width so dense previews are not re-shrunk."""
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    heading_font = ImageFont.truetype(str(font_path), 50)
    label_font = ImageFont.truetype(str(font_path), 42)
    canvas_width = 2700
    margin = 110
    image_width = canvas_width - 2 * margin
    top_gap = 90
    label_height = 70
    image_gap = 80
    section_gap = 150
    prepared: list[tuple[str, Image.Image, Image.Image]] = []
    for title, old_path, new_path in pairs:
        with Image.open(old_path) as old_image, Image.open(new_path) as new_image:
            old = _fit(old_image.convert("RGB"), image_width)
            new = _fit(new_image.convert("RGB"), image_width)
        prepared.append((title, old, new))
    height = top_gap
    for _, old, new in prepared:
        height += 3 * label_height + old.height + image_gap + new.height + section_gap
    canvas = Image.new("RGB", (canvas_width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = top_gap
    for title, old, new in prepared:
        draw.text((margin, y), title, fill="#222222", font=heading_font, anchor="la")
        y += label_height
        draw.text((margin, y), "Current", fill="#555555", font=label_font, anchor="la")
        y += label_height
        canvas.paste(old, (margin, y))
        y += old.height + image_gap
        draw.text(
            (margin, y),
            "Typography-only preview",
            fill="#555555",
            font=label_font,
            anchor="la",
        )
        y += label_height
        canvas.paste(new, (margin, y))
        y += new.height + section_gap
    path = FIGURES / filename
    canvas.crop((0, 0, canvas_width, y)).save(path, dpi=(200, 200), optimize=True)
    return str(path.relative_to(ROOT))


def _build_comparison_boards(new_paths: dict[str, Path]) -> list[str]:
    current = {
        "Figure 1D": ROOT / "results/paper-ready-figure/fig1/figures/fig1-paneld.png",
        "Figure 2E": ROOT / "results/paper-ready-figure/fig2/figures/fig2-panele.png",
        "Figure 3E": ROOT / "results/paper-ready-figure/fig3/figures/fig3-panele.png",
    }
    all_pairs = [(label, current[label], new_paths[label]) for label in current]
    all_board = _vertical_comparison(
        all_pairs,
        filename="fig1-3-current-vs-final-typography-preview.png",
    )
    fig2_board = _vertical_comparison(
        [("Figure 2E", current["Figure 2E"], new_paths["Figure 2E"])],
        filename="fig2e-current-vs-panel-aware-preview.png",
    )
    return [all_board, fig2_board]


def _write_readme(outputs: dict[str, object]) -> None:
    spec = FINAL_MAIN_FIGURE_TYPOGRAPHY.as_dict()
    (FIGURES / "README.md").write_text(
        f"""# Figure 1--3 最终字号预览

本目录是视觉验收 preview，不替换 `fig1/fig2/fig3` canonical 输出。三张样例分别覆盖统计括号、密集场图与 colorbar、以及带 legend 的时程图；数据、统计、坐标和配色均复用已验收 producer。

### fig1-3-current-vs-final-typography-preview.png

每个 current / preview 都占满同一显示宽度并上下排列，避免多轴 panel 在右列被再次压缩。

**关注点**：先判断 tick、axis label、统计标记和 legend 是否已经达到无需放大即可阅读；本轮不据此改变科学合同。

### fig2e-current-vs-panel-aware-preview.png

Figure 2E 的 current / typography-only preview 全宽上下对照。比较板只改变展示方式；panel 本身严格保持 canonical 排版。

**关注点**：四列受试者 × 两行 TA/TB 的既有排版必须保持不变；只比较字号与留白。

### fig1-paneld-final-typography-preview.png / .pdf

Figure 1D 的 masked MI data-vs-null 统计图。保留 violin、box/IQR、whisker、逐患者点与显著性括号。

**关注点**：24 pt ticks、26 pt axis label、28 pt title 和 24 pt 显著性是否过大或合适。

### fig2-panele-final-typography-preview.png / .pdf

Figure 2E 的四例 TA/TB shared-axis field。严格保留 canonical 的四列受试者 × 两行 TA/TB、右侧共享 colorbar 和原阅读顺序；字号按画布宽度做有界换算。

**关注点**：不重新排版；第二至第四列隐藏重复 y tick labels，列/行间距只从 0.27/0.22 微调到 0.32/0.28，避免放大后的 ticks 覆盖相邻 field。

### fig3-panele-final-typography-preview.png / .pdf

Figure 3E 的 field-similarity trajectory。两组 legend 移到各自 axes 上方的预留空白，不再压住数据。

**关注点**：24 pt legend 与 ticks、26 pt axis labels 是否平衡，顶端 legend 是否比图内 legend 更易读。

## 字号合同（最终显示参考 points）

- panel letter / identity label: {spec['panel_letter']:.0f} / {spec['identity_label']:.0f}
- condition / quantity header: {spec['condition_label']:.0f}
- axis label: {spec['axis_label']:.0f}
- ticks / legend: {spec['tick_label']:.0f} / {spec['legend']:.0f}
- colorbar label / ticks: {spec['colorbar_label']:.0f} / {spec['colorbar_tick']:.0f}
- annotation / significance: {spec['annotation']:.0f} / {spec['significance']:.0f}
- dense channel/contact ticks only: {spec['dense_tick']:.0f}（绝不低于此值）

producer 不逐图硬编码这组数值。对宽画布使用有上限的平方根宽度补偿；metadata 记录解析字号、每个原子轴的最终显示尺寸和 layout-lock。已验收排版不因字号调整而换行或换列；需要更多空间时只调整整个 panel 在 complete layout 中的分配面积、留白或重复 tick/label。
""",
        encoding="utf-8",
    )


def build() -> dict[str, object]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fig1d": _build_fig1d_preview(),
        "fig2e": _build_fig2e_preview(),
        "fig3e": _build_fig3e_preview(),
    }
    new_paths = {
        "Figure 1D": ROOT / outputs["fig1d"][0],
        "Figure 2E": ROOT / outputs["fig2e"][0],
        "Figure 3E": ROOT / outputs["fig3e"][0],
    }
    outputs["comparison_boards"] = _build_comparison_boards(new_paths)
    outputs["typography"] = {
        "reference_final_display_pt": FINAL_MAIN_FIGURE_TYPOGRAPHY.as_dict(),
        "policy": FINAL_VISUAL_TYPOGRAPHY_POLICY.as_dict(),
        "resolved_diagnostics": DIAGNOSTICS,
    }
    _write_readme(outputs)
    metadata = OUT_ROOT / "typography_preview_metadata.json"
    metadata.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    print(json.dumps(build(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
