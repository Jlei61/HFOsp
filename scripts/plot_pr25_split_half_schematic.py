"""PR-2.5 split-half / odd-even reproducibility schematic for PPT use.

Single-panel methods illustration explaining how
`compute_time_split_reproducibility()` in `src/interictal_propagation.py`
works. Not a result figure — purely didactic.

Output: results/interictal_propagation/figures/ppt/
        fig_pr25_split_half_schematic.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.font_manager import FontProperties

from src.plot_style import (
    COL_EPILEPSIAE,
    COL_NEUTRAL,
    COL_SIG,
    COL_YUQUAN,
    DPI_PUB,
)


# Chinese font: matplotlib indexes the Noto Sans CJK .ttc under the JP family
# name, but glyphs for common Hanzi are shared across SC/TC/JP/KR, so this
# renders simplified Chinese correctly.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# Noto Sans CJK has no italic variant; disable italic globally so CJK text
# does not silently fall back to a glyph-less DejaVu Sans Italic.
plt.rcParams["font.style"] = "normal"

COL_A = COL_YUQUAN          # half A — blue
COL_B = COL_EPILEPSIAE      # half B — terracotta
COL_OUT = COL_SIG           # output / anti-corr emphasis
COL_BG = "#F2EFEA"          # very light Morandi cream
COL_BORDER = "#3F3F3F"

# Figure-space coordinates: (0..16) x (0..10)
FIG_W, FIG_H = 16.0, 10.0


def rounded_box(ax, x, y, w, h, facecolor, edgecolor=COL_BORDER, lw=1.2, alpha=1.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha,
    )
    ax.add_patch(box)
    return box


def arrow(ax, x0, y0, x1, y1, color=COL_BORDER, lw=1.6):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=15,
        color=color, linewidth=lw, zorder=5,
    )
    ax.add_patch(a)


def step_header(ax, x, y, text):
    ax.text(
        x, y, text,
        fontsize=15, fontweight="bold", color=COL_BORDER,
        ha="left", va="center",
    )


def tier_band(ax, y_bottom, h, color=COL_BG):
    ax.add_patch(Rectangle(
        (0.15, y_bottom), FIG_W - 0.3, h,
        facecolor=color, edgecolor="none", alpha=0.55, zorder=0,
    ))


def main():
    out_dir = Path("results/interictal_propagation/figures/ppt")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        FIG_W / 2, FIG_H - 0.35,
        "PR-2.5  Split-half / Odd-even template reproducibility",
        fontsize=18, fontweight="bold", ha="center", va="center",
        color=COL_BORDER,
    )
    ax.text(
        FIG_W / 2, FIG_H - 0.85,
        "把同一个 subject 的事件切两半，各自重跑 KMeans，看模板是否一致",
        fontsize=12, ha="center", va="center",
        color="#555555",
    )

    # ----------------------------------------------------------------
    # Tier 1 (y=7.0 .. 8.6) — Step 1: two split modes
    # ----------------------------------------------------------------
    tier_band(ax, 6.95, 1.75)
    step_header(ax, 0.35, 8.4, "Step 1 · 切两半（两种切法各做一次）")

    # (a) split-half by event time
    ax.text(0.55, 7.9, "(a) 按事件时间排序", fontsize=11, fontweight="bold", color=COL_BORDER)
    # event dots
    n_dots = 24
    for i in range(n_dots):
        cx = 0.55 + i * 0.20
        col = COL_A if i < n_dots // 2 else COL_B
        ax.plot(cx, 7.45, marker="o", markersize=7, color=col, markeredgecolor=COL_BORDER, markeredgewidth=0.5)
    # arrows under
    mid_x = 0.55 + (n_dots // 2 - 0.5) * 0.20
    ax.annotate("", xy=(0.5, 7.1), xytext=(mid_x, 7.1),
                arrowprops=dict(arrowstyle="<->", color=COL_A, lw=1.2))
    ax.annotate("", xy=(mid_x, 7.1), xytext=(0.55 + (n_dots - 1) * 0.20, 7.1),
                arrowprops=dict(arrowstyle="<->", color=COL_B, lw=1.2))
    ax.text((0.55 + mid_x) / 2, 6.85, "first half (A)", fontsize=10, ha="center", color=COL_A, fontweight="bold")
    ax.text((mid_x + 0.55 + (n_dots - 1) * 0.20) / 2, 6.85, "second half (B)", fontsize=10, ha="center", color=COL_B, fontweight="bold")
    ax.text(0.55 + n_dots * 0.20 + 0.1, 7.45, "→ t", fontsize=10, color=COL_BORDER, va="center")

    # (b) odd/even block split
    ax.text(6.6, 7.9, "(b) 按 1h block 奇偶", fontsize=11, fontweight="bold", color=COL_BORDER)
    n_blocks = 12
    block_w = 0.45
    block_x0 = 6.6
    for i in range(n_blocks):
        bx = block_x0 + i * block_w
        col = COL_A if i % 2 == 0 else COL_B
        ax.add_patch(Rectangle((bx, 7.3), block_w * 0.94, 0.4,
                               facecolor=col, edgecolor=COL_BORDER, linewidth=0.8, alpha=0.85))
        ax.text(bx + block_w * 0.47, 7.5, f"{i+1}", fontsize=8, ha="center", va="center", color="white", fontweight="bold")
    ax.text(block_x0, 7.05, "奇数 block → A", fontsize=10, color=COL_A, fontweight="bold")
    ax.text(block_x0 + n_blocks * block_w * 0.5, 7.05, "偶数 block → B", fontsize=10, color=COL_B, fontweight="bold")
    ax.text(block_x0 + 0.1, 6.75, "防止 A=白天 / B=夜晚 这种混杂", fontsize=9, color="#666666")

    # ----------------------------------------------------------------
    # Tier 2 (y=4.6 .. 6.5) — Step 2 + Step 3
    # ----------------------------------------------------------------
    tier_band(ax, 4.55, 2.0)
    step_header(ax, 0.35, 6.35, "Step 2 · 每一半独立跑 KMeans (k = stable_k)         Step 3 · 匈牙利匹配（Spearman 相关）")

    # KMeans box A (left)
    rounded_box(ax, 0.5, 4.85, 4.4, 1.25, facecolor="white", edgecolor=COL_A, lw=2.0)
    ax.text(2.7, 5.9, "Half A · KMeans", fontsize=11, fontweight="bold", color=COL_A, ha="center")
    # template chips
    template_labels = ["T_A1", "T_A2", "T_A3"]
    for i, lbl in enumerate(template_labels):
        cx = 1.0 + i * 1.3
        ax.add_patch(FancyBboxPatch((cx, 5.0), 1.05, 0.55,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    facecolor=COL_A, edgecolor=COL_BORDER, lw=0.8, alpha=0.85))
        ax.text(cx + 0.52, 5.28, lbl, fontsize=11, ha="center", va="center", color="white", fontweight="bold")

    # KMeans box B (right)
    rounded_box(ax, 10.6, 4.85, 4.4, 1.25, facecolor="white", edgecolor=COL_B, lw=2.0)
    ax.text(12.8, 5.9, "Half B · KMeans", fontsize=11, fontweight="bold", color=COL_B, ha="center")
    template_labels_b = ["T_B1", "T_B2", "T_B3"]
    for i, lbl in enumerate(template_labels_b):
        cx = 11.1 + i * 1.3
        ax.add_patch(FancyBboxPatch((cx, 5.0), 1.05, 0.55,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    facecolor=COL_B, edgecolor=COL_BORDER, lw=0.8, alpha=0.85))
        ax.text(cx + 0.52, 5.28, lbl, fontsize=11, ha="center", va="center", color="white", fontweight="bold")

    # Hungarian matching middle box
    rounded_box(ax, 5.3, 4.85, 5.0, 1.25, facecolor="white", edgecolor=COL_BORDER, lw=1.6)
    ax.text(7.8, 5.9, "Hungarian on Spearman r", fontsize=11, fontweight="bold", color=COL_BORDER, ha="center")
    # show three matched pairs
    pairs = [("T_A1", "T_B2"), ("T_A2", "T_B3"), ("T_A3", "T_B1")]
    for i, (a, b) in enumerate(pairs):
        y_row = 5.55 - i * 0.22
        ax.text(5.55, y_row, a, fontsize=10, color=COL_A, fontweight="bold", va="center")
        ax.annotate("", xy=(7.5, y_row), xytext=(6.2, y_row),
                    arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.0))
        ax.text(6.85, y_row + 0.10, "r", fontsize=8, color="#666666", ha="center")
        ax.text(7.65, y_row, b, fontsize=10, color=COL_B, fontweight="bold", va="center")
    ax.text(9.8, 4.95, "→ 1-1 配对", fontsize=9, color="#666666", ha="right")

    # arrows A -> matching -> B
    arrow(ax, 4.9, 5.45, 5.3, 5.45, color=COL_A, lw=1.6)
    arrow(ax, 10.6, 5.45, 10.3, 5.45, color=COL_B, lw=1.6)

    # ----------------------------------------------------------------
    # Tier 3 (y=2.15 .. 4.2) — Step 4 outputs
    # ----------------------------------------------------------------
    tier_band(ax, 2.1, 2.15)
    step_header(ax, 0.35, 4.0, "Step 4 · 两个读数")

    # Box 4a: template match corr
    rounded_box(ax, 0.5, 2.35, 7.0, 1.45, facecolor="white", edgecolor=COL_OUT, lw=2.0)
    ax.text(0.85, 3.55, "(4a)  template match corr", fontsize=12, fontweight="bold", color=COL_OUT)
    ax.text(0.85, 3.20,
            "配对后每对模板的 Spearman 相关，取中位",
            fontsize=10.5, color=COL_BORDER)
    ax.text(0.85, 2.92,
            r"$\mathrm{median}_{i}\ \rho\,(T_{A_i},\ T_{B_{\sigma(i)}})$",
            fontsize=12, color=COL_BORDER)
    ax.text(0.85, 2.55,
            "→ 两半画出的模板形状有多像",
            fontsize=10, color="#555555")

    # Box 4b: assignment agreement
    rounded_box(ax, 8.0, 2.35, 7.5, 1.45, facecolor="white", edgecolor=COL_OUT, lw=2.0)
    ax.text(8.35, 3.55, "(4b)  assignment agreement", fontsize=12, fontweight="bold", color=COL_OUT)
    ax.text(8.35, 3.20,
            "用 A 的模板给 B 的每个事件贴标签，",
            fontsize=10.5, color=COL_BORDER)
    ax.text(8.35, 2.92,
            "看是否对得上 B 自己 KMeans 的标签",
            fontsize=10.5, color=COL_BORDER)
    ax.text(8.35, 2.55,
            "→ 模板预测他半事件归属的命中率",
            fontsize=10, color="#555555")

    # ----------------------------------------------------------------
    # Tier 4 (y=0.3 .. 1.85) — Step 5: forward/reverse check
    # ----------------------------------------------------------------
    tier_band(ax, 0.25, 1.65, color="#EFE7DC")
    step_header(ax, 0.35, 1.65, "Step 5 · forward / reverse 复现检查（条件性）")

    rounded_box(ax, 0.5, 0.45, 15.0, 0.85, facecolor="white", edgecolor=COL_OUT, lw=1.6)
    ax.text(0.85, 1.05,
            "若 PR-2 在全数据上找到一对反相关模板 (T_i, T_j)  →  检查两半的模板里",
            fontsize=11, color=COL_BORDER)
    ax.text(0.85, 0.72,
            "是否仍能找到这对反相关关系  →  输出 ",
            fontsize=11, color=COL_BORDER)
    # inline rounded label for the boolean field name
    fld_x = 0.85 + 5.4
    ax.add_patch(FancyBboxPatch((fld_x, 0.58), 2.7, 0.32,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=COL_OUT, edgecolor=COL_BORDER, lw=0.6, alpha=0.9))
    ax.text(fld_x + 1.35, 0.74, "forward_reverse_reproduced", fontsize=10, color="white", fontweight="bold", ha="center", va="center")
    ax.text(fld_x + 2.85, 0.74, ": True / False", fontsize=11, color=COL_BORDER, va="center")

    # Down-arrows connecting tiers
    arrow(ax, FIG_W / 2, 6.9, FIG_W / 2, 6.65, color="#888888", lw=1.4)
    arrow(ax, FIG_W / 2, 4.50, FIG_W / 2, 4.25, color="#888888", lw=1.4)
    arrow(ax, FIG_W / 2, 2.05, FIG_W / 2, 1.80, color="#888888", lw=1.4)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    png_path = out_dir / "fig_pr25_split_half_schematic.png"
    pdf_path = out_dir / "fig_pr25_split_half_schematic.pdf"
    fig.savefig(png_path, dpi=DPI_PUB, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
