"""Render one Figure 3A case with readable single-case axes; extraction unchanged.

This diagnostic entry point reuses the canonical raw/TFR producer. The runtime
saver replacement changes layout only, and never edits accepted Figure 3 files.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import plot_fig3_raw_spectral_context as producer


def save_single_case(fig, panel_axes, out_dir):
    axes = panel_axes["a"]
    if len(axes) != 3:
        raise ValueError("This diagnostic saver requires exactly one seizure")
    raw, tfr, cbar = axes
    for ax in fig.axes:
        ax.set_visible(ax in axes)
    fig.set_size_inches(9.5, 8.5)
    for ax, pos in zip(axes, ([.12, .47, .75, .43], [.12, .12, .75, .23], [.90, .12, .023, .23])):
        ax.set_position(pos)
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.title.set_fontsize(13)
    raw.tick_params(axis="y", labelsize=9)
    raw.set_title(raw.get_title(), fontsize=14, pad=29)
    for txt in raw.texts:
        if "BASELINE" in txt.get_text().upper() or "ONSET" in txt.get_text().upper():
            txt.set_y(1.01)
            txt.set_verticalalignment("bottom")
            txt.set_fontsize(9)
    cbar.title.set_fontsize(9)
    outputs = []
    for ext in ("png", "pdf"):
        path = out_dir / f"fig3-panela.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        outputs.append(str(path.relative_to(ROOT)))
    (out_dir / "README.md").write_text(
        "### fig3-panela.png / fig3-panela.pdf\n\n"
        "复用当前 Figure 3A 原始信号和频谱计算，只调整单次发作的排版。显示全部 15 个冻结接触点，"
        "两次发作均固定 ICL2 为 TFR 通道；CAR、1–150 Hz、临床起点为零，显示 −110 至 −90 s 与 −10 至 +20 s。"
        "TFR 基线为临床起点前 −120 至 −90 s；它不同于发作空间标签计算使用的 EEG 起点前基线。\n\n"
        "**关注点**：检查起始活动的空间位置及频谱；这张波形图本身不证明发作复现了间期传播顺序。待用户目视检查。\n",
        encoding="utf-8",
    )
    return {"a": outputs}


if __name__ == "__main__":
    producer._save_independent_panel_crops = save_single_case
    producer.main()
