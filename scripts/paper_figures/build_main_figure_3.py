#!/usr/bin/env python3
"""Build label-free Figure 3 panels and one lettered complete layout."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PAPER_ROOT = ROOT / "results/paper-ready-figure"
FIG3_ROOT = PAPER_ROOT / "fig3"
FIGURES = FIG3_ROOT / "figures"
LATEST_SOURCE_ROOT = (
    PAPER_ROOT
    / "archive/2026-09-02_noncanonical_staging_qa_cleanup"
    / "fig3_all_events_timing_plus_space_sources"
)

SOURCES = {
    "c": LATEST_SOURCE_ROOT / "fig3c/figures/epilepsiae_1146_seizure_02_interictal_ictal_shared_field.pdf",
    "d": LATEST_SOURCE_ROOT / "fig3d/figures/clinical_onset_gradient_field_cohort_stat.pdf",
    "e": LATEST_SOURCE_ROOT / "fig3e/figures/epilepsiae_1146_peri_onset_template_expression_paper_ready_journal_clean.pdf",
    "f": LATEST_SOURCE_ROOT / "fig3f/figures/fig3f_ab_dominance_heatmap.pdf",
}
C_SOURCE_METADATA = SOURCES["c"].with_name(
    "epilepsiae_1146_seizure_02_interictal_ictal_shared_field_metadata.json"
)


def _copy_vector_and_rasterize(panel_id: str, source_pdf: Path) -> list[str]:
    if not source_pdf.exists():
        raise FileNotFoundError(source_pdf)
    pdf = FIGURES / f"fig3-panel{panel_id}.pdf"
    shutil.copy2(source_pdf, pdf)
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "600", str(pdf),
         str(FIGURES / f"fig3-panel{panel_id}")],
        check=True,
    )
    return [
        str((FIGURES / f"fig3-panel{panel_id}.png").relative_to(ROOT)),
        str(pdf.relative_to(ROOT)),
    ]


def _canonicalize_panel_c_metadata() -> str:
    if not C_SOURCE_METADATA.exists():
        raise FileNotFoundError(C_SOURCE_METADATA)
    payload = json.loads(C_SOURCE_METADATA.read_text(encoding="utf-8"))
    payload["status"] = "paper-ready Figure 3C locked"
    payload["paper_role"] = "Figure 3C interictal timing versus early-ictal shared field"
    payload["source_metadata"] = str(C_SOURCE_METADATA.relative_to(ROOT))
    payload["display"]["outputs"] = {
        "png": str((FIGURES / "fig3-panelc.png").relative_to(ROOT)),
        "pdf": str((FIGURES / "fig3-panelc.pdf").relative_to(ROOT)),
    }
    target = FIG3_ROOT / "fig3_panelc_metadata.json"
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(target.relative_to(ROOT))


def _build_panels_ab() -> tuple[dict[str, list[str]], Path]:
    command = [
        sys.executable,
        str(ROOT / "scripts/paper_figures/plot_fig3_raw_spectral_context.py"),
        "--subject", "epilepsiae_1146",
        "--seizure-idx", "7",
        "--spectral-channel", "SCL9",
        "--compact-main",
        "--comparison-subject", "epilepsiae_635",
        "--comparison-seizure-idx", "7",
        "--comparison-spectral-channel", "HRB1",
        "--comparison-spectral-profile", "gamma",
        "--output-dir", str(FIGURES),
        "--independent-only",
    ]
    subprocess.run(command, check=True)
    summary = FIGURES / "epilepsiae_1146_seizure_07_raw_spectral_context_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return {
        "a": payload["outputs"]["a"],
        "b": payload["outputs"]["b"],
    }, summary


def _compose_complete_layout() -> list[str]:
    from scripts.paper_figures.build_main_figures_1_2 import _compose_complete_layout

    return _compose_complete_layout(
        figures_dir=FIGURES,
        stem="fig3-complete-layout",
        canvas_size=(7200, 6000),
        placements={
            "a": (FIGURES / "fig3-panela.png", (140, 100, 4680, 1900)),
            "b": (FIGURES / "fig3-panelb.png", (4810, 160, 7060, 1900)),
            "c": (FIGURES / "fig3-panelc.png", (140, 2140, 4680, 3910)),
            "d": (FIGURES / "fig3-paneld.png", (4810, 2130, 7060, 3900)),
            "e": (FIGURES / "fig3-panele.png", (140, 4170, 4680, 5840)),
            "f": (FIGURES / "fig3-panelf.png", (4810, 4170, 7060, 5840)),
        },
        labels={
            "A": (35, 25), "B": (3905, 25),
            "C": (35, 1995), "D": (4705, 1995),
            "E": (35, 4035), "F": (4705, 4035),
        },
        anchors={"a": "top-left", "b": "top", "c": "top-left", "d": "top", "e": "top-left", "f": "top"},
        fit_to_cell=True,
        fit_width_panels={"a", "c", "e"},
    )


def _write_readme() -> None:
    (FIGURES / "README.md").write_text(
        """# Figure 3 panel 与完整排版输出

独立 panel 文件不写左上角 A–F；字母只出现在 `fig3-complete-layout`。所有独立 PNG 均由矢量 PDF 或原始 producer 以 600 dpi 生成。

本版按各 panel 进入拼板后的实际缩放比例分别校准坐标字体，而不是把同一个 producer 字号硬套到所有 panel。A/B 的既定排版不变；C/E 与 D/F 分别补偿左、右列的实际缩放，C 与 E 共用左侧列宽，D 与 F 共用更紧凑的右侧列宽，且 E/F 的最终显示高度匹配。

### fig3-panela.png / .pdf

两个代表性发作模式的并列 signal context：左为 E10 | SZ8 broadband-type 的 raw SEEG 与 SCL9 baseline-normalized TFR，右为 supplementary 已接受的 E20 | SZ8 gamma-type raw SEEG 与 HRB1 TFR。两例都只显示 20 s baseline 邻域 −110 至 −90 s 和 clinical onset 邻域 −10 至 +20 s；中间 −90 至 −10 s 用成对斜线断轴明确标为未显示。

**关注点**：每个内部示例的两段式 raw SEEG 与 TFR 必须严格共轴，横轴统一写作 `Time (s)`；`BASELINE` 在 20 s baseline 段居中；病例/类型标题必须显著大于 `BASELINE` / `CLINICAL ONSET` 区间标注；E20/SZ8/HRB1 应清楚显示 gamma-dominant 快活动增强。淡灰省略带、居中省略号和断轴斜线共同表示删去的显示区间，不表示数据缺失或时间连续。A 的两个示例使用与 C、E 相同的左右列槽。

### fig3-panelb.png / .pdf

两个代表性发作的 low bands、gamma、high-gamma 与 broadband 能量轨迹：E10/SZ8 的代表通道 SCL9 为 broadband-type，E20/SZ8 的代表通道 HRB1 为 gamma-type。

**关注点**：B 连续显示 −120 至 +20 s，不使用 A 的断轴；四图在 0 s 统一画黑色竖直虚线，左列 ylabel 简写为 `dB`。颜色只编码发作表型，不编码频带；legend 在 low-bands 图左上角的无曲线区纵向排列，只写 `Broadband` / `Gamma`，避免覆盖 onset 附近的核心变化。患者/SZ/通道身份由 A 标题给出。两例来自不同患者，只是代表性形态对照，不是患者内或 cohort 统计。

### fig3-panelc.png / .pdf

all-event Timing+Space 冻结间期 TA timing field 与固定 SZ3 的 early-ictal broadband power field。C 不使用总标题；右图以 `E10 | SZ3` / `Early ictal field` 两行子图标题标识病例与语义。左色条与 Fig2 统一为 `0 early / 0.5 / 1 late` normalized ranks，右色条标题简写为 `power` / `z`，空间 y label 为 `Y (mm)`。

**关注点**：右图色条中的 `z` 指 baseline-normalized robust z power，不是传播 rank；两条 colorbar 均与各自的正方形 field 绘图区等高，并留有一致的小间隙。左侧 field 组向内收紧，且两图都显示 `Y (mm)`；完整拼板以右列为锚，使 C/E 两列中心对齐。该病例经过形态选择，只作空间读出桥。

### fig3-paneld.png / .pdf

all-event Timing+Space 场下 clinical onset 后 0–10 s 的 gradient-field cohort Data–Null 比较：Pooled n=17、Broadband n=16、Gamma n=11。

**关注点**：Pooled/Broadband 显著、Gamma 为 n.s.；不得替换成旧 endpoint n=20 三组全显著版本。

### fig3-panele.png / .pdf

E10 peri-onset amplitude-aware template expression：左为 `max(|q_A|, |q_B|)`，右为 signed TA/TB projection；两图各自把 legend 纵向放在右上角，并使用白底细框。

**关注点**：两个时程 panel 的内部间距随 C 同步收紧；legend、axis label 与 ticks 均按 E 进入左栏后的实际缩放补偿，legend 仍位于每图右上角的白底细框内。这是单病例描述性轨迹，不支持 onset-emergent alignment 或机制结论。

### fig3-panelf.png / .pdf

17 名可评估患者在 −120 至 +20 s 的 all-event Timing+Space signed A/B contrast heatmap。

**关注点**：虚线为 clinical onset；主图使用 heatmap，paired inferential companion 留作补充材料。F 与 D 共用右侧列宽，最终显示高度与 E 匹配。

### fig3-complete-layout.png / .pdf

将 A–F 六个无角标独立 panel 组装为带统一 A–F 字母的完整 Figure 3。

**关注点**：完整排版只负责版面与字母，不改变各 panel 的数据、统计或坐标合同。
""",
        encoding="utf-8",
    )


def build() -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    panels, ab_summary = _build_panels_ab()
    for panel_id in ("c", "d", "e", "f"):
        panels[panel_id] = _copy_vector_and_rasterize(panel_id, SOURCES[panel_id])
    panel_c_metadata = _canonicalize_panel_c_metadata()
    complete = _compose_complete_layout()
    _write_readme()
    registry = {
        "schema_version": "paper_figure3_panels_and_complete_layout_v12",
        "producer": "scripts/paper_figures/build_main_figure_3.py",
        "panel_letters_in_individual_files": False,
        "panel_letters_in_complete_layout": True,
        "individual_panel_png_dpi": 600,
        "complete_layout_dpi": 600,
        "panels": panels,
        "panel_metadata": {"c": panel_c_metadata},
        "complete_layout": complete,
        "sources": {
            "a_b": str(ab_summary.relative_to(ROOT)),
            "c": str(SOURCES["c"].relative_to(ROOT)),
            "d": str(SOURCES["d"].relative_to(ROOT)),
            "e": str(SOURCES["e"].relative_to(ROOT)),
            "f": str(SOURCES["f"].relative_to(ROOT)),
        },
        "panel_contract": {
            "a": "two compact broken-axis raw SEEG plus TFR examples: E10 SZ8 SCL9 broadband-type and E20 SZ8 HRB1 gamma-type",
            "b": "continuous representative E10 SZ8 SCL9 versus E20 SZ8 HRB1 band-power trajectories with 0-s onset markers, dB y labels, and unobstructed Broadband/Gamma legend",
            "c": "title-free all-event Timing+Space interictal TA field with Fig2-matched normalized ranks versus E10 | SZ3 early ictal field; power colorbar shown as z",
            "d": "all-event Timing+Space clinical-onset gradient-field cohort, n=17/16/11",
            "e": "E10 all-event Timing+Space amplitude-aware template expression trajectories with per-axis upper-right framed stacked legends",
            "f": "17-subject all-event Timing+Space signed A/B contrast heatmap",
        },
        "final_typography_contract": {
            "principle": "normalize visible type after each panel is scaled into the locked complete layout",
            "standard_axis_label_target_pt": [9.0, 10.5],
            "standard_tick_target_pt": [7.5, 8.5],
            "dense_tick_target_pt": [6.5, 7.5],
            "exceptions": {
                "a_raw_channel_names": "dense tick tier",
                "f_subject_labels": "dense tick tier",
            },
        },
        "layout_changed": "A/C/E share the same two-panel left-block columns; B/D/F share the compact right block; rendered E/F heights are matched",
    }
    (FIG3_ROOT / "figure3_panel_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(build(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
