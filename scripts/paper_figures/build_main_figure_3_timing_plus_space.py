#!/usr/bin/env python3
"""Assemble a Timing+Space refresh of Figure 3 in the locked main layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _copy_and_rasterize(
    panel_id: str,
    source_pdf: Path,
    figures_dir: Path,
) -> list[str]:
    if not source_pdf.exists():
        raise FileNotFoundError(source_pdf)
    pdf = figures_dir / f"fig3-panel{panel_id}.pdf"
    shutil.copy2(source_pdf, pdf)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            "600",
            str(pdf),
            str(figures_dir / f"fig3-panel{panel_id}"),
        ],
        check=True,
    )
    return [_display_path(figures_dir / f"fig3-panel{panel_id}.png"), _display_path(pdf)]


def _compose(figures_dir: Path) -> list[str]:
    from scripts.paper_figures.build_main_figures_1_2 import _compose_complete_layout

    return _compose_complete_layout(
        figures_dir=figures_dir,
        stem="fig3-complete-layout",
        canvas_size=(7200, 6000),
        placements={
            "a": (figures_dir / "fig3-panela.png", (140, 160, 4800, 1900)),
            "b": (figures_dir / "fig3-panelb.png", (4930, 160, 7060, 1900)),
            "c": (figures_dir / "fig3-panelc.png", (140, 2130, 4550, 3900)),
            "d": (figures_dir / "fig3-paneld.png", (4680, 2130, 7060, 3900)),
            "e": (figures_dir / "fig3-panele.png", (140, 4170, 4680, 5840)),
            "f": (figures_dir / "fig3-panelf.png", (4810, 4170, 7060, 5840)),
        },
        labels={
            "A": (35, 25), "B": (4825, 25),
            "C": (35, 1995), "D": (4575, 1995),
            "E": (35, 4035), "F": (4705, 4035),
        },
        anchors={key: "top" for key in "abcdef"},
    )


def build(args: argparse.Namespace) -> dict:
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "a": args.reference_figure_dir / "fig3-panela.pdf",
        "b": args.reference_figure_dir / "fig3-panelb.pdf",
        "c": args.panel_c,
        "d": args.panel_d,
        "e": args.panel_e,
        "f": args.panel_f,
    }
    panels = {
        panel_id: _copy_and_rasterize(panel_id, source, figures_dir)
        for panel_id, source in sources.items()
    }
    complete = _compose(figures_dir)
    readme = """# Timing+Space Figure 3

各独立 panel 不含 A–F 字母；`fig3-complete-layout` 复用原主图的 7200×6000、600 dpi 版式和固定 panel 位置。

### fig3-panela.png / .pdf

原 Figure 3A signal context，数据与版式均未改变。

**关注点**：该 panel 不依赖间期模板。

### fig3-panelb.png / .pdf

原 Figure 3B 频带能量轨迹，数据与版式均未改变。

**关注点**：该 panel 不依赖间期模板。

### fig3-panelc.png / .pdf

使用 Timing+Space 间期模板重新冻结 E1146 的共享方向和早期发作能量场。

**关注点**：发作期数据未参与模板或轴拟合。

### fig3-paneld.png / .pdf

使用 Timing+Space 场重算 onset 后 0–10 s 的 Data–channel-shuffle null 比较。

**关注点**：患者、发作、路由和显著性均来自本次重算，不沿用旧数字。

### fig3-panele.png / .pdf

E1146 在 Timing+Space 共享场上的 amplitude-aware TA/TB 表达轨迹。

**关注点**：q 保留场的空间对比幅度；相关系数 r 另存为审计图。

### fig3-panelf.png / .pdf

使用 Timing+Space TA/TB 稠密传播序位重算的全患者 signed A/B contrast heatmap。

**关注点**：只有满足新二维场合同的患者进入统计。

### fig3-complete-layout.png / .pdf

更新后的完整 Figure 3；版式与原 paper-ready 主图一致，模板依赖的 C–F 均来自 Timing+Space 重算。

**关注点**：主图只改变模板依赖分析，不改变 A/B 的信号背景。
"""
    (figures_dir / "README.md").write_text(readme, encoding="utf-8")
    registry = {
        "schema_version": "paper_figure3_timing_plus_space_refresh_v1",
        "producer": "scripts/paper_figures/build_main_figure_3_timing_plus_space.py",
        "layout_reference": "locked Figure 3 7200x6000 complete layout",
        "panel_letters_in_individual_files": False,
        "panel_letters_in_complete_layout": True,
        "individual_panel_png_dpi": 600,
        "complete_layout_dpi": 600,
        "panels": panels,
        "complete_layout": complete,
        "sources": {
            panel_id: {
                "path": _display_path(source),
                "sha256": _sha256(source),
            }
            for panel_id, source in sources.items()
        },
    }
    (args.out_dir / "figure3_panel_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-figure-dir", type=Path, required=True)
    for panel_id in "cdef":
        parser.add_argument(f"--panel-{panel_id}", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.reference_figure_dir = args.reference_figure_dir.resolve()
    for panel_id in "cdef":
        setattr(args, f"panel_{panel_id}", getattr(args, f"panel_{panel_id}").resolve())
    args.out_dir = args.out_dir.resolve()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
