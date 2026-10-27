#!/usr/bin/env python3
"""Compose the author's six-panel Fig5 sketch from frozen scientific exports.

No simulation, field estimation, event selection, or statistical test is run.
PDF panels retain their native vector/raster layers and uniform aspect ratio.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

try:
    import pymupdf as fitz
except ImportError:
    sys.path.insert(0, "/tmp/hfosp_fig5_pdf_tools")
    import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "results/topic4_sef_hfo/fig5_fig4f_historical_zm_20260930/right_column_offset_20261008"
COHORT = ROOT / "results/paper-ready-figure/fig5/candidates/broadband_phase_statistics_20261009"
DEFAULT_OUT = ROOT / "results/paper-ready-figure/fig5/candidates/reference_layout_20261011"
MM = 72 / 25.4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    figures, frozen = out / "figures", out / "source"
    figures.mkdir(parents=True, exist_ok=True)
    frozen.mkdir(exist_ok=True)

    # Left A/B/C and right D/E/F follow the requested reading order. The raster
    # is absent from the supplied sketch; it remains intact in its source pack.
    sources = {
        "A": NATIVE / "figures/fig5-panela.pdf",
        "B": NATIVE / "figures/fig5-panelb.pdf",
        "C": NATIVE / "figures/fig5-paneld.pdf",
        "D": NATIVE / "figures/fig5-panele.pdf",
        "E": NATIVE / "figures/fig5-panelf.pdf",
        "F": COHORT / "two_model_tfr/figures/cohort_db_statistic.pdf",
    }
    originals = {k: sha(p) for k, p in sources.items()}
    support = {
        "native_metadata.json": NATIVE / "fig5_metadata.json",
        "display_selection.json": NATIVE / "display_selection.json",
        "field_comparison.json": NATIVE / "field_comparison.json",
        "native_panel_manifest.json": NATIVE / "panel_manifest.json",
        "cohort_statistics.json": COHORT / "statistics.json",
        "cohort_current_version.json": COHORT / "two_model_tfr/current_version.json",
    }
    for name, path in support.items():
        target = frozen / name
        if target.exists():
            assert sha(target) == sha(path), f"Frozen source changed: {path}"
        else:
            shutil.copy2(path, target)
    statistics = json.loads((frozen / "cohort_statistics.json").read_text())
    stat = next(s for s in statistics if s["analysis"] == "full_energy_plus200ms__db")
    assert (stat["n_patients"], stat["n_seizures"], stat["p"]) == (13, 140, 0.00708)
    selection = json.loads((frozen / "display_selection.json").read_text())
    assert selection["times_s"] == [14.6, 15.045, 15.075, 15.11, 15.3]

    document = fitz.open()
    page = document.new_page(width=320 * MM, height=238 * MM)
    scale = 0.59
    # A and B have slightly different tight-export bounds; restore their
    # original common time-axis coordinates instead of fitting each to width.
    locations = {
        "A": (5.0, 12.0, scale),
        "B": (5.0 + (80.66 - 78.95) * scale / MM, 103.0, scale),
        "C": (5.0 + (80.66 - 62.85) * scale / MM, 168.0, scale),
        "D": (204.0, 12.0, scale),
        "E": (204.0, 108.0, scale),
        "F": (218.0, 163.0, 0.56),
    }
    letters = {"A": (6, 16), "B": (6, 112), "C": (6, 170),
               "D": (204, 16), "E": (204, 112), "F": (204, 170)}
    records = {}
    for letter, path in sources.items():
        src = frozen / f"fig5-panel{letter.lower()}.pdf"
        if src.exists():
            assert sha(src) == originals[letter], f"Frozen panel changed: {letter}"
        else:
            shutil.copy2(path, src)
        with fitz.open(src) as panel:
            clip = fitz.Rect(panel[0].rect)
            if letter == "A":
                # Only redundant standalone time ticks/title are omitted. The
                # baseline at y=405.674 pt and every waveform sample remain.
                clip.y1 = 408.0
            elif letter == "F":
                # Trim only the empty PDF page margins so the statistic can
                # use the same visual font scale as the neighboring panels.
                clip.y0, clip.y1 = 18.0, 390.0
            x, y, panel_scale = locations[letter]
            box = fitz.Rect(x * MM, y * MM,
                            x * MM + clip.width * panel_scale,
                            y * MM + clip.height * panel_scale)
            page.show_pdf_page(box, panel, 0, clip=clip, keep_proportion=True)
            standalone = figures / f"fig5-panel{letter.lower()}.pdf"
            shutil.copy2(src, standalone)
            panel[0].get_pixmap(dpi=240).save(standalone.with_suffix(".png"))
            standalone.with_suffix(".svg").write_text(panel[0].get_svg_image(text_as_path=False))
            records[letter] = {
                "source": str(path), "source_sha256": originals[letter],
                "frozen_pdf": str(src.relative_to(out)),
                "source_pdf_rect_pt": list(panel[0].rect),
                "composite_clip_pt": list(clip), "box_pt": list(box),
                "uniform_scale": panel_scale,
                "independent_pdf_byte_identical": sha(standalone) == originals[letter],
            }
    for letter, (x, y) in letters.items():
        page.insert_text((x * MM, y * MM), letter, fontname="hebo", fontsize=14)
    output_pdf = figures / "fig5-complete-layout.pdf"
    document.save(output_pdf, garbage=4, deflate=True)
    page.get_pixmap(dpi=240).save(output_pdf.with_suffix(".png"))
    page.get_pixmap(dpi=120).save(figures / "fig5-complete-layout-preview.png")
    output_pdf.with_suffix(".svg").write_text(page.get_svg_image(text_as_path=False))

    # Check layout geometry and the frozen content, including actual PDF text.
    rects = [fitz.Rect(v["box_pt"]) for v in records.values()]
    assert all(page.rect.contains(box) for box in rects)
    assert all(not a.intersects(b) for i, a in enumerate(rects) for b in rects[i + 1:])
    text = page.get_text()
    assert "0.0071" in text and "13 patients" in text
    for value in ["15.045", "15.075", "15.110", "15.300", "Normalized", "SCL9", "ICL1"]:
        assert value in text, value
    a_left = records["A"]["box_pt"][0] + 80.66 * scale
    b_left = records["B"]["box_pt"][0] + 78.95 * scale
    assert abs(a_left - b_left) < 1e-6
    unchanged = all(sha(path) == originals[k] for k, path in sources.items())
    assert unchanged
    write(out / "layout_manifest.json", {
        "canvas_mm": [320, 238], "panels": records,
        "panel_mapping": {"A": "old A", "B": "old B", "C": "old D", "D": "old E", "E": "old F", "F": "13-patient dB statistic"},
        "producer": str(Path(__file__).resolve()),
        "source_files_unchanged": unchanged, "new_simulations": 0,
        "statistical_recalculation": False,
        "native_model": "Fig4F xy_left_20, topology 2511, dynamics 847401, historical Z/M",
        "field_observable": "Within-field min-max normalized robust z, retained from source",
        "statistic_observable": "dB without MAD division; selection-aware full-contact null",
        "statistic": {k: v for k, v in stat.items() if k != "matches"},
        "scientific_note": "Example field and cohort statistic retain their distinct source readout contracts; layout does not establish new scientific acceptance.",
    })
    write(out / "validation.json", {
        "status": "PASS", "all_panels_inside_canvas": True,
        "panel_boxes_nonoverlapping": True, "A_B_axis_alignment_error_pt": abs(a_left - b_left),
        "single_panel_pdfs_byte_identical": True, "source_files_unchanged": unchanged,
        "all_panel_aspect_ratios_preserved": True,
        "A_crop": "Redundant standalone x tick labels and title only; waveform axes retained",
        "agent_visual_review": "PENDING", "human_visual_acceptance": "PENDING",
    })
    write(out / "current_revision.json", {
        "version": "reference_layout_20261011", "status": "CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW",
        "producer": str(Path(__file__).resolve()), "main_figure": "figures/fig5-complete-layout.pdf",
        "replaces_formal_current": False, "layout": "left A/B/C; right D/E/F",
    })
    descriptions = {
        "fig5-complete-layout": "按作者参考图组成两列六面板：左列波形、慢变量、四帧空间图；右列三维轨迹、模型与患者起始场、13人统计。A/B共享时间轴，完整拼版带连续A–F角标；不显示原栅格面板。",
        "fig5-panela": "保留14.3–15.4秒、15触点80–1000 Hz模型电流读出，以及原先固定的P99.5显示定标和1–5时刻标记。独立图保留时间刻度，合图仅在B显示共享时间轴。",
        "fig5-panelb": "保留上方Z及下方有效adaptation，All E实线、Core A虚线。与A的实际时间轴精确对齐，数值和图例均沿用冻结图层。",
        "fig5-panelc": "对应原D的15.045、15.075、15.110及15.300秒四帧原生空间发放率；每帧20 ms。保留core位置、magma配色和0–500 Hz共享色条。",
        "fig5-paneld": "对应原E的Z、有效adaptation及E发放率三维轨迹，颜色为原0–18.1秒轨迹的归一化时间。视角、显示范围及1–5状态标记不变。",
        "fig5-panele": "对应原F的模型起始场与Seizure onset 26示例场。保留各自的robust z场0–1归一化显示、空间坐标、电极和色条；该示例的z定义与队列dB统计分别记录。",
        "fig5-panelf": "保留作者指定的13人、140次候选dB统计，p=0.00708，图中显示0.0071。患者为统计单位，保留全部配对点、violin/box与99999次选择校正全触点置换口径。",
    }
    (figures / "README.md").write_text("\n\n".join(
        f"### {name}.png / .pdf / .svg\n\n{description}\n\n**关注点**：仅修改布局，科学内容沿用来源；待作者目视检查。"
        for name, description in descriptions.items()) + "\n")
    (out / "README.md").write_text(
        "# Fig5参考布局修订（2026-10-11）\n\n"
        "入口：[完整PDF](figures/fig5-complete-layout.pdf)；[预览](figures/fig5-complete-layout-preview.png)。"
        "按附件省去原栅格行，空间图上移，右下放已有dB统计，连续编号A–F。\n\n"
        "原始PDF、数值定义和metadata保存在source/；单panel PDF与来源逐字节一致。"
        "A仅在合图隐藏重复时间刻度，不裁波形；所有图层等比例缩放。"
        "模型/患者例图保持z场，队列保持作者指定的dB统计，未用拼版修改读出定义或科学结论。\n\n"
        "复现：`python scripts/paper_figures/build_fig5_reference_layout.py`（依赖PyMuPDF）。"
        "独立候选，当前正式入口未覆盖；作者目视验收待完成。\n")
    shutil.copy2(Path(__file__), frozen / Path(__file__).name)
    print(output_pdf)


if __name__ == "__main__":
    main()
