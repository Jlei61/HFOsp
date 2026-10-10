#!/usr/bin/env python3
"""Shorten patient identities to Y1 in the author-approved Figure 1 layout.

Only native text overlays are changed. All scientific pixels and vector layers
are retained from the current data-driven producers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextToPath
import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/fig1_pdf_deps")
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "results/paper-ready-figure/fig1"
BASE = CANON / "candidates/y1_b_visual_alignment_20261010"
OUT = CANON / "revisions/y1_final_short_labels_20261010"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def change_titles(name, dpi):
    pdf_path = BASE / "figures" / f"{name}.pdf"
    png_path = BASE / "figures" / f"{name}.png"
    page = PdfReader(pdf_path).pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    occurrences = []

    def collect(text, cm, tm, font, size):
        if "Yuquan Y1" in text and 0 < cm[4] < width and 0 < cm[5] < height:
            occurrences.append((text.strip(), cm[4], cm[5], size))

    page.extract_text(visitor_text=collect)
    if name == "fig1-complete-layout":
        a = min((v for v in occurrences if v[2] > 1000), key=lambda v: v[1])
        b = max((v for v in occurrences if v[2] > 1000), key=lambda v: v[1])
        c = next(v for v in occurrences if "n=18,190" in v[0])
        titles = [(a, "center"), (b, "center"), (c, "left")]
    else:
        assert len(occurrences) == 1, (name, occurrences)
        titles = [(occurrences[0], "center" if name in ("fig1-panela", "fig1-panelb") else "left")]
    fig = plt.figure(figsize=(width/72, height/72))
    rectangles = []
    replacements = []
    for (old, x, baseline, size), anchor in titles:
        prop = FontProperties(family="DejaVu Sans", weight="bold", size=size)
        text_width = TextToPath().get_text_width_height_descent(old, prop, False)[0]
        left, bottom = x-2, baseline-.30*size
        right, top = x+text_width+2, min(height, baseline+1.02*size)
        rectangles.append([left, bottom, right, top])
        fig.add_artist(Rectangle((left/width, bottom/height), (right-left)/width,
                                 (top-bottom)/height, transform=fig.transFigure,
                                 facecolor="white", edgecolor="none", zorder=-1))
        new = old.replace("Yuquan Y1", "Y1")
        anchor_x = x+text_width/2 if anchor == "center" else x
        fig.text(anchor_x/width, baseline/height, new, ha=anchor, va="baseline",
                 fontproperties=prop, color="black")
        replacements.append(dict(old=old, new=new, baseline_pt=baseline,
                                 horizontal_anchor=anchor, anchor_x_pt=anchor_x))
    overlay_pdf = OUT / "source" / f"{name}_labels.pdf"
    overlay_png = OUT / "source" / f"{name}_labels.png"
    fig.savefig(overlay_pdf, transparent=True)
    fig.savefig(overlay_png, dpi=dpi, transparent=True)
    plt.close(fig)
    original = Image.open(png_path).convert("RGBA")
    overlay = Image.open(overlay_png).convert("RGBA")
    # Cropped legacy PNGs round physical PDF dimensions down independently.
    # Match that outer pixel grid without scaling either the title or data.
    assert max(abs(a-b) for a,b in zip(original.size, overlay.size)) <= 1
    if original.size != overlay.size:
        aligned = Image.new("RGBA", original.size, (255, 255, 255, 0))
        aligned.paste(overlay, (0, 0))
        overlay = aligned
    result = Image.alpha_composite(original, overlay)
    result.convert("RGB").save(OUT / "figures" / f"{name}.png")
    changes = np.any(np.asarray(original) != np.asarray(result), axis=2)
    ys, xs = np.where(changes)
    allowed = np.zeros(len(xs), dtype=bool)
    for left, bottom, right, top in rectangles:
        allowed |= ((xs >= left*dpi/72-1) & (xs <= right*dpi/72+1) &
                    (ys >= (height-top)*dpi/72-1) & (ys <= (height-bottom)*dpi/72+1))
    assert allowed.all(), name
    page.merge_page(PdfReader(overlay_pdf).pages[0])
    writer = PdfWriter(); writer.add_page(page)
    with (OUT / "figures" / f"{name}.pdf").open("wb") as handle:
        writer.write(handle)
    return dict(replacements=replacements, allowed_rectangles_pt=rectangles,
                changed_pixels=int(changes.sum()), pixels_outside_titles_identical=True)


def main():
    (OUT / "source").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    hashes = {str(p): sha(p) for p in BASE.rglob("*") if p.is_file()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    audit = {}
    for name in ("complete-layout", "panela", "panelb", "panelc", "spectrum-full-window-check"):
        audit[name] = change_titles(f"fig1-{name}", 220 if name == "spectrum-full-window-check" else 300)
    for letter in "def":
        for suffix in ("png", "pdf"):
            name = f"fig1-panel{letter}.{suffix}"
            shutil.copy2(BASE / "figures" / name, OUT / "figures" / name)
            assert sha(BASE / "figures" / name) == sha(OUT / "figures" / name)
    shutil.copy2(BASE / "spectrum_contract.json", OUT / "spectrum_contract.json")
    assert hashes == {str(p): sha(p) for p in BASE.rglob("*") if p.is_file()}
    metadata = json.loads((BASE / "metadata.json").read_text())
    metadata.update(status="AUTHOR_ACCEPTED_LAYOUT_LABELS_UPDATED",
        source_revision=str(BASE), previous_layout=str(BASE), producer=str(Path(__file__).resolve()),
        author_layout_acceptance="ACCEPTED_2026-10-10",
        human_visual_acceptance="PRIOR_LAYOUT_ACCEPTED_FINAL_LABEL_EXPORT_PENDING_CHECK",
        changed_panels=["A patient title", "B patient title", "C patient title"],
        display_patient_label="Y1", spectrum_contract=str(OUT / "spectrum_contract.json"),
        title_edit_audit=audit, validation=str(OUT / "validation.json"),
        preservation=dict(all_pixels_outside_title_rectangles_identical=True,
                          D_E_F_standalone_files_byte_identical=True,
                          all_data_algorithms_and_layout_preserved=True),
        outputs={str(p.relative_to(OUT)): sha(p) for p in (OUT / "figures").glob("*")},
        input_hashes={str(p): sha(p) for p in [Path(__file__), BASE / "metadata.json"]})
    metadata["subtitle_alignment"]["labels"] = ["Y1", "HFO n = 178", "Y1"]
    for name, value in [("metadata", metadata), ("validation", dict(status="PASS",
        title_edit_audit=audit, scientific_content_and_layout_unchanged=True,
        D_E_F_standalone_files_byte_identical=True, previous_outputs_unchanged=True))]:
        (OUT / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    shutil.copy2(Path(__file__), OUT / "source" / Path(__file__).name)
    descriptions = {
        "complete-layout": "沿用作者已认可的Figure 1布局，仅将A、B及C中的患者名称由Yuquan Y1改成Y1，C保留n=18,190。所有标题以外的像素、数据、坐标轴及色条布局保持。",
        "panela": "患者标题缩写为Y1并保留原水平中心及基线。脑模型、电极、A7/A9标记与真实双极波形保持。",
        "panelb": "患者标题缩写为Y1，仍与左侧HFO标题同基线。三个真实事件、完整事件S³质心、谱图、Time (s)对齐及色条位置均保持。",
        "panelc": "标题改为Y1 | n=18,190，保留原左对齐锚点。18通道显示rank、热图及峰高归一分布保持。",
        "paneld": "原40人MI对照及permutation示意保持。Yuquan是队列名称，与Epilepsiae对应，因此保留。",
        "panele": "冻结TA/TB标签、18通道热图及rank均值和总体标准差保持。独立PNG/PDF与上一版逐字节一致。",
        "panelf": "原40人MI散点、配对inset及与E对齐的轴框保持。Yuquan与Epilepsiae队列图例保持。",
        "spectrum-full-window-check": "完整500 ms谱图核对图的患者标题同步缩写为Y1。事件、质心和显示范围保持。",
    }
    (OUT / "figures/README.md").write_text("# Figure 1：最终患者标题缩写\n\n"+
        "原布局已获作者认可；本次仅执行最后的标题修改。\n\n"+
        "\n\n".join(f"### fig1-{name}.png / .pdf\n\n{text}\n\n**关注点**：患者身份统一为Y1，数据和已认可的排版保持。"
                     for name, text in descriptions.items())+"\n")
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
