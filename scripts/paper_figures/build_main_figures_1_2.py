#!/usr/bin/env python3
"""Build label-free panels and lettered complete layouts for Figures 1 and 2."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER_ROOT = ROOT / "results/paper-ready-figure"
FIG1_ROOT = PAPER_ROOT / "fig1"
FIG2_ROOT = PAPER_ROOT / "fig2"
FIG2C_ACCEPTED_SCHEMA = "fig2c_interictal_event_envelope_field_candidate_v10"


def _move(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def _compose_complete_layout(
    *,
    figures_dir: Path,
    stem: str,
    canvas_size: tuple[int, int],
    placements: dict[str, tuple[Path, tuple[int, int, int, int]]],
    labels: dict[str, tuple[int, int]],
    anchors: dict[str, str] | None = None,
    fit_to_cell: bool = False,
    label_font_size: int = 132,
) -> list[str]:
    """Assemble label-free panels on an aligned, whitespace-trimmed canvas.

    Source panels intentionally keep their own export margins.  Those margins
    are useful for standalone files but must not become invisible gutters in a
    complete figure.  We therefore trim only uniform near-white outer space,
    then fit the data-bearing rectangle into its assigned grid cell.
    """
    from PIL import Image, ImageChops, ImageDraw, ImageFont

    anchors = anchors or {}

    def trim_outer_whitespace(image: Image.Image) -> Image.Image:
        background = Image.new("RGB", image.size, "white")
        mask = ImageChops.difference(image, background).convert("L")
        mask = mask.point(lambda value: 255 if value > 10 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            return image.copy()
        pad_x = max(8, int(round(image.width * 0.008)))
        pad_y = max(8, int(round(image.height * 0.008)))
        left, top, right, bottom = bbox
        bbox = (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(image.width, right + pad_x),
            min(image.height, bottom + pad_y),
        )
        return image.crop(bbox)

    def anchored_origin(
        box: tuple[int, int, int, int], size: tuple[int, int], anchor: str,
    ) -> tuple[int, int]:
        x0, y0, x1, y1 = box
        width, height = size
        if "left" in anchor:
            x = x0
        elif "right" in anchor:
            x = x1 - width
        else:
            x = x0 + (x1 - x0 - width) // 2
        if "top" in anchor:
            y = y0
        elif "bottom" in anchor:
            y = y1 - height
        else:
            y = y0 + (y1 - y0 - height) // 2
        return x, y

    canvas = Image.new("RGB", canvas_size, "white")
    for panel_id, (path, box) in placements.items():
        if not path.exists():
            raise FileNotFoundError(f"missing panel {panel_id}: {path}")
        with Image.open(path) as source:
            image = trim_outer_whitespace(source.convert("RGB"))
            x0, y0, x1, y1 = box
            resampling = getattr(Image, "Resampling", Image)
            target = (x1 - x0, y1 - y0)
            if fit_to_cell:
                scale = min(target[0] / image.width, target[1] / image.height)
                image = image.resize(
                    (
                        max(1, round(image.width * scale)),
                        max(1, round(image.height * scale)),
                    ),
                    resampling.LANCZOS,
                )
            else:
                image.thumbnail(target, resampling.LANCZOS)
            x, y = anchored_origin(box, image.size, anchors.get(panel_id, "center"))
            canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), size=label_font_size)
    for label, position in labels.items():
        draw.text(position, label, fill="#111111", font=font, anchor="la")

    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    canvas.save(png, dpi=(600, 600), optimize=True)
    canvas.save(pdf, "PDF", resolution=600.0)
    return [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))]


def build_figure1() -> dict:
    from scripts.paper_figures import plot_fig1_interictal_hfo_temporal_scaffold as fig1

    metadata = fig1.build(
        output_dir=FIG1_ROOT / "figures",
        single_hfo_png=fig1.DEFAULT_SINGLE_HFO,
        group_event_png=fig1.DEFAULT_GROUP_EVENT,
        c1_exemplar_subject="442",
        c1_exemplar_label="Epilepsiae E7",
        max_events=2000,
    )
    figures = FIG1_ROOT / "figures"
    complete = _compose_complete_layout(
        figures_dir=figures,
        stem="fig1-complete-layout",
        canvas_size=(6000, 4800),
        placements={
            "b1": (figures / "fig1-panelb1.png", (1700, 190, 2250, 1500)),
            "b2": (figures / "fig1-panelb2.png", (2330, 190, 4150, 1500)),
            "c": (figures / "fig1-panelc.png", (180, 1710, 4230, 3060)),
            "d": (figures / "fig1-paneld.png", (4450, 1710, 5840, 3060)),
            "e": (figures / "fig1-panele.png", (180, 3290, 4230, 4640)),
            "f": (figures / "fig1-panelf.png", (4450, 3290, 5840, 4640)),
        },
        labels={
            "B": (1540, 35),
            "C": (45, 1560), "D": (4310, 1560),
            "E": (45, 3140), "F": (4310, 3140),
        },
        anchors={"c": "top", "d": "top", "e": "top", "f": "top"},
    )
    metadata["complete_layout"] = complete
    metadata["composite_emitted"] = True
    metadata["panel_letters_in_individual_files"] = False
    metadata["panel_letters_in_complete_layout"] = True
    (FIG1_ROOT / "figure1_panel_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return metadata


def _write_fig2_readme(figures: Path) -> None:
    text = """# Figure 2 独立 panel 输出

本目录保存 Figure 2A–F 的无字母逐 panel PNG/PDF，以及带 A–F 字母的 `fig2-complete-layout` 完整拼版。独立 panel PNG 均以 600 dpi 输出。

### fig2-panela.png / .pdf

E1146 的冻结间期模板轴从三维触点空间投影到 shared patient plane，再生成 TA/TB 连续 rank field。坐标、轴、support 与 rank 均读取冻结 artifact，不在画图时重拟合。

**关注点**：连续表面是 support-limited interpolation，不代表未采样组织中的直接测量。

### fig2-panelb.png / .pdf

左侧以 E1146/E548 的同一组 fold-0 留出事件方向对比仅时序模板轴（同色虚线）和时序--空间模板轴（同色实线），右侧显示 25 名可评估患者的绝对留出方向得分、患者内配对变化和记录块内方向置换零模型。底部同一行叠加蓝色 Timing、橙色 +Space 的患者 bootstrap cohort-median 分布及灰色方向置换 cohort-median Null；底部分布区的长横括号表示 +Space 相对零模型的检验，短横括号表示 +Space 相对 Timing 的患者内配对检验。配对小提琴分布进入 Supplementary Fig. 4B。

**关注点**：该 panel 支持真实三维电极信息提高患者内跨记录块的方向一致性；不是未见患者预测，也不证明连续组织轨迹、传播速度或机制因果。

### fig2-panelc.png / .pdf

E1146 的 TA/TB 单事件 readout、4 个严格等间距的 participant-only HFO envelope 场及时不变的冻结 template-rank field。当前 canonical v10 使用固定 gamma=0.5 的蓝灰包络场；静态时刻由 all-participant full-field selector 在 2 ms 网格上选择，要求每一步的全参与触点质心和 top-3 热点均相反移动。每幅静态场再按本帧最强三个参与触点的均值显示相对包络，避免完整窗尺度把有效后帧压成近白色；因此静态帧只读空间位置，不读帧间绝对幅度。

**关注点**：这是 raw-EEG-derived timing 在既有冻结轴上的 representative cross-check，不是独立验证。

### fig2-paneld.png / .pdf

E1146 的冻结 TA/TB shared-plane rank fields，作为静态模板对照。两幅场使用同一物理平面和统一 6 mm display kernel。

**关注点**：模板场与 panel C 的单事件 envelope 场含义不同，不得把插值表面写成真实组织传播轨迹。

### fig2-panele.png / .pdf

四个锁定案例的 TA/TB shared-axis rank-field 配对展示；案例只用于说明可读的反向场形态，队列推断不由这四例承担。

**关注点**：患者选择和完整 12 人 denominator 写在 `fig2_panel_ef_metadata.json`，不能把 4 个显示例当独立抽样验证。

### fig2-panelf.png / .pdf

完整 shared-axis、二维几何可评估队列的逐患者 signed field correlation，以及 full-contact shuffle 的 cohort-median-shift null。

**关注点**：安全口径是 cohort median 比全触点随机化更负；不能升级成所有患者或所有 null 均显著。

### fig2-complete-layout.png / .pdf

将 A–F 六个独立 panel 排为完整 Figure 2，并只在完整画布上添加 A–F 字母。

**关注点**：独立 panel 内不应重复出现字母；完整排版应保留各 panel 的相对信息层级和可读字号。
"""
    (figures / "README.md").write_text(text, encoding="utf-8")


def _canonicalize_fig2c_metadata(metadata_path: Path, figures: Path) -> None:
    """Rewrite producer staging paths after Fig2-C assets receive canonical panel names."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    static = metadata.get("static")
    if isinstance(static, dict):
        static["figure"] = str((figures / "fig2-panelc.png").resolve())
        static["extra_outputs"] = [str((figures / "fig2-panelc.pdf").resolve())]
    gif = metadata.get("gif")
    if isinstance(gif, dict):
        gif["figure"] = str((figures / "fig2-panelc.gif").resolve())
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _reuse_accepted_fig2c(figures: Path, *, include_gif: bool = False) -> list[str]:
    canonical_pdf = figures / "fig2-panelc.pdf"
    canonical_metadata = FIG2_ROOT / "fig2_panelc_metadata.json"
    if canonical_pdf.exists() and canonical_metadata.exists():
        metadata = json.loads(canonical_metadata.read_text())
        if metadata.get("schema_id") == FIG2C_ACCEPTED_SCHEMA:
            subprocess.run(
                ["pdftoppm", "-png", "-singlefile", "-r", "600", str(canonical_pdf),
                 str(figures / "fig2-panelc")],
                check=True,
            )
            _canonicalize_fig2c_metadata(canonical_metadata, figures)
            outputs = [
                str((figures / "fig2-panelc.png").relative_to(ROOT)),
                str(canonical_pdf.relative_to(ROOT)),
            ]
            canonical_gif = figures / "fig2-panelc.gif"
            if include_gif and canonical_gif.exists():
                outputs.append(str(canonical_gif.relative_to(ROOT)))
            return outputs

    candidates = [
        PAPER_ROOT / "fig2c_interictal_event_envelope_field/figures",
        PAPER_ROOT / "archive/2026-08-09_fig2_pre_panel_contract/fig2c_interictal_event_envelope_field/figures",
    ]
    stem = "fig2c_candidate_E1146_interictal_event_envelope_field"
    source_dir = None
    for path in candidates:
        source_metadata = path / f"{stem}_metadata.json"
        if not source_metadata.exists():
            continue
        metadata = json.loads(source_metadata.read_text())
        if metadata.get("schema_id") == FIG2C_ACCEPTED_SCHEMA:
            source_dir = path
            break
    if source_dir is None:
        raise FileNotFoundError(
            f"accepted Fig2-C schema {FIG2C_ACCEPTED_SCHEMA} is absent; "
            "rerun with --recompute-fig2c"
        )
    pdf = figures / "fig2-panelc.pdf"
    shutil.copy2(source_dir / f"{stem}.pdf", pdf)
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "600", str(pdf), str(figures / "fig2-panelc")],
        check=True,
    )
    shutil.copy2(source_dir / f"{stem}_metadata.json", FIG2_ROOT / "fig2_panelc_metadata.json")
    source_gif = source_dir / f"{stem}.gif"
    if include_gif and source_gif.exists():
        shutil.copy2(source_gif, figures / "fig2-panelc.gif")
    _canonicalize_fig2c_metadata(FIG2_ROOT / "fig2_panelc_metadata.json", figures)
    outputs = [str((figures / "fig2-panelc.png").relative_to(ROOT)), str(pdf.relative_to(ROOT))]
    if include_gif and (figures / "fig2-panelc.gif").exists():
        outputs.append(str((figures / "fig2-panelc.gif").relative_to(ROOT)))
    return outputs


def build_figure2(*, make_gif: bool = False, recompute_fig2c: bool = False) -> dict:
    figures = FIG2_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[str]] = {}

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/paper_figures/plot_fig2_e1146_template_projection_composite.py"),
         "--output-dir", str(figures), "--stem", "fig2-panela"],
        check=True,
    )
    a_png = figures / "fig2-panela.png"
    a_pdf = figures / "fig2-panela.pdf"
    a_svg = figures / "fig2-panela.svg"
    a_meta = figures / "fig2-panela_metadata.json"
    _move(a_meta, FIG2_ROOT / "fig2_panela_metadata.json")
    outputs["a"] = [str(a_png.relative_to(ROOT)), str(a_pdf.relative_to(ROOT))]
    # SVG is retained as the editable vector counterpart of panel A.
    outputs["a"].append(str(a_svg.relative_to(ROOT)))
    gc.collect()

    spatial_gain_root = PAPER_ROOT / "fig2b_spatial_information_gain"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/paper_figures/plot_interictal_spatial_information_gain.py"),
            "--paper-root",
            str(spatial_gain_root),
        ],
        check=True,
    )
    b_outputs = {
        "png": spatial_gain_root / "figures/fig2b-spatial-information-gain.png",
        "pdf": spatial_gain_root / "figures/fig2b-spatial-information-gain.pdf",
        "metadata": (
            spatial_gain_root
            / "figures/fig2b-spatial-information-gain_metadata.json"
        ),
    }
    b_png = figures / "fig2-panelb.png"
    b_pdf = figures / "fig2-panelb.pdf"
    shutil.copy2(b_outputs["png"], b_png)
    shutil.copy2(b_outputs["pdf"], b_pdf)
    shutil.copy2(b_outputs["metadata"], FIG2_ROOT / "fig2_panelb_metadata.json")
    outputs["b"] = [str(b_png.relative_to(ROOT)), str(b_pdf.relative_to(ROOT))]
    del b_outputs
    gc.collect()

    if recompute_fig2c:
        command = [
            sys.executable,
            str(ROOT / "scripts/paper_figures/plot_fig2c_interictal_event_envelope_field.py"),
            "--output-dir", str(figures),
        ]
        if not make_gif:
            command.append("--no-gif")
        subprocess.run(command, check=True)
        source_stem = "fig2c_candidate_E1146_interictal_event_envelope_field"
        c_png = _move(figures / f"{source_stem}.png", figures / "fig2-panelc.png")
        c_pdf = _move(figures / f"{source_stem}.pdf", figures / "fig2-panelc.pdf")
        _move(figures / f"{source_stem}_metadata.json", FIG2_ROOT / "fig2_panelc_metadata.json")
        outputs["c"] = [str(c_png.relative_to(ROOT)), str(c_pdf.relative_to(ROOT))]
        source_gif = figures / f"{source_stem}.gif"
        if source_gif.exists():
            c_gif = _move(source_gif, figures / "fig2-panelc.gif")
            outputs["c"].append(str(c_gif.relative_to(ROOT)))
        _canonicalize_fig2c_metadata(FIG2_ROOT / "fig2_panelc_metadata.json", figures)
    else:
        outputs["c"] = _reuse_accepted_fig2c(figures, include_gif=make_gif)
    gc.collect()

    d_script = str(ROOT / "scripts/plot_topic5_interictal_template_ab_fields.py")
    for output_format in ("png", "pdf"):
        subprocess.run(
            [sys.executable, d_script, "--output-dir", str(figures),
             "--subjects", "epilepsiae_1146", "--format", output_format, "--no-atlas"],
            check=True,
        )
    d_png_source = figures / "epilepsiae_1146_interictal_AB.png"
    d_pdf_source = figures / "epilepsiae_1146_interictal_AB.pdf"
    d_png = _move(d_png_source, figures / "fig2-paneld.png")
    d_pdf = _move(d_pdf_source, figures / "fig2-paneld.pdf")
    outputs["d"] = [str(d_png.relative_to(ROOT)), str(d_pdf.relative_to(ROOT))]
    gc.collect()

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/paper_figures/plot_fig2_shared_field_reversal_row.py"),
         "--output-dir", str(FIG2_ROOT)],
        check=True,
    )
    ef_meta = json.loads((FIG2_ROOT / "fig2_panel_ef_metadata.json").read_text())
    outputs["e"] = [ef_meta["outputs"]["panel_e_png"], ef_meta["outputs"]["panel_e_pdf"]]
    outputs["f"] = [ef_meta["outputs"]["panel_f_png"], ef_meta["outputs"]["panel_f_pdf"]]

    _write_fig2_readme(figures)
    complete = _compose_complete_layout(
        figures_dir=figures,
        stem="fig2-complete-layout",
        canvas_size=(7000, 6000),
        placements={
            "a": (figures / "fig2-panela.png", (160, 180, 3420, 1800)),
            "b": (figures / "fig2-panelb.png", (3600, 180, 6840, 1800)),
            "c": (figures / "fig2-panelc.png", (160, 2080, 4920, 3830)),
            "d": (figures / "fig2-paneld.png", (5100, 2080, 6840, 3830)),
            "e": (figures / "fig2-panele.png", (160, 4100, 4920, 5850)),
            "f": (figures / "fig2-panelf.png", (5100, 4100, 6840, 5850)),
        },
        labels={
            "A": (45, 35), "B": (3485, 35),
            "C": (45, 1930), "D": (4965, 1930),
            "E": (45, 3950), "F": (4965, 3950),
        },
        anchors={"c": "top", "d": "top", "e": "top", "f": "top"},
    )
    registry = {
        "schema_version": "paper_figure2_panels_and_complete_layout_v2",
        "composite_emitted": True,
        "png_dpi": 600,
        "panels": outputs,
        "complete_layout": complete,
        "panel_letters_in_individual_files": False,
        "panel_letters_in_complete_layout": True,
        "panel_c_version_note": (
            "canonical v10 producer uses four equally spaced all-participant hotspot-selected "
            "frames, per-frame participant-top3 relative scaling, and a fixed-gamma blue-gray "
            "envelope colormap; older assembled screenshots "
            "used five or seven frames. "
            "Default packaging re-rasterizes the accepted vector PDF at 600 dpi; pass "
            "--recompute-fig2c only when raw-data regeneration is required."
        ),
    }
    (FIG2_ROOT / "figure2_panel_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("1", "2", "all"), default="all")
    parser.add_argument("--fig2-gif", action="store_true")
    parser.add_argument("--recompute-fig2c", action="store_true")
    args = parser.parse_args()
    result = {}
    if args.figure in ("1", "all"):
        built = build_figure1()
        result["figure1"] = {
            "panels": built["outputs"],
            "complete_layout": built["complete_layout"],
        }
    if args.figure in ("2", "all"):
        built = build_figure2(
            make_gif=args.fig2_gif, recompute_fig2c=args.recompute_fig2c,
        )
        result["figure2"] = {
            "panels": built["panels"],
            "complete_layout": built["complete_layout"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
