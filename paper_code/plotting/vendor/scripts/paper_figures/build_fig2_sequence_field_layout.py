"""Assemble a four-panel Figure 2 candidate from frozen vector source panels.

Only layout and representation names change. No data or statistical renderer is
rerun. Install PyMuPDF to run this standalone composition script.
"""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pymupdf as pdf
from PIL import Image, ImageChops
ROOT = INPUT_ROOT
SOURCE = ROOT / 'results/paper-ready-figure/fig2'
OUTPUT = SOURCE / 'candidates/sequence_to_spatial_field_20261009'

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot_sources(source: Path, output: Path) -> dict:
    snapshot = output / 'source_snapshot'
    manifest_path = snapshot / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, entry in manifest.items():
            if sha256(snapshot / name) != entry['sha256']:
                raise RuntimeError(f'Frozen source changed: {name}')
        return manifest
    names = [f'figures/fig2-panel{k}.{ext}' for k in 'abcdef' for ext in ('pdf', 'png')]
    names += ['figure2_panel_registry.json', 'figures/README.md']
    names += [f'fig2_panel{k}_metadata.json' for k in 'abcef']
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    manifest = {}
    for name in names:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
        manifest[name] = {'source': str(source / name), 'sha256': sha256(destination)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest

def content_rect(page: pdf.Page, clip: pdf.Rect | None=None, pad: float=2) -> pdf.Rect:
    """Find exterior white margins without altering any source graphics."""
    clip = clip or page.rect
    pix = page.get_pixmap(matrix=pdf.Matrix(2, 2), clip=clip, alpha=False)
    im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    difference = ImageChops.difference(im, Image.new('RGB', im.size, 'white'))
    box = difference.convert('L').point(lambda v: 255 if v > 10 else 0).getbbox()
    if box is None:
        raise ValueError('Empty panel crop')
    return pdf.Rect(max(clip.x0, pix.x / 2 + box[0] / 2 - pad), max(clip.y0, pix.y / 2 + box[1] / 2 - pad), min(clip.x1, pix.x / 2 + box[2] / 2 + pad), min(clip.y1, pix.y / 2 + box[3] / 2 + pad))

def place(page: pdf.Page, document: pdf.Document, target, clip=None) -> dict:
    clip = pdf.Rect(clip) if clip is not None else content_rect(document[0])
    target = pdf.Rect(target)
    scale = min(target.width / clip.width, target.height / clip.height)
    width, height = (clip.width * scale, clip.height * scale)
    rect = pdf.Rect(target.x0 + (target.width - width) / 2, target.y0 + (target.height - height) / 2, target.x0 + (target.width + width) / 2, target.y0 + (target.height + height) / 2)
    page.show_pdf_page(rect, document, 0, clip=clip, keep_proportion=True)
    return {'source_clip_pt': list(clip), 'placed_rect_pt': list(rect), 'scale': scale}

def label(page: pdf.Page, text: str, rect, size: float, *, bold=False) -> None:
    remaining = page.insert_textbox(pdf.Rect(rect), text, fontsize=size, fontname='hebo' if bold else 'helv', align=1, color=(0.12, 0.12, 0.12))
    if remaining < 0:
        raise RuntimeError(f'Label does not fit: {text}')

def export(document: pdf.Document, stem: Path, dpi: int) -> list[str]:
    document.save(stem.with_suffix('.pdf'), garbage=4, deflate=True)
    page = document[0]
    page.get_pixmap(dpi=dpi, alpha=False).save(stem.with_suffix('.png'))
    stem.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=True))
    return [str(stem.with_suffix(ext)) for ext in ('.png', '.pdf', '.svg')]

def build(source: Path, output: Path, dpi: int=600) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = snapshot_sources(source, output)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    inputs = output / 'source_snapshot/figures'
    docs = {k: pdf.open(inputs / f'fig2-panel{k}.pdf') for k in 'abcdef'}
    a = pdf.open()
    page = a.new_page(width=660, height=365)
    label(page, 'Electrode-sequence template (1D)', (0, 0, 215, 36), 12, bold=True)
    label(page, 'Local electrode geometry', (219, 0, 421, 36), 12, bold=True)
    label(page, 'Spatial propagation field (2D)', (438, 0, 660, 36), 12, bold=True)
    layout_a = {'sequence_ta': place(page, docs['c'], (15, 37, 186, 184), (0, 0, 160, 153)), 'sequence_tb': place(page, docs['c'], (15, 207, 186, 354), (0, 176.1, 160, 329.1)), 'anatomy': place(page, docs['a'], (217, 33, 417, 193), content_rect(docs['a'][0], pdf.Rect(235, 0, 400, 134))), 'projection': place(page, docs['a'], (215, 209, 419, 355), content_rect(docs['a'][0], pdf.Rect(235, 137, 400, 247))), 'field_ta': place(page, docs['d'], (437, 37, 660, 168), (0, 0, 220.7876, 136.5)), 'field_tb': place(page, docs['d'], (437, 207, 660, 357), (0, 176.1, 220.7876, 332.2459))}
    files = {'a': export(a, figures / 'fig2-panela', dpi)}
    for new, old in (('b', 'b'), ('c', 'e'), ('d', 'f')):
        files[new] = []
        for ext in ('png', 'pdf'):
            path = figures / f'fig2-panel{new}.{ext}'
            shutil.copy2(inputs / f'fig2-panel{old}.{ext}', path)
            assert sha256(path) == manifest[f'figures/fig2-panel{old}.{ext}']['sha256']
            files[new].append(str(path.relative_to(ROOT)))
        svg = figures / f'fig2-panel{new}.svg'
        svg.write_text(docs[old][0].get_svg_image(text_as_path=True))
        files[new].append(str(svg.relative_to(ROOT)))
    complete = pdf.open()
    page = complete.new_page(width=1240, height=790)
    layout = {'a': place(page, a, (28, 53, 680, 414), (0, 0, 660, 365)), 'b': place(page, docs['b'], (710, 60, 1221, 400)), 'c': place(page, docs['e'], (35, 475, 785, 762)), 'd': place(page, docs['f'], (837, 449, 1182, 775))}
    page.insert_font(fontname='FigureTwoPanelBold', fontfile=str(INPUT_ROOT / 'fonts' / 'DejaVuSans-Bold.ttf'))
    for letter, x, y in (('A', 12, 33), ('B', 695, 33), ('C', 12, 447), ('D', 695, 447)):
        page.insert_text((x, y), letter, fontname='FigureTwoPanelBold', fontsize=24, color=(0, 0, 0))
    complete_files = export(complete, figures / 'fig2-complete-layout', dpi)
    preview = figures / 'fig2-complete-layout-preview.png'
    page.get_pixmap(matrix=pdf.Matrix(1800 / page.rect.width, 1800 / page.rect.width), alpha=False).save(preview)
    metadata = {'schema_version': 'fig2_sequence_to_spatial_field_layout_v1', 'status': 'CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW', 'human_visual_acceptance': 'PENDING', 'source_contract': json.loads((output / 'source_snapshot/figure2_panel_registry.json').read_text())['source_contract'], 'producer': str(Path(__file__).relative_to(CODE_ROOT)), 'source_manifest': str((output / 'source_snapshot/manifest.json').relative_to(ROOT)), 'panel_mapping': {'A': 'original C readouts + original A E10 anatomy and projection + original D fields', 'B': 'original B, unchanged', 'C': 'original E, unchanged: E1, E12, E5, Y9', 'D': 'original F, unchanged: complete 18-patient shared-field cohort'}, 'representation_names': {'1D': {'en': 'Electrode-sequence template', 'zh': '电极序列模板'}, '2D': {'en': 'Spatial propagation field', 'zh': '空间传播场'}}, 'scientific_boundary': ['1D denotes a contact-indexed sequence, not anatomical linearity.', 'The two left readouts remain individual TA/TB samples, not average templates.', '2D is a geometry-constrained interpolation of frozen propagation ranks, not a measured continuous tissue field.', 'All A components use E10 (epilepsiae_1146); the original Y9 overview is omitted.', "The latest E12 example is retained; the sketch's earlier E20 selection is not reinstated.", 'No statistical analysis, field interpolation, exemplars, or cohort membership changed.'], 'panel_a_layout': layout_a, 'complete_layout': layout, 'canvas_pt': [1240, 790], 'png_dpi': dpi, 'panels': files, 'complete_files': complete_files, 'preview': str(preview.relative_to(ROOT)), 'validation': {'standalone_b_c_d_match_source_bytes': True, 'source_hashes_verified': True, 'vector_pdf_composition': True, 'canonical_assets_replaced': False}}
    (output / 'figure2_layout_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    (figures / 'README.md').write_text('# Figure 2 四面板重排候选\n\n本版按 2026-10-09 作者示意拼成 A–D，状态为待作者目视检查。PNG、PDF、SVG 来自同一矢量拼版；源面板及 metadata 冻结在同级 `source_snapshot/`，可由 `scripts/paper_figures/build_fig2_sequence_field_layout.py` 复现。\n\n### fig2-complete-layout.png / .pdf / .svg\n\n上排 A 将电极序列示例、局部电极几何与空间模板场连起来，B 保留原方向验证图；下排 C、D 分别为原 E、F。四例沿用当前版本 E1、E12、E5、Y9；候选图未替换原 A–F 文件。\n**关注点**：检查整图阅读顺序、上下排比例、B 的字号与 A 中三列关系。\n\n### fig2-panela.png / .pdf / .svg\n\n将原 C 的 TA/TB readout、原 A 右列的 E10 解剖与二维投影、原 D 的 TA/TB rank field 合成一个面板；所有内容均来自 E10。暂定名称为 Electrode-sequence template（电极序列模板，1D）与 Spatial propagation field（空间传播场，2D），左侧仍写 TA/TB samples。\n**关注点**：1D 指触点索引上的序列，不表示电极几何共线；2D 表示受真实触点几何约束的秩插值场，不是直接测得的连续组织活动。左侧每行是一次示例事件，不是群体平均模板。\n\n### fig2-panelb.png / .pdf / .svg\n\n保留原 Figure 2B 的两位患者方向玫瑰图、26 人留出得分、配对变化及零模型。独立 PNG/PDF 与本次冻结的原 B 逐字节相同。\n**关注点**：只改变整图中的放置与缩放，仍是患者内跨记录块的方向验证。\n\n### fig2-panelc.png / .pdf / .svg\n\n原 E 原样编号为 C，保持 E1、E12、E5、Y9 四列和每列上下 TA/TB 场。独立 PNG/PDF 与本次冻结的原 E 逐字节相同。\n**关注点**：这些是形态展示案例，完整队列推断见 D；未按草图恢复旧 E20 案例。\n\n### fig2-paneld.png / .pdf / .svg\n\n原 F 原样编号为 D，保留 18 人相关分布、IQR、raw cohort median、空间零模型及两项显著性标注。独立 PNG/PDF 与本次冻结的原 F 逐字节相同。\n**关注点**：上下菱形均表示同一 observed raw median r；15/18 与空间零模型检验的原统计含义不变。\n\n### fig2-complete-layout-preview.png\n\n完整拼图的 1800 px 宽预览，用于快速目视检查。高分辨率交付请用同目录完整 PNG 或矢量 PDF/SVG。\n**关注点**：本轮完成 Agent 自查后，人工视觉验收仍待作者。\n', encoding='utf-8')
    for document in [a, complete, *docs.values()]:
        document.close()
    return metadata

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    result = build(args.source_root, args.output_root, args.dpi)
    print(json.dumps({'complete_files': result['complete_files'], 'status': result['status']}, indent=2))
if __name__ == '__main__':
    main()
