#!/usr/bin/env python3
"""Split frozen held-out statistics and roses into Figure 2 panels B and C."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pymupdf as pdf

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/paper-ready-figure/fig2/candidates/nearest_direction_compact_roses_20261023'
OUT = SOURCE.parent / 'split_holdout_panels_20261023'
SPLIT_X = 360
CLIPS = {'b': [0, 0, SPLIT_X, 350], 'c': [SPLIT_X, 0, 583, 350]}
RENAME = {'a': 'a', 'd': 'c', 'e': 'd'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def raster(path, dpi, preview_width):
    with pdf.open(path) as doc:
        page = doc[0]
        page.get_pixmap(matrix=pdf.Matrix(dpi / 72, dpi / 72), alpha=False).save(path.with_suffix('.png'))
        scale = preview_width / page.rect.width
        page.get_pixmap(matrix=pdf.Matrix(scale, scale), alpha=False).save(path.with_name(path.stem + '-preview.png'))


def build(source, out, dpi):
    figures = out / 'figures'
    snapshot = out / 'source_snapshot'
    figures.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    old = json.loads((source / 'metadata.json').read_text())
    assert old['direction_assignment'] == 'nearest_3d_angle'
    assert old['rose_layout']['circle_gap_pt'] == 40
    combined = source / 'figures/fig2b-direction-target.pdf'
    frozen = {}
    for new_letter, old_letter in RENAME.items():
        for path in (source / 'figures').glob(f'fig2-panel{old_letter}.*'):
            dest = figures / path.name.replace(f'panel{old_letter}', f'panel{new_letter}')
            shutil.copy2(path, dest)
            frozen[dest.name] = {'source': str(path), 'sha256': sha(path)}
    with pdf.open(combined) as src:
        for letter, coords in CLIPS.items():
            clip = pdf.Rect(coords)
            doc = pdf.open()
            page = doc.new_page(width=clip.width, height=clip.height)
            page.show_pdf_page(page.rect, src, 0, clip=clip)
            path = figures / f'fig2-panel{letter}.pdf'
            doc.save(path)
            doc.close()
            raster(path, dpi, 1100 if letter == 'b' else 700)

    layout = {new: copy.deepcopy(old['complete_layout'][prior]) for new, prior in RENAME.items()}
    original_b = old['complete_layout']['b']
    origin_x, origin_y = original_b['placed_rect_pt'][:2]
    scale = original_b['scale']
    for letter, coords in CLIPS.items():
        x0, y0, x1, y1 = coords
        layout[letter] = {
            'source_clip_pt': [0, 0, x1 - x0, y1 - y0],
            'placed_rect_pt': [origin_x + scale*x0, origin_y + scale*y0,
                               origin_x + scale*x1, origin_y + scale*y1],
            'scale': scale,
        }
    labels = [('A', 12, 32), ('B', 667, 32),
              ('C', origin_x + SPLIT_X*scale, 32), ('D', 12, 443), ('E', 824, 443)]
    doc = pdf.open()
    page = doc.new_page(width=1240, height=860)
    for letter in 'abcde':
        placement = layout[letter]
        with pdf.open(figures / f'fig2-panel{letter}.pdf') as part:
            page.show_pdf_page(pdf.Rect(placement['placed_rect_pt']), part, 0,
                               clip=pdf.Rect(placement['source_clip_pt']))
    for label, x, y in labels:
        page.insert_text((x, y), label, fontsize=27, fontname='hebo')
    complete = figures / 'fig2-complete-layout.pdf'
    doc.save(complete)
    doc.close()
    raster(complete, dpi, 1800)

    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    shutil.copy2(source / 'metadata.json', snapshot / 'holdout_metadata.json')
    shutil.copy2(source / 'validation.json', snapshot / 'holdout_layout_validation.json')
    shutil.copy2(combined, snapshot / combined.name)
    meta = {
        'status': 'PENDING_AUTHOR_VISUAL_REVIEW',
        'schema_version': 'fig2_split_holdout_panels_v1',
        'producer': str(Path(__file__)), 'source_candidate': str(source),
        'direction_assignment': old['direction_assignment'],
        'analysis_root': old['analysis_root'],
        'panel_mapping': {'a': 'previous A', 'b': 'previous B left statistics',
                          'c': 'previous B right roses', 'd': 'previous C', 'e': 'previous D'},
        'combined_source_sha256': sha(combined), 'split_clips_pt': CLIPS,
        'complete_layout': layout, 'panel_labels': labels, 'canvas_pt': [1240, 860],
        'frozen_panel_exports': frozen, 'panel_b': old['plots']['angular_mae_deg'],
        'panel_c': {key: old[key] for key in ('rose_semantics', 'rose_rays', 'rose_legend', 'rose_layout')},
        'cohort_summary': old['cohort_summary'],
        'png_source': 'same-state delivered PDFs', 'dpi': dpi,
        'agent_visual_review': 'PENDING',
    }
    write_json(out / 'figure2_layout_metadata.json', meta)
    (figures / 'README.md').write_text(
        '# Figure 2：统计与玫瑰图拆分为 B、C\n\n'
        '状态：PENDING_AUTHOR_VISUAL_REVIEW。完整图为A–E，上排A/B/C、下排D/E；由上一版冻结PDF原生矢量拆分，未重算或改变数据、统计、字号及摆放比例。复现：`PYTHONPATH=/tmp/hfosp_fig5_pdf_tools python scripts/paper_figures/build_fig2_split_holdout_panels.py --dpi 600`。\n\n'
        '### fig2-complete-layout.pdf / fig2-complete-layout.png\n'
        'B为三维方向留出误差统计，C为两个方向玫瑰图；原C/D顺延为D/E。保持1240×860 pt画布及原面板实际位置，PNG直接从同一PDF生成。\n'
        '**关注点**：A–E阅读顺序、独立B/C身份和字母位置，待作者目视确认。\n\n'
        '### fig2-panela.pdf / fig2-panela.png\n'
        '沿用原A的Rank SEQ、解剖投影及Spatial TRAJ.示例。源文件字节不变。\n'
        '**关注点**：序列与空间表示的对应。\n\n'
        '### fig2-panelb.pdf / fig2-panelb.png\n'
        '原左侧统计独立成为B：每个留出事件取两个训练模板中的最小有向三维夹角，每折平均后两折等权，再以患者为统计单位。保留26位患者、各自null、Holm星号与Relative density；未重算分析。\n'
        '**关注点**：两种训练模板对留出方向的代表性，非未知方向预测。\n\n'
        '### fig2-panelc.pdf / fig2-panelc.png\n'
        '原右侧E10/E14玫瑰图独立成为C，保留40 pt圆周间距与圆内图例。灰柱为第一留出折的方向二维投影，红蓝实线为同折Spatial TRAJ.模板方向；图中不显示fold字样，二维投影不能替代B的三维误差。\n'
        '**关注点**：事件方向分布与模板方向的直观对应。\n\n'
        '### fig2-paneld.pdf / fig2-paneld.png\n'
        '原C的四位患者TA/TB空间场与轨迹改为D。源文件字节、色标与空间坐标保持。\n'
        '**关注点**：不同患者的两类空间轨迹。\n\n'
        '### fig2-panele.pdf / fig2-panele.png\n'
        '原D的队列空间场统计改为E。源文件字节、患者点、null及显著性保持。\n'
        '**关注点**：沿用原队列空间统计口径，与B留出误差分别解释。\n'
    )
    return meta


def validate(source, out):
    meta = json.loads((out / 'figure2_layout_metadata.json').read_text())
    old = json.loads((source / 'metadata.json').read_text())
    for path, record in meta['frozen_panel_exports'].items():
        assert sha(out / 'figures' / path) == record['sha256']
    for new, prior in RENAME.items():
        assert meta['complete_layout'][new] == old['complete_layout'][prior]
    assert meta['panel_b'] == old['plots']['angular_mae_deg']
    assert meta['cohort_summary'] == old['cohort_summary']
    differences = {}
    with pdf.open(source / 'figures/fig2b-direction-target.pdf') as src:
        for letter, coords in CLIPS.items():
            with pdf.open(out / f'figures/fig2-panel{letter}.pdf') as part:
                before = src[0].get_pixmap(matrix=pdf.Matrix(2, 2), clip=pdf.Rect(coords), alpha=False)
                after = part[0].get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False)
                assert (before.width, before.height) == (after.width, after.height)
                a = np.frombuffer(before.samples, np.uint8).astype(int)
                b = np.frombuffer(after.samples, np.uint8).astype(int)
                differences[letter] = {'max_pixel_channel_difference': int(abs(a-b).max()),
                                       'changed_channel_fraction': float(np.mean(a != b))}
                assert np.max(abs(a-b)) <= 3 and np.mean(a != b) < .0001
                text = part[0].get_text()
                assert ('Held-out error' in text) == (letter == 'b')
                assert ('Event directions' in text) == (letter == 'c')
                if letter == 'b':
                    assert text.count('***') == 3
                assert 'Fold-1' not in text
    # Inspect only the header bands; the frozen A/D/E content is hash-verified.
    with pdf.open(out / 'figures/fig2-complete-layout.pdf') as doc:
        labels = []
        for expected, x, y in meta['panel_labels']:
            text = doc[0].get_text(clip=pdf.Rect(x-1, y-30, x+25, y+9)).strip()
            assert text == expected, (expected, text)
            labels.append(text)
    report = {'status': 'PASS', 'scope': 'panel split and lettering only',
              'panel_labels': labels, 'frozen_exports_verified': len(meta['frozen_panel_exports']),
              'A_D_E_positions_and_scales_unchanged': True,
              'B_C_split_raster_checks': differences, 'statistics_nulls_and_stars_unchanged': True,
              'rose_gap_pt': 40, 'legend_inside_rose': True,
              'human_visual_acceptance': 'PENDING_AUTHOR_VISUAL_REVIEW'}
    write_json(out / 'validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    build(args.source, args.output, args.dpi)
    print(json.dumps(validate(args.source, args.output), indent=2), flush=True)
    print(args.output, flush=True)
