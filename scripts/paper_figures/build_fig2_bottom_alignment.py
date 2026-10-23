#!/usr/bin/env python3
"""Match Figure 2 E to D's plot height and xlabel baseline."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pymupdf as pdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig2_aligned_columns as renderer
from scripts.paper_figures.build_fig2_split_holdout_panels import raster, sha, write_json

SOURCE = ROOT / 'results/paper-ready-figure/fig2/candidates/narrow_columns_20261023'
OUT = SOURCE.parent / 'aligned_bottom_row_20261023'


def text_span(page, text):
    matches = [s for block in page.get_text('dict', flags=pdf.TEXTFLAGS_TEXT)['blocks']
               for line in block.get('lines', []) for s in line['spans'] if s['text'] == text]
    assert len(matches) == 1, (text, len(matches))
    return matches[0]


def d_reference(source, old):
    scale = old['complete_layout']['d']['scale']
    with pdf.open(source / 'figures/fig2-paneld.pdf') as doc:
        page = doc[0]
        frames = {tuple(d['rect']) for d in page.get_drawings()
                  if abs(d['rect'].width-120.) < .01 and abs(d['rect'].height-120.) < .01}
        assert len(frames) == 8
        label = text_span(page, 'Shared TA axis (mm)')
        return dict(axes_top_pt=min(r[1] for r in frames)*scale,
                    axes_bottom_pt=max(r[3] for r in frames)*scale,
                    xlabel_baseline_pt=label['origin'][1]*scale,
                    xlabel_fontsize=label['size']*scale, xtick_fontsize=12.*scale,
                    xtick_length_pt=2.5*scale, xtick_pad_pt=2.5*scale)


def build(source, out, dpi):
    figures = out / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    snapshot = out / 'source_snapshot'; snapshot.mkdir(exist_ok=True)
    old = json.loads((source / 'figure2_layout_metadata.json').read_text())
    frozen = {}
    for letter in 'abcd':
        for path in (source / 'figures').glob(f'fig2-panel{letter}*'):
            shutil.copy2(path, figures / path.name)
            frozen[path.name] = sha(path)
    reference = d_reference(source, old)
    layout = copy.deepcopy(old['complete_layout'])
    d_rect = pdf.Rect(layout['d']['placed_rect_pt'])
    e_rect = pdf.Rect(layout['e']['placed_rect_pt'])
    e_meta = renderer.build_e(figures, out, dpi, width=e_rect.width, height=d_rect.height,
                             type_scale=.85, axis_alignment=reference)
    for key in ('distribution', 'display_raw_cohort_median_null', 'significance_display', 'patient_order'):
        assert e_meta[key] == old['panel_e'][key], key
    layout['e'] = dict(source_clip_pt=[0, 0, e_rect.width, d_rect.height],
                       placed_rect_pt=[e_rect.x0, d_rect.y0, e_rect.x1, d_rect.y1], scale=1.)
    canvas = [old['canvas_pt'][0], d_rect.y1+16.]
    doc = pdf.open(); page = doc.new_page(width=canvas[0], height=canvas[1])
    for letter, placement in layout.items():
        with pdf.open(figures / f'fig2-panel{letter}.pdf') as part:
            page.show_pdf_page(pdf.Rect(placement['placed_rect_pt']), part, 0,
                               clip=pdf.Rect(placement['source_clip_pt']))
    for letter, x, y in old['panel_labels']:
        page.insert_text((x, y), letter, fontsize=23, fontname='hebo')
    complete = figures / 'fig2-complete-layout.pdf'
    doc.save(complete); doc.close()
    raster(complete, dpi, 1900)
    for stem in ('fig2-panele', 'fig2-complete-layout'):
        path = figures / f'{stem}.png'
        pixels = pdf.Pixmap(str(path)); pixels.set_dpi(dpi, dpi); pixels.save(path)
    meta = copy.deepcopy(old)
    meta.update(schema_version='fig2_aligned_bottom_row_v1', producer=str(Path(__file__)),
                source_candidate=str(source), complete_layout=layout, canvas_pt=canvas,
                frozen_panel_exports=frozen, panel_e=e_meta, dpi=dpi,
                agent_visual_review='PENDING', status='PENDING_AUTHOR_VISUAL_REVIEW')
    meta.pop('agent_visual_review_notes', None)
    meta['compaction']['lower_panel_heights_independent'] = False
    meta['alignment'].update(D_E_bottom_pt=d_rect.y1, D_E_height_pt=d_rect.height,
        D_E_axes_top_pt=d_rect.y0+reference['axes_top_pt'],
        D_E_axes_bottom_pt=d_rect.y0+reference['axes_bottom_pt'],
        D_E_xlabel_baseline_pt=d_rect.y0+reference['xlabel_baseline_pt'])
    write_json(out / 'figure2_layout_metadata.json', renderer.plain(meta))
    for path in (Path(__file__), Path(renderer.__file__), renderer.FIELD_RENDERER):
        shutil.copy2(path, snapshot / path.name)
    shutil.copy2(source / 'figure2_layout_metadata.json', snapshot / 'previous_layout_metadata.json')
    shutil.copy2(renderer.FIELD_META, snapshot / 'field_statistics_metadata.json')
    (figures / 'README.md').write_text(
        '# Figure 2：D/E等高及X轴标签对齐\n\n状态：PENDING_AUTHOR_VISUAL_REVIEW。'
        '沿用上一版收窄布局，将E高度由310 pt降至与D一致的261.268 pt；两图绘图区上下边缘及X轴标签基线对齐。'
        'A–D单图文件及放置不变，E仍复用原统计绘图函数，重新避让患者名称。'
        '复现：`LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/lib PYTHONPATH=/tmp/hfosp_fig5_pdf_tools python scripts/paper_figures/build_fig2_bottom_alignment.py`。\n\n'
        '### fig2-complete-layout.pdf / fig2-complete-layout.png\n'
        '完整画布1000.7×663.268 pt；A/D等宽、BC/E同列，并对齐D/E高度与X轴标签。PNG由同一PDF渲染。\n'
        '**关注点**：D/E绘图区上下边缘与标签基线，待作者目视验收。\n\n'
        '### fig2-panela.pdf / fig2-panela.png\nA的序列、MRI与空间场保持上一版。\n**关注点**：正常图形比例与紧凑列间距。\n\n'
        '### fig2-panelb.pdf / fig2-panelb.png\nB的26人留出三维方向误差、null与Holm星号完整保留。\n**关注点**：患者值及统计未改。\n\n'
        '### fig2-panelc.pdf / fig2-panelc.png\n两个rose、圆内图例、灰色事件方向分布和红蓝实线保持上一版。\n**关注点**：圆形与紧凑间距未改。\n\n'
        '### fig2-paneld.pdf / fig2-paneld.png\n四位患者的TA/TB场及放置均不变，作为E的高度和标签对齐参照。\n**关注点**：空间坐标保持等比。\n\n'
        '### fig2-panele.pdf / fig2-panele.png\nE重新安排纵向轴框和X轴标签，18人r、中位数、IQR、空间null及显著性与上一版逐项核对一致。\n'
        '**关注点**：患者名称不重叠，X轴标签和D位于同一水平线。\n')
    return meta


def validate(out):
    meta = json.loads((out / 'figure2_layout_metadata.json').read_text())
    for name, digest in meta['frozen_panel_exports'].items():
        assert sha(out/'figures'/name) == digest
    d = pdf.Rect(meta['complete_layout']['d']['placed_rect_pt'])
    e = pdf.Rect(meta['complete_layout']['e']['placed_rect_pt'])
    np.testing.assert_allclose([e.y0, e.y1], [d.y0, d.y1], atol=1e-6)
    with pdf.open(out / 'figures/fig2-panele.pdf') as doc:
        page = doc[0]
        span = text_span(page, 'Cohort median r vs spatial null')
        actual_baseline = e.y0+span['origin'][1]
        expected = meta['alignment']['D_E_xlabel_baseline_pt']
        assert abs(actual_baseline-expected) < .03, (actual_baseline, expected)
        for word in page.get_text('words'):
            assert page.rect.contains(pdf.Rect(word[:4])), word[4]
    result = dict(status='PASS', A_B_C_D_exports_and_placement_unchanged=True,
        E_statistics_identical_to_previous=True, E_patient_label_overlaps=0,
        D_E_height_pt=d.height, D_E_axes_top_pt=meta['alignment']['D_E_axes_top_pt'],
        D_E_axes_bottom_pt=meta['alignment']['D_E_axes_bottom_pt'],
        D_E_xlabel_baseline_difference_pt=actual_baseline-expected,
        no_nonuniform_scaling=True, human_visual_acceptance='PENDING_AUTHOR_VISUAL_REVIEW')
    write_json(out / 'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    build(args.source, args.output, args.dpi)
    print(json.dumps(validate(args.output), indent=2), flush=True)
