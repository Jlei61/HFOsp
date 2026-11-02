#!/usr/bin/env python3
"""Restore E colors and label the two x categories and shared patient count."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_fig3_stat_style as previous

base, layout, plt = previous.base, previous.layout, previous.plt
normalized, compositor = previous.normalized, previous.compositor
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_model_null_labels_20261102')
LABELS = ['Model–seizure\nsimilarity', 'NULL']
DISPLAY_CHECKS = {}


def export_e(fig, stem):
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba
    from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import _p_stars

    assert stem == 'fig6-panele'
    ax = fig.axes[0]
    stat = json.loads((SOURCE/'cohort_statistics_selected.json').read_text())
    fig.set_size_inches(62/25.4, 71/25.4)
    ax.set_position([18/62, 15/71, 40.6/62, 46/71])
    ax.set_ylabel('Field similarity |r|', fontsize=layout.LABEL, labelpad=4)
    ax.set_xticks([1,2], LABELS)
    ax.tick_params(axis='x', pad=3, labelsize=layout.TICK, length=2)
    for label in ax.get_xticklabels():
        label.set_linespacing(1.05)
    ax.set_xlabel('n = 13 patients', fontsize=layout.TICK, labelpad=3)
    assert len(ax.texts) == 1 and 'p =' in ax.texts[0].get_text()
    ax.texts[0].set_text(_p_stars(stat['p']))
    ax.texts[0].set_fontsize(layout.LABEL*1.2)
    # The historical producer supplies the original brown/gray artists.
    points = [c for c in ax.collections if isinstance(c,PathCollection) and len(c.get_offsets()) == 13]
    assert len(points) == 4
    for point,color in zip(points[-2:],['#b45b40','#a2a2a2']):
        assert list(point.get_facecolor()[0]) == list(to_rgba(color))
    assert not fig.legends and ax.get_legend() is None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    sample_box = ax.xaxis.label.get_window_extent(renderer)
    assert not boxes[0].overlaps(boxes[1]), 'Keep the category labels separate.'
    assert sample_box.y1 < min(b.y0 for b in boxes) and sample_box.y0 > 0
    DISPLAY_CHECKS.update(original_brown_gray_colors_restored=True, no_legend=True,
                          category_labels_do_not_overlap=True, sample_size_on_separate_bottom_line=True)
    normalized.matched.OUT = OUT
    normalized.NATIVE_EXPORT(fig,stem)


def main():
    pointer = base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    frozen = pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    stat = json.loads((SOURCE/'cohort_statistics_selected.json').read_text())
    assert stat['n_patients'] == len(stat['matches']) == len({r['subject'] for r in stat['matches']}) == 13
    layout.OUT, layout.export = OUT, export_e
    layout.draw_e(stat)
    meta = json.loads((SOURCE/'figure_metadata.json').read_text())
    compositor.OUT, compositor.SOURCE = OUT, SOURCE
    compositor.C_DISPLAY.update(meta['C_display'])
    compositor.compose()
    display = meta['E_display']
    for key in ['visual_reference','visual_reference_sha256','reference_scope','legend_font_pt']:
        display.pop(key,None)
    display.update(y_label='Field similarity |r|', x_tick_labels=LABELS, x_tick_positions=[1,2],
        x_label='n = 13 patients', x_label_font_pt=layout.TICK, legend_labels=[],
        legend_location=None, legend_colors=[], legend=False,
        point_colors=['#b45b40','#a2a2a2'], sample_size_label='n = 13 patients',
        sample_size_location='shared x-axis label below both category labels')
    meta.update(producer=str(Path(__file__).resolve()), previous_candidate=str(SOURCE),
        revision_scope='Restore brown/gray E; two category labels Model–seizure similarity and NULL; separate bottom n; no legend.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json',meta)
    unchanged=[]
    for stem in ['fig6-panela','fig6-panelb','fig6-panelc','fig6-paneld',
                 'fig6-panelb-interictal','fig6-panelb-full-trajectory']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}'
            assert (OUT/'figures'/name).read_bytes() == (SOURCE/'figures'/name).read_bytes()
            unchanged.append(name)
    for name in ['cohort_statistics_selected.json','stage_readouts.npz','normalized_propagation_fields.npz',
                 'model_selected_patient_arrays.npz','trajectory_data.npz']:
        assert (OUT/name).read_bytes() == (SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-panele','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem] = all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
            text=doc[0].get_text()
            assert all(t in text for t in ['Model–seizure','similarity','NULL','n = 13 patients'])
            assert 'Observed' not in text and 'Channel-shuffle null' not in text
    assert all(checks.values()),checks
    assert pointer.read_bytes() == frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=unchanged, all_data_statistics_unchanged=True,
        n_unique_patients=13, original_permutation_p=stat['p'], **DISPLAY_CHECKS,
        PDF_text_inside_canvas=checks, formal_pointer_unchanged=True, human_visual_acceptance='PENDING'))
    caption=(SOURCE/'caption_zh.md').read_text()
    old='沿用 Figure 3C 的展示方式：配对数据下方标 Model–patient 及 n = 13 patients，图内右下角用蓝色 Observed、灰色 Channel-shuffle null 标识两组，纵轴为 Field similarity |r|。'
    caption=caption.replace(old,'恢复原来的棕灰配色，X 轴左侧标 Model–seizure similarity，右侧标 NULL；两组标签下方独立一行显示 n = 13 patients，不加图例。')
    (OUT/'caption_zh.md').write_text(caption)
    readme=(SOURCE/'figures/README.md').read_text()
    start=readme.index('### fig6-panele.png')
    end=readme.find('\n### ',start+1)
    if end<0:end=len(readme)
    readme=readme[:start]+('### fig6-panele.png / .pdf / .svg\n\n'
        '恢复棕灰配色，左侧 X 轴标签为 Model–seizure similarity，右侧为 NULL。'
        '最下面独立一行显示 n = 13 patients，无图例；原配对数据和统计保持。\n'
        '**关注点**：核对左右标签、人数及原配色。\n')+readme[end:]
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text('# Figure 6：E 双组标签与患者数\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf)\n\n'
        '左侧 Model–seizure similarity，右侧 NULL，下方 n = 13 patients。恢复棕灰配色，无图例。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks),indent=2))


if __name__ == '__main__':
    main()
