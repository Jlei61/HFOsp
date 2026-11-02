#!/usr/bin/env python3
"""Match Fig3C comparison-label, sample-size, legend and color grammar in Fig6E."""
import hashlib
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_legend_sample as previous

base, layout, plt = previous.base, previous.layout, previous.plt
normalized = previous.normalized
compositor = previous.previous.compositor
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_fig3_stat_style_20261102')
REFERENCE = base.MAIN/'results/paper-ready-figure/fig3/figures/fig3-panelc.pdf'
LEGEND_LABELS = ['Observed', 'Channel-shuffle null']
X_LABEL = 'Model–patient\n$n = 13$ patients'
DISPLAY_CHECKS = {}


def export_e(fig, stem):
    from matplotlib.collections import PathCollection, PolyCollection
    from matplotlib.patches import Patch, PathPatch
    from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import _p_stars

    assert stem == 'fig6-panele'
    ax = fig.axes[0]
    stat = json.loads((SOURCE/'cohort_statistics_selected.json').read_text())
    fig.set_size_inches(62/25.4, 71/25.4)
    ax.set_position([18/62, 15/71, 40.6/62, 46/71])
    ax.set_ylabel('Field similarity |r|', fontsize=layout.LABEL, labelpad=4)
    ax.set_xticks([1.5], [X_LABEL])
    ax.tick_params(axis='x', pad=4, labelsize=layout.TICK, length=2)
    ax.get_xticklabels()[0].set_linespacing(1.1)
    ax.set_xlabel('')
    assert len(ax.texts) == 1 and 'p =' in ax.texts[0].get_text()
    ax.texts[0].set_text(_p_stars(stat['p']))
    ax.texts[0].set_fontsize(layout.LABEL*1.2)
    # Restyle the existing artists only: retain each point, line and distribution.
    bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
    boxes = [p for p in ax.patches if isinstance(p, PathPatch)]
    points = [c for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets()) == 13]
    assert len(bodies) == len(boxes) == 2 and len(points) == 4
    fills, edges, dots = ['#9fbdcf','#d8d8d8'], ['#6f8fa3','#9a9a9a'], ['#5f86a3','#888888']
    for body, box, point, fill, edge, dot in zip(bodies, boxes, points[-2:], fills, edges, dots):
        body.set_facecolor(fill)
        box.set_facecolor(fill)
        box.set_edgecolor(edge)
        point.set_facecolor(dot)
        point.set_edgecolor('white')
    legend = ax.legend(handles=[Patch(facecolor=f, edgecolor=e, label=s)
                       for f,e,s in zip(fills,edges,LEGEND_LABELS)],
                       loc='lower right', bbox_to_anchor=(.995,.025), frameon=False,
                       fontsize=layout.TICK, handlelength=1.05, handletextpad=.45,
                       labelspacing=.22, borderaxespad=0, borderpad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    axes_box = ax.get_window_extent(renderer)
    xlabel_box = ax.get_xticklabels()[0].get_window_extent(renderer)
    min_value = min(min(r['best_abs_r'], r['null_median']) for r in stat['matches'])
    min_point_y = ax.transData.transform((1,min_value))[1]
    assert axes_box.contains(legend_box.x0,legend_box.y0) and axes_box.contains(legend_box.x1,legend_box.y1)
    assert legend_box.y1 < min_point_y-5, 'Keep the legend clear of all patient points.'
    assert xlabel_box.y1 < axes_box.y0 and xlabel_box.y0 > 0
    DISPLAY_CHECKS.update(legend_inside_lower_right=True, legend_clear_of_data=True,
                          comparison_and_n_below_pair=True,
                          original_point_offsets_and_plot_limits_retained=True)
    normalized.matched.OUT = OUT
    normalized.NATIVE_EXPORT(fig, stem)


def main():
    pointer = base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    frozen = pointer.read_bytes()
    shutil.copytree(SOURCE, OUT, dirs_exist_ok=True,
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
    meta['E_display'].update(y_label='Field similarity |r|', x_tick_labels=[X_LABEL],
        x_tick_positions=[1.5], x_label='', legend_labels=LEGEND_LABELS,
        legend_location='inside lower right', legend_colors=['#9fbdcf','#d8d8d8'],
        point_colors=['#5f86a3','#888888'], sample_size_label='n = 13 patients',
        sample_size_location='second line of centered comparison tick label',
        visual_reference=str(REFERENCE), visual_reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        statistical_definition_unchanged=True,
        reference_scope='Fig3C display grammar only; keep original Fig6 paired data and permutation p, not Fig3 Wilcoxon.')
    meta.update(producer=str(Path(__file__).resolve()), previous_candidate=str(SOURCE),
        revision_scope='Match Fig3C grouped comparison and sample-size labels, lower-right legend and blue/gray colors.',
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
            assert all(t in text for t in ['Model–patient','patients','Field similarity |r|',*LEGEND_LABELS])
    assert all(checks.values()),checks
    assert pointer.read_bytes() == frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=unchanged, all_data_statistics_unchanged=True,
        n_unique_patients=13, original_permutation_p=stat['p'], **DISPLAY_CHECKS,
        PDF_text_inside_canvas=checks, formal_pointer_unchanged=True, human_visual_acceptance='PENDING'))
    caption=(SOURCE/'caption_zh.md').read_text().replace(
        '右上角用棕色 Model–patient 和灰色 Within-patient shuffle 图例标识两组；左下角标注 n = 13 patients。',
        '沿用 Figure 3C 的展示方式：配对数据下方标 Model–patient 及 n = 13 patients，图内右下角用蓝色 Observed、灰色 Channel-shuffle null 标识两组，纵轴为 Field similarity |r|。仍为各患者最大绝对场相关，统计定义与原置换检验不变。')
    (OUT/'caption_zh.md').write_text(caption)
    readme=(SOURCE/'figures/README.md').read_text()
    start=readme.index('### fig6-panele.png')
    end=readme.find('\n### ',start+1)
    if end<0:end=len(readme)
    readme=readme[:start]+('### fig6-panele.png / .pdf / .svg\n\n'
        '沿用 Figure 3 左下角的展示方式，X 轴在配对数据中央标 Model–patient，下一行标 n = 13 patients。'
        '纵轴为 Field similarity |r|，图内右下角的蓝灰图例区分 Observed 与 Channel-shuffle null。'
        '原患者配对点、最佳匹配值和置换检验保持。\n'
        '**关注点**：统一展示语法，不将 Figure 3 的统计方法套用到 Figure 6。\n')+readme[end:]
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text('# Figure 6：E 与 Figure 3C 统一标法\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf)\n\n'
        '配对数据下方显示比较对象和患者数，图内右下角显示 Observed / Channel-shuffle null。数据与统计保持。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__, OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks),indent=2))


if __name__ == '__main__':
    main()
