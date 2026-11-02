#!/usr/bin/env python3
"""Put E comparison labels in an upper-right legend and patient n below left."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_concise_labels as previous
import build_fig6_selected_r08_r09_normalized_onsets as normalized

base, layout, plt = previous.base, previous.layout, previous.plt
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_legend_sample_20261102')
LABELS = ['Model–patient', 'Within-patient shuffle']
ARTIST_BOUNDS = {}


def export_e(fig, stem):
    from matplotlib.lines import Line2D
    from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import _p_stars

    assert stem == 'fig6-panele'
    stat = json.loads((SOURCE/'cohort_statistics_selected.json').read_text())
    ax = fig.axes[0]
    fig.set_size_inches(62/25.4, 71/25.4)
    ax.set_position([18/62, 15/71, 40.6/62, 46/71])
    ax.set_xticks([])
    ax.set_xlabel('')
    assert len(ax.texts) == 1 and 'p =' in ax.texts[0].get_text()
    ax.texts[0].set_text(_p_stars(stat['p']))
    ax.texts[0].set_fontsize(layout.LABEL*1.2)
    handles = [Line2D([], [], linestyle='none', marker='o', markersize=4.8,
                      markerfacecolor=color, markeredgecolor='#333333', markeredgewidth=.45)
               for color in ['#b45b40', '#a2a2a2']]
    legend = fig.legend(handles, LABELS, loc='upper right',
                        bbox_to_anchor=(58.6/62, 70/71), bbox_transform=fig.transFigure,
                        frameon=False, fontsize=layout.TICK, borderaxespad=0,
                        borderpad=0, handlelength=.8, handletextpad=.45, labelspacing=.22)
    n_text = ax.text(0, -.075, f'n = {stat["n_patients"]} patients',
                     transform=ax.transAxes, ha='left', va='top', fontsize=layout.TICK)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    axes_box = ax.get_window_extent(renderer)
    n_box = n_text.get_window_extent(renderer)
    assert legend_box.y0 > axes_box.y1, 'Legend must clear the plot and significance bracket.'
    assert n_box.y1 < axes_box.y0 and n_box.x0 >= axes_box.x0-.01
    ARTIST_BOUNDS.update(legend_above_plot=True, sample_size_below_left=True,
                        legend_bbox_px=list(legend_box.bounds), n_bbox_px=list(n_box.bounds))
    normalized.matched.OUT = OUT
    normalized.NATIVE_EXPORT(fig, stem)


def main():
    pointer = base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    frozen = pointer.read_bytes()
    shutil.copytree(SOURCE, OUT, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('validation.json', 'visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42, 'svg.fonttype':'none', 'path.simplify':False})
    stat = json.loads((SOURCE/'cohort_statistics_selected.json').read_text())
    assert stat['n_patients'] == len(stat['matches']) == len({r['subject'] for r in stat['matches']}) == 13
    layout.OUT = OUT
    layout.export = export_e
    layout.draw_e(stat)
    meta = json.loads((SOURCE/'figure_metadata.json').read_text())
    compositor = previous.compositor
    compositor.OUT = OUT
    compositor.SOURCE = SOURCE
    compositor.C_DISPLAY.update(meta['C_display'])
    compositor.compose()
    meta['E_display'].update(x_tick_labels=[], x_label='',
        legend_labels=LABELS, legend_location='upper right, above axes',
        legend_colors=['#b45b40','#a2a2a2'], legend_font_pt=layout.TICK,
        sample_size_label='n = 13 patients', sample_size_location='below axes, left aligned',
        sample_size_font_pt=layout.TICK, explanatory_text_below_categories=False,
        final_size_mm=[62,71], axis_bounds=[18/62,15/71,40.6/62,46/71])
    meta.update(producer=str(Path(__file__).resolve()), previous_candidate=str(SOURCE),
        revision_scope='Upper-right E color legend and lower-left patient count; remove category text below the two groups.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json', meta)
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
            content=doc[0].get_text()
            assert 'n = 13 patients' in content and all(label in content for label in LABELS)
    assert all(checks.values()), checks
    assert pointer.read_bytes() == frozen
    base.write(OUT/'validation.json', dict(unchanged_panel_files=unchanged,
        all_data_statistics_unchanged=True, n_unique_patients=13,
        legend_labels=LABELS, category_labels_removed=True, **ARTIST_BOUNDS,
        PDF_text_inside_canvas=checks, formal_pointer_unchanged=True, human_visual_acceptance='PENDING'))
    caption=(SOURCE/'caption_zh.md').read_text().replace(
        '两组横轴直接标为 Patient model similarity 和 Patient shuffle similarity；图内不另加底部说明文字。',
        '右上角用棕色 Model–patient 和灰色 Within-patient shuffle 图例标识两组；左下角标注 n = 13 patients。')
    (OUT/'caption_zh.md').write_text(caption)
    readme=(SOURCE/'figures/README.md').read_text()
    start=readme.index('### fig6-panele.png')
    end=readme.find('\n### ',start+1)
    if end<0: end=len(readme)
    readme=readme[:start]+('### fig6-panele.png\n\n'
        '右上角图例用棕色标识 Model–patient、灰色标识 Within-patient shuffle，左下角标注 n = 13 patients。'
        '移除两组下方的长标签，保留原始配对点、统计和绘图区尺寸。\n'
        '**关注点**：图例与点颜色对应，n 表示患者人数。\n')+readme[end:]
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text('# Figure 6：E 图例与患者数\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf)\n\n'
        'E 右上角显示颜色图例，左下角显示 n = 13 patients。其他面板、数据和统计保持。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__, OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks),indent=2))


if __name__ == '__main__':
    main()
