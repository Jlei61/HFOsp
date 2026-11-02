#!/usr/bin/env python3
"""Clarify the patient-level paired units and within-patient E comparison."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_normalized_onsets as previous

base,layout,matched = previous.base,previous.layout,previous.matched
np,plt = previous.np,previous.plt
MM = 72/25.4
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_patient_labels_20261102')
E_HEIGHT = 76.
TICK_LABELS = ['Model–seizure\nbest match','Contact\nshuffle null']
X_LABEL = 'Within-patient comparisons\n13 patients; one pair each'


def export_e(fig,stem):
    assert stem == 'fig6-panele'
    from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import _p_stars
    fig.set_size_inches(62/25.4,E_HEIGHT/25.4)
    ax=fig.axes[0]
    # Add room below the axes; retain the plotted region at the original size and top.
    ax.set_position([18/62,20/E_HEIGHT,40.6/62,46/E_HEIGHT])
    ax.set_xticks([1,2],TICK_LABELS)
    ax.set_xlabel(X_LABEL,fontsize=layout.TICK,labelpad=3,linespacing=1.15)
    stat=json.loads((SOURCE/'cohort_statistics_selected.json').read_text())
    assert len(ax.texts)==1 and 'p =' in ax.texts[0].get_text()
    ax.texts[0].set_text(_p_stars(stat['p']))
    ax.texts[0].set_fontsize(layout.LABEL*1.2)
    matched.OUT=OUT
    previous.NATIVE_EXPORT(fig,stem)


def compose():
    from matplotlib.font_manager import findfont,FontProperties
    manifest=json.loads((SOURCE/'layout_manifest.json').read_text())
    manifest['canvas_mm']=[320,299]
    manifest['panels']['E']['box_mm']=[255,220,317,220+E_HEIGHT]
    doc=base.fitz.open();page=doc.new_page(width=320*MM,height=299*MM)
    font=findfont(FontProperties(family='DejaVu Sans',weight='bold'))
    page.insert_font(fontname='FigureBold',fontfile=font)
    letters={'B':(3,106),'C':(140,106),'D':(3,220),'E':(254,220)}
    for label,record in manifest['panels'].items():
        path=OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(path) as src:
            rect=base.fitz.Rect([v*MM for v in record['box_mm']])
            page.show_pdf_page(rect,src,0,keep_proportion=True)
        record['source']=str(path)
        if label in letters:
            page.insert_text(tuple(v*MM for v in letters[label]),label,fontsize=layout.LETTER,fontname='FigureBold')
    path=OUT/'figures/fig6-complete-layout.pdf';doc.save(path,garbage=4,deflate=True);doc.close()
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72,220/72),alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    manifest.update(layout_unchanged_above_E_labels=True,E_extra_lower_label_margin_mm=5.,
        D_E_plot_height_mm=46.,bottom_row_plot_top_mm=230.,bottom_row_plot_bottom_mm=276.,
        layout_source=str(SOURCE))
    base.write(OUT/'layout_manifest.json',manifest)


def main():
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json';frozen=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    stat=json.loads((SOURCE/'cohort_statistics_selected.json').read_text())
    assert stat['n_patients']==len(stat['matches'])==13 and stat['n_seizures']==140
    assert len({r['subject'] for r in stat['matches']})==13
    layout.OUT=OUT;layout.export=export_e
    layout.draw_e(stat)
    compose()
    meta=json.loads((SOURCE/'figure_metadata.json').read_text())
    meta['E_display'].update(final_size_mm=[62,E_HEIGHT],plot_bottom_mm=20,
        axis_bounds=[18/62,20/E_HEIGHT,40.6/62,46/E_HEIGHT],x_tick_labels=TICK_LABELS,x_label=X_LABEL,
        x_label_font_pt=layout.TICK,statistical_unit='patient',one_pair_per_patient=True,
        observed_definition='Maximum absolute model-seizure field correlation over the same patient eligible seizures.',
        null_point_definition='Median of 99999 within-patient model-contact permutations; shuffle before spatial smoothing and reselect the best own seizure in every permutation.',
        cohort_p_definition='Upper-tail permutation test of the median observed best match across the 13 patients.',
        null_is_cross_patient_shuffle=False)
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),
        revision_scope='Clarify E patient-level pairs and within-patient null without changing the data or statistic.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json',meta)
    unchanged=[]
    for stem in ['fig6-panela','fig6-panelb','fig6-panelc','fig6-paneld','fig6-panelb-interictal','fig6-panelb-full-trajectory']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}'
            assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes()
            unchanged.append(name)
    for name in ['cohort_statistics_selected.json','stage_readouts.npz','normalized_propagation_fields.npz',
                 'model_selected_patient_arrays.npz','trajectory_data.npz']:
        assert (OUT/name).read_bytes()==(SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-panele','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=unchanged,all_data_statistics_unchanged=True,
        n_unique_patients=13,one_pair_per_patient=True,within_patient_null_confirmed_from_analysis_source=True,
        E_plot_size_and_top_unchanged=True,PDF_text_inside_canvas=checks,
        formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    caption=(SOURCE/'caption_zh.md').read_text()
    paragraph=next(p for p in caption.split('\n\n') if p.startswith('E：'))
    caption=caption.replace(paragraph,'E：13 位患者的配对展示，每位患者一对点。左侧取该患者模型与其自身合格发作之间的最大绝对场相关；右侧是同一患者内 99,999 次模型触点置换所得最佳匹配分布的中位数。置换发生在原空间平滑之前，每次仍从该患者同一候选发作池中重选最佳匹配，不交换患者身份。所有患者合计 140 次合格发作；p=0.00708 来自以 13 位患者最佳相关的中位数为统计量的队列置换检验，图面保留 **。沿用 full_energy_plus200ms__db 数值、点线、小提琴和箱线，不重算统计；横轴明确患者内比较和每位患者一对点。')
    (OUT/'caption_zh.md').write_text(caption)
    readme=(SOURCE/'figures/README.md').read_text().replace('历史患者统计保持。与上一版导出字节一致。',
        '每位患者一对点，共 13 位患者：左侧为患者内最佳模型—发作匹配，右侧为患者内触点置换 null 的中位数。横轴补明比较和统计单位，全部数值及统计保持。')
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text('# Figure 6：明确 E 的患者统计单位\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf)\n\n'
        'E 为 13 位患者，每位患者一对观测最佳匹配与自身置换 null 中位数。仅更新标签及所需下边距。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks),indent=2))


if __name__=='__main__':main()
