#!/usr/bin/env python3
"""Use direct E category labels with no shared explanatory text beneath them."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_patient_labels as previous
import build_fig6_selected_r08_r09_large_maps as resize
import build_fig6_selected_r08_r09_enlarged_c as compositor

base,layout,plt = previous.base,previous.layout,previous.plt
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_concise_labels_20261102')
LABELS = ['Patient\nmodel\nsimilarity','Patient\nshuffle\nsimilarity']


def export_e(fig,stem):
    assert stem=='fig6-panele'
    ax=fig.axes[0]
    ax.set_xticks([1,2],LABELS)
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_linespacing(1.05)
    resize.OUT=OUT
    resize.export_resized(fig,stem)


def main():
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json';frozen=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    stat=json.loads((SOURCE/'cohort_statistics_selected.json').read_text())
    layout.OUT=OUT;layout.export=export_e
    layout.draw_e(stat)
    meta=json.loads((SOURCE/'figure_metadata.json').read_text())
    compositor.OUT=OUT;compositor.SOURCE=SOURCE;compositor.C_DISPLAY.update(meta['C_display'])
    compositor.compose()
    meta['E_display'].update(final_size_mm=[62,71],plot_bottom_mm=15,
        axis_bounds=[18/62,15/71,40.6/62,46/71],x_tick_labels=LABELS,x_label='',
        explanatory_text_below_categories=False,footnote=False)
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),
        revision_scope='Direct Patient model similarity / Patient shuffle similarity category labels; remove shared explanatory lines.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json',meta)
    copied=[]
    for stem in ['fig6-panela','fig6-panelb','fig6-panelc','fig6-paneld','fig6-panelb-interictal','fig6-panelb-full-trajectory']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}'
            assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes()
            copied.append(name)
    assert (OUT/'cohort_statistics_selected.json').read_bytes()==(SOURCE/'cohort_statistics_selected.json').read_bytes()
    checks={}
    for stem in ['fig6-panele','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
            text=doc[0].get_text()
            assert 'Within-patient comparisons' not in text and '13 patients' not in text
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=copied,statistics_unchanged=True,
        direct_category_labels=LABELS,shared_x_label_removed=True,original_compact_canvas_restored_mm=[320,294],
        PDF_text_inside_canvas=checks,formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    caption=(SOURCE/'caption_zh.md').read_text().replace('横轴明确患者内比较和每位患者一对点。',
        '两组横轴直接标为 Patient model similarity 和 Patient shuffle similarity；图内不另加底部说明文字。')
    (OUT/'caption_zh.md').write_text(caption)
    readme=(SOURCE/'figures/README.md').read_text().replace('横轴补明比较和统计单位，全部数值及统计保持。',
        '两组下方直接显示 Patient model similarity 和 Patient shuffle similarity，取消共用横轴下的两行说明；全部数值及统计保持。')
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text('# Figure 6：E 使用简短分组标签\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf)\n\n'
        '两组直接标为 Patient model similarity / Patient shuffle similarity，无额外底部说明。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks),indent=2))


if __name__=='__main__':main()
