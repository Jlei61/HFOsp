#!/usr/bin/env python3
"""Remove only the patient-count line from the approved Fig6E label layout."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_model_null_labels as previous

base, layout, plt = previous.base, previous.layout, previous.plt
normalized, compositor = previous.normalized, previous.compositor
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_without_n_20261102')
NATIVE_EXPORT = normalized.NATIVE_EXPORT


def export_without_n(fig, stem):
    # Keep the immediately preceding renderer intact and remove just its xlabel.
    fig.axes[0].set_xlabel('')
    normalized.matched.OUT = OUT
    NATIVE_EXPORT(fig,stem)


def main():
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    frozen=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    stat=json.loads((SOURCE/'cohort_statistics_selected.json').read_text())
    layout.OUT=OUT
    layout.export=previous.export_e
    previous.OUT=OUT
    normalized.NATIVE_EXPORT=export_without_n
    try:
        layout.draw_e(stat)
    finally:
        normalized.NATIVE_EXPORT=NATIVE_EXPORT
    meta=json.loads((SOURCE/'figure_metadata.json').read_text())
    compositor.OUT,compositor.SOURCE=OUT,SOURCE
    compositor.C_DISPLAY.update(meta['C_display'])
    compositor.compose()
    meta['E_display'].update(x_label='',sample_size_label='',sample_size_location=None,
                             sample_size_displayed=False)
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),
        revision_scope='Remove only the n = 13 patients line from E; preserve the complete layout.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json',meta)
    unchanged=[]
    for stem in ['fig6-panela','fig6-panelb','fig6-panelc','fig6-paneld',
                 'fig6-panelb-interictal','fig6-panelb-full-trajectory']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}'
            assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes()
            unchanged.append(name)
    assert (OUT/'cohort_statistics_selected.json').read_bytes()==(SOURCE/'cohort_statistics_selected.json').read_bytes()
    checks={}
    for stem in ['fig6-panele','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
            content=doc[0].get_text()
            assert 'patients' not in content and 'Model–seizure' in content and 'NULL' in content
    assert all(checks.values()) and pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=unchanged,statistics_unchanged=True,
        patient_count_line_removed=True,original_colors_and_no_legend_retained=True,
        PDF_text_inside_canvas=checks,formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    caption=(SOURCE/'caption_zh.md').read_text().replace(
        '两组标签下方独立一行显示 n = 13 patients，不加图例。','去掉图面最下方的患者数一行，不加图例。')
    (OUT/'caption_zh.md').write_text(caption)
    readme=(SOURCE/'figures/README.md').read_text().replace(
        '最下面独立一行显示 n = 13 patients，无图例；原配对数据和统计保持。',
        '去掉最下面的患者数一行，无图例；原配对数据和统计保持。').replace(
        '**关注点**：核对左右标签、人数及原配色。','**关注点**：仅去掉患者数文字，左右标签和原配色保持。')
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text('# Figure 6：完整布局，E 去掉患者数文字\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf)\n\n'
        '仅去掉 E 最下方的 n = 13 patients 文字，所有其他图层、布局和统计保持。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks),indent=2))


if __name__=='__main__':
    main()
