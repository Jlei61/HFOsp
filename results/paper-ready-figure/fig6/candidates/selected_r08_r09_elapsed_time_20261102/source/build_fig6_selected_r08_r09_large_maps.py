#!/usr/bin/env python3
"""Enlarge D, narrow/right-align E and display the frozen p value as stars."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_integrated_detail as previous

base=previous.base
np,plt=previous.np,previous.plt
MM=previous.MM
SOURCE=previous.OUT.with_name('selected_r08_r09_entry_runup_20261031')
OUT=SOURCE.with_name('selected_r08_r09_large_maps_20261031')
HEIGHT=71.
PLOT_BOTTOM,PLOT_HEIGHT=15.,46.
E_WIDTH,E_PLOT_WIDTH=62.,40.6
LAYOUT_AXES={}


def export_resized(fig,stem):
    # Keep the accepted artists and data; change their physical allocation.
    if stem=='fig6-paneld':
        width=245.
        fig.set_size_inches(width/25.4,HEIGHT/25.4)
        assert len(fig.axes)==5
        for ax,x in zip(fig.axes[:4],[15.,66.,119.,170.]):
            ax.set_position([x/width,PLOT_BOTTOM/HEIGHT,PLOT_HEIGHT/width,PLOT_HEIGHT/HEIGHT])
        fig.axes[-1].set_position([220/width,PLOT_BOTTOM/HEIGHT,1.7/width,PLOT_HEIGHT/HEIGHT])
        assert len(fig.texts)==1
        fig.texts[0].set_position((115.5/width,8.5/HEIGHT))
    elif stem=='fig6-panele':
        from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import _p_stars
        width=E_WIDTH
        fig.set_size_inches(width/25.4,HEIGHT/25.4)
        ax=fig.axes[0]
        ax.set_position([18/width,PLOT_BOTTOM/HEIGHT,E_PLOT_WIDTH/width,PLOT_HEIGHT/HEIGHT])
        stat,_=previous.load_stat()
        stars=_p_stars(stat['p']);assert stars=='**'
        assert len(ax.texts)==1 and 'p =' in ax.texts[0].get_text()
        ax.texts[0].set_text(stars);ax.texts[0].set_fontsize(previous.LABEL*1.2)
    else:
        raise ValueError(stem)
    fig.canvas.draw()
    axes=[ax.get_position().bounds for ax in fig.axes]
    for bounds in axes:
        np.testing.assert_allclose(np.array([bounds[1],bounds[1]+bounds[3]])*HEIGHT,
            [PLOT_BOTTOM,PLOT_BOTTOM+PLOT_HEIGHT],atol=1e-8)
    LAYOUT_AXES[stem]=axes
    previous.previous.OUT=OUT
    previous.previous.export(fig,stem)


def compose():
    from matplotlib.font_manager import findfont,FontProperties
    doc=base.fitz.open();page=doc.new_page(width=320*MM,height=324*MM)
    font=findfont(FontProperties(family='DejaVu Sans',weight='bold'))
    page.insert_font(fontname='FigureBold',fontfile=font)
    boxes={'A':(0,1,320,108),'B':(3,108,142,232),'C':(144,115,317,231),
        'D':(3,250,248,321),'E':(255,250,317,321)}
    letters={'B':(3,112),'C':(144,112),'D':(3,247),'E':(254,247)}
    records={}
    for label,box in boxes.items():
        source=OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(source) as src:
            x0,y0,x1,y1=box
            scale=min((x1-x0)*MM/src[0].rect.width,(y1-y0)*MM/src[0].rect.height)
            rect=base.fitz.Rect(x0*MM,y0*MM,x0*MM+src[0].rect.width*scale,y0*MM+src[0].rect.height*scale)
            page.show_pdf_page(rect,src,0,keep_proportion=True)
        if label in letters:
            page.insert_text(tuple(v*MM for v in letters[label]),label,fontsize=previous.LETTER,fontname='FigureBold')
        if label!='A':np.testing.assert_allclose(scale,1,atol=1e-6)
        records[label]=dict(source=str(source),box_mm=[v/MM for v in rect],scale=scale)
    # A's accepted waveform axes end at x=0.98 of its 320 mm panel.
    A_RIGHT=320*.98
    E_RIGHT=boxes['E'][0]+18+E_PLOT_WIDTH
    np.testing.assert_allclose(A_RIGHT,E_RIGHT,atol=1e-10)
    path=OUT/'figures/fig6-complete-layout.pdf';doc.save(path,garbage=4,deflate=True);doc.close()
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72,220/72),alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    old=json.loads((SOURCE/'layout_manifest.json').read_text())
    for label in 'ABC':
        np.testing.assert_allclose(records[label]['box_mm'],old['panels'][label]['box_mm'],atol=1e-5)
    base.write(OUT/'layout_manifest.json',dict(canvas_mm=[320,324],panels=records,
        D_E_plot_height_mm=PLOT_HEIGHT,bottom_row_plot_top_mm=260.,bottom_row_plot_bottom_mm=306.,
        A_E_right_plot_edge_mm=A_RIGHT,E_right_panel_edge_mm=317.,
        same_field_and_statistic_plot_height=True,typography=old['typography']))


def documents(meta):
    caption=(SOURCE/'caption_zh.md').read_text()
    old=caption.split('\n\n')[-1]
    new='D 四幅场图及色条绘图区高均为 46 mm，每幅场图边长由 40 mm 增至 46 mm。E 绘图区为 40.6 × 46 mm，保留与 D 相同的上下边缘；其右侧坐标轴对齐 A 右边界 313.6 mm，panel 外边界与 C 同为 317 mm。E 使用原历史 13 人、140 次合格发作的 full_energy_plus200ms__db 结果和原有小提琴、箱线、配对点线绘图函数；原始 p=0.00708，图面按既有规则显示 **（p<0.01）。星号映射为 * p<0.05、** p<0.01、*** p<0.001。本次只调整展示，原始数值和统计没有重算。A/B/C 与上一版字节一致。本版为候选，待作者目视检查。\n'
    (OUT/'caption_zh.md').write_text(caption.replace(old,new))
    readme=(SOURCE/'figures/README.md').read_text()
    parts=readme.split('\n\n### ')
    replacements={
        'fig6-complete-layout':'完整 A–E 的前三个面板保持。底部 D 四幅地图放大，E 收窄并右对齐，显著性改用星号。',
        'fig6-paneld':'四幅模型与患者场图各由 40 mm 放大到 46 mm。原始场、配对标题颜色与竖色条语义保持。',
        'fig6-panele':'历史队列的原始统计与配对点保持，绘图区收窄为 40.6 × 46 mm。图面显示 **，代表原 p=0.00708 满足 p<0.01。'}
    for j,part in enumerate(parts):
        head=part.split('\n',1)[0].removeprefix('### ')
        for stem,desc in replacements.items():
            if head.startswith(stem+'.png'):
                parts[j]=('### ' if j==0 else '')+head+'\n\n'+desc+'\n**关注点**：本次调整尺寸及显著性显示，不重算数据。'
    (OUT/'figures/README.md').write_text('\n\n### '.join(parts))
    (OUT/'README.md').write_text('# Figure 6：放大地图与右对齐统计图\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf)\n\n'
        'D 地图放大，E 收窄并与 A 右边界对齐，p 值改为 **。A/B/C 保持。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')


def main():
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json';frozen=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,
        'svg.fonttype':'none','path.simplify':False})
    previous.OUT=OUT;previous.export=export_resized
    stat,alias=previous.load_stat();matches=json.loads((SOURCE/'model_selected_patient_matches.json').read_text())
    dmeta=previous.draw_d(matches,alias);emeta=previous.draw_e(stat)
    dmeta.update(final_size_mm=[245,HEIGHT],field_plot_height_mm=PLOT_HEIGHT,colorbar_height_mm=PLOT_HEIGHT,
        field_axes_bounds=LAYOUT_AXES['fig6-paneld'][:4],colorbar_bounds=LAYOUT_AXES['fig6-paneld'][-1])
    emeta.update(final_size_mm=[E_WIDTH,HEIGHT],plot_width_mm=E_PLOT_WIDTH,plot_height_mm=PLOT_HEIGHT,
        significance_text='**',significance_rule='* p<0.05; ** p<0.01; *** p<0.001',
        numeric_p_displayed=False,plot_right_global_mm=313.6,axis_bounds=LAYOUT_AXES['fig6-panele'][0])
    compose()
    meta=json.loads((SOURCE/'figure_metadata.json').read_text())
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),D=dmeta,E_display=emeta,
        revision_scope='Larger maps, narrower right-aligned E and significance stars; data unchanged.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json',meta)
    copied=[]
    for stem in ['fig6-panela','fig6-panelb','fig6-panelc','fig6-panelb-interictal','fig6-panelb-full-trajectory']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}';assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes();copied.append(name)
    for name in ['trajectory_data.npz','stage_readouts.npz','field_comparisons.json','A_C_stage_correspondence.csv',
        'model_selected_patient_matches.json','model_selected_patient_arrays.npz','all_patient_seizure_matches.csv','cohort_statistics_selected.json']:
        assert (OUT/name).read_bytes()==(SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-paneld','fig6-panele','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
            if stem=='fig6-panele':
                text=doc[0].get_text();assert '**' in text and '0.0071' not in text
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=copied,
        all_underlying_trajectory_fields_stage_times_matches_and_statistics_unchanged=True,
        A_B_C_placement_unchanged=True,D_E_equal_plot_height_mm=46.,E_right_aligned_to_A_mm=313.6,
        stars='**',original_p=stat['p'],significance_helper='plot_fig3_field_concordance_cohort_stat._p_stars',
        PDF_text_inside_canvas=checks,formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    documents(meta)
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks),indent=2))


if __name__=='__main__':main()
