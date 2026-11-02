#!/usr/bin/env python3
"""Compact Fig6 rows, align the time bar and annotate the interictal inset."""
import itertools
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_large_maps as large
import build_fig6_selected_r08_r09_entry_runup as entry

layout=large.previous
matched=layout.previous
base,recovery=layout.base,layout.recovery
np,plt=layout.np,layout.plt
MM=layout.MM
SOURCE=large.OUT
OUT=SOURCE.with_name('selected_r08_r09_compact_rows_20261031')
NATIVE_EXPORT=matched.export
C_HEIGHT=98.
DETAIL_RECORD={}


def export(fig,stem):
    matched.OUT=OUT
    NATIVE_EXPORT(fig,stem)


def draw_b(meta):
    with np.load(SOURCE/'trajectory_data.npz') as data:
        take=data['time_s']<recovery.HORIZON
        t,xyz=data['time_s'][take],data['Z_etaM_rate'][take]
    fig=plt.figure(figsize=np.array([139.,124.])/25.4)
    ax=fig.add_axes([.065,.11,.75,.80],projection='3d',computed_zorder=False)
    bounds=np.array([[0.,1.],[0.,1000.],[0.,500.]])
    segments,keep=recovery.add_curve(ax,t,xyz,bounds);assert keep.all()
    norm=base.Normalize(0,recovery.HORIZON)
    main_curve=ax.collections[-1];main_curve.set_norm(norm)
    ax.set_box_aspect((1.6,1.,1.25),zoom=1.03)
    ax.set(xlabel='$Z$',ylabel=r'$\eta_M M$ (mV)',xticks=[0,.5,1.],yticks=[0,500,1000],zticks=[0,250,500])
    ax.set_yticklabels(['','500','1000']);ax.tick_params(labelsize=layout.TICK,pad=1)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.label.set_fontsize(layout.LABEL);axis.labelpad=4
    fig.text(.012,.525,r'$r_E$ (Hz)',rotation=90,va='center',fontsize=layout.LABEL)
    for sec in meta['trajectory']['first_returned_brief_s']:
        pt=[np.interp(sec,t,xyz[:,k]) for k in range(3)]
        ax.scatter(*pt,marker='D',s=28,color=plt.colormaps['viridis'](sec/recovery.HORIZON),
            edgecolor='black',linewidth=.6,depthshade=False,zorder=12)
    main_labels=layout.parent.reference.prior.label_states(ax,meta['states'][3:4],t,xyz,fontsize=layout.TITLE)
    zoom=fig.add_axes([.49,.565,.37,.31],projection='3d',computed_zorder=False,facecolor='none')
    inset=entry.add_detail(zoom,t,xyz,inset=True)
    inset_curve=zoom.collections[0];inset_curve.set_norm(norm)
    for axis in [zoom.xaxis,zoom.yaxis,zoom.zaxis]:
        axis.pane.set_facecolor('#f2f5fa');axis.pane.set_alpha(1)
    fig.text(.675,.917,'Interictal',fontsize=layout.LABEL,ha='center',va='top')
    valid_states=[]
    for state in meta['states'][:4]:
        point=np.array([np.interp(state['time_s'],t,xyz[:,k]) for k in range(3)])
        visible=entry.WINDOW[0]<=state['time_s']<entry.WINDOW[1] and np.all((point>=entry.BOUNDS[:,0])&(point<=entry.BOUNDS[:,1]))
        if visible:valid_states.append(state)
    assert [s['number'] for s in valid_states]==[1,2,3]
    inset_labels=layout.parent.reference.prior.label_states(zoom,valid_states,t,xyz,fontsize=layout.TITLE)
    fig.canvas.draw()
    # Match the bar to the projection of the full 3D coordinate box.
    corners=np.array(list(itertools.product(*bounds)))
    px,py,_=base.proj3d.proj_transform(*corners.T,ax.get_proj())
    projected=fig.transFigure.inverted().transform(ax.transData.transform(np.c_[px,py]))
    bottom,top=float(projected[:,1].min()),float(projected[:,1].max())
    cb=fig.colorbar(main_curve,cax=fig.add_axes([.865,bottom,.023,top-bottom]),ticks=[0,60,recovery.HORIZON])
    cb.set_label('Time (s)',fontsize=layout.LABEL,labelpad=4)
    cb.ax.tick_params(labelsize=layout.TICK,length=2,pad=3)
    assert main_curve.norm is inset_curve.norm is cb.norm
    np.testing.assert_allclose(cb.ax.get_position().extents[[1,3]],[bottom,top],atol=1e-12)
    recovery.check_projected(fig,ax,segments)
    export(fig,'fig6-panelb')
    inset.update(title='Interictal',annotation_state_numbers=[1,2,3],states=inset_labels)
    # The separate detail shares the exact same states and trajectory.
    fig=plt.figure(figsize=np.array([160.,125.])/25.4)
    detail_ax=fig.add_axes([.10,.09,.76,.78],projection='3d',computed_zorder=False)
    detail=entry.add_detail(detail_ax,t,xyz,inset=False)
    detail_labels=layout.parent.reference.prior.label_states(detail_ax,valid_states,t,xyz,fontsize=layout.TITLE)
    fig.text(.07,.98,'Interictal',fontsize=layout.LABEL,va='top')
    fig.text(.02,.51,r'$r_E$ (Hz)',fontsize=layout.LABEL,rotation=90,va='center')
    export(fig,'fig6-panelb-interictal')
    detail.update(title='Interictal',annotation_state_numbers=[1,2,3],states=detail_labels)
    return dict(final_size_mm=[139,124],limits=bounds,segment_count=int(keep.sum()),
        unchanged_full_trajectory=True,main_state_numbers=[4],inset_state_numbers=[1,2,3],
        states=main_labels+inset_labels,inset=inset,standalone_interictal=detail,
        inset_axes_position=[.49,.565,.37,.31],shared_colorbar=True,shared_normalization_object=True,
        shared_colormap='viridis',colorbar_bounds=[.865,bottom,.023,top-bottom],
        colorbar_aligned_to_projected_3D_box=True,colorbar_height_mm=(top-bottom)*124,
        font_label_pt=layout.LABEL,font_tick_pt=layout.TICK)


def compact_c_export(fig,stem):
    assert stem=='fig6-panelc' and len(fig.axes)==10
    width=173.
    fig.canvas.draw()
    old_positions=[ax.get_position().bounds for ax in fig.axes]
    field_width=old_positions[0][2]*width
    old_gap=(old_positions[0][1]-old_positions[4][1]-old_positions[4][3])*116
    fig.set_size_inches(width/25.4,C_HEIGHT/25.4)
    positions=[]
    for j,ax in enumerate(fig.axes[:8]):
        row=j//4;top_from_page=11. if row==0 else 56.
        x=old_positions[j][0];w=old_positions[j][2]
        y=(C_HEIGHT-top_from_page-field_width)/C_HEIGHT
        ax.set_position([x,y,w,field_width/C_HEIGHT]);positions.append(ax.get_position().bounds)
    for row,cbax in enumerate(fig.axes[8:]):
        pos=fig.axes[4*row+3].get_position()
        cbax.set_position([.902,pos.y0,.012,pos.height])
        fig.texts[row].set_position((.021,pos.y0+pos.height/2))
    fig.canvas.draw()
    for row in range(2):
        np.testing.assert_allclose(fig.axes[8+row].get_position().extents[[1,3]],fig.axes[4*row+3].get_position().extents[[1,3]],atol=1e-12)
    DETAIL_RECORD['C']=dict(final_size_mm=[width,C_HEIGHT],field_size_mm=field_width,
        previous_row_gap_mm=old_gap,row_gap_mm=56.-11.-field_width,
        field_axes_bounds=positions,colorbar_bounds=[a.get_position().bounds for a in fig.axes[8:]])
    export(fig,stem)


def draw_c(meta):
    matched.SOURCE=SOURCE
    matched.export=compact_c_export
    try:
        result=matched.draw_c(meta)
    finally:
        matched.export=NATIVE_EXPORT
    result.update(DETAIL_RECORD['C'])
    result['individual_row_colorbars']=result['colorbar_bounds']
    return result


def draw_d(matches,alias):
    def black_model_export(fig,stem):
        assert stem=='fig6-paneld'
        for ax in [fig.axes[0],fig.axes[2]]:ax.title.set_color('black')
        large.export_resized(fig,stem)
    large.OUT=OUT;layout.OUT=OUT;layout.export=black_model_export
    result=layout.draw_d(matches,alias)
    result.update(final_size_mm=[245,71],field_plot_height_mm=46.,colorbar_height_mm=46.,
        field_axes_bounds=large.LAYOUT_AXES['fig6-paneld'][:4],colorbar_bounds=large.LAYOUT_AXES['fig6-paneld'][-1],
        model_title_color='black',patient_title_colors=layout.PAIR_COLORS)
    return result


def compose():
    from matplotlib.font_manager import findfont,FontProperties
    doc=base.fitz.open();page=doc.new_page(width=320*MM,height=302*MM)
    font=findfont(FontProperties(family='DejaVu Sans',weight='bold'));page.insert_font(fontname='FigureBold',fontfile=font)
    boxes={'A':(0,1,320,108),'B':(3,102,142,226),'C':(144,109,317,207),
        'D':(3,228,248,299),'E':(255,228,317,299)}
    letters={'B':(3,106),'C':(144,106),'D':(3,225),'E':(254,225)}
    records={}
    for label,box in boxes.items():
        source=OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(source) as src:
            x0,y0,x1,y1=box;scale=min((x1-x0)*MM/src[0].rect.width,(y1-y0)*MM/src[0].rect.height)
            rect=base.fitz.Rect(x0*MM,y0*MM,x0*MM+src[0].rect.width*scale,y0*MM+src[0].rect.height*scale)
            page.show_pdf_page(rect,src,0,keep_proportion=True)
        if label in letters:page.insert_text(tuple(v*MM for v in letters[label]),label,fontsize=layout.LETTER,fontname='FigureBold')
        if label!='A':np.testing.assert_allclose(scale,1,atol=1e-6)
        records[label]=dict(source=str(source),box_mm=[v/MM for v in rect],scale=scale)
    path=OUT/'figures/fig6-complete-layout.pdf';doc.save(path,garbage=4,deflate=True);doc.close()
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72,220/72),alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    prior=json.loads((SOURCE/'layout_manifest.json').read_text())
    base.write(OUT/'layout_manifest.json',dict(canvas_mm=[320,302],panels=records,
        middle_row_shift_up_mm=6.,bottom_row_shift_up_mm=22.,
        row_separation_reduction_mm=dict(A_BC=6.,BC_DE=16.),
        D_E_plot_height_mm=46.,bottom_row_plot_top_mm=238.,bottom_row_plot_bottom_mm=284.,
        A_E_right_plot_edge_mm=313.6,E_right_panel_edge_mm=317.,
        same_field_and_statistic_plot_height=True,typography=prior['typography']))


def documents(meta):
    caption=(SOURCE/'caption_zh.md').read_text()
    first=caption.split('\n\n')[1]
    caption=caption.replace(first,'A 保持原版，B/C 整行上移 6 mm，D/E 整行上移 22 mm，使两处行间距分别缩小 6 mm 与 16 mm。C 两行场图之间的空隙从约 26.13 mm 缩至约 15.26 mm，每幅场图仍约 29.74 mm，原生数据、采样时刻与字体大小保持。B 主图保持完整 0–118 s 原生轨迹，小窗保持连续 5.90–9.24 s 的间期至第一次进入段，标题恢复 Interictal 且不写时间。小窗向左移动，1/2/3 放在其真实状态坐标；4 超出小窗的时间和电流范围，只放在主图。主图与小窗共用同一时间色条，色条上下边缘对齐主三维坐标框的投影范围。')
    caption=caption.replace('第一对标题为深绿，第二对为深蓝','两个 Model Onset 标题均为黑色，E10 SZ13 标题为深绿、E10 SZ14 标题为深蓝')
    caption=caption.replace('A/B/C 与上一版字节一致。','A/E 独立面板与上一版字节一致；B/C/D 仅调整展示。')
    (OUT/'caption_zh.md').write_text(caption)
    readme=(SOURCE/'figures/README.md').read_text()
    parts=readme.split('\n\n### ')
    replacements={
        'fig6-complete-layout':'完整 A–E 收紧两处行间距，C 两个模型之间的间距也缩小。地图与 E 的尺寸和右侧对齐保持。',
        'fig6-panelb':'主轨迹和小窗数据范围保持，小窗左移并只标 Interictal。1–3 在小窗原生位置，4 留在主图；时间色条上移并与主三维坐标框等高。',
        'fig6-panelb-interictal':'独立展示同一间期至进入轨迹，标题为 Interictal。标出真实的 1–3，4 不在此窗口内。',
        'fig6-panelb-full-trajectory':'独立全范围轨迹图与前版保持一致。完整原生状态仍可在此检查。',
        'fig6-panelc':'两行 Model Onset 1/2 的地图大小、数据与字号保持。两行之间的空隙缩小，每行色条仍与场图等高。',
        'fig6-paneld':'四幅场图尺寸与数据保持。Model Onset 1/2 标题恢复黑色，两个患者标题继续使用深绿与深蓝。',
        'fig6-panele':'历史队列的原始统计、配对点及绘图区尺寸保持，与前版导出字节一致。图面保留 **，代表原 p=0.00708 满足 p<0.01。'}
    for j,part in enumerate(parts):
        head=part.split('\n',1)[0].removeprefix('### ')
        for stem,desc in replacements.items():
            if head.startswith(stem+'.png'):
                parts[j]=('### ' if j==0 else '')+head+'\n\n'+desc+'\n**关注点**：本版只调整展示，待作者目视检查。'
    (OUT/'figures/README.md').write_text('\n\n### '.join(parts))
    (OUT/'README.md').write_text('# Figure 6：收紧行距与小窗阶段标记\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · '
        '[B 图](figures/fig6-panelb.png)\n\n'
        f'复现：`python {Path(__file__).resolve()}`。原生数据、阶段时刻及统计保持。\n')


def main():
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json';frozen=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    meta=json.loads((SOURCE/'figure_metadata.json').read_text())
    print('Rendering compact B/C and black model titles.',flush=True)
    bmeta=draw_b(meta);cmeta=draw_c(meta)
    matches=json.loads((SOURCE/'model_selected_patient_matches.json').read_text());_,alias=layout.load_stat()
    dmeta=draw_d(matches,alias);compose()
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),B_display=bmeta,C_display=cmeta,D=dmeta,
        revision_scope='Compact row spacing, native inset state labels, full-height time bar and black model titles.',status='PENDING_AUTHOR_VISUAL_REVIEW')
    meta['trajectory'].update(display_revision=bmeta,inset_title='Interictal',main_state_numbers=[4],inset_state_numbers=[1,2,3])
    meta['trajectory']['outputs']['fig6-panelb']=bmeta
    meta['trajectory']['outputs']['fig6-panelb-interictal']=bmeta['standalone_interictal']
    base.write(OUT/'figure_metadata.json',meta);base.write(OUT/'interictal_display_selection.json',bmeta['inset'])
    copied=[]
    for stem in ['fig6-panela','fig6-panele','fig6-panelb-full-trajectory']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}';assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes();copied.append(name)
    for name in ['trajectory_data.npz','stage_readouts.npz','field_comparisons.json','A_C_stage_correspondence.csv',
        'model_selected_patient_matches.json','model_selected_patient_arrays.npz','all_patient_seizure_matches.csv','cohort_statistics_selected.json']:
        assert (OUT/name).read_bytes()==(SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-panelb','fig6-panelb-interictal','fig6-panelc','fig6-paneld','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
            if stem.startswith('fig6-panelb'):
                text=doc[0].get_text();assert 'Interictal' in text and 'entry' not in text and '5.90' not in text
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=copied,all_native_data_times_fields_and_statistics_unchanged=True,
        inset_state_numbers=[1,2,3],main_state_numbers=[4],colorbar_aligned_to_projected_box=True,
        C_field_sizes_preserved=True,C_row_gap_reduced=True,D_model_titles_black=True,
        PDF_text_inside_canvas=checks,formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    documents(meta);shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks,colorbar_height_mm=bmeta['colorbar_height_mm'],C=cmeta['row_gap_mm']),indent=2))


if __name__=='__main__':main()
