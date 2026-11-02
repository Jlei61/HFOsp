#!/usr/bin/env python3
"""Show a short native interictal trajectory and align the D/E plotting areas."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_matched_row as previous

base, parent, recovery = previous.base, previous.parent, previous.previous
np, plt = previous.np, previous.plt
TICK, LABEL, TITLE, LETTER = previous.TICK, previous.LABEL, previous.TITLE, previous.LETTER
MM = previous.MM
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_event_detail_20261031')
INTERICTAL = (7.50, 8.27)
BOUNDS = np.array([[.775, .800], [.018, .026], [0., 200.]])
BOTTOM_HEIGHT = 65.
PLOT_BOTTOM, PLOT_HEIGHT = 15., 40.
STATS = base.MAIN/'results/paper-ready-figure/fig6/candidates/broadband_phase_statistics_20261009/statistics.json'


def export(fig, stem):
    previous.OUT = OUT
    previous.export(fig, stem)


def add_interictal(ax, t, xyz, *, inset):
    from scipy.signal import find_peaks
    keep_time = (t >= INTERICTAL[0]) & (t < INTERICTAL[1])
    st, sx = t[keep_time], xyz[keep_time]
    segments, keep = recovery.add_curve(ax, st, sx, BOUNDS)
    assert keep.all(), 'Retain every adjacent native sample within the selected interval.'
    ax.collections[-1].set_linewidth(1.15 if inset else 1.4)
    ax.view_init(elev=22, azim=-115)
    ax.set_box_aspect((1.55, 1., 1.25), zoom=1.04 if inset else 1.06)
    ax.set(xticks=[.78,.80], yticks=[.018,.026], zticks=[0,200],
        xlabel='$Z$', ylabel=r'$\eta_M M$' if inset else r'$\eta_M M$ (mV)',
        zlabel=r'$r_E$' if inset else '')
    ax.set_yticklabels(['', '0.026'])
    ax.tick_params(labelsize=TICK, pad=-1 if inset else 1)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_fontsize(TITLE if inset else LABEL)
        axis.labelpad = -2 if inset else 4
    ax.figure.canvas.draw()
    peaks, _ = find_peaks(sx[:,2], height=100., prominence=80., distance=30)
    assert len(peaks) == 3
    arrows = []
    for i in peaks:
        # Arrows follow actual adjacent trajectory samples on each ascent.
        candidates = np.arange(max(1,i-10),i)
        middle = candidates[np.argmin(abs(sx[candidates,2]-.5*sx[i,2]))]
        lo, hi = middle-1, middle+1
        q = np.array([base.proj3d.proj_transform(*p,ax.get_proj())[:2] for p in sx[[lo,hi]]])
        ax.annotate('', xy=q[1], xytext=q[0],
            arrowprops=dict(arrowstyle='-|>', color=plt.colormaps['viridis'](st[middle]/recovery.HORIZON),
                lw=1., mutation_scale=9 if inset else 13, shrinkA=0, shrinkB=0), zorder=15)
        arrows.append(dict(from_time_s=float(st[lo]),to_time_s=float(st[hi])))
    recovery.check_projected(ax.figure,ax,segments)
    history = recovery.history()
    events = [{k:event[k] for k in ['event_detection_index','start_s','end_s','phase','participation']}
        for event in history['briefs'] if INTERICTAL[0] <= event['start_s'] < INTERICTAL[1]]
    assert sum(e['participation']=='both_cores' for e in events)==3
    assert all(e['phase']=='initial_interictal' and e['end_s']<INTERICTAL[1] for e in events)
    return dict(time_s=INTERICTAL,limits=BOUNDS,samples=len(st),segment_count=int(keep.sum()),
        selection='One contiguous interval before first entry; all samples and background activity retained.',
        detected_brief_events=events,dominant_rate_peak_times_s=st[peaks],
        both_core_event_count=3,all_native_segments_visible=True,direction_arrows=arrows,
        time_color_scale_s=[0,recovery.HORIZON],
        minimum_Z_etaM_rate=sx.min(0),maximum_Z_etaM_rate=sx.max(0),
        annotation_state_numbers=[],rate_bin_ms=5,display_smoothing_sigma_ms=5)


def draw_b(meta):
    with np.load(SOURCE/'trajectory_data.npz') as data:
        take = data['time_s'] < recovery.HORIZON
        t,xyz = data['time_s'][take],data['Z_etaM_rate'][take]
    fig=plt.figure(figsize=np.array([139.,124.])/25.4)
    ax=fig.add_axes([.065,.065,.75,.80],projection='3d',computed_zorder=False)
    bounds=np.array([[0.,1.],[0.,1000.],[0.,500.]])
    segments,keep=recovery.add_curve(ax,t,xyz,bounds)
    assert keep.all()
    ax.set_box_aspect((1.6,1.,1.25),zoom=1.03)
    ax.set(xlabel='$Z$',ylabel=r'$\eta_M M$ (mV)',xticks=[0,.5,1.],yticks=[0,500,1000],zticks=[0,250,500])
    ax.set_yticklabels(['','500','1000']);ax.tick_params(labelsize=TICK,pad=1)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.label.set_fontsize(LABEL);axis.labelpad=4
    fig.text(.012,.48,r'$r_E$ (Hz)',rotation=90,va='center',fontsize=LABEL)
    for sec in meta['trajectory']['first_returned_brief_s']:
        pt=[np.interp(sec,t,xyz[:,k]) for k in range(3)]
        ax.scatter(*pt,marker='D',s=28,color=plt.colormaps['viridis'](sec/recovery.HORIZON),
            edgecolor='black',linewidth=.6,depthshade=False,zorder=12)
    labels=parent.reference.prior.label_states(ax,meta['states'][:4],t,xyz,fontsize=TITLE)
    fig.text(.07,.97,'Full trajectory',fontsize=LABEL,va='top')
    zoom=fig.add_axes([.565,.64,.37,.31],projection='3d',computed_zorder=False,facecolor='white')
    inset=add_interictal(zoom,t,xyz,inset=True)
    fig.text(.75,.992,'Interictal',fontsize=LABEL,ha='center',va='top')
    fig.text(.75,.958,'7.50–8.27 s',fontsize=TICK,ha='center',va='top')
    cb=fig.colorbar(base.ScalarMappable(norm=base.Normalize(0,recovery.HORIZON),cmap='viridis'),
        cax=fig.add_axes([.865,.19,.023,.40]),ticks=[0,60,recovery.HORIZON])
    cb.set_label('Time (s)',fontsize=LABEL,labelpad=4);cb.ax.tick_params(labelsize=TICK,length=2,pad=3)
    recovery.check_projected(fig,ax,segments)
    export(fig,'fig6-panelb')
    # A separate full-resolution view uses exactly the same selected samples.
    fig=plt.figure(figsize=np.array([160.,125.])/25.4)
    ax=fig.add_axes([.10,.09,.76,.78],projection='3d',computed_zorder=False)
    detail=add_interictal(ax,t,xyz,inset=False)
    fig.text(.07,.98,'Interictal · 7.50–8.27 s',fontsize=LABEL,va='top')
    fig.text(.02,.51,r'$r_E$ (Hz)',fontsize=LABEL,rotation=90,va='center')
    export(fig,'fig6-panelb-interictal')
    return dict(final_size_mm=[139,124],limits=bounds,segment_count=int(keep.sum()),
        unchanged_full_trajectory=True,main_state_numbers=[1,2,3,4],states=labels,
        inset=inset,standalone_interictal=detail,font_label_pt=LABEL,font_tick_pt=TICK)


def load_stat():
    stat=next(r for r in json.loads(STATS.read_text()) if r['analysis']=='full_energy_plus200ms__db')
    assert (stat['n_patients'],stat['n_seizures'],stat['p'])==(13,140,.00708)
    alias=next(r['patient'] for r in stat['matches'] if r['subject']=='epilepsiae_1146')
    assert alias=='E10'
    return stat,alias


def draw_d(matches, alias):
    clinical=base.native.energy.clinical;fz=clinical.load_frozen('epilepsiae_1146')
    width,height=215.,BOTTOM_HEIGHT
    fig=plt.figure(figsize=(width/25.4,height/25.4))
    axes=[];fields=[];titles=[]
    for j,match in enumerate(matches['matches']):
        fields.extend([(f'Model Onset {j+1}',np.asarray(match['model_robust_z'])),
            (f"{alias} {match['patient']['public_seizure']} · {match['patient']['existing_type']}",
             np.asarray(match['patient']['values']))])
    for x,(title,values) in zip([15.,60.,107.,152.],fields):
        ax=fig.add_axes([x/width,PLOT_BOTTOM/height,PLOT_HEIGHT/width,PLOT_HEIGHT/height]);axes.append(ax)
        clinical._draw_field(ax,fz,clinical._normalize_minmax(values),np.asarray(fz['support_a']),
            cmap='Blues',colorbar_values=np.array([0.,1.]),title=title,title_color='black',show_y=x==15.)
        ax.set_xticks([-20,0,20]);ax.set_yticks([-10,0,10,20]);ax.set_xlabel('')
        ax.tick_params(labelsize=TICK,length=2,pad=2)
        ax.yaxis.label.set_fontsize(LABEL);ax.yaxis.labelpad=3
        ax.title.set_fontsize(TITLE);ax.title.set_fontweight('normal');titles.append(title)
    fig.text(103.5/width,8.5/height,'Shared TA axis (mm)',ha='center',va='top',fontsize=LABEL)
    fig.canvas.draw();pos=axes[-1].get_position()
    cb=fig.colorbar(base.ScalarMappable(norm=base.Normalize(0,1),cmap='Blues'),
        cax=fig.add_axes([196/width,pos.y0,1.7/width,pos.height]),ticks=[0,.5,1])
    cb.set_label('Normalized z field',fontsize=LABEL,labelpad=3)
    cb.ax.tick_params(labelsize=TICK,length=2,pad=2)
    for ax in axes:
        np.testing.assert_allclose(np.array(ax.get_position().extents)[[1,3]]*height,[PLOT_BOTTOM,PLOT_BOTTOM+PLOT_HEIGHT],atol=1e-8)
    export(fig,'fig6-paneld')
    return dict(final_size_mm=[width,height],titles=titles,model_numbers=[1,2],
        numeric_correlations_displayed=False,reference_prefix_displayed=False,
        same_selected_patient_fields=True,patient_alias=dict(E10='epilepsiae_1146'),
        field_plot_bottom_mm=PLOT_BOTTOM,field_plot_height_mm=PLOT_HEIGHT,
        colorbar_orientation='vertical',colorbar_height_mm=PLOT_HEIGHT,
        font_label_pt=LABEL,font_tick_pt=TICK,font_title_pt=TITLE)


def draw_e(stat):
    from scripts.paper_figures import build_fig5g_full_ramp_energy as style
    from matplotlib.collections import PathCollection
    width,height=92.,BOTTOM_HEIGHT
    fig=plt.figure(figsize=(width/25.4,height/25.4))
    # Reuse the exact historical violin, box, paired-line and swarm producer.
    ax=style.draw_stat(fig,[18/width,PLOT_BOTTOM/height,72/width,PLOT_HEIGHT/height],stat)
    point_collections=[a for a in ax.collections if isinstance(a,PathCollection) and len(a.get_offsets())==13]
    assert len(point_collections)==4
    values=np.array([r['best_abs_r'] for r in stat['matches']]);null=np.array([r['null_median'] for r in stat['matches']])
    np.testing.assert_array_equal(point_collections[-2].get_offsets()[:,1],values)
    np.testing.assert_array_equal(point_collections[-1].get_offsets()[:,1],null)
    # Match the final rendered marker and stroke scale of the previous E.
    size_scale=92*MM/402.
    for line in ax.lines:line.set_linewidth(line.get_linewidth()*size_scale)
    for collection in ax.collections:
        collection.set_linewidth(collection.get_linewidth()*size_scale)
        if isinstance(collection,PathCollection):collection.set_sizes(collection.get_sizes()*size_scale**2)
    for spine in ax.spines.values():spine.set_linewidth(spine.get_linewidth()*size_scale)
    ax.set_xlabel('');ax.set_ylabel('Best field similarity |r|',fontsize=LABEL,labelpad=4)
    ax.tick_params(labelsize=TICK,length=2,pad=2)
    for text in ax.texts:text.set_fontsize(LABEL)
    export(fig,'fig6-panele')
    base.write(OUT/'cohort_statistics_selected.json',stat)
    return dict(source=str(STATS),analysis=stat['analysis'],n_patients=13,n_seizures=140,p=stat['p'],
        final_size_mm=[width,height],plot_bottom_mm=PLOT_BOTTOM,plot_height_mm=PLOT_HEIGHT,
        same_statistics=True,same_historical_producer=str(Path(style.__file__).resolve()),
        same_patient_points_and_nulls=True,footnote=False,font_label_pt=LABEL,font_tick_pt=TICK)


def compose():
    from matplotlib.font_manager import findfont,FontProperties
    doc=base.fitz.open();page=doc.new_page(width=320*MM,height=318*MM)
    font=findfont(FontProperties(family='DejaVu Sans',weight='bold'));page.insert_font(fontname='FigureBold',fontfile=font)
    boxes={'A':(0,1,320,108),'B':(3,115,142,239),'C':(144,115,317,231),
        'D':(3,250,218,315),'E':(225,250,317,315)}
    letters={'B':(3,112),'C':(144,112),'D':(3,247),'E':(224,247)}
    records={}
    for label,box in boxes.items():
        source=OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(source) as src:
            x0,y0,x1,y1=box;scale=min((x1-x0)*MM/src[0].rect.width,(y1-y0)*MM/src[0].rect.height)
            rect=base.fitz.Rect(x0*MM,y0*MM,x0*MM+src[0].rect.width*scale,y0*MM+src[0].rect.height*scale)
            page.show_pdf_page(rect,src,0,keep_proportion=True)
        if label in letters:page.insert_text(tuple(v*MM for v in letters[label]),label,fontsize=LETTER,fontname='FigureBold')
        if label!='A':np.testing.assert_allclose(scale,1,atol=1e-6)
        records[label]=dict(source=str(source),box_mm=[v/MM for v in rect],scale=scale)
    path=OUT/'figures/fig6-complete-layout.pdf';doc.save(path,garbage=4,deflate=True);doc.close()
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72,220/72),alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    base.write(OUT/'layout_manifest.json',dict(canvas_mm=[320,318],panels=records,
        bottom_row_plot_top_mm=260.,bottom_row_plot_bottom_mm=300.,D_E_plot_height_mm=40.,
        same_field_and_statistic_plot_height=True,
        typography=dict(family='DejaVu Sans',label_pt=LABEL,tick_pt=TICK,title_pt=TITLE,panel_letter_pt=LETTER)))


def documents(meta):
    (OUT/'caption_zh.md').write_text('''Figure 6：连续发作、间期轨迹与患者能量场。

A、C 与上一版完全一致，八个时刻保持对应，C 两行仍为 Model Onset 1/2。B 主图显示 0–118 s 全部原生轨迹，1–4 标记放在主图；Interictal 小窗只显示第一次进入前连续 7.50–8.27 s，包含三次双 core 间期事件及其间的背景活动，不拼接、不删除该时段的任何采样点。小窗 Z=0.775–0.800，eta_M M=0.018–0.026 mV，E 放电率 0–200 Hz；三条短箭头沿真实相邻样本标示上升方向。时间颜色仍使用完整 0–118 s 色条。该例展示原生轨迹的短暂活动及回落，不将其称为闭合周期轨道；主图的两次发作及恢复保持完整。

D 依次为 Model Onset 1、E10 SZ13 · TB、Model Onset 2、E10 SZ14 · TA。E10 对应 epilepsiae_1146；患者选择、模型能量窗口与场算法沿用上一版，全部排名及相关系数仍保留于 model_selected_patient_matches.json 和 all_patient_seizure_matches.csv，图面省略相关系数及 Reference seizure 前缀。两例是从该患者 25 次合格发作中按最高有符号场相关选出的描述性匹配，并非独立验证或新的模板分类。模型和患者各场单独 min–max 显示，评分使用原始 robust-z 场。

D 四幅场图及色条的绘图区高均为 40 mm，E 绘图区也为 40 mm，上下边缘对齐。E 使用原历史 13 人、140 次合格发作的 full_energy_plus200ms__db 结果和原有小提琴、箱线、配对点线绘图函数，p=0.00708；仅调整尺寸、字体和间距，原始数据及统计没有重算。本版为候选，待作者目视检查。
''')
    desc={
        'fig6-complete-layout':'完整 A–E 采用与上一版一致的上两行，底部 D/E 的绘图区上下对齐。地图标题简化并移除数值相关。',
        'fig6-panela':'两次连续发作的原生波形和八个阶段标记保持原版。与前版导出字节一致。',
        'fig6-panelb':'主图保留 0–118 s 完整轨迹，1–4 移至主图。Interictal 小窗显示连续 7.50–8.27 s 的三次双 core 间期事件，坐标范围按该时段收紧。',
        'fig6-panelb-interictal':'独立放大小窗的同一连续时段，包含三次双 core 事件及其间的背景活动。箭头沿真实相邻轨迹样本指向，未拼接或平移事件。',
        'fig6-panelb-full-trajectory':'独立全范围轨迹图与前版保持一致。完整原生状态仍可在此检查。',
        'fig6-panelc':'两行 Model Onset 1/2 的八个空间状态及色条保持不变。时刻与 A 逐一对应。',
        'fig6-paneld':'四张场图依次为 Model Onset 1、E10 SZ13 · TB、Model Onset 2、E10 SZ14 · TA。图面不再显示参考前缀或相关系数，选择结果仍在数值记录中保留。',
        'fig6-panele':'复用历史 13 人队列的原数据和原统计绘图函数。绘图区高调整至 40 mm，与 D 的四个场图对齐。'}
    (OUT/'figures/README.md').write_text('\n\n'.join(f'### {k}.png / .pdf / .svg\n\n{v}\n**关注点**：'
        +('仅减少展示时段，原生轨迹及统计语义不变。' if 'panelb' in k else '本版只调整展示，待作者目视检查。')
        for k,v in desc.items())+'\n')
    (OUT/'README.md').write_text('# Figure 6：少量间期事件与 D/E 等高\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · '
        '[间期细节](figures/fig6-panelb-interictal.png)\n\n'
        f'复现：`python {Path(__file__).resolve()}`。A/C、患者配对和历史 E 统计保持不变。\n')


def main():
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json';frozen=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    meta=json.loads((SOURCE/'figure_metadata.json').read_text())
    print('Rendering selected interictal interval.',flush=True)
    bmeta=draw_b(meta)
    stat,alias=load_stat();matches=json.loads((SOURCE/'model_selected_patient_matches.json').read_text())
    print('Aligning D and E with unchanged fields and cohort statistics.',flush=True)
    dmeta=draw_d(matches,alias);emeta=draw_e(stat);compose()
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),
        B_display=bmeta,D=dmeta,E_display=emeta,status='PENDING_AUTHOR_VISUAL_REVIEW',
        revision_scope='Sparse contiguous interictal view; simplified map titles; aligned D/E plot height.',
        E_source=str(STATS),E_scope='Same historical 13-patient statistic; layout only.')
    meta['trajectory'].update(main_state_numbers=[1,2,3,4],inset_state_numbers=[],display_revision=bmeta,
        inset_bounds_rationale='A contiguous 7.50-8.27 s interval with three both-core brief events; tight linear bounds cover every selected sample.',
        inset_time_s=INTERICTAL)
    meta['trajectory']['outputs']['fig6-panelb']=bmeta
    meta['trajectory']['outputs']['fig6-panelb-interictal']=bmeta['standalone_interictal']
    base.write(OUT/'figure_metadata.json',meta)
    base.write(OUT/'interictal_display_selection.json',bmeta['inset'])
    copied=[]
    for stem in ['fig6-panela','fig6-panelc','fig6-panelb-full-trajectory']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}';assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes();copied.append(name)
    for name in ['trajectory_data.npz','stage_readouts.npz','field_comparisons.json','A_C_stage_correspondence.csv',
        'model_selected_patient_matches.json','model_selected_patient_arrays.npz','all_patient_seizure_matches.csv']:
        assert (OUT/name).read_bytes()==(SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-panelb','fig6-panelb-interictal','fig6-paneld','fig6-panele','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(span['bbox'])) for page in doc
                for block in page.get_text('dict')['blocks'] if block['type']==0
                for line in block['lines'] for span in line['spans'])
            if stem=='fig6-paneld':
                text=''.join(page.get_text() for page in doc)
                assert 'Reference seizure' not in text and '0.850' not in text and '0.920' not in text
                assert 'Model Onset 1' in text and 'Model Onset 2' in text and 'E10 SZ13' in text and 'E10 SZ14' in text
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=copied,
        all_underlying_trajectory_fields_stage_times_and_matches_unchanged=True,
        inset_is_one_contiguous_native_time_slice=True,all_selected_samples_within_view=True,
        D_E_plot_bounds_mm=[260.,300.],D_E_equal_plot_height=True,
        E_statistic_and_patient_points_unchanged=True,PDF_text_inside_canvas=checks,
        formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    documents(meta)
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),inset_time_s=INTERICTAL,checks=checks),indent=2))


if __name__=='__main__':main()
