#!/usr/bin/env python3
"""Author revision: shorter A context, second-entry zoom, resolved D, model-only E."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import build_fig6_unified_adaptation_two_entry_rows as previous
base=previous.base
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PARENT=base.MAIN/'results/paper-ready-figure/fig6/candidates/unified_current_adaptation_two_entry_rows_s9108406_20261031'
OUT=PARENT.with_name('unified_current_adaptation_entry_zoom_s9108406_20261031')
WINDOWS=[(7.8,13.65),(53.61,54.91)]
REMOVED=['ICL2','ICL1']


def draw_a(a,states,meta):
    native=base.native
    native.RECRUIT=8.84;native.ETA=.0005
    native.MAIN_BAND_HZ=(80.,1000.);native.GAIN_PERCENTILE=99.5;native.DISPLAY_GAIN=.65
    native.HIDDEN_STATE_NUMBERS=set(range(1,9))
    def marks(ax,ss,onset):
        for start,end,color in [(8.4,meta['termination_s'],'#2166ac'),(54.,58.,'#b2182b')]:
            ax.axvspan(start,end,fc=color,alpha=.065,lw=0)
        for s in ss:ax.axvline(s['time_s'],ls=':',color=s['color'],lw=.85)
        ax.set_xlim(native.WINDOW)
    native.marks=marks
    fig=plt.figure(figsize=(15.7,5.45))
    axes=[fig.add_axes([.075,.15,.4325,.77]),fig.add_axes([.5475,.15,.4325,.77])]
    waveforms=[]
    for k,(ax,window) in enumerate(zip(axes,WINDOWS)):
        native.WINDOW=window;ss=states[k*4:k*4+4]
        # Invoke the accepted filter/gain producer on all15 contacts, then perform
        # only the author-requested display deletion. Analysis still uses all15.
        wm=native.draw_a(ax,a,ss,8.4)
        assert wm['contact_order'][-2:]==REMOVED
        traces=list(ax.lines[:15]);separator=ax.lines[15]
        for line in traces[-2:]:line.remove()
        for line in traces[:13]:line.set_ydata(line.get_ydata()-2)
        separator.set_ydata(np.asarray(separator.get_ydata())-2)
        plotted=np.array([line.get_ydata() for line in traces[:13]])
        ax.set(yticks=np.arange(12,-1,-1),yticklabels=wm['contact_order'][:13],
               ylim=(min(-.65,plotted.min()-.12),max(13.,plotted.max()+.12)),xlabel='')
        ax.tick_params(axis='x',labelbottom=True,labelsize=13)
        ax.set_xticks([8,10,12,13.5] if k==0 else [53.8,54.2,54.6,54.9])
        ax.spines[['top','right']].set_visible(False)
        if k:
            ax.set_ylabel('');ax.tick_params(axis='y',left=False,labelleft=False)
            ax.spines['left'].set_visible(False)
        positions=[8.45,8.78,9.1,13.22] if k==0 else [s['time_s'] for s in ss]
        for s,x in zip(ss,positions):
            ax.annotate(str(s['number']),xy=(s['time_s'],1),xycoords=ax.get_xaxis_transform(),
                xytext=(x,1.025),textcoords=ax.get_xaxis_transform(),ha='center',va='bottom',
                color=s['color'],fontsize=15,weight='bold',annotation_clip=False,
                arrowprops=dict(arrowstyle='-',color=s['color'],lw=.65))
        wm.update(display_contact_order=wm['contact_order'][:13],display_removed=REMOVED,
                  displayed_channel_count=13,visible_state_numbers=[s['number'] for s in ss],
                  titles_removed=True,scalar_vertical_offset_for_remaining_contacts=-2,
                  analysis_contact_order_unchanged=True)
        assert np.array_equal(wm['gain_per_contact'],meta['waveform']['segments'][0]['gain_per_contact'])
        waveforms.append(wm)
    limits=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:ax.set_ylim(limits)
    for ax,x in [(axes[0],1.),(axes[1],0.)]:
        for y in [0.,1.]:
            ax.plot([x-.009,x+.009],[y-.012,y+.012],transform=ax.transAxes,
                    color='#555',lw=1.1,clip_on=False)
    fig.text(.5275,.065,'13.65–53.61 s omitted',ha='center',fontsize=11,color='#555')
    fig.text(.53,.012,'Time (s)',ha='center',va='bottom',fontsize=16)
    base.save(fig,'fig6-panela')
    return dict(segments=waveforms,visible_windows_s=WINDOWS,omitted_s=[13.65,53.61],
        same_y_limits=limits,same_contact_gains=True,filter_before_windowing=True,
        seconds_per_width_equal=False,right_temporal_magnification=np.ptp(WINDOWS[0])/np.ptp(WINDOWS[1]),
        time_scale_disclosed='Actual absolute-time ticks on both axes; right window spans1.30s, left5.85s.',
        display_removed=REMOVED,analysis_contacts_removed=[],titles_removed=True,continuous_simulation=True)


def label_states(ax,states,t,xyz,fontsize=9):
    fig=ax.figure;fig.canvas.draw();boxes=[];records=[]
    for s in states:
        pt=np.array([np.interp(s['time_s'],t,xyz[:,k]) for k in range(3)])
        ax.scatter(*pt,c=s['color'],s=20,edgecolor='white',linewidth=.5,depthshade=False,zorder=10,clip_on=False)
        x,y,_=proj3d.proj_transform(*pt,ax.get_proj());anchor=ax.transData.transform((x,y));unit=fig.dpi/72
        for off in [(12,12),(-20,12),(18,-20),(-20,-20),(0,31),(33,0),(-33,0),(0,-33),(33,28),(-33,28),(43,-30),(-43,-30)]:
            c=anchor+np.array(off)*unit;box=(c[0]-7*unit,c[1]-7*unit,c[0]+7*unit,c[1]+7*unit)
            if all(box[2]<b[0] or box[0]>b[2] or box[3]<b[1] or box[1]>b[3] for b in boxes):
                boxes.append(box);break
        else:raise RuntimeError('Trajectory labels overlap')
        ax.annotate(str(s['number']),xy=(x,y),xytext=off,textcoords='offset points',ha='center',va='center',
            fontsize=fontsize,weight='bold',bbox=dict(boxstyle='circle,pad=.12',fc='white',ec='#777',lw=.5),
            arrowprops=dict(arrowstyle='-',color='#777',lw=.5),zorder=20)
        records.append(dict(**s,Z_etaM_rate=pt,label_offset_pt=off))
    return records


def draw_d(states):
    a=np.load(PARENT/'trajectory_data.npz');t=a['time_s'];xyz=a['Z_etaM_rate'];duration=58.
    seg=np.stack([xyz[:-1],xyz[1:]],axis=1);color_time=(t[:-1]+t[1:])/(2*duration)
    fig=plt.figure(figsize=(4.7,5.6))
    configs=[dict(box=[.10,.56,.83,.40],xlim=(max(0,float(xyz[:,0].min())-.02),1.02),ylim=(0,float(xyz[:,1].max()*1.08)),
                  zlim=(0,float(xyz[:,2].max()*1.06)),numbers=[4,8],name='Full trajectory'),
             dict(box=[.10,.10,.83,.40],xlim=(.70,1.00),ylim=(0,.10),zlim=(0,250),
                  numbers=[1,2,3,5,6,7],name='Interictal / entry zoom')]
    records=[];views=[]
    for k,cfg in enumerate(configs):
        ax=fig.add_axes(cfg['box'],projection='3d');ax.computed_zorder=False
        bounds=np.array([cfg['xlim'],cfg['ylim'],cfg['zlim']])
        if k:
            inside=np.all((xyz>=bounds[:,0])&(xyz<=bounds[:,1]),axis=1)
            keep=inside[:-1]&inside[1:]
        else:
            assert np.all((xyz>=bounds[:,0])&(xyz<=bounds[:,1])), 'Full view clips native coordinates'
            keep=np.ones(len(seg),bool)
        ax.add_collection3d(Line3DCollection(seg[keep],cmap='viridis',norm=Normalize(0,1),
            array=color_time[keep],linewidths=.65 if k==0 else .9,zorder=6,clip_on=False))
        ax.set(xlim=cfg['xlim'],ylim=cfg['ylim'],zlim=cfg['zlim'],xlabel='$Z$',ylabel='')
        ax.view_init(elev=20,azim=-105);ax.set_box_aspect((3.2,1,1.3),zoom=1.25)
        ax.set_proj_type('ortho')
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.label.set_fontsize(10);axis.labelpad=1
            axis.pane.set_facecolor('#f2f5fa');axis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=8,pad=0)
        if k:
            ax.set_xticks([.7,.8,.9,1]);ax.set_yticks([0,.05,.1]);ax.set_zticks([0,125,250])
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        else:ax.set_xticks([.4,.7,1])
        fig.text(.14,.978 if k==0 else .505,cfg['name'],fontsize=10,ha='left')
        fig.text(.12,.775 if k==0 else .315,r'$r_E$ (Hz)',fontsize=10,rotation=90,va='center')
        fig.text(.19,.54 if k==0 else .08,r'$\eta_M M$ (mV)',fontsize=9,ha='center')
        records.extend(label_states(ax,[s for s in states if s['number'] in cfg['numbers']],t,xyz))
        # Axes3D's square clipping rectangle can truncate a wide projected box.
        # Disable that clip, while ensuring every displayed vertex is on canvas.
        vertices=np.unique(seg[keep].reshape(-1,3),axis=0)
        px,py,_=proj3d.proj_transform(*vertices.T,ax.get_proj())
        screen=ax.transData.transform(np.c_[px,py]);fb=fig.bbox
        assert np.all((screen[:,0]>=fb.x0)&(screen[:,0]<=fb.x1)&
                      (screen[:,1]>=fb.y0)&(screen[:,1]<=fb.y1))
        views.append(dict(name=cfg['name'],limits=bounds,linear_axes=True,
                          elevation_deg=20,azimuth_deg=-105,box_aspect=[3.2,1,1.3],display_zoom=1.25,
                          orthographic=True,segment_count=int(keep.sum()),total_segments=len(seg),
                          axes_clipping_disabled=True,all_displayed_vertices_inside_canvas=True,
                          no_bridging_across_hidden_segments=True,numbers=cfg['numbers']))
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),
                   cax=fig.add_axes([.26,.034,.49,.018]),orientation='horizontal',ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=8,pad=2,length=2);cb.set_label('Normalized time',fontsize=9,labelpad=2)
    base.save(fig,'fig6-paneld')
    shutil.copy2(PARENT/'trajectory_data.npz',OUT/'trajectory_data.npz')
    return dict(coordinates=['mean Z','eta_M mean M (mV)','all-E rate (Hz)'],time_s=[0,duration],
        rate_bin_ms=5,display_smoothing_sigma_ms=5,normalized_time='t / full duration',
        views=views,states=sorted(records,key=lambda s:s['number']),full_raw_trajectory_preserved=True,
        interpretation='Full native trajectory above; linear-coordinate low-adaptation zoom below. No coordinate warp.')


def draw_e(comparisons):
    clinical=base.native.energy.clinical;fz=clinical.load_frozen('epilepsiae_1146')
    assert all(c['frozen_fingerprint']==fz['fingerprint'] for c in comparisons)
    fig=plt.figure(figsize=(8.6,3.5))
    gs=fig.add_gridspec(1,2,left=.08,right=.88,bottom=.22,top=.83,wspace=.28)
    for i,c in enumerate(comparisons):
        ax=fig.add_subplot(gs[i]);values=np.asarray(c['model_robust_z'])
        lo,hi=c['model_early_s']
        clinical._draw_field(ax,fz,clinical._normalize_minmax(values),np.asarray(fz['support_a']),
            cmap='Blues',colorbar_values=np.array([0.,1.]),title=f'Model onset {i+1}\n{lo:.2f}–{hi:.2f} s',
            title_color='black',show_y=i==0)
        ax.set_xticks([-20,0,20]);ax.set_yticks([-10,0,10,20]);ax.tick_params(labelsize=11)
        ax.xaxis.label.set_fontsize(12);ax.yaxis.label.set_fontsize(12)
        ax.title.set_fontsize(13);ax.title.set_fontweight('normal')
        ax.text(.5,-.35,f"$r$ = {c['selected']['field_r']:+.3f}",transform=ax.transAxes,ha='center',fontsize=12)
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='Blues'),
        cax=fig.add_axes([.92,.25,.018,.49]),ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=11);cb.set_label('Normalized z field',fontsize=12,labelpad=5)
    base.save(fig,'fig6-panele')
    return dict(layout='two model onset fields only',patient_field_displayed=False,
        display='per-field min-max normalized robust z',analysis_contact_count=15,
        correlations=[c['selected']['field_r'] for c in comparisons],
        correlation_reference='Unchanged E1146 SZ26 field, omitted from display at author request.',
        model_windows_s=[c['model_early_s'] for c in comparisons])


def compose():
    mm=72/25.4;doc=base.fitz.open();page=doc.new_page(width=320*mm,height=330*mm)
    boxes={'A':(3,8,317,117),'D':(3,125,110,243),'C':(112,125,317,243),
           'E':(3,253,218,327),'F':(225,253,317,327)}
    letters={'A':(3,7),'D':(3,125),'C':(111,125),'E':(3,251),'F':(224,251)}
    manifest={}
    for label,box in boxes.items():
        path=OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(path) as src:page.show_pdf_page(base.fitz.Rect(*(v*mm for v in box)),src,0,keep_proportion=True)
        page.insert_text(tuple(v*mm for v in letters[label]),label,fontsize=16,fontname='hebo')
        manifest[label]=dict(source=str(path),box_mm=box)
    path=OUT/'figures/fig6-complete-layout.pdf';doc.save(path,garbage=4,deflate=True);doc.close()
    previous.layout.export_pdf(path)
    base.write(OUT/'layout_manifest.json',dict(canvas_mm=[320,330],panels=manifest,
        A='two windows; right onset magnified4.5x in time;13 display contacts; no titles',
        C='two horizontal rows of four states',D='full trajectory and linear low-adaptation zoom',
        E='two model fields only',B='supplementary'))


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    pointer_before=pointer.read_bytes()
    meta=json.loads((PARENT/'figure_metadata.json').read_text())
    route=json.loads((PARENT/'core_recruitment.json').read_text())
    comparisons=json.loads((PARENT/'field_comparisons.json').read_text())
    states=copy.deepcopy(meta['states']);states[-1]['time_s']=54.8
    assert states[-1]['label']=='Ictal' and states[-1]['time_s']>route['episodes'][1]['high_threshold_onset_s']+.2
    base.ROOT=base.analysis.ROOT=previous.audit.RAW;base.RUN_NAME='seed9108406';base.OUT=OUT
    a=base.load(base.RUN_NAME)
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':14,'xtick.labelsize':12,
        'ytick.labelsize':12,'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    wave=draw_a(a,states,meta);maps=previous.draw_c(a,states)
    for old,new in zip(meta['spatial_maps'][:7],maps[:7]):assert np.array_equal(old['E_rate_Hz'],new['E_rate_Hz'])
    traj=draw_d(states);emeta=draw_e(comparisons)
    for ext in ['pdf','png','svg']:shutil.copy2(PARENT/'figures'/f'fig6-panelf.{ext}',OUT/'figures'/f'fig6-panelf.{ext}')
    rates,_=previous.audit.recruitment(Path(meta['source_run']),a)
    previous.supplements(a,rates,route,states,meta['termination_s'])
    compose()
    for name in ['core_recruitment.json','scientific_result.json','field_comparisons.json']:
        shutil.copy2(PARENT/name,OUT/name)
    newmeta=copy.deepcopy(meta)
    newmeta.update(parent=str(PARENT),states=states,waveform=wave,spatial_maps=maps,trajectory=traj,E=emeta,
        state8_revision=dict(previous_time_s=57.985,time_s=54.8,meaning='Ictal unchanged; now an early post-confirmation sample for the onset zoom'),
        source_simulation_unchanged=True,human_visual_acceptance='PENDING',producer=str(Path(__file__).resolve()))
    base.write(OUT/'figure_metadata.json',newmeta)
    checks={}
    for path in (OUT/'figures').glob('*.pdf'):
        with base.fitz.open(path) as doc:
            checks[path.name]=all(pg.rect.contains(base.fitz.Rect(s['bbox'])) for pg in doc
                for block in pg.get_text('dict')['blocks'] if block['type']==0 for line in block['lines'] for s in line['spans'])
    assert all(checks.values()),checks
    with base.fitz.open(OUT/'figures/fig6-panela.pdf') as doc:
        txt=doc[0].get_text()
        assert all(word not in txt for word in ['ICL1\n','ICL2\n','Interictal','Core B','Core A'])
    assert pointer.read_bytes()==pointer_before
    assert (OUT/'trajectory_data.npz').read_bytes()==(PARENT/'trajectory_data.npz').read_bytes()
    assert (OUT/'figures/fig6-panelf.pdf').read_bytes()==(PARENT/'figures/fig6-panelf.pdf').read_bytes()
    base.write(OUT/'validation.json',dict(status='PASS',same_native_simulation=True,
        unchanged_first_seven_maps=True,state8_native20ms_count_conserved=True,
        state8_follows_second_entry_confirmation=True,full_trajectory_byte_identical=True,
        full_D_limits_contain_all_native_coordinates=True,
        A_ICL1_ICL2_display_only_removed=True,A_has_no_titles=True,A_parent_gains_identical=True,
        E_two_model_fields_only=True,E_values_correlations_and_contact_set_unchanged=True,
        F_PDF_byte_identical=True,formal_pointer_unchanged=True,PDF_text_inside=checks,human_visual_acceptance='PENDING'))
    (OUT/'source').mkdir(exist_ok=True)
    for p in [Path(__file__),Path(previous.__file__),Path(base.__file__)]:shutil.copy2(p,OUT/'source'/p.name)
    write_docs(states)
    print(json.dumps(dict(output=str(OUT),windows_s=WINDOWS,state8_s=54.8,
                         zoom_limits=traj['views'][1]['limits'].tolist())),flush=True)


def write_docs(states):
    (OUT/'caption_zh.md').write_text('''本候选沿用seed9108406的连续58 s统一适应仿真。A显示7.80–13.65 s与53.61–54.91 s，左侧保留第一轮进入与终止，右侧聚焦第二轮进入；左右显示宽度相同，真实时间刻度明确右侧4.5倍时间放大。断轴省略13.65–53.61 s。两段使用相同滤波、固定触点增益及纵坐标；应作者要求仅在A隐藏ICL2、ICL1，保留其余13触点既定顺序，并去掉顶部标题。滤波仍在完整10 kHz原生轨迹上进行，增益仍由原15触点规则求得，能量场分析仍使用全部15触点。

C保持两行四帧：1–4与5–8分别对应Local、Spread、Entry、Ictal。为匹配第二轮进入的放大窗口，8号从57.985 s移到54.800 s（第二轮全局高活动确认54.700 s之后），Ictal语义不变；A/C/D同步更新。其他七帧的时间、原生20 ms率与颜色映射完全不变。

D上方保留完整轨迹，下方放大Z=0.70–1.00、适应电流0–0.10 mV、E率0–250 Hz的低适应区域。两图均为线性坐标，颜色仍为完整58 s的归一化时间。下方仅画两端都落在显示范围的连续线段，跨出范围后不连接不相邻的点；它不表示完整恢复轨迹。4、8号标在完整视图，1–3与5–7标在局部视图，所有状态坐标均来自原始轨迹。

E只显示两个模型onset场（8.40–9.40 s、54.00–55.00 s），不再绘制患者参考场；两次同一患者SZ26场相关仍为−0.435、+0.767，r的参考对象未改变。原15触点CAR、0.5–3.5 s基线、1–150 Hz能量算子、固定空间平滑及逐场min-max显示均不变。F仍为历史13人模型队列，统计与小脚注去除版本保持不变；慢状态B在补图候选中。

核间B→A指100 Hz持续200 ms的招募先后，第一次进入前仍有A核短暂放电；两次能量场均偏TA。此图仅为同一轨迹的显示调整，不改变种子搜索或模型机制结论。正式入口未替换，候选待作者目视检查。
''')
    (OUT/'README.md').write_text(f'''# Figure 6：进入放大与间期轨迹局部显示

[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [图注](caption_zh.md)。

A缩短两段前间期，移除两个显示通道与顶部标题，右侧4.5倍时间放大。D增加线性坐标局部放大，E仅绘制两个模型场。8号同步改为54.800 s的早期Ictal，其他七帧、完整轨迹、能量场数值及F保持原值。正式入口未替换。

复现：`python {Path(__file__).resolve()}`。来源与核对记录见figure_metadata.json和validation.json。
''')
    descriptions={
      'fig6-complete-layout':'顶部A无标题并聚焦第二次进入；中排D为完整/局部轨迹，C为两行四帧；底排E/F。',
      'fig6-panela':'仅绘制13个触点，去掉ICL2、ICL1。前间期缩短，右侧第二次进入采用更短的1.30 s窗口。',
      'fig6-panelc':'两行重复Local、Spread、Entry、Ictal语义。8号移到54.800 s，其他七帧与上一版相同。',
      'fig6-paneld':'上方完整轨迹，下方Z=0.7–1、低适应电流区域的局部放大。真实数值与线性坐标保留，出界线段不跨接。',
      'fig6-panele':'仅保留两个模型onset能量场。患者参考场不显示，原相似度仍以固定SZ26为参考。',
      'fig6-panelf':'历史模型队列参考，统计量和图层与上一版相同。小脚注保持去除。'}
    (OUT/'figures/README.md').write_text('\n\n'.join(f'### {k}.png / .pdf\n{v}\n**关注点**：'+(
       '左右真实时间比例不同，触点增益相同。' if k=='fig6-panela' else
       '局部放大不替代上方完整轨迹。' if k=='fig6-paneld' else
       '本轮为同一数据的显示修订，候选待作者目视检查。') for k,v in descriptions.items())+'\n')
    (OUT/'supplementary_candidates/figures/README.md').write_text('''### slow-state-traces.png / .pdf
显示完整58 s全体E及双核Z与适应电流。8号时点对应的竖线同步到54.800 s。
**关注点**：A省略的中间恢复期仍在完整曲线中。

### two-entry-core-recruitment.png / .pdf
保留两次进入的固定细胞raster与三分区原生率。首次短暂A核放电及后续B核持续招募均可见。
**关注点**：持续招募领先与首次spike不是同一判据。
''')


if __name__=='__main__':main()
