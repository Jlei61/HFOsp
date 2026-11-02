#!/usr/bin/env python3
"""Two native entry sequences, onset-aligned states and uncluttered mean trajectory."""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import build_fig6_unified_adaptation_equal_entry_zoom as previous

base=previous.base
rows=previous.rows
prior=previous.prior
SEED=9108402
SOURCE=base.MAIN/f'results/paper-ready-figure/fig6/candidates/unified_current_adaptation_paired_s{SEED}_20261030'
OUT=SOURCE.with_name(f'unified_current_adaptation_origin_review_s{SEED}_20261031')
REVIEW=base.MAIN/'results/topic4_sef_hfo/entry_origin_review_20261031'
DISPLAY_PARENT=previous.OUT
WINDOWS=[(11.00,12.40),(61.22,62.62)]
TIMES=[11.605,11.640,11.820,12.080,61.805,61.840,62.130,62.450]


def draw_a(a,states,selection):
    native=base.native
    native.RECRUIT=selection['episodes'][0]['high_threshold_onset_s']
    native.ETA=.0005
    native.MAIN_BAND_HZ=(80.,1000.)
    native.GAIN_PERCENTILE=99.5
    native.DISPLAY_GAIN=.65
    native.HIDDEN_STATE_NUMBERS=set(range(1,9))
    def marks(ax,ss,onset):
        for ep,end,color in zip(selection['episodes'],[19.57,66.],['#b2182b','#2166ac']):
            ax.axvspan(ep['last20ms_quiet_end_s'],end,fc=color,alpha=.065,lw=0)
        for s in ss:ax.axvline(s['time_s'],ls=':',color=s['color'],lw=.85)
        ax.set_xlim(native.WINDOW)
    native.marks=marks
    fig=plt.figure(figsize=(15.7,5.45))
    gap=.065;width=(.905-gap)/2
    axes=[fig.add_axes([.075,.15,width,.77]),fig.add_axes([.075+width+gap,.15,width,.77])]
    records=[]
    for k,(ax,window) in enumerate(zip(axes,WINDOWS)):
        native.WINDOW=window;ss=states[k*4:k*4+4]
        wm=native.draw_a(ax,a,ss,selection['episodes'][k]['last20ms_quiet_end_s'])
        traces=list(ax.lines[:15]);separator=ax.lines[15]
        assert wm['contact_order'][-2:]==['ICL2','ICL1']
        for line in traces[-2:]:line.remove()
        for line in traces[:13]:line.set_ydata(line.get_ydata()-2)
        separator.set_ydata(np.asarray(separator.get_ydata())-2)
        plotted=np.array([line.get_ydata() for line in traces[:13]])
        ax.set(yticks=np.arange(12,-1,-1),yticklabels=wm['contact_order'][:13],xlabel='',
            ylim=(min(-.65,plotted.min()-.12),max(13.,plotted.max()+.12)))
        ax.tick_params(axis='x',labelbottom=True,labelsize=13,pad=3)
        ax.set_xticks([11.,11.4,11.8,12.2] if k==0 else [61.4,61.8,62.2,62.6])
        ax.spines[['top','right']].set_visible(False)
        if k:
            ax.set_ylabel('');ax.tick_params(axis='y',left=False,labelleft=False)
            ax.spines['left'].set_visible(False)
        for j,s in enumerate(ss):
            # Small text offsets separate the first two close samples; dotted
            # lines always stay at the actual native frame times.
            x=s['time_s']+(-.012 if j==0 else .012 if j==1 else 0.)
            ax.annotate(str(s['number']),xy=(s['time_s'],1),xycoords=ax.get_xaxis_transform(),
                xytext=(x,1.025),textcoords=ax.get_xaxis_transform(),ha='center',va='bottom',
                fontsize=15,weight='bold',color=s['color'],annotation_clip=False,
                arrowprops=dict(arrowstyle='-',color=s['color'],lw=.5))
        wm.update(display_contact_order=wm['contact_order'][:13],display_removed=['ICL2','ICL1'],
            displayed_channel_count=13,visible_state_numbers=[s['number'] for s in ss])
        records.append(wm)
    ylim=(min(ax.get_ylim()[0] for ax in axes),max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:ax.set_ylim(ylim)
    for ax,x in [(axes[0],1.),(axes[1],0.)]:
        for y in [0.,1.]:
            ax.plot([x-.009,x+.009],[y-.012,y+.012],transform=ax.transAxes,
                    color='#555',lw=1.1,clip_on=False)
    fig.text(.5275,.09,'Time (s)',ha='center',va='top',fontsize=16)
    assert np.array_equal(records[0]['gain_per_contact'],records[1]['gain_per_contact'])
    base.save(fig,'fig6-panela')
    return dict(segments=records,visible_windows_s=WINDOWS,omitted_s=[WINDOWS[0][1],WINDOWS[1][0]],
        seconds_per_width_equal=True,same_contact_gains=True,same_y_limits=ylim,
        break_annotation='break marks only; no omitted-duration text',time_label_y=.09,
        analysis_contacts_removed=[],displayed_channel_count=13,titles_removed=True,
        continuous_simulation=True,filter_before_windowing=True)


def draw_b(states):
    raw=np.load(SOURCE/'trajectory_data.npz');t=raw['time_s'];xyz=raw['Z_etaM_rate']
    bounds=np.array([[.5,1.],[0.,300.],[0.,350.]])
    inside=(xyz[:,0]>=.5)&(xyz[:,0]<=1.)
    assert np.all((xyz[inside]>=bounds[:,0])&(xyz[inside]<=bounds[:,1]))
    keep=inside[:-1]&inside[1:];segments=np.stack([xyz[:-1],xyz[1:]],axis=1)
    fig=plt.figure(figsize=(5.5,5.1))
    ax=fig.add_axes([.07,.13,.71,.82],projection='3d');ax.computed_zorder=False
    ax.add_collection3d(Line3DCollection(segments[keep],cmap='viridis',norm=Normalize(0,1),
        array=((t[:-1]+t[1:])/(2*66.))[keep],linewidths=.8,zorder=6,clip_on=False))
    ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2],xlabel='$Z$',ylabel=r'$\eta_M M$ (mV)',
        xticks=[.5,.75,1.],yticks=[0,150,300],zticks=[0,175,350])
    ax.set_yticklabels(['','150','300'])
    ax.view_init(elev=26,azim=-125);ax.set_box_aspect((1.65,1,1.25));ax.set_proj_type('ortho')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.label.set_fontsize(11);axis.labelpad=5;axis.pane.set_facecolor('#f2f5fa')
    ax.tick_params(labelsize=9,pad=0)
    fig.text(.015,.64,r'$r_E$ (Hz)',rotation=90,va='center',fontsize=11)
    labels=prior.label_states(ax,states[:4],t,xyz,fontsize=10)
    vertices=np.unique(segments[keep].reshape(-1,3),axis=0)
    px,py,_=proj3d.proj_transform(*vertices.T,ax.get_proj())
    screen=ax.transData.transform(np.c_[px,py]);fb=fig.bbox
    assert np.all((screen[:,0]>=fb.x0)&(screen[:,0]<=fb.x1)&(screen[:,1]>=fb.y0)&(screen[:,1]<=fb.y1))
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),
        cax=fig.add_axes([.85,.30,.025,.46]),ticks=[0,.5,1])
    cb.set_label('Normalized time',fontsize=11,labelpad=4)
    cb.ax.tick_params(labelsize=10,pad=3,length=2)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/'figures'/f'fig6-panelb.{ext}',dpi=220,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    shutil.copy2(SOURCE/'trajectory_data.npz',OUT/'trajectory_data.npz')
    return dict(coordinates=['mean Z','eta_M mean M (mV)','all-E rate (Hz)'],time_s=[0,66],
        rate_bin_ms=5,display_smoothing_sigma_ms=5,view_count=1,limits=bounds,linear_axes=True,
        visible_state_numbers=[1,2,3,4],states=labels,legend=False,colorbar_orientation='vertical',
        segment_count=int(keep.sum()),no_bridging_across_hidden_segments=True,
        all_displayed_vertices_inside_canvas=True,only_Z_controls_segment_selection=True,
        full_raw_trajectory_preserved=True,normalized_time='t / 66 s')


def draw_c(a,states):
    fig=plt.figure(figsize=(8.8,4.65))
    gs=fig.add_gridspec(2,4,left=.095,right=.91,bottom=.12,top=.89,wspace=.18,hspace=.57)
    records=[]
    for j,s in enumerate(states):
        row,col=divmod(j,4);ax=fig.add_subplot(gs[row,col]);idx=round(s['time_s']/.005)
        count=a['field_5ms'][idx-2:idx+2].sum(0);hz=count/a['cell_e_counts']/.02
        assert count.sum()==a['spikes_1ms'][(idx-2)*5:(idx+2)*5,0].sum()
        im=ax.imshow(hz.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',
            norm=PowerNorm(.6,0,500),interpolation='nearest')
        for k,xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1.1))
            ax.text(xy[0],xy[1]+2.1,'AB'[k],color='#10656a',fontsize=9,ha='center',weight='bold',
                bbox=dict(fc='white',ec='none',pad=.15,alpha=.85))
        ax.set(xticks=[0,10,20],yticks=[0,10,20]);ax.tick_params(labelsize=9,length=2,pad=2,
            labelbottom=row==1,labelleft=col==0)
        if row==1:ax.set_xlabel('x (mm)',fontsize=10,labelpad=2)
        if col==0:ax.set_ylabel('y (mm)',fontsize=10,labelpad=2)
        ax.set_title(f"{s['number']}  {s['label']}\n{s['time_s']:.3f} s",color=s['color'],fontsize=11,pad=5)
        records.append(dict(**s,E_rate_Hz=hz,time_window_s=[(idx-2)*.005,(idx+2)*.005],native_spike_count=int(count.sum())))
    fig.text(.018,.70,'Entry 1',ha='center',va='center',rotation=90,fontsize=11)
    fig.text(.018,.25,'Entry 2',ha='center',va='center',rotation=90,fontsize=11)
    cb=fig.colorbar(im,cax=fig.add_axes([.94,.155,.018,.66]),ticks=[0,250,500])
    cb.set_label('E rate (Hz)',fontsize=11,labelpad=5);cb.ax.tick_params(labelsize=10)
    base.save(fig,'fig6-panelc')
    return records


def draw_d(comparisons):
    clinical=base.native.energy.clinical;fz=clinical.load_frozen('epilepsiae_1146')
    fig=plt.figure(figsize=(8.6,3.5));gs=fig.add_gridspec(1,2,left=.08,right=.88,bottom=.22,top=.83,wspace=.28)
    axes=[]
    for i,c in enumerate(comparisons):
        ax=fig.add_subplot(gs[i]);axes.append(ax);lo,hi=c['model_early_s']
        clinical._draw_field(ax,fz,clinical._normalize_minmax(np.asarray(c['model_robust_z'])),
            np.asarray(fz['support_a']),cmap='Blues',colorbar_values=np.array([0.,1.]),
            title=f'Model onset {i+1}\n{lo:.3f}–{hi:.3f} s',title_color='black',show_y=i==0)
        ax.set_xticks([-20,0,20]);ax.set_yticks([-10,0,10,20]);ax.tick_params(labelsize=11)
        ax.xaxis.label.set_fontsize(12);ax.yaxis.label.set_fontsize(12)
        ax.title.set_fontsize(13);ax.title.set_fontweight('normal')
        ax.text(.5,-.35,f"$r$ = {c['selected']['field_r']:+.3f}",transform=ax.transAxes,ha='center',fontsize=12)
    # Match the colorbar to the drawn field rectangles after equal-aspect layout.
    fig.canvas.draw();pos=axes[-1].get_position()
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='Blues'),
        cax=fig.add_axes([.92,pos.y0,.018,pos.height]),ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=11,length=3,pad=3)
    cb.set_label('Normalized z field',fontsize=12,labelpad=5)
    fig.canvas.draw()
    assert np.allclose(cb.ax.get_position().extents[[1,3]],pos.extents[[1,3]],atol=1e-12)
    record=dict(layout='two model onset fields',patient_field_displayed=False,
        analysis_contact_count=15,correlations=[c['selected']['field_r'] for c in comparisons],
        model_windows_s=[c['model_early_s'] for c in comparisons],colorbar_aligned_to_field_height=True,
        field_bounds=[ax.get_position().bounds for ax in axes],colorbar_bounds=cb.ax.get_position().bounds,
        colorbar_tick_fontsize=11,field_tick_fontsize=11,correlation_reference='Fixed E1146 SZ26')
    base.save(fig,'fig6-paneld')
    return record


def supplements(a,rates,states):
    base.OUT=OUT/'supplementary_candidates'
    st=a['slow_time_ms']/1000.;tt=(np.arange(len(rates))+.5)*.01
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True)
    for label,color,j,zi,mi in [('All E','#555',0,0,0),('Core A','#b2182b',1,5,1),('Core B','#2166ac',2,6,2)]:
        axes[0].plot(tt,rates[:,j],lw=.8,color=color,label=label)
        axes[1].plot(st,a['Z'][:,zi],lw=1.2,color=color)
        axes[2].plot(st,.0005*a['M'][:,mi],lw=1.2,color=color)
    axes[0].set_ylabel('E rate (Hz)');axes[0].legend(ncol=3,frameon=False)
    axes[1].set_ylabel('Z');axes[2].set(ylabel=r'$\eta_M M$ (mV)',xlabel='Time (s)')
    for ax in axes:
        ax.set_xlim(0,66);ax.spines[['top','right']].set_visible(False)
        for s in states:ax.axvline(s['time_s'],color=s['color'],ls=':',lw=.6)
    fig.tight_layout();base.save(fig,'full-continuous-state')
    fig,axes=plt.subplots(1,2,figsize=(12,3.5),sharey=True)
    for k,ax in enumerate(axes):
        take=(tt>=WINDOWS[k][0])&(tt<=WINDOWS[k][1])
        for j,label,color in [(0,'All E','#555'),(1,'Core A','#b2182b'),(2,'Core B','#2166ac')]:
            ax.plot(tt[take],rates[take,j],lw=1.4,color=color,label=label)
        for s in states[k*4:k*4+4]:ax.axvline(s['time_s'],color=s['color'],ls=':',lw=.7)
        ax.set(xlabel='Time (s)',title=f'Entry {k+1}',xlim=WINDOWS[k]);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Native E rate (Hz)');axes[1].legend(frameon=False,ncol=3,fontsize=10)
    fig.tight_layout();base.save(fig,'two-entry-native-rates')
    base.OUT=OUT


def write_docs(meta):
    (OUT/'caption_zh.md').write_text('''本候选改用同一固定网络、同一统一适应参数下已完成的连续种子9108402，0–66 s不中断、不重置。A显示11.00–12.40 s与61.22–62.62 s；每段保留进入前两次主要短事件及随后进入过程，左右同为1.40 s、相同触点增益。两段之间仅用断轴符号，不显示省略时长。保留80–1000 Hz原生全程滤波、既定触点顺序和13个显示通道，全部15触点仍用于能量分析。

1–4与5–8沿用Local、Spread、Entry、Ictal。时点依次为11.605、11.640、11.820、12.080 s与61.805、61.840、62.130、62.450 s。Local标在最后一次短事件之后的局部招募刚出现时；Spread取35 ms后的早期扩展；Entry取随后持续招募扩展到全体E的阶段；Ictal取全体E进入高活动后的早期。3/7是进入过程的展示帧，不再把较晚的全局200 Hz起点当成唯一进入位置。正式高活动定义仍为全体E≥200 Hz持续200 ms，其起点分别为12.060、62.430 s，没有通过移动图上编号改变检测结果。

B为全体E平均Z、适应电流eta_M M及放电率的单个三维轨迹，Z仅显示0.5–1；另两轴0–300 mV及0–350 Hz覆盖该Z区间内全部原生数据。仅标第一轮1–4，去掉5–8与图例，右侧竖色条表示完整66 s的归一化时间。跨出显示范围的线段不跨接，完整轨迹保存于trajectory_data.npz。C保留两行四帧的20 ms原生空间率，核位置与0–500 Hz色标不变，时点与A/B一致。

D显示最后20 ms全局低活动结束后的1 s模型场：11.546–12.546 s与61.793–62.793 s。两个模型均用同一0.5–3.5 s基线、全部15触点CAR、1–150 Hz谱功率及固定患者E1146 SZ26参考；只显示模型场，不绘制患者参考场。各场独立min–max显示，色条严格匹配场图实际高度，刻度字号与场图一致。E原样保留历史13人模型队列及原统计，不用这条新示例重算。

原生核率支持第一轮A先持续招募（A=11.60、B=11.81 s），第二轮B先持续招募（B=61.91、A=62.14 s）。第二轮最初短波亦由B先激活。第一轮最早还有核外短暂活动，因此这里展示核的持续招募顺序变化，不把它写成已证实的单核因果起源。两轮一秒能量场在原TA/TB评分规则下均偏TA，且在−20、+20、+50 ms窗口平移时保持TA；不能写为TA→TB能量类型切换。另一已完成种子9108407虽有TB/TA能量偏好，早期两轮均A先激活，且首轮平移50 ms即变为TA，未用它冒充相反起始传播。

这是对作者所指起始传播差别的示例修订，不估计发生概率。原数据、旧候选与正式入口保留；本版待作者目视检查。
''')
    (OUT/'README.md').write_text(f'''# Figure 6：按最后短事件后的实际进入过程重排

[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [图注](caption_zh.md)。

改用已完成连续种子9108402，重新核对局部再激活与初次传播；Local/Spread/Entry/Ictal四阶段语义保留。A每段两次主要短事件后进入，断轴不写省略时长；B为Z=0.5–1单图，只标1–4，竖色条、无图例；D色条与场图等高同字号。E统计原样保留。

两轮持续核招募顺序为A→B，但一秒能量分类均为TA。实际核对见entry_origin_review.json、onset_field_review.json与supplementary_candidates。未新增仿真，正式入口未替换。

复现：`python {Path(__file__).resolve()}`。
''')
    descriptions={
        'fig6-complete-layout':'A两段进入，B单个平均状态轨迹，C两行空间状态，D两次模型onset场，E历史统计。种子改为连续9108402，所有模型面板同步来源。',
        'fig6-panela':'每段保留进入前两次主要短事件，随后标出Local、Spread、Entry、Ictal。只留断轴符号，Time标签靠近轴线。',
        'fig6-panelb':'全体E平均状态，Z范围0.5–1。只标第一轮1–4，竖色条、无重复状态图例。',
        'fig6-panelc':'两轮进入各四个20 ms原生空间率窗口。与A/B时点一致，不用全局200 Hz阈值替代所有早期传播阶段。',
        'fig6-paneld':'最后20 ms静默结束之后的两个1 s模型能量场。色条与场图实际上下边缘对齐，刻度字号一致。',
        'fig6-panele':'历史13人模型队列统计原样保留。不是当前种子的队列检验。'}
    (OUT/'figures/README.md').write_text('\n\n'.join(f'### {k}.png / .pdf\n{v}\n**关注点**：'
        +('核招募先后与TA/TB能量分类分开解释。' if k!='fig6-panele' else '数据与统计未重算。') for k,v in descriptions.items())+'\n')
    (OUT/'supplementary_candidates/figures/README.md').write_text('''### full-continuous-state.png / .pdf
完整66 s全体E与双核放电率、Z和适应电流。保留主图断轴省略的终止与恢复过程。
**关注点**：两次进入来自连续轨迹，没有状态重置。

### two-entry-native-rates.png / .pdf
两次进入的原生双核及全体E率，与主图相同时间窗口及状态线。第一轮A先保持高活动，第二轮B先保持高活动。
**关注点**：最初短促波与随后持续招募不能混为一个起点。
''')


def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json';pointer_before=pointer.read_bytes()
    review=json.loads((REVIEW/'existing_pair_review.json').read_text())
    selection=next(x for x in review if x['seed']==SEED)
    field_review=json.loads((REVIEW/'onset_field_review.json').read_text())
    score=next(x for x in field_review if x['seed']==SEED)
    base.ROOT=base.analysis.ROOT=rows.audit.RAW;base.RUN_NAME=f'seed{SEED}';base.OUT=OUT
    a=base.load(base.RUN_NAME);rates,route=rows.audit.recruitment(base.ROOT/'runs'/base.RUN_NAME,a)
    assert route['core_order'][:2]=='AB' and route['sustained_200ms_core_order'][:2]=='AB'
    assert a['spikes_1ms'][:,0].sum()==a['field_5ms'].sum()==a['regions_1ms'][:,:3].sum()
    states=[dict(number=i+1,label=rows.PHASES[i%4],episode=i//4+1,time_s=t,color=rows.COLORS[i%4]) for i,t in enumerate(TIMES)]
    for k in range(2):
        assert WINDOWS[k][0]<min(TIMES[k*4:k*4+4])<max(TIMES[k*4:k*4+4])<WINDOWS[k][1]
        assert states[k*4]['time_s']>selection['episodes'][k]['last20ms_quiet_end_s']
        for s in states[k*4:k*4+4]:
            ix=round(s['time_s']*1000)
            s['core_A_B_rate20ms_Hz']=(a['regions_1ms'][ix-10:ix+10,:2].sum(0)/a['region_counts'][:2]/.02).tolist()
    assert states[0]['core_A_B_rate20ms_Hz'][0]>states[0]['core_A_B_rate20ms_Hz'][1]
    assert states[4]['core_A_B_rate20ms_Hz'][1]>states[4]['core_A_B_rate20ms_Hz'][0]
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':14,'xtick.labelsize':12,
        'ytick.labelsize':12,'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    wave=draw_a(a,states,selection);trajectory=draw_b(states);maps=draw_c(a,states)
    comparisons=[]
    for k,ep in enumerate(selection['episodes']):
        base.OUT=OUT/'onset_comparisons'/f'entry{k+1}';base.OUT.mkdir(parents=True,exist_ok=True)
        comparison=base.onset_comparison(a,ep['last20ms_quiet_end_s'])
        comparison['contract'].update(model_source=f'Same continuous66s unified-adaptation trajectory; seed{SEED}',
            model_onset='End of the last >=20ms global quiet interval before high activity; centered20ms rate<5Hz. Fixed1s window.')
        comparison.update(episode=k+1,energy_template=score['episodes'][k][0]['best_abs_template'],
            high_threshold_onset_s=ep['high_threshold_onset_s'])
        assert np.allclose(comparison['model_robust_z'],score['episodes'][k][0]['robust_z'],atol=1e-10)
        base.write(base.OUT/'field_comparison.json',comparison);comparisons.append(comparison)
    base.OUT=OUT
    dmeta=draw_d(comparisons)
    assert comparisons[0]['patient_source_sha256']==comparisons[1]['patient_source_sha256']
    assert np.array_equal(comparisons[0]['model_baseline_log_power'],comparisons[1]['model_baseline_log_power'])
    for ext in ['png','pdf','svg']:
        src=DISPLAY_PARENT/'figures'/f'fig6-panele.{ext}'
        dst=OUT/'figures'/f'fig6-panele.{ext}'
        shutil.copy2(src,dst)
        assert src.read_bytes()==dst.read_bytes()
    supplements(a,rates,states)
    previous.OUT=OUT;previous.compose()
    manifest=json.loads((OUT/'layout_manifest.json').read_text())
    manifest.update(A='Two equal1.4s windows, two preceding major events, break marks only',
        B='Single global-mean trajectory; Z0.5–1; states1–4 only; vertical colorbar; no legend',
        D='Two model fields; colorbar matches drawn field height and tick size')
    manifest.pop('relabel_from_parent',None);base.write(OUT/'layout_manifest.json',manifest)
    meta=dict(seed=SEED,source_run=str(base.ROOT/'runs'/base.RUN_NAME),same_run_panels=['A','B','C','D'],
        source_previous_candidate=str(SOURCE),states=states,waveform=wave,trajectory=trajectory,spatial_maps=maps,D=dmeta,
        continuous_no_reset=True,persistent_core_order=route['sustained_200ms_core_order'],
        energy_template_order=[ep[0]['best_abs_template'] for ep in score['episodes']],
        state_selection='Local: initial core recruitment after the final brief; Spread:35ms later; Entry:early sustained global recruitment; Ictal:20ms after200Hz high onset. Phase samples are distinct from event detection.',
        high_detection_unchanged='All-E>=200Hz for200ms',E_scope='Unchanged historical13-patient model cohort',
        source_simulation_rerun=False,producer=str(Path(__file__).resolve()),human_visual_acceptance='PENDING')
    meta['author_confirmed_phase_semantics']='1/5: local reactivation after the last interictal event; preserve Local/Spread/Entry/Ictal.'
    base.write(OUT/'figure_metadata.json',meta);base.write(OUT/'core_recruitment.json',route)
    base.write(OUT/'entry_origin_review.json',selection);base.write(OUT/'onset_field_review.json',score)
    base.write(OUT/'field_comparisons.json',comparisons)
    shutil.copy2(SOURCE/'scientific_result.json',OUT/'scientific_result.json')
    checks={}
    for pdf in (OUT/'figures').glob('*.pdf'):
        with base.fitz.open(pdf) as doc:
            checks[pdf.name]=all(pg.rect.contains(base.fitz.Rect(s['bbox'])) for pg in doc
                for block in pg.get_text('dict')['blocks'] if block['type']==0 for line in block['lines'] for s in line['spans'])
    assert all(checks.values()),checks
    with base.fitz.open(OUT/'figures/fig6-panela.pdf') as doc:
        txt=doc[0].get_text();assert 'omitted' not in txt and 'ICL1\n' not in txt and 'ICL2\n' not in txt
    from PIL import Image
    with base.fitz.open(OUT/'figures/fig6-complete-layout.pdf') as doc:
        pix=doc[0].get_pixmap(matrix=base.fitz.Matrix(2.6,2.6),alpha=False)
        rendered=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)
    assert np.array_equal(np.asarray(Image.open(OUT/'figures/fig6-complete-layout.png')),rendered)
    assert [(s['number'],s['time_s']) for s in trajectory['states']]==[(s['number'],s['time_s']) for s in states[:4]]
    assert [(s['number'],s['time_s']) for s in maps]==[(s['number'],s['time_s']) for s in states]
    assert pointer.read_bytes()==pointer_before
    assert (OUT/'trajectory_data.npz').read_bytes()==(SOURCE/'trajectory_data.npz').read_bytes()
    base.write(OUT/'validation.json',dict(status='PASS',one_continuous66s_run=True,no_new_simulation=True,
        native_E_spikes_conserved=True,all_eight_native_maps_count_conserved=True,A_B_C_states_synchronized=True,
        Local1_A_dominant_Local5_B_dominant=True,
        A_two_equal_windows=True,A_no_omitted_duration_text=True,A13_display15_analysis_contacts=True,
        B_Z_limits=[.5,1.],B_labels=[1,2,3,4],B_vertical_colorbar=True,B_no_legend=True,
        D_colorbar_aligned_to_field=True,D_common_patient_and_baseline=True,
        E_historical_statistic_unchanged=True,PDF_text_inside=checks,same_state_PNG_PDF_pixel_identity=True,
        formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    (OUT/'source').mkdir(exist_ok=True)
    for p in [Path(__file__),Path(previous.__file__),Path(prior.__file__),Path(rows.__file__),Path(base.__file__)]:
        shutil.copy2(p,OUT/'source'/p.name)
    write_docs(meta)
    for i in [1,2]:
        (OUT/'onset_comparisons'/f'entry{i}'/'figures/README.md').write_text('''### fig6-panele.png / .pdf
模型与固定患者场的计算核对图，保留原生计算器输出。主图D只展示模型场，此处不作为主面板。
**关注点**：患者未重新选择。

### onset-field-signed-z.png / .pdf
同一窗口未经逐场min–max显示归一化的带符号z场。用于核对显示与原始能量数值之间的区别。
**关注点**：不把颜色深浅当成跨场绝对能量比较。
''')
    print(json.dumps(dict(output=str(OUT),states=TIMES,correlations=dmeta['correlations'],
        persistent_order=route['sustained_200ms_core_order'],field_types=meta['energy_template_order'])),flush=True)


if __name__=='__main__':main()
