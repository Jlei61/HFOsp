#!/usr/bin/env python3
"""Fig. 5: physical core outlines, compact trajectory, spatial comparison and latency."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, Rectangle
import plot_topic4_fig5_manual_release_layout_v9 as previous
from analyze_topic4_fig5_early_spatial import OUT as ANALYSIS

ROOT = previous.ROOT
BASE = previous.OUT.parent
OUT = BASE / 'layout_v10'
FIG = OUT / 'figures'
SIGMA_MM = 2.5


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.16)
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.16)
    plt.close(fig)


def core_outlines(ax, centers, radius, labels=True, color='#08aebc'):
    for k, xy in enumerate(centers):
        ax.add_patch(Circle(xy,radius,facecolor='none',edgecolor=color,lw=1.3,zorder=7))
        if labels:
            ax.text(xy[0],xy[1]+radius+.35,'AB'[k],ha='center',va='bottom',
                    color='#125965',fontsize=11,weight='bold',zorder=8,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.3))


def contact_field(ax, xy, values, centers, radius, norm, cmap, title, ylabel=True):
    grid=np.linspace(0,20,180);xx,yy=np.meshgrid(grid,grid)
    valid=np.isfinite(values)
    d2=(xx[...,None]-xy[valid,0])**2+(yy[...,None]-xy[valid,1])**2
    weights=np.exp(-d2/(2*SIGMA_MM**2))
    support=weights.sum(-1)
    field=np.sum(weights*values[valid],axis=-1)/np.maximum(support,1e-12)
    alpha=np.minimum(1,support/(.35*support.max()))
    ax.imshow(field,origin='lower',extent=[0,20,0,20],norm=norm,cmap=cmap,
              alpha=alpha,interpolation='nearest',zorder=1)
    ax.scatter(xy[:,0],xy[:,1],s=28,facecolors='white',edgecolors='#343434',lw=.7,zorder=5)
    ax.scatter(xy[valid,0],xy[valid,1],s=28,c=values[valid],norm=norm,cmap=cmap,
               edgecolors='#343434',lw=.7,zorder=6)
    core_outlines(ax,centers,radius)
    ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
    ax.set_ylabel('y (mm)' if ylabel else '')
    if not ylabel:ax.set_yticklabels([])
    ax.tick_params(labelsize=11)
    ax.set_title(title,fontsize=13,pad=8)
    return field


def comparison(fig, spec, data, summary, radius, heading=None):
    gs=spec.subgridspec(2,1,height_ratios=[.25,1],hspace=.30)
    titleax=fig.add_subplot(gs[0]);titleax.axis('off')
    text=heading or 'Interictal recruitment order and early high-state power'
    titleax.text(0,.9,text,fontsize=15,weight='bold',va='center')
    titleax.text(0,.20,'Same early window: 10.48–10.73 s; correspondence scored at contacts',fontsize=11,color='#555555')
    maps=gs[1].subgridspec(1,5,width_ratios=[1,1,.045,1,.045],wspace=.22)
    ranknorm=Normalize(0,1)
    power=data['early_delta_power']/1e6
    powernorm=Normalize(0,np.ceil(power.max()))
    for k,family in enumerate(['A','B']):
        group=summary['groups'][family]
        title=f'{family}-leading events (n={group["n_events"]})\nρ = {group["template_contact_rho"]:+.2f}'
        ax=fig.add_subplot(maps[k])
        contact_field(ax,data['contact_xy'],data['template_rank'][k],data['centers_mm'],
                      radius,ranknorm,'viridis',title,ylabel=k==0)
    cbax=fig.add_subplot(maps[2]);cb=fig.colorbar(ScalarMappable(norm=ranknorm,cmap='viridis'),cax=cbax)
    cb.set_ticks([0,.5,1]);cb.set_ticklabels(['0','.5','1'])
    cb.ax.tick_params(labelsize=9);cb.ax.set_title('Rank\n0 early',fontsize=9,pad=9)
    ax=fig.add_subplot(maps[3])
    contact_field(ax,data['contact_xy'],power,data['centers_mm'],radius,powernorm,'Blues',
                  'Early high-state\npower increase',ylabel=False)
    cbax=fig.add_subplot(maps[4]);cb=fig.colorbar(ScalarMappable(norm=powernorm,cmap='Blues'),cax=cbax)
    cb.set_ticks([0,powernorm.vmax/2,powernorm.vmax]);cb.ax.tick_params(labelsize=10)
    cb.ax.set_title('ΔP\n×10⁶',fontsize=10,pad=7)
    return dict(display_kernel_sigma_mm=SIGMA_MM, display='Gaussian contact interpolation with support fade; actual contacts overlaid',
                scoring='Direct contact values, not smoothed field pixels',
                shared_high_state_map=True, rank_limits=[0,1],power_scale=1e6,
                power_units='virtual SEEG proxy squared',power_limits_scaled=[0,float(powernorm.vmax)])


def latency(fig,spec,data,heading='F  Z kinetics and transition time'):
    gs=spec.subgridspec(1,3,width_ratios=[1,.055,1.1],wspace=.38)
    ax=fig.add_subplot(gs[0]);v=data['restricted_mean_time_s']
    norm=Normalize(0,24);im=ax.imshow(v,origin='lower',cmap='viridis',norm=norm,aspect='equal')
    ax.set(xticks=range(3),xticklabels=['2.5','5','10'],yticks=range(3),yticklabels=['75','95.2','120'],
           xlabel=r'$\tau_Z$ (s)',ylabel=r'Depletion threshold $I_{th}$')
    ax.tick_params(labelsize=12)
    ax.set_title(heading,fontsize=15,loc='left',weight='bold',pad=14)
    for y in range(3):
        for x in range(3):
            n=int(data['parameter_n'][y,x]);nobs=round(data['transition_fraction'][y,x]*n)
            ax.text(x,y,f'{v[y,x]:.1f} s\n{nobs}/{n}',ha='center',va='center',fontsize=12,
                    color='white' if v[y,x]<12 else '#222222')
    ax.add_patch(Rectangle((.5,.5),1,1,fc='none',ec='#ee6d38',lw=2))
    cbax=fig.add_subplot(gs[1]);cb=fig.colorbar(im,cax=cbax,ticks=[0,8,16,24])
    cb.set_label('Restricted mean time (s)',fontsize=12,labelpad=8);cb.ax.tick_params(labelsize=11)
    tx=fig.add_subplot(gs[2]);tx.axis('off')
    tx.text(0,.92,'3 noise seeds per cell',fontsize=13,weight='bold',va='top')
    tx.text(0,.74,'High-state criterion:\nE rate ≥200 Hz for 200 ms',fontsize=12,va='top',linespacing=1.4)
    tx.text(0,.49,'Each cell: restricted mean time\nand runs reaching the criterion.\nUnreached runs contribute 24 s.',fontsize=11.5,va='top',linespacing=1.5)
    tx.text(0,.17,'τZ controls depletion and recovery.\nIth is a current threshold\n(mV equivalent).',fontsize=11.5,va='top',linespacing=1.4)


def left_with_boundaries(fig,spec,a,run,windows,radius,early):
    before=len(fig.axes)
    meta=previous.previous.prior.old.left_panels(fig,spec,a,run,windows)
    axes=fig.axes[before:]
    maps=[ax for ax in axes if ax.images]
    assert len(maps)==5
    for k,ax in enumerate(maps):
        for artist in list(ax.collections):artist.remove()
        core_outlines(ax,a['centers_mm'],radius,labels=(k==0),color='#42d5d5')
    # Mark the independently selected power window on the common time axes.
    for ax in axes[:4]:
        if ax.get_xlim()[1]>25:
            ax.axvline(early[0],color='#2a566c',lw=.8,ls='--',alpha=.8)
            ax.axvline(early[1],color='#2a566c',lw=.8,ls='--',alpha=.8)
    meta.update(core_markers='Physical circular threshold-support boundary, not screen-size center markers',
                physical_core_radius_mm=radius, early_power_window_s=early)
    return meta


def main():
    data=np.load(ANALYSIS/'analysis_arrays.npz')
    summary=json.loads((ANALYSIS/'summary.json').read_text())
    radius=summary['actual_core_radius_mm']
    metadata=json.loads((previous.OUT/'figure_metadata.json').read_text())
    a,run,*_=previous.previous.prior.old.previous.source.load_main()
    t,xyz,_=previous.previous.current_coordinates(a)
    paths=previous.previous.previous.previous.complete_paths(t,xyz,run,metadata['windows'])
    saved=np.load(BASE/'layout_v8/trajectory_arrays.npz')
    assert all(np.array_equal(p['coords'],saved[f'path{k}_Z_H_E']) for k,p in enumerate(paths))

    fig=plt.figure(figsize=(13,5.7))
    spec=fig.add_gridspec(1,1,left=.055,right=.945,top=.94,bottom=.16)[0]
    spatial=comparison(fig,spec,data,summary,radius)
    fig.text(.055,.025,'ΔP: baseline-referenced mean-square virtual SEEG proxy; one high-state map compared with both families.',fontsize=11)
    save(fig,'interictal_order_vs_early_power')

    fig=plt.figure(figsize=(11,4.8))
    spec=fig.add_gridspec(1,1,left=.09,right=.97,top=.84,bottom=.18)[0]
    latency(fig,spec,data,heading='Z kinetics and time to sustained high activity')
    save(fig,'z_parameter_transition_time')

    fig=plt.figure(figsize=(26,17))
    outer=fig.add_gridspec(1,2,width_ratios=[1.37,1],left=.054,right=.94,
                          top=.94,bottom=.055,wspace=.18)
    left=left_with_boundaries(fig,outer[0],a,run,metadata['windows'],radius,summary['early_window_s'])
    right=outer[1].subgridspec(3,1,height_ratios=[1.28,1.,.78],hspace=.42)
    # Only this axis occupies the upper block: smaller than the previous large E.
    phase_slot=right[0].subgridspec(1,3,width_ratios=[.10,1,.20])[1]
    ax=fig.add_subplot(phase_slot,projection='3d')
    previous.plot_summary(ax,t,xyz,paths,run,metadata['windows'],panel_letter=False)
    ax.set_title('E1  Firing–inhibition trajectory',fontsize=15,loc='left',weight='bold',pad=12)
    # Annotation point offsets scale with the reduced panel, not the state values.
    for artist in ax.artists:
        if hasattr(artist,'xyann'):
            artist.xyann=tuple(.68*np.array(artist.xyann))
            artist.set_fontsize(10 if '\n' in artist.get_text() else 12)
    comparison(fig,right[1],data,summary,radius,heading='E2  Interictal order versus early high-state power')
    latency(fig,right[2],data)
    fig.suptitle('Slow inhibitory depletion, spatial recruitment and high-state transition in a dual-core SNN',fontsize=20,y=.985)
    save(fig,'fig5_manual_core_release_layout_v10')

    # Direct native grids are a separate check of the tempting readout-level claim.
    replaypath=ANALYSIS/'native_current_replay.npz'
    replaymeta=ANALYSIS/'native_current_replay_summary.json'
    if replaypath.exists() and replaymeta.exists():
        replay=np.load(replaypath);rs=json.loads(replaymeta.read_text())
        support=json.loads((ANALYSIS/'matched_support_audit.json').read_text())
        fig=plt.figure(figsize=(14,7.8));gs=fig.add_gridspec(2,3,left=.07,right=.91,top=.88,bottom=.09,wspace=.27,hspace=.35)
        for k,family in enumerate(['A','B']):
            ax=fig.add_subplot(gs[0,k]);rank=data['native_template_rank'][k]
            im=ax.imshow(np.ma.masked_invalid(rank.reshape(20,20)),origin='lower',extent=[0,20,0,20],cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
            core_outlines(ax,data['centers_mm'],radius)
            ax.set_title(f'{family}-leading recruitment',fontsize=13)
            ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20])
            cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04,ticks=[0,.5,1])
            cb.ax.set_yticklabels(['Early','.5','Late']);cb.ax.set_title('Rank',fontsize=10,pad=8)
        ax=fig.add_subplot(gs[0,2]);v=replay['native_current_delta_power']/1e6
        if v.min()>=0: norm=Normalize(0,v.max());cmap='Blues'
        else:norm=TwoSlopeNorm(vmin=v.min(),vcenter=0,vmax=v.max());cmap='RdBu_r'
        im=ax.imshow(v.reshape(20,20),origin='lower',extent=[0,20,0,20],norm=norm,cmap=cmap,interpolation='nearest')
        core_outlines(ax,data['centers_mm'],radius);ax.set_title('Native current-power increase',fontsize=13)
        ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20])
        cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label('ΔP (mV² ×10⁶)')
        ax=fig.add_subplot(gs[1,0]);v=data['native_early_delta_hz']
        im=ax.imshow(v.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',interpolation='nearest')
        core_outlines(ax,data['centers_mm'],radius);ax.set_title('Native rate increase (Hz)',fontsize=13)
        ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20]);fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
        ax=fig.add_subplot(gs[1,1:]);ax.axis('off')
        ax.text(0,.95,'Direct spatial correspondence: earlier recruitment vs high-state signal',fontsize=13,weight='bold',va='top')
        text='                              A-leading  B-leading\n'
        text+=f'Contact power (main panel)       {summary["groups"]["A"]["template_contact_rho"]:+.2f}      {summary["groups"]["B"]["template_contact_rho"]:+.2f}\n'
        text+=f'Native power, full grid          {rs["native_template_power_rho"]["A"]:+.2f}      {rs["native_template_power_rho"]["B"]:+.2f}\n'
        text+=f'Native power, at contacts        {support["groups"]["A"]["native_order_native_power_at_contacts"]:+.2f}      {support["groups"]["B"]["native_order_native_power_at_contacts"]:+.2f}\n'
        text+=f'Native spike rate, full grid     {summary["groups"]["A"]["template_native_rho"]:+.2f}      {summary["groups"]["B"]["template_native_rho"]:+.2f}'
        ax.text(0,.76,text,fontsize=11.5,family='monospace',va='top',linespacing=1.5)
        ax.text(0,.14,'B correspondence survives native sampling at contacts, but not over the full grid.\nNative power: baseline 8.00–8.25 s; main contact/rate: 0.5–8 s.\nOne realization. No spatial smoothing of the native grid.',fontsize=10.5,va='top',linespacing=1.4)
        fig.suptitle('Native spatial audit of the same 10.48–10.73 s transition window',fontsize=17,y=.98)
        save(fig,'native_spatial_correspondence_audit')

    metadata.update(source=str(previous.OUT),left=left,spatial_comparison=spatial,
        spatial_analysis=str(ANALYSIS),core_radius_mm=radius,phase_panel_smaller=True,
        parameter_panel='F: all 27 existing runs, 3x3 cells; restricted mean at 24s, no interpolated sweep',
        layout_revision='E1 compact phase / E2 paired spatial comparison / F parameter heatmap',
        human_acceptance='PENDING_USER_REVIEW')
    write('figure_metadata.json',metadata)
    write('producer_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path(previous.__file__),ROOT/'scripts/analyze_topic4_fig5_early_spatial.py',ROOT/'scripts/replay_topic4_fig5_native_current_field.py',ROOT/'scripts/analyze_topic4_fig5_native_support.py']})
    (FIG/'README.md').write_text('''### fig5_manual_core_release_layout_v10.png / .pdf
完整图补上物理半径1.5毫米的双核边界，缩小右侧三维轨迹，并在其下加入A先行、B先行间期事件与同一次早期高态功率图的比较。F复用27条既有运行，显示τZ与耗竭电流阈值对24秒限制平均首入时间的影响，各格保留进入数/总数。
**关注点**：E2的同一高态图没有按两类事件分别选窗；A/B标签为1.75毫米邻域的先行观测，不是因果源头证明。实际阈值core半径1.5毫米与raster/Z统计邻域1.75毫米区分；不能将电极功率对应直接升级为全网传播恢复。

### interictal_order_vs_early_power.png / .pdf
两类间期事件使用各自的参与掩码和等事件权重中位rank模板，右侧共用模型首段持续200Hz活动的前250毫秒，即10.48–10.73秒的功率升高。展示用相同2.5毫米高斯插值核，统计直接在15个触点上计算；功率是虚拟SEEG电流proxy的基线校正均方值，不是患者1–150Hz频带功率。
**关注点**：保留A/B两边包括不匹配的结果；真实core边界按物理毫米显示，不能将插值像素视为独立数据。

### z_parameter_transition_time.png / .pdf
已有τZ=2.5/5/10秒和电流阈值75/95.2/120的九格扫描，每格三条配对噪声运行。无转变运行按24秒进入限制均值，不剔除；没有插值成未经仿真的连续相图。
**关注点**：τZ同时改变耗竭与恢复，阈值不是每事件耗竭量；颜色表示首入高态判据时间，不能照抄示意图对纯恢复时间常数的解释。
''',encoding='utf-8')
    if replaypath.exists() and replaymeta.exists():
        with (FIG/'README.md').open('a') as f:
            f.write('''
### native_spatial_correspondence_audit.png / .pdf
直接使用原生1毫米网格比较间期招募时序、早期突触电流功率与放电率，不经过电极插值。电流场由两个合计800毫秒的检查点重放补录，要求所采spikes、全E/I计数及虚拟SEEG与原轨迹逐点一致。
**关注点**：同基线检查中，B先行的对应在触点位置的原生网格仍成立，而在全网格不成立，说明采样范围会改变对应；不能把差异直接归因于电极平滑。电流场与放电率是不同观测；图中电流基线为8.0–8.25秒，主面板触点功率和放电率基线为0.5–8秒，完整同基线对照另存matched_support_audit.json。
''')
    write('artifact_qa.json',dict(trajectory_unchanged=True,physical_core_radius_mm=radius,
        parameter_cells=9,parameter_runs=27,early_window_before_refill=True,
        agent_visual_review='PENDING',human_acceptance='PENDING_USER_REVIEW'))
    write('delivery_status.json',dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',human_acceptance='PENDING_USER_REVIEW'))
    print(json.dumps(dict(out=str(OUT),spatial=spatial,native_audit_included=replaypath.exists())))


if __name__=='__main__':main()
