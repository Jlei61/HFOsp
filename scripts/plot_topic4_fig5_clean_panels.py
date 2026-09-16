#!/usr/bin/env python3
"""Readable single transition: honest early spatial snapshots and native raster zooms."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle,ConnectionPatch,Rectangle
import plot_topic4_fig5_single_transition as old
import analyze_topic4_fig5_preentry_events as audit
import plot_topic4_fig5_readable_preentry as previous
import analyze_topic4_fig5_log_m_scan as logscan
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
f=old.f
OUT=audit.OUT/'clean_panels_v3_broadband_z'
COL=['#616873','#267ba8','#dd871c','#ba263c']

def choose_states(a,first,end):
    snaps,selection=previous.choose_states(a,first,end)
    for snap,label in zip(snaps,['Rest','Interictal','Pre-ictal','Onset']):
        snap['label']=label
    original=snaps[1]['time_s'];snaps[1]['time_s']=round(original+.020,3)
    selection['prior_state2_time_s']=original
    selection['state2_display_revision']='Same event, 20ms later than previous localized early frame, to show established onset. No waveform or field editing.'
    selection['prior_locality_criterion_not_reapplied_after_shift']=True
    selection['display_state2_time_s']=snaps[1]['time_s']
    return snaps,selection

def full_raster(ax,a,snaps,end):
    it,ix=np.where(a['raster']);tt=it*.0001
    # Fixed neurons and row order, with more vertical room allocated to the two cores.
    mapping=np.r_[np.linspace(0,33,20),np.linspace(36,69,20),np.linspace(72,84,20),np.linspace(87,99,20)]
    for lo,hi,col in [(0,20,f.REGCOL[1]),(20,40,f.REGCOL[2]),(40,60,'#225b7f'),(60,80,'#c17730')]:
        take=(ix>=lo)&(ix<hi)
        ax.scatter(tt[take],mapping[ix[take]],s=7,marker='o',c=col,lw=0,rasterized=True)
    for y in [34.5,70.5,85.5]:ax.axhline(y,c='#bbb',lw=.8)
    ax.set(ylim=(-2,101),yticks=[16.5,52.5,78,93],yticklabels=['Core A E','Core B E','Other E','I'],xlim=(0,end))
    ax.set_title('A',loc='left',weight='bold',pad=16)
    ax.tick_params(axis='x',labelbottom=False)
    for s in snaps:
        ax.text(s['time_s'],.99,str(s['number']),transform=ax.get_xaxis_transform(),ha='center',va='top',
            color=s['color'],weight='bold',fontsize=19,bbox=dict(fc='white',ec='none',alpha=.85,pad=.3))
    return tt,ix,mapping

def zooms(fig,spec,a,snaps,tt,ix):
    gs=spec.subgridspec(1,2,wspace=.30);records=[]
    for i,s in enumerate(snaps[1:3]):
        ax=fig.add_subplot(gs[i]);lo=s['time_s']-.05;hi=lo+.30
        for low,high,col in [(0,20,f.REGCOL[1]),(20,40,f.REGCOL[2])]:
            take=(ix>=low)&(ix<high)&(tt>=lo)&(tt<hi)
            ax.scatter(tt[take],ix[take],s=15,marker='|',c=col,linewidth=.9,rasterized=True)
        ax.axhline(19.5,c='#bbb',lw=.7);ax.axvspan(s['time_s']-.025,s['time_s']+.025,color=s['color'],alpha=.12,lw=0)
        ax.axvline(s['time_s'],color=s['color'],ls=':',lw=1.2)
        ax.set(ylim=(-1,40),xlim=(lo,hi),yticks=[9.5,29.5],yticklabels=['Core A E','Core B E'],xlabel='Time (s)')
        ax.set_title(str(s['number']),loc='left',fontsize=18,color=s['color'],pad=9)
        ax.set_xticks([round(lo+.05,2),round(lo+.15,2),round(lo+.25,2)])
        records.append(dict(state=s['number'],time_window_s=[lo,hi],sample_ids=a['sample_ids'][:40],same_spikes_and_order=True))
    return records

def make_grid():
    extended=f.ROOT/'results/topic4_sef_hfo/fig5_log_m_entry_extension_20260915/measured_grid.json'
    if extended.exists():
        grid=f.read(extended)
        if grid['all_complete']:
            for key in ['mean','count','entered']:grid[key]=np.asarray(grid[key])
            return grid
        import analyze_topic4_fig5_entry_progress as progress
        return progress.collect()
    return logscan.collect()

def draw_grid(fig,spec,g,job):
    if g.get('quantity_kind')=='mean_confirmation_time_lower_bound':
        import analyze_topic4_fig5_entry_progress as progress
        return progress.draw(fig,spec,g,job,logscan.draw)
    return logscan.draw(fig,spec,g,letter='E',working_point=(job['tau_M_s'],job['eta_m']))

def grid_caption(grid):
    if grid.get('quantity_kind')=='mean_confirmation_time_lower_bound':
        p=grid['progress_summary'];times=p['audited_running_followup_s']
        span=f'{min(times):g}–{max(times):g}秒' if times else '无在跑轨迹'
        return (f'E为截至当前完整落盘计数的进度图，目标观察窗1000秒；已确认进入{p["observed_total"]}/48条，'
            f'其中新增晚进入{p["newly_observed"]}条，续跑完成{p["completed_extensions"]}/22条，'
            f'在跑{p["running"]}条（已核对随访{span}），排队{p["queued"]}条。'
            '无斜线格为两个噪声种子的已观测平均确认时间；斜线格的≥数值为平均首次确认时间下界，'
            '未进入的种子只使用各自已核对的随访时长，排队种子保留原300秒。'
            '各格随访尚不等长，不是共同1000秒限制均值；颜色统一使用1–1000秒log尺度。')
    return f'E为已完成的共同{grid["horizon_s"]:g}秒观察窗限制均值图；斜线表示有右删失，不能解释为永久不进入。'

def draw_f(fig,spec,a,comparison):
    import analyze_topic4_fig5_onset_z_field as onset_field
    clinical=onset_field.clinical
    _,fz,_=onset_field.patient_cases()
    gs=spec.subgridspec(2,1,height_ratios=[.10,1],hspace=.15)
    title=fig.add_subplot(gs[0]);title.axis('off');title.text(0,.3,'F',fontsize=24,weight='bold')
    maps=gs[1].subgridspec(1,4,width_ratios=[1,.045,1,.045],wspace=.5)
    fields=[np.asarray(comparison['model_robust_z']),np.asarray(comparison['patient_robust_z'])]
    titles=['Model\nOnset broadband field',f'E10 | {comparison["selected"]["public_seizure"]}\nEarly ictal field']
    bars=[];limits=[]
    for i,(values,heading) in enumerate(zip(fields,titles)):
        ax=fig.add_subplot(maps[2*i])
        mapped=clinical._draw_field(ax,fz,clinical._normalize_minmax(values),np.asarray(fz['support_a']),
            cmap='Blues',colorbar_values=values,title=heading,title_color='black',show_y=i==0)
        cb=fig.colorbar(mapped,cax=fig.add_subplot(maps[2*i+1]))
        ticks=([float(values.min()),0.,float(values.max())] if values.min()<0<values.max()
            else np.linspace(float(values.min()),float(values.max()),3))
        if values.min()<0<values.max() and values.max()/np.ptp(values)<.10:
            # Keep baseline zero readable when it sits very near the upper end.
            ticks=[float(values.min()),float(values.min())/2,0.]
        cb.set_ticks(ticks);cb.set_ticklabels([f'{value:.1f}' for value in ticks])
        cb.ax.set_title('power\nz',fontsize=18,pad=12)
        ax.title.set_fontsize(18)
        bars.append((cb.ax,ax));limits.append([float(values.min()),float(values.max())])
    return dict(quantity='Baseline-normalized1-150Hz log-band-power robust z',model_window_s=comparison['model_early_s'],
        model_baseline_s=comparison['model_baseline_s'],model_baseline_frames=comparison['model_baseline_frames'],
        patient_window_s=[0,10],patient_baseline_eeg_s=[-120,-90],patient=comparison['selected'],
        colorbar_limits=limits,colormap='Blues',display_geometry=comparison['display_geometry'],
        display_sigma_mm=float(fz['display_sigma_mm']),continuous_minmax=True,rank_transform=False,
        negative_values_clipped=False,amplitude_equivalence_claim=False),bars

def render(row,grid):
    a,r=audit.load_small(row['source'],keys=('spikes_1ms','regions_1ms','field_1ms','raster','slow_time_ms','Z','M','currents','lfp_time_ms','lfp_raw','time_ms'))
    first=r['tracker']['entries'][0];end=round(first['confirmation_s']+2,2)
    if r['tracker']['restore_s'] is not None:end=min(end,round(r['tracker']['restore_s']-.01,2))
    old.layout.trim_display(a,end)
    assert np.array_equal(a['field_1ms'].sum(1),a['spikes_1ms'][:,0])
    snaps,selection=choose_states(a,first['onset_s'],end)
    import analyze_topic4_fig5_onset_z_field as onset_field
    comparison=onset_field.compute(a,first['onset_s'],row['name'])
    plt.rcParams.update({'font.size':19,'axes.labelsize':21,'axes.titlesize':22,'xtick.labelsize':18,'ytick.labelsize':18})
    fig=plt.figure(figsize=(30,19))
    outer=fig.add_gridspec(1,2,width_ratios=[1.10,1.08],left=.065,right=.954,top=.955,bottom=.10,wspace=.20)
    left=outer[0].subgridspec(6,1,height_ratios=[2.5,1.0,.16,1.35,.34,1.25],hspace=.32)
    right=outer[1].subgridspec(2,1,height_ratios=[1.10,1],hspace=.33)
    upper=right[0].subgridspec(1,2,width_ratios=[1.18,1],wspace=.70)
    ra=fig.add_subplot(left[0]);tt,ix,mapping=full_raster(ra,a,snaps,end)
    ra.set_title('',loc='right')
    zm=fig.add_subplot(left[3],sharex=ra);ma=zm.twinx()
    zoomrecords=zooms(fig,left[1],a,snaps,tt,ix)
    for record,s in zip(zoomrecords,snaps[1:3]):
        lo,hi=record['time_window_s']
        ra.add_patch(Rectangle((lo,-1),hi-lo,71,fc='none',ec=s['color'],lw=2.1,zorder=10))

    zt=a['slow_time_ms']/1000;z=a['Z'];m=a['M']*row['eta_m']
    zm.fill_between(zt,z[:,2],z[:,4],color=f.REGCOL[0],alpha=.13,lw=0)
    for zi,mi,col in [(0,0,f.REGCOL[0]),(5,1,f.REGCOL[1]),(6,2,f.REGCOL[2])]:
        zm.plot(zt,z[:,zi],c=col,lw=1.8);ma.plot(zt,m[:,mi],c=col,ls='--',lw=1.6)
    zm.set(ylabel='Resource Z',ylim=(0,1.05),xlabel='Time (s)');ma.set_ylabel('ηM × M (mV equiv.)',labelpad=12)
    assert m.max()<.5,'Shared effective-M display limit would clip data.'
    ma.set(ylim=(0,.5),yticks=[0,.25,.5])
    ma.spines['right'].set_visible(True);zm.set_title('B',loc='left',weight='bold',pad=16)
    handles=[Line2D([],[],c=c,label=n) for c,n in zip(f.REGCOL,['All E','Core A','Core B'])]
    handles += [Line2D([],[],c='black',label='Z'),Line2D([],[],c='black',ls='--',label='ηM × M')]
    zm.legend(handles=handles,ncol=1,loc='upper right',fontsize=13,frameon=True,facecolor='white',edgecolor='none',framealpha=.92,handlelength=2.,borderpad=.25,labelspacing=.25)
    for ax in [ra,zm]:
        ax.axvspan(first['onset_s'],end,fc='#ba263c',alpha=.10,lw=0)
        for s in snaps:
            ax.axvspan(s['time_s']-.025,s['time_s']+.025,color=s['color'],alpha=.12,lw=0)
            ax.axvline(s['time_s'],ls=':',lw=1.2,c=s['color'],alpha=.9)
    header=fig.add_subplot(left[4]);header.axis('off');header.text(0,.4,'C',fontsize=22,weight='bold',bbox=dict(fc='white',ec='none',pad=2))
    mapgrid=left[5].subgridspec(1,4,wspace=.25);maps=[];links=[]
    for i,s in enumerate(snaps):
        ax=fig.add_subplot(mapgrid[i]);lo=round(s['time_s']*1000)-25
        field=a['field_1ms'][lo:lo+50].sum(0)/a['cell_e_counts']/.05
        im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for j,xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1.8))
            ax.text(xy[0],xy[1]+2.1,'AB'[j],color='#10656a',fontsize=13,ha='center',weight='bold',bbox=dict(fc='white',ec='none',pad=.25,alpha=.85))
        ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        if i==0:ax.set_ylabel('y (mm)')
        else:ax.set_yticklabels([])
        ax.set_title(f'{s["number"]}  {s["label"]}\n{s["time_s"]:.3f} s',c=s['color'],fontsize=18,pad=12)
        ax.title.set_bbox(dict(fc='white',ec='none',pad=1))
        connector=ConnectionPatch(xyA=(s['time_s'],-.18),coordsA=zm.get_xaxis_transform(),xyB=(.5,1.29),coordsB=ax.transAxes,
            arrowstyle='-',color=s['color'],lw=1.4,alpha=.8,clip_on=False,zorder=-1)
        fig.add_artist(connector)
        maps.append(dict(number=s['number'],time_window_s=[lo/1000,(lo+50)/1000],rate_Hz=field,
            sheet_fraction_above20Hz=float(np.mean(field>=20))))
        links.append((s,ax))
        if i==3:
            ca=ax.inset_axes([1.10,0,.06,1]);cb=fig.colorbar(im,cax=ca,ticks=[0,250,500]);cb.set_label('E rate (Hz)',fontsize=19)
    before=len(fig.axes);trajectory=old.draw_trajectory(fig,upper[0],a,snaps,end);ta=fig.axes[before];ta.set_title('',loc='left')
    for axis in [ta.xaxis,ta.yaxis,ta.zaxis]:axis.labelpad=22
    grid_ax=draw_grid(fig,upper[1],grid,r['job'])
    display,colorbar_axes=draw_f(fig,right[1],a,comparison)
    # Uniform larger x/y labels and ticks, including all colorbars and 3D axes.
    allaxes=list(fig.axes)
    for parent in list(fig.axes):allaxes.extend(parent.child_axes)
    for ax in allaxes:
        for axis in [ax.xaxis,ax.yaxis]+([ax.zaxis] if hasattr(ax,'zaxis') else []):axis.label.set_fontsize(21)
        ax.tick_params(labelsize=18)
        if ax.title.get_text() and ax.title.get_fontsize()<18:ax.title.set_fontsize(18)
    fig.canvas.draw()
    grid_box=grid_ax.get_window_extent();bar_box=grid_ax.child_axes[0].get_window_extent()
    assert np.isclose(grid_box.width,grid_box.height,atol=1e-6)
    assert np.isclose(grid_box.height,bar_box.height,atol=1e-6)
    bar_ax=grid_ax.child_axes[0]
    assert bar_ax.get_yscale()=='log'
    decade_y=bar_ax.transData.transform(np.c_[np.ones(3),[1,10,100]])[:,1]
    assert np.isclose(decade_y[1]-decade_y[0],decade_y[2]-decade_y[1])
    grid_layout=dict(width_px=grid_box.width,height_px=grid_box.height,
        height_over_width=grid_box.height/grid_box.width,colorbar_height_px=bar_box.height,
        square=True,parameter_axes='logarithmic',colorbar_axis_scale=bar_ax.get_yscale(),
        colorbar_ticks=bar_ax.get_yticks().tolist(),colorbar_decade_y_px=decade_y.tolist())
    for cbaxis,mapaxis in colorbar_axes:
        p=cbaxis.get_position();mp=mapaxis.get_position();cbaxis.set_position([p.x0,mp.y0,p.width,mp.height])
    ta.set_title('',loc='left')
    fig.text(ta.get_position().x0,upper[0].get_position(fig).y1+.012,'D',fontsize=24,weight='bold')
    state_links=[]
    for s,ax in links:
        rx=ra.get_xaxis_transform().transform((s['time_s'],0))[0];zx=zm.get_xaxis_transform().transform((s['time_s'],0))[0]
        assert abs(rx-zx)<1e-6
        state_links.append(dict(number=s['number'],time_s=s['time_s'],raster_x_px=rx,ZM_x_px=zx))
    dest=OUT/row['name'];figdir=dest/'figures';figdir.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figdir/f'fig5.{ext}',dpi=150,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)
    meta=dict(source=str(row['source']),job=r['job'],display_time_window_s=[0,end],snapshots=snaps,selection=selection,
        native_maps=maps,state_time_links=state_links,raster_zoom_windows=zoomrecords,
        raster=dict(fixed_neurons=80,core_vertical_fraction=.70,row_order_unchanged=True,marker_area_pt2=7,
            sample_ids=a['sample_ids'],raster_sha256=hashlib.sha256(a['raster'].tobytes()).hexdigest(),row_display_coordinates=mapping),
        panel_mapping={'A':'E-only core raster with boxed full300ms zoom windows; I separately above','B':'Z and effective M','C':'Rest / Interictal / Pre-ictal / Onset native50ms fields; vertical colorbar right of4','D':'Actual SNN trajectory','E':'M first-entry timings','F':'Model onset broadband robust-z field and most similar enhancement seizure field within E1146'},
        fonts=dict(xyz_axis_labels_pt=21,tick_labels_pt=18,panel_title_pt=22),effective_M_axis_shared_limits=[0,.5],E1=trajectory,F_grid=grid,E_grid_layout=grid_layout,E2=comparison,E2_display=display,all_descriptive_titles_removed=False,legend_location='upper right, vertical',time_axis_label='Time (s)',
        event_frequency_audit=str(audit.OUT/'event_audit.json'),event_frequency_increase_claim=False,
        scope='Figure selection is illustrative, not a training target or an independent claim of mode confinement. High rate is not proof of oscillatory clinical seizure.',
        no_simulation_state_change_in_plotting=True,manual_reset_in_display=False,agent_visual_review='PENDING',human_review='PENDING',
        producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    f.write(dest/'fig5_metadata.json',f.safe(meta))
    (figdir/'README.md').write_text(f'### fig5.png / .pdf\nC保留1–4编号和Rest、Interictal、Pre-ictal、Onset小标题，E发放率色条竖放于4右侧；A/B及实际状态时刻沿用上一版。F左为模型首次高态起点后1秒的1–150Hz log-power robust-z，基线为0.5–3.5秒，右为E1146内按预先固定规则选择的{comparison["selected"]["public_seizure"]}早期发作场；两图均用Fig3C右图画法及真实robust-z色条。\n'+grid_caption(grid)+'\n**关注点**：模型3秒基线仅5个重叠PSD窗，是Fig3C公式的短窗适配；患者保持原30秒远端基线与0–10秒临床窗，所选最近似病例仅作示例，候选待人工检查。\n')
    print(row['name'],[(s['label'],s['time_s']) for s in snaps],flush=True)
    return dict(name=row['name'],eta_m=row['eta_m'],seed=row['seed'],figure=str(figdir/'fig5.png'),pdf=str(figdir/'fig5.pdf'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--eta',type=float,nargs='+',default=[.001,.0005]);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);grid=make_grid()
    f.write(OUT/'current_grid.json',f.safe(grid))
    rows=[render(row,grid) for row in audit.sources() if row['eta_m'] in args.eta and (row['source']/'result.json').exists()]
    f.write(OUT/'delivery_manifest.json',dict(versions=rows,human_review='PENDING'))
    (OUT/'README.md').write_text('# Fig5 状态标题与broadband robust-z场\n\n'+ '\n'.join(f'- {r["name"]}：[PNG]({r["figure"]}) · [PDF]({r["pdf"]})' for r in rows)+f'\n\nC色条竖放于4右侧，1–4为Rest / Interictal / Pre-ictal / Onset。F为模型与E1146患者的早期broadband robust-z场，按Fig3C右图画法；模型1秒早期窗、0.5–3.5秒基线是短窗适配，患者仍用原始远端基线和0–10秒临床窗。\n\n'+grid_caption(grid)+'\n\n[1000秒续跑进度](../../fig5_log_m_entry_extension_20260915/README.md)。后台每10分钟检查刷新，新增终点立即触发重绘；候选待人工检查。\n')
if __name__=='__main__':main()
