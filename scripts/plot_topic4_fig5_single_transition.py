#!/usr/bin/env python3
"""One pre-intervention transition, raster first, and reference-style E2."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import copy
import hashlib
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, gaussian_filter1d
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plot_topic4_m_parameter_modes as f
import plot_topic4_fig5_resting_recovery as layout
import plot_topic4_weaker_M_original_fig5 as original
import analyze_topic4_weaker_M_onset_pilot as pilot
from analyze_topic4_fig5_early_spatial import contact_order, correlation
from plot_topic4_fig5_spatial_latency_layout_v10 import contact_field

OUT=f.ROOT/'results/topic4_sef_hfo/fig5_single_transition_20260914/raster_alignment_v2'
REFERENCE=f.ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'


def compare_families(a,onset,signal_name='Original unfiltered |AMPA| + |GABA| contact current proxy'):
    """Same event/window/mean-square definitions as the supplied v10 reference."""
    assert onset>8.25
    lt=a['lfp_time_ms']/1000;signal=np.asarray(a['lfp_raw'],float)
    rate5=a['spikes_1ms'][:,0].reshape(-1,5).sum(1)/32000/.005
    t5=(np.arange(len(rate5))+.5)*.005
    base=(lt>=.5)&(lt<8.)
    quiet=base&(np.interp(lt,t5,rate5)<1.)
    assert quiet.sum()>=100
    center=np.median(signal[quiet],axis=0)
    noise=1.4826*np.median(abs(signal[quiet]-center),axis=0)
    early=(lt>=onset)&(lt<onset+.25)
    b=signal[base]-center;h=signal[early]-center
    dp=np.mean(h*h,0)-np.mean(b*b,0)
    dvar=np.var(h,0)-np.var(b,0)
    dmean=np.mean(h,0)**2-np.mean(b,0)**2
    assert np.allclose(dp,dvar+dmean,rtol=1e-10,atol=1e-6)
    local=gaussian_filter1d(a['regions_1ms'][:,:2]/a['region_counts'][:2]*1000,2,axis=0)
    active=binary_closing(rate5>1,structure=np.ones(2))
    events=[];ranks=[]
    for lo,hi in f.spans(active):
        start,end=lo*.005,hi*.005
        if start<.5 or end>8. or rate5[lo:hi].max()<20:continue
        piece=local[round(start*1000):round(end*1000)]
        local_onsets=[]
        for c in range(2):
            hits=np.flatnonzero(piece[:,c]>=max(5,.2*piece[:,c].max()))
            local_onsets.append(float(hits[0]) if len(hits) else None)
        lag=local_onsets[1]-local_onsets[0] if all(v is not None for v in local_onsets) else None
        family='A' if lag is not None and lag>=5 else 'B' if lag is not None and lag<=-5 else 'unclassified'
        times,rank=contact_order(lt,signal,start,end,noise,np.ones(signal.shape[1],bool))
        ranks.append(rank)
        events.append(dict(start_s=start,end_s=end,family=family,B_minus_A_local_onset_ms=lag,
                           contact_onset_s=times,rank=rank,contact_rho=correlation(-rank,dp)))
    rank_array=np.array(ranks);groups={}
    for family in ['A','B']:
        ix=np.array([v['family']==family for v in events])
        if ix.sum():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                template=np.nanmedian(rank_array[ix],axis=0)
            support=np.isfinite(rank_array[ix]).mean(0);template[support<.5]=np.nan
        else:template=np.full(signal.shape[1],np.nan);support=np.zeros(signal.shape[1])
        groups[family]=dict(n_events=int(ix.sum()),template_rank=template,support_fraction=support,
                            contact_rho=correlation(-template,dp))
    return dict(groups=groups,events=events,baseline_s=[.5,8.],early_window_s=[onset,onset+.25],
        early_delta_power=dp,baseline_power=np.mean(b*b,0),quiet_baseline=center,
        delta_variance=dvar,delta_squared_mean=dmean,mean_square_decomposition_verified=True,
        contacts_with_increased_power=int((dp>0).sum()),quiet_samples=int(quiet.sum()),signal=signal_name,
        power_definition='Delta P = mean_early[(L-b)^2] - mean_0.5-8s[(L-b)^2]; b = quiet-bin median.',
        units='Unfiltered current-proxy units squared. Includes sustained mean shifts; not 1–150 Hz band power or integrated energy.',
        event_definition='Global E >1 Hz at5ms, one-bin closing, peak >=20Hz, wholly within0.5–8s.',
        family_definition='Core-neighborhood smoothed rate crosses max(5Hz,20% local peak); >=5ms A/B lead; otherwise unclassified.',
        template_definition='Equal-event median normalized masked rank; contact participation in at least half family events.',
        correlation_definition='Signed Spearman correlation of negative template rank and the same early Delta P across directly observed contacts.',
        selection='Both families shown. First250ms of first sustained-high entry, no spatial/window optimization.',
        legacy_difference='The supplied v10 used the separately recorded Z-applied proxy. Current trials saved original pre-Z synaptic-current readout; no regional-mean-Z reconstruction is substituted.',
        scope='One fixed network/noise realization. A/B-leading are model event families, not patient TA/TB labels; this panel contains no patient data.')


def verify_reference():
    saved=np.load(REFERENCE/'early_spatial_v1/analysis_arrays.npz')
    with np.load(REFERENCE/'runs/continuous_refill_release.npz') as old:
        n=round(11.*10000)
        ec=np.rint(old['rate_e_hz'][:n]*32000*.0001).astype(int).reshape(-1,10).sum(1)
        a=dict(spikes_1ms=ec[:,None],lfp_raw=old['lfp_effective'][:n//5],
               lfp_time_ms=old['lfp_time_ms'][:n//5],regions_1ms=old['region_spikes_1ms'][:n//10],
               region_counts=old['region_counts'])
        measured=compare_families(a,10.48,signal_name='Historical Z-applied contact current proxy')
    summary=f.read(REFERENCE/'early_spatial_v1/summary.json')
    assert np.allclose(measured['early_delta_power'],saved['early_delta_power'],rtol=1e-10,atol=1e-6)
    for k,family in enumerate(['A','B']):
        assert measured['groups'][family]['n_events']==summary['groups'][family]['n_events']
        assert np.allclose(measured['groups'][family]['template_rank'],saved['template_rank'][k],equal_nan=True)
        assert np.isclose(measured['groups'][family]['contact_rho'],summary['groups'][family]['template_contact_rho'])
    qa=dict(status='PASS',same_reference_power_and_templates=True,
            reference=str(REFERENCE/'layout_v10/figures/fig5_manual_core_release_layout_v10.png'),
            historical_groups=measured['groups'],new_run_readout_difference=measured['legacy_difference'])
    f.write(OUT/'reference_semantics_qa.json',f.safe(qa))


def draw_e2(fig,spec,a,comparison):
    gs=spec.subgridspec(2,1,height_ratios=[.12,1],hspace=.26)
    title=fig.add_subplot(gs[0]);title.axis('off')
    title.text(0,.35,'E2  Interictal order and early high-state power',weight='bold',fontsize=18)
    maps=gs[1].subgridspec(1,5,width_ratios=[1,1,.035,1,.04],wspace=.28)
    for i,family in enumerate(['A','B']):
        group=comparison['groups'][family];ax=fig.add_subplot(maps[i])
        rho=group['contact_rho'];rho_text=f'ρ = {rho:+.2f}' if rho is not None else 'ρ not estimable'
        contact_field(ax,a['contact_xy'],np.array(group['template_rank'],float),a['centers_mm'],float(a['core_radius_mm']),
                      Normalize(0,1),'viridis',f'{family}-leading events (n={group["n_events"]})\n{rho_text}',ylabel=i==0)
        ax.tick_params(labelsize=14);ax.title.set_fontsize(15);ax.xaxis.label.set_fontsize(16);ax.yaxis.label.set_fontsize(16)
        if not group['n_events']:ax.text(.5,.5,'Not observed',transform=ax.transAxes,ha='center')
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),cax=fig.add_subplot(maps[2]),ticks=[0,.5,1])
    cb.ax.set_title('Rank\n0 early',fontsize=12,pad=12);cb.ax.tick_params(labelsize=12)
    p=np.asarray(comparison['early_delta_power'])/1e6
    limit=max(1.,float(np.ceil(np.max(abs(p)))))
    if np.min(p)>=0:norm=Normalize(0,limit);cmap='Blues'
    else:norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit);cmap='RdBu'
    ax=fig.add_subplot(maps[3]);lo,hi=comparison['early_window_s']
    contact_field(ax,a['contact_xy'],p,a['centers_mm'],float(a['core_radius_mm']),norm,cmap,
                  f'Early high-state power\n{lo:.2f}–{hi:.2f} s',ylabel=False)
    ax.tick_params(labelsize=14);ax.title.set_fontsize(15);ax.xaxis.label.set_fontsize(16)
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_subplot(maps[4]))
    cb.set_label('ΔP (unfiltered proxy² ×10⁶)',fontsize=13,labelpad=8);cb.ax.tick_params(labelsize=12)
    cb.set_ticks([0,limit/2,limit] if norm.vmin==0 else [-limit,0,limit])
    return dict(rank_cmap='viridis',rank_limits=[0,1],power_cmap=cmap,power_limits=[norm.vmin,norm.vmax],
                power_divisor=1e6,interpolation_sigma_mm=2.5,contact_values_not_pixels_used_for_rho=True,
                negative_values_clipped=False,patients_in_this_panel=False)


def draw_trajectory(fig,spec,a,snaps,end):
    ax=fig.add_subplot(spec,projection='3d');ax.computed_zorder=False
    zt=a['slow_time_ms']/1000
    rate=a['spikes_1ms'][:,0].reshape(-1,5).sum(1)/32000/.005
    t=(np.arange(len(rate))+.5)*.005
    xyz=np.c_[np.interp(t,zt,a['Z'][:,0]),np.interp(t,zt,a['currents'][:,2]),gaussian_filter1d(rate,1)]
    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz/[.5,1000,500],axis=0),axis=1))]
    keep=np.unique(np.r_[0,np.flatnonzero(np.diff(np.floor(length/.035))>0)+1,len(t)-1])
    points=xyz[keep];times=t[keep];norm=Normalize(0,end);cmap=plt.get_cmap('viridis')
    ax.add_collection3d(Line3DCollection(np.stack([points[:-1],points[1:]],axis=1),
        colors=cmap(norm((times[:-1]+times[1:])/2)),linewidths=1.1,zorder=6))
    ax.set(xlim=(max(0,xyz[:,0].min()-.03),1.025),ylim=(-25,xyz[:,1].max()*1.05),
           zlim=(-15,xyz[:,2].max()*1.08),xlabel='Mean Z',ylabel=r'$H_E$ (mV equiv.)',zlabel=r'E rate, $r_E$ (Hz)')
    ax.view_init(elev=26,azim=-125);ax.set_box_aspect((1.25,1,1.05))
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.labelpad=10;axis.label.set_fontsize(14);axis.pane.set_facecolor('#f2f5fa');axis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=12);ax.set_title('E1  State trajectory',loc='left',weight='bold',pad=18)
    fig.canvas.draw();boxes=[];point_records=[]
    for s in snaps:
        pt=np.array([np.interp(s['time_s'],t,xyz[:,k]) for k in range(3)])
        ax.scatter(*pt,c=s['color'],s=30,edgecolor='white',depthshade=False,zorder=10)
        x,y,_=proj3d.proj_transform(*pt,ax.get_proj());anchor=ax.transData.transform((x,y));unit=fig.dpi/72
        for offset in [(12,12),(-20,12),(20,-20),(-20,-20),(0,34),(34,0),(-34,0),(0,-34),(40,30),(-40,30)]:
            center=anchor+np.array(offset)*unit
            box=(center[0]-12*unit,center[1]-12*unit,center[0]+12*unit,center[1]+12*unit)
            if all(box[2]<b[0] or box[0]>b[2] or box[3]<b[1] or box[1]>b[3] for b in boxes):boxes.append(box);break
        ax.annotate(str(s['number']),xy=(x,y),xytext=offset,textcoords='offset points',ha='center',va='center',
            fontsize=13,weight='bold',bbox=dict(boxstyle='circle,pad=.2',fc='white',ec='#777'),
            arrowprops=dict(arrowstyle='-',color='#777',lw=.6),zorder=20)
        point_records.append(dict(number=s['number'],time_s=s['time_s'],Z_H_E=pt))
    ca=ax.inset_axes([1.04,.14,.033,.68]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ca)
    cb.set_label('Time (s)',fontsize=14);ca.tick_params(labelsize=12)
    return dict(time_window_s=[0,end],coordinates='Mean Z / applied inhibition H_E / all-E rate',
                states=point_records,interpretation='Actual stochastic SNN trajectory, not an autonomous vector field or bifurcation diagram.')


def render(row,grid):
    source_meta=f.read(pilot.OUT/'original_fig5'/row['name']/'fig5_metadata.json')
    r=f.read(row['source']/'result.json');first=r['tracker']['entries'][0]
    end=round(min(first['confirmation_s']+2,r['tracker']['restore_s']-.01),2)
    a=f.load(row['source']);layout.trim_display(a,end)
    # Extra raw-resolution records are not used for this unfiltered contact-power panel.
    a.pop('field_0p1ms',None);a.pop('population_0p1ms',None)
    observed=f.first_entry_from_counts(a['spikes_1ms'])
    assert np.isclose(observed['onset_s'],first['onset_s'])
    assert end<r['tracker']['restore_s']
    assert np.array_equal(a['spikes_1ms'][:,0],a['field_1ms'].sum(1))
    snaps=copy.deepcopy(source_meta['snapshots'][:4]);assert [s['number'] for s in snaps]==[1,2,3,4]
    previous_state2=snaps[1]['time_s']
    candidates=[event for event in source_meta['metrics']['events']
                if event['start_s']>=2 and event['end_s']<min(6,first['onset_s']-2)]
    assert candidates,'No early finite event for state2'
    early_event=min(candidates,key=lambda event:abs(event['peak_s']-end/3))
    snaps[1]['time_s']=early_event['peak_s']
    assert snaps[0]['time_s']<snaps[1]['time_s']<snaps[2]['time_s']<snaps[3]['time_s']
    assert all(s['time_s']+.025<end for s in snaps)
    comparison=compare_families(a,first['onset_s'])
    fig=plt.figure(figsize=(27,13.5))
    outer=fig.add_gridspec(1,2,width_ratios=[1,1.18],left=.053,right=.952,top=.935,bottom=.095,wspace=.20)
    left=outer[0].subgridspec(4,1,height_ratios=[1.75,1.15,.30,1],hspace=.25)
    right=outer[1].subgridspec(2,1,height_ratios=[1.15,1],hspace=.37)
    upper=right[0].subgridspec(1,2,width_ratios=[1.05,1],wspace=.60)
    ax=fig.add_subplot(left[0]);za=fig.add_subplot(left[1],sharex=ax);ma=za.twinx()
    it,ix=np.where(a['raster']);tt=it*.0001
    for lo,hi,col in [(0,60,'#225b7f'),(60,80,'#c17730')]:
        take=(ix>=lo)&(ix<hi);ax.scatter(tt[take],ix[take],s=5.5,marker='o',c=col,lw=0,rasterized=True,zorder=3)
    for y in [19.5,39.5,59.5]:ax.axhline(y,c='#bbb',lw=.7)
    ax.set(ylim=(-1,80),yticks=[10,30,50,70],yticklabels=['Core A E','Core B E','Other E','I'])
    ax.set_title('B  Continuous spike raster',loc='left',weight='bold',pad=24)
    ax.set_title(f'ηM = {row["eta_m"]:g}, τM = 1 s',loc='right',fontsize=13,pad=24)
    ax.tick_params(axis='x',labelbottom=False)
    zt=a['slow_time_ms']/1000;z=a['Z'];m=a['M']*row['eta_m']
    za.fill_between(zt,z[:,2],z[:,4],color=f.REGCOL[0],alpha=.13,lw=0)
    for zi,mi,col in [(0,0,f.REGCOL[0]),(5,1,f.REGCOL[1]),(6,2,f.REGCOL[2])]:
        za.plot(zt,z[:,zi],c=col,lw=1.5);ma.plot(zt,m[:,mi],c=col,ls='--',lw=1.4)
    za.set(ylabel='Resource Z',ylim=(0,1.05),xlabel='Time, t (s)');ma.set_ylabel('ηM × M (mV equiv.)',labelpad=10)
    ma.spines['right'].set_visible(True)
    if row['eta_m']==0:ma.set(ylim=(0,1),yticks=[0])
    else:ma.set_ylim(bottom=0)
    za.set_title('C  Inhibition resource and adaptation',loc='left',weight='bold',pad=12)
    handles=[Line2D([],[],c=c,label=n) for c,n in zip(f.REGCOL,['All E','Core A','Core B'])]
    handles += [Line2D([],[],c='black',label='Z'),Line2D([],[],c='black',ls='--',label='ηM × M')]
    za.legend(handles=handles,ncol=5,loc='center left',bbox_to_anchor=(.01,.48),fontsize=11.5,
              frameon=True,facecolor='white',edgecolor='none',framealpha=.9)
    header=fig.add_subplot(left[2]);header.axis('off')
    header.text(0,-.04,'D  Spatial activity · 50 ms',fontsize=17,weight='bold',
                bbox=dict(fc='white',ec='none',pad=2),zorder=8)
    for axis in [ax,za]:
        axis.set_xlim(0,end);axis.axvspan(first['onset_s'],end,fc='#ba263c',alpha=.10,lw=0)
        for s in snaps:
            axis.axvspan(s['time_s']-.025,s['time_s']+.025,color=s['color'],alpha=.12,lw=0)
            axis.axvline(s['time_s'],ls=':',lw=1.1,c=s['color'],alpha=.9)
    last=-10;label_row=0
    for s in snaps:
        label_row=(label_row+1)%2 if s['time_s']-last<end*.035 else 0;last=s['time_s']
        ax.text(s['time_s'],.99-.08*label_row,str(s['number']),transform=ax.get_xaxis_transform(),ha='center',va='top',
                c=s['color'],weight='bold',bbox=dict(fc='white',ec='none',alpha=.85,pad=.3))
    mapgrid=left[3].subgridspec(1,4,wspace=.22);map_records=[];map_axes=[]
    for i,s in enumerate(snaps):
        q=fig.add_subplot(mapgrid[i]);lo=round(s['time_s']*1000)-25
        field=a['field_1ms'][lo:lo+50].sum(0)/a['cell_e_counts']/.05
        im=q.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for xy in a['centers_mm']:q.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1.5))
        q.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        if i==0:q.set_ylabel('y (mm)')
        else:q.set_yticklabels([])
        q.tick_params(labelsize=13);q.xaxis.label.set_fontsize(15);q.yaxis.label.set_fontsize(15)
        q.set_title(f'{s["number"]}  {s["label"]}\n{s["time_s"]:.2f} s',c=s['color'],fontsize=14,pad=10)
        q.title.set_bbox(dict(fc='white',ec='none',pad=1))
        map_axes.append(q)
        connector=ConnectionPatch(xyA=(s['time_s'],-.12),coordsA=za.get_xaxis_transform(),
            xyB=(.5,1.26),coordsB=q.transAxes,arrowstyle='-',connectionstyle='arc3,rad=0',
            color=s['color'],lw=1.25,alpha=.8,clip_on=False,zorder=-1)
        fig.add_artist(connector)
        map_records.append(dict(state=s['number'],window_s=[lo/1000,(lo+50)/1000],E_rate_Hz=field))
        if i==0:
            ca=q.inset_axes([0,-.42,1,.07]);cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[0,250,500])
            cb.set_label('E rate (Hz)',fontsize=12);ca.tick_params(labelsize=11)
    trajectory=draw_trajectory(fig,upper[0],a,snaps,end)
    before=len(fig.axes);original.draw_grid(fig,upper[1],grid,r['job'])
    faxis=fig.axes[before]
    for note in list(faxis.texts):
        if note.get_text().startswith('Entered by 180 s:'):note.remove()
    faxis.set_title('F  M kinetics',loc='left',weight='bold',pad=12)
    e2display=draw_e2(fig,right[1],a,comparison)
    fig.canvas.draw();assert np.allclose(ax.get_position().x0,za.get_position().x0)
    assert np.allclose(ax.get_position().x1,za.get_position().x1)
    linked_states=[]
    for s,q in zip(snaps,map_axes):
        raster_x=ax.get_xaxis_transform().transform((s['time_s'],0))[0]
        slow_x=za.get_xaxis_transform().transform((s['time_s'],0))[0]
        assert abs(raster_x-slow_x)<1e-6
        map_center=q.transAxes.transform((.5,1.26))
        linked_states.append(dict(number=s['number'],time_s=s['time_s'],raster_x_px=raster_x,
                                 ZM_x_px=slow_x,map_connector_end_px=map_center,
                                 linked_by='Color-matched connector from true time coordinate to map title'))
    dest=OUT/row['name'];figdir=dest/'figures';figdir.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figdir/f'fig5.{ext}',dpi=160,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)
    preserved=source_meta['E2']
    meta=dict(job=r['job'],source_run=str(row['source']),source_duration_s=r['end_s'],display_time_window_s=[0,end],
        removed=['Contact time-series A','Recovery','State5','Second entry','All refill/release annotations'],
        first_entry=first,source_refill_starts_s=r['tracker']['restore_s'],manual_intervention_in_display=False,
        simulation_rerun=False,source_arrays_preserved=True,display_snapshots=snaps,native_maps=map_records,
        state2_revision=dict(previous_time_s=previous_state2,new_event=early_event,
                             selection='Finite pre-entry event closest to one third of displayed timeline, within2–6s; no spatial-match selection.'),
        raster_display=dict(marker='o',area_pt2=5.5,previous_marker='.',previous_area_pt2=2.2,
                            all_saved_spikes_retained=True,row_order_unchanged=True,
                            sample_ids=a['sample_ids'],raster_sha256=hashlib.sha256(a['raster'].tobytes()).hexdigest()),
        state_time_map_links=linked_states,
        raster_fixed_neurons=80,spatial_count_conservation='PASS',raster_ZM_axes_aligned=True,
        E1=trajectory,F=dict(measured_cells=23,first_endpoints=46,source=str(pilot.OUT/'original_fig5/F_measured_grid.json')),
        E2=comparison,E2_display=e2display,
        previous_band_power_result_preserved=dict(source=str(pilot.OUT/'original_fig5'/row['name']/'fig5_metadata.json'),
            band_Hz=[1,150],window_s=preserved['target_s'],contact_robust_z=preserved['contact_robust_z'],
            warning='Positive unfiltered Delta P does not replace or overturn the existing band-power decrease.'),
        producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scientific_scope='Current manual-core SNN, Z and M unchanged; high rate is an operational state, not demonstrated clinical ictal oscillation.',
        agent_visual_review='PENDING',human_review='PENDING')
    f.write(dest/'fig5_metadata.json',f.safe(meta))
    (figdir/'README.md').write_text('### fig5.png / .pdf\n'
        '仅展示同一手放双核、弱快M工作点在第一次人工补Z之前的连续轨迹；删除电极时序与恢复、再次进入，保留raster、Z/M和四个原生空间观察位置。Raster实心圆点放大，固定80个神经元与全部已保存放电保持不变；状态2改到更早的自限事件，各真实采样时刻用同色线连接到下方空间图，并同步更新E1。\n'
        'E2的ΔP是未滤波突触电流代理相对静息中位数的平方均值增量，包含持续电流水平升高；原附件曾用另存的Z加权代理，本次使用实际保存的原始读出。既有1–150Hz功率降低结果保留在metadata，不能用本图替代。\n'
        '**关注点**：模型A/B先行不等同于患者TA/TB；秩只在参与触点中计算，两个模板对照同一次起始窗。图为候选，等待用户目视与科学验收。\n')
    summary=dict(name=row['name'],seed=row['seed'],eta_m=row['eta_m'],end_s=end,first_onset_s=first['onset_s'],
                 groups={k:dict(n=v['n_events'],rho=v['contact_rho']) for k,v in comparison['groups'].items()},
                 figure=str(figdir/'fig5.png'),pdf=str(figdir/'fig5.pdf'))
    print(f.safe(summary),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);verify_reference()
    raw=f.read(pilot.OUT/'original_fig5/F_measured_grid.json')
    grid={**raw,**{k:np.array(raw[k],float) for k in ['mean','prob','count']}}
    cases=[r for r in pilot.sources() if args.all or r['eta_m']==.001]
    rows=[render(row,grid) for row in cases]
    f.write(OUT/'delivery_manifest.json',dict(versions=rows,human_review='PENDING'))
    lines=['# Fig5 单次转变版','','删除原A电极时序；仅保留第一次高态进入前后、人工补Z之前。主展示工作点ηM=0.001，τM=1秒。','']
    for row in rows:lines.append(f'- {row["name"]}：[PNG]({row["figure"]}) · [PDF]({row["pdf"]})')
    lines+=['','E2按用户附件三联图重算；ΔP为未滤波代理平方均值增量，包含均值上升，不是频带能量增强。附件公式已用原数组复算核对；当前数据源差异及旧频带结论见各版metadata。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
