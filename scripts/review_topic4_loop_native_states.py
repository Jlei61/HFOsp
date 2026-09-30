#!/usr/bin/env python3
"""Native spike/field/contact review of completed conditional and autonomous runs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import run_topic4_loop_zk_conditional as native
import analyze_topic4_interictal_recurrence as audit
import run_topic4_rhythm_preserving_feedback as rhythm
from zoom_topic4_return_core_propagation import event_metrics,summarize

ORDER=['SCL9','SCL8','SCL7','SCL6']+[f'ICL{i}' for i in range(11,0,-1)]
COLORS=['#8952ab','#cf3e87','#159cbe','#37698b']
PRODUCER_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load(folder,keys,subdir='chunks',first_step=None):
    parts={k:[] for k in keys};last=first_step
    for path in sorted((folder/subdir).glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as z:
            if 'start_step' in z:
                if last is not None:assert int(z['start_step'])==last
                last=int(z['end_step'])
            for key in keys:parts[key].append(z[key])
    return {k:np.concatenate(v) for k,v in parts.items()} if parts[keys[0]] else None


def draw_event(folder,geo,data,r5,contacts,first_s,event,label,dest,fixed_window=None,fixed_position='final'):
    plt.rcParams.update({'font.size':9,'axes.labelsize':10,'axes.titlesize':10,
                         'xtick.labelsize':8,'ytick.labelsize':8})
    if fixed_window is None:
        assert event is not None
        lo=max(0.,event['start_s']-.04);lo=np.floor(lo/.005)*.005
        hi=min(len(r5)*.005,lo+.3);reference=event['start_s']
        basename=f'{label}_first_brief'
        title=f'{label} • first complete brief event at {first_s+reference:.2f} s'
        baseline_label='Change from pre-event (mV equiv.)'
    else:
        assert event is None
        lo,hi=fixed_window;reference=lo+.04
        basename=f'{label}_fixed_window'
        title=f'{label} • no complete brief events; fixed {fixed_position} 300 ms\nSpatial frame reference: {first_s+reference:.3f} s (not an event onset)'
        baseline_label='Change from window baseline (mV equiv.)'
    fig=plt.figure(figsize=(12,6.6))
    gs=fig.add_gridspec(3,6,height_ratios=[1,1,1.3],left=.07,right=.92,bottom=.1,top=.87,hspace=.54,wspace=.3)
    ax=fig.add_subplot(gs[0,:3])
    a,b=round(lo*10000),round(hi*10000)
    it,ix=np.where(data['raster'][a:b])
    mapping=np.r_[np.linspace(0,33,20),np.linspace(36,69,20),np.linspace(72,84,20),np.linspace(87,99,20)]
    for low,high,color in [(0,20,COLORS[1]),(20,40,COLORS[2]),(40,60,COLORS[3]),(60,80,'#c17730')]:
        use=(ix>=low)&(ix<high)
        ax.scatter((it[use]+a)*.0001+first_s,mapping[ix[use]],s=4,lw=0,c=color,rasterized=True)
    ax.set(xlim=(first_s+lo,first_s+hi),ylim=(-2,101),yticks=[16.5,52.5,78,93],
           yticklabels=['Core A E','Core B E','Other E','I'])
    ax.tick_params(axis='x',labelbottom=False);ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[1,:3]);t=(np.arange(len(r5))+.5)*.005
    use=(t>=lo)&(t<hi)
    for j,name in enumerate(['All E','Core A','Core B','Other E']):
        ax.plot(t[use]+first_s,r5[use,j],c=COLORS[j],lw=1,label=name)
    if event is not None:
        ax.axvspan(event['start_s']+first_s,event['end_s']+first_s,color='#c4cbd1',alpha=.18,lw=0)
    ax.set(xlim=(first_s+lo,first_s+hi),ylim=(0,510),xlabel='Time (s)',ylabel='E rate (Hz)')
    ax.legend(frameon=False,fontsize=7,ncol=4,loc='upper left');ax.spines[['top','right']].set_visible(False)
    ax=fig.add_subplot(gs[:2,3:])
    names=geo['contact_names'].astype(str).tolist();indices=[names.index(n) for n in ORDER]
    tc=contacts['time_ms']/1000-first_s
    view=(tc>=lo)&(tc<hi);baseline=(tc>=lo)&(tc<reference)
    assert baseline.any() and view.any()
    delta=contacts['contact_current'][view][:,indices]-np.median(contacts['contact_current'][baseline][:,indices],axis=0)
    scale=max(float(np.max(abs(delta))),1e-8)
    im=ax.imshow(delta.T,origin='upper',aspect='auto',interpolation='nearest',extent=(first_s+lo,first_s+hi,14.5,-.5),
                 cmap='RdBu_r',vmin=-scale,vmax=scale)
    ax.set(yticks=np.arange(15),yticklabels=ORDER,xlabel='Time (s)',title='Contact current proxy',)
    ax.tick_params(axis='y',labelsize=7);ax.axhline(3.5,color='black',lw=.8)
    fig.colorbar(im,ax=ax,label=baseline_label,fraction=.035,pad=.025)
    frames=[]
    for j,offset in enumerate([0,25,50,75,100,125]):
        ax=fig.add_subplot(gs[2,j]);index=round(reference*200)+offset//5
        if index>=len(data['field_5ms']):
            ax.text(.5,.5,'Not observed',ha='center',va='center',transform=ax.transAxes,fontsize=8)
            ax.set_title(f'+{offset}–{offset+5} ms',fontsize=9);ax.set_axis_off()
            continue
        rates=data['field_5ms'][index]/geo['cell_e_counts']/.005
        field=ax.imshow(rates.reshape(20,20),origin='lower',extent=(0,20,0,20),interpolation='nearest',cmap='magma',vmin=0,vmax=500)
        for name,center in zip('AB',geo['centers_mm']):
            ax.add_patch(Circle(center,float(geo['core_radius_mm']),fill=False,ec='#36dbd3',lw=.9))
            ax.text(center[0],center[1]+2.,name,c='#36dbd3',ha='center',fontsize=8)
        ax.set(title=f'+{offset}–{offset+5} ms',xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        ax.tick_params(labelsize=8)
        if j==0:ax.set_ylabel('y (mm)')
        else:ax.set_yticklabels([])
        frames.append(dict(time_s=first_s+index*.005,requested_offset_ms=offset,counts=data['field_5ms'][index].tolist()))
    cax=fig.add_axes([.935,.12,.01,.205]);fig.colorbar(field,cax=cax,label='E rate (Hz)')
    fig.suptitle(title+'\nNative counts and original contact operator; no interpolation or selection by morphology',fontsize=11)
    path=dest/f'{basename}.png';fig.savefig(path,dpi=170,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    np.savez_compressed(dest/f'{basename}_inputs.npz',contact_time_s=tc[view]+first_s,
        contact_change_mV_equiv=delta,contact_names=np.array(ORDER),field_counts=np.array([r['counts'] for r in frames]),
        field_start_s=np.array([r['time_s'] for r in frames]),cell_e_counts=geo['cell_e_counts'])
    return dict(file=str(path),native_event=event,frames=frames,
        selection='first_complete_brief' if event is not None else f'fixed_{fixed_position}300ms_no_brief_events',
        frame_reference_s=first_s+reference,
        contact_readout='Original normalized spatial weighted absIE+absZII+absGcurrent,2ms sampling; per-contact median over the first40ms of this view subtracted. One common color scale for all15contacts in this example; scales may differ across examples.',
        not_HFO_spectral_validation=True,contact_order=ORDER)


def review(root,name):
    folder=root/'runs'/name;result_path=folder/'result.json'
    if not result_path.exists():return None
    result=native.base.read(result_path);job=result['job'];first=float(job.get('branch_start_s',0.))
    output=root/'spatial_review'/name;output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(result_path.read_bytes()).hexdigest()
    if (output/'review.json').exists():
        old=native.base.read(output/'review.json')
        if old.get('result_sha256')==digest and old.get('producer_sha256')==PRODUCER_SHA:return old
    data=load(folder,['spikes_1ms','regions_1ms','field_5ms','raster','slow_time_ms','Z'],first_step=round(first*10000))
    with np.load(root/'geometry.npz') as g:geo={k:g[k] for k in g.files}
    counts=np.r_[32000,geo['region_counts'][:3]]
    raw=np.c_[data['spikes_1ms'][:,0],data['regions_1ms'][:,:3]]
    r5=raw.reshape(-1,5,4).sum(1)/counts/.005;r10=raw.reshape(-1,10,4).sum(1)/counts/.01
    assert np.array_equal(data['field_5ms'].sum(1),raw[:,0].reshape(-1,5).sum(1))
    assert np.array_equal(data['regions_1ms'][:,:3].sum(1),raw[:,0])
    duration=len(raw)/1000.;events=rhythm.strict_events(r10,duration)
    skipped=[];reference=None
    if job.get('conditional_clamp'):
        windows=[('conditional_tail',max(0,duration-10),duration)]
    else:
        primary=audit.temporal_audit(r10);onset=primary['entries'][0]['onset_s'] if primary['entries'] else duration
        windows=[('initial',.5,max(.5,min(8.,onset)))]
        ts=data['slow_time_ms']/1000;zc=data['Z'][:,[5,6]]
        with np.load(native.SOURCE/f'references/native_s{job["seed"]}.npz') as ref:
            reference=ref['Z'][np.searchsorted(ref['slow_time_ms'],8000.,side='right')-1,[5,6]]
        for i,ex in enumerate(primary['low_activity_exits'],1):
            end=next((e['onset_s'] for e in primary['entries'] if e['onset_s']>ex['confirmation_s']),duration)
            hit=np.flatnonzero((ts>=ex['start_s'])&(ts<end)&np.all(zc>=reference,axis=1))
            if len(hit):windows.append((f'post_exit{i}',max(float(ts[hit[0]]),ex['confirmation_s']),end))
            else:skipped.append(dict(exit_number=i,exit=ex,window_end_s=end,
                reason='Both cores did not regain the common current-graph8s Z reference before the next entry or observation end.'))
    contacts=load(folder,['time_ms','contact_current'],'actual_current_chunks')
    figures=output/'figures';figures.mkdir(exist_ok=True)
    groups=[];descriptions=[]
    for label,lo,hi in windows:
        part=audit.interval_events(events,lo,hi)
        metrics=[event_metrics(dict(rates=r5),e) for e in part['brief_events']]
        summary=summarize(metrics) if metrics else None
        example=None
        if metrics:
            example=draw_event(folder,geo,data,r5,contacts,first,part['brief_events'][0],label,figures)
            descriptions.append(f'### {label}_first_brief.png\n\n窗口{first+lo:.2f}–{first+hi:.2f}秒内按时间顺序第一个完整短事件，未按漂亮程度筛选。固定80细胞raster、两核和核外率、六帧原生5ms空间计数及同一15触点电流代理同步显示；触点按杆固定排序。该窗口共{len(metrics)}个短事件，其中{summary["core_over100"]}个至少一核峰值超过100Hz；这不等于所有事件均由核起源。\n\n**关注点**：低活动之后是否返回群体传播；触点是2ms电流代理，不代表已验证HFO能量或患者对应。图待人工审阅。\n')
        elif hi-lo>=.3:
            # Also expose native spatial/contact observations for high, quiet
            # and merged-burst responses that produce no isolated brief event.
            # This is a fixed-time view, never an invented detected event.
            initial=label=='initial'
            example=draw_event(folder,geo,data,r5,contacts,first,None,label,figures,
                fixed_window=(lo,lo+.3) if initial else (hi-.3,hi),
                fixed_position='first' if initial else 'final')
            which='最初' if initial else '最后'
            descriptions.append(f'### {label}_fixed_window.png\n\n声明窗口{first+lo:.2f}–{first+hi:.2f}秒中没有完整短事件，因此展示该窗口{which}300ms的固定时间切片（原生初始窗口取最初，其余窗口取最后）。固定80细胞raster、区域率、原生空间计数和15触点电流代理仍完整保留；空间帧以切片起点后40ms为参照，这不是事件起始时刻，触点也仅扣除该窗最初40ms的基线。\n\n**关注点**：没有独立短事件时，区分持续高活动、安静与尾部相连的爆发；该固定切片不代表完整长轨迹状态，需同时看full_rate_context。图待人工审阅。\n')
        groups.append(dict(label=label,relative_window_s=[lo,hi],absolute_window_s=[first+lo,first+hi],
            events=part,core_recruitment=summary,event_metrics=metrics,example=example))
    plt.rcParams.update({'font.size':9,'axes.labelsize':10,'axes.titlesize':10,
                         'xtick.labelsize':8,'ytick.labelsize':8})
    fig,ax=plt.subplots(figsize=(10,3));t=(np.arange(len(r10))+.5)*.01+first
    for j,label in enumerate(['All E','Core A','Core B','Other E']):ax.plot(t,r10[:,j],c=COLORS[j],lw=.6,label=label)
    for _,lo,hi in windows:ax.axvspan(first+lo,first+hi,color='#e2ecef',alpha=.35,lw=0,zorder=0)
    ax.set(xlim=(first,first+duration),ylim=(0,510),xlabel='Time (s)',ylabel='E rate (Hz)',
        title='Prescribed Z/K; conditional response' if job.get('conditional_clamp') else 'Autonomous native trajectory')
    ax.legend(frameon=False,ncol=4,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(figures/'full_rate_context.png',dpi=160);plt.close(fig)
    descriptions.insert(0,'### full_rate_context.png\n\n完整实测窗口的全E、两核和核外10ms率，浅色区标示分析窗口。未把前20秒过渡或末端截尾当成稳定状态，条件钳制与自主轨迹在标题中区分。\n\n**关注点**：短事件是否分离、是否持续高平台，不能只用平均率判断间期。\n')
    (figures/'README.md').write_text('\n'.join(descriptions))
    row=dict(name=name,source=str(folder),result_sha256=digest,producer_sha256=PRODUCER_SHA,
        observed_duration_s=duration,groups=groups,field_spike_integrity='PASS',
        common_current_graph_core_Z_reference=None if reference is None else reference.tolist(),
        postexit_windows_without_Z_reference_recovery=skipped,
        window_rule='Conditional:last10s; autonomous:initial0.5-8s beforeentry,then each post-exit window after both cores regain the original current-graph8s Z reference. A post-exit window may contain zero short events and is not automatically a return.',
        statistic_unit='One trajectory; all short events in each declared window are nested descriptive observations.',
        interpretation='Regional threshold order describes recruitment, not unique source identification. Raw spatial frames and contact traces remain candidates for visual review.',
        core_observer_radius_mm=1.75,physical_core_circle_radius_mm=float(geo['core_radius_mm']),
        temporal_bins_ms=dict(regional=5,field=5,contact=2),human_review='PENDING')
    native.write(output/'review.json',row)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    row=review(a.root,a.name)
    print(json.dumps(None if row is None else dict(name=row['name'],groups=[dict(label=g['label'],brief=g['events']['brief_count'],core=g['core_recruitment']) for g in row['groups']])),flush=True)


if __name__=='__main__':main()
