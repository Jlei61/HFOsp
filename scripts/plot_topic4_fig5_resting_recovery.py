#!/usr/bin/env python3
"""User-requested five-state display on two unchanged Z+M trajectories."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import copy
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import plot_topic4_m_parameter_modes as f
from analyze_topic4_fig5_early_spatial import contact_order
from plot_topic4_fig5_spatial_latency_layout_v10 import contact_field

WINDOW=f.ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
OUT=f.ROOT/'results/topic4_sef_hfo/fig5_resting_recovery_layout_20260914'


def trim_display(a,end_s):
    """Crop observations only; retain the full saved physical simulation."""
    n=round(end_s*1000)
    assert n%10==0
    for key in ['time_ms','spikes_1ms','regions_1ms','field_1ms']:
        a[key]=a[key][:n]
    a['raster']=a['raster'][:n*10]
    for key in ['lfp_time_ms','lfp_raw']:
        a[key]=a[key][:n*2]
    take=a['slow_time_ms']<=end_s*1000+1e-7
    for key in ['slow_time_ms','Z','M','currents']:
        a[key]=a[key][take]
    # Input samples are retained as provenance and never plotted or analyzed.
    return a


def select_states(a,m,r):
    original=f.snapshots(m,r)
    rate=a['spikes_1ms'].reshape(-1,10,2).sum(1)[:,0]/320
    quiet=[(max(lo,100),hi) for lo,hi in f.spans(rate<5)
           if hi-max(lo,100)>=5 and hi*.01<m['entries'][0]['onset_s']]
    assert quiet,'No actual post-initialization resting interval'
    resting=(quiet[0][0]+2.5)*.01
    times=[resting,original[0]['time_s'],original[1]['time_s'],original[2]['time_s'],original[3]['time_s']]
    assert all(t is not None for t in times) and np.all(np.diff(times)>0)
    labels=['Resting','Interictal HFO','Entry','Ictal','Recovery']
    colors=['#616873','#267ba8','#dd871c','#ba263c','#248d78']
    snapshots=[dict(number=i+1,time_s=t,label=label,color=color) for i,(t,label,color) in enumerate(zip(times,labels,colors))]
    event=next(v for v in m['events'] if abs(v['peak_s']-times[1])<1e-7)
    lo=round(resting*1000)-25
    assert np.all(rate[lo//10:(lo+50)//10]<5)
    return snapshots,event


def model_event_order(a,m,event):
    lt=a['lfp_time_ms']/1000
    rate=a['spikes_1ms'].reshape(-1,10,2).sum(1)[:,0]/320
    rt=(np.arange(len(rate))+.5)*.01
    baseline=[1.,min(30.,m['entries'][0]['onset_s']-2)]
    quiet=(lt>=baseline[0])&(lt<baseline[1])&(np.interp(lt,rt,rate)<1)
    assert quiet.sum()>=100,'Insufficient quiet baseline for contact participation'
    raw=a['lfp_raw']
    med=np.median(raw[quiet],axis=0)
    noise=1.4826*np.median(abs(raw[quiet]-med),axis=0)
    onset,rank=contact_order(lt,raw,event['start_s'],event['end_s'],noise,
                           np.ones(raw.shape[1],dtype=bool))
    assert np.isfinite(rank).sum()>=2,'Selected event does not define a propagation order'
    return dict(event=event,contact_names=a['contact_names'].tolist(),
                onset_s=onset,normalized_rank=rank,participating_contacts=int(np.isfinite(rank).sum()),
                timing_observer='Original current-proxy last upward 20-percent crossing before contact peak',
                participation='Peak-to-peak >= max(5 quiet MAD, 0.1 maximum contact peak-to-peak)',
                quiet_baseline_s=baseline,quiet_samples=int(quiet.sum()),
                selection='Last self-limited event before first entry; same event as display state 2; no spatial concordance selection',
                scope='Single model event illustration; not patient HFO detector timing or TA/TB distribution validation')


def draw_e2(fig,spec,a,m,en):
    assert en['status']=='MEASURED'
    event=a['event_order'];ranks=np.asarray(event['normalized_rank'],float)
    values=np.asarray(en['contact_robust_z'],float)
    gs=spec.subgridspec(3,1,height_ratios=[.10,1,1.14],hspace=.45)
    title=fig.add_subplot(gs[0]);title.axis('off')
    title.text(0,.15,'E2  Propagation and early energy',fontsize=18,weight='bold')
    maps=gs[1].subgridspec(1,2,wspace=.62)
    limit=max(1.,float(np.ceil(np.nanmax(abs(values)))))
    styles=[(ranks,Normalize(0,1),'viridis','Model interictal order',
             f"{event['event']['start_s']:.2f}–{event['event']['end_s']:.2f} s",'Rank · 0 early'),
            (values,TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),'RdBu','Model early energy',
             f"{en['target_s'][0]:.2f}–{en['target_s'][1]:.2f} s",'Log-power robust z')]
    for i,(v,norm,cmap,label,time,labelbar) in enumerate(styles):
        ax=fig.add_subplot(maps[i])
        contact_field(ax,a['contact_xy'],v,a['centers_mm'],float(a['core_radius_mm']),
                      norm,cmap,label+'\n'+time,ylabel=i==0)
        ax.tick_params(labelsize=14)
        ax.xaxis.label.set_fontsize(15);ax.yaxis.label.set_fontsize(15)
        ax.title.set_fontsize(15)
        ca=ax.inset_axes([1.035,0,.05,1])
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ca)
        cb.set_label(labelbar,fontsize=13,labelpad=8);ca.tick_params(labelsize=12)
        if i==0:cb.set_ticks([0,.5,1])
    ref=f.CANON/'figures/fig3-panelc.png'
    meta=f.read(f.CANON/'fig3_panelc_metadata.json')
    assert meta['seizure_idx']==2 and meta['ictal_extraction']['clinical_window_sec']==[0.,10.]
    ax=fig.add_subplot(gs[2]);ax.imshow(plt.imread(ref),interpolation='none');ax.axis('off')
    ax.set_title('Patient · original Fig. 3C · E1146 / SZ3',fontsize=14,pad=5)
    en['interictal_contact_order']=event
    en['native_field_diagnostic']=a['native_diagnostic']
    return en


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    g=f.grid_summary()
    assert g['established_first_endpoints']==40 and np.all(g['count']==2)
    f.write(OUT/'M_parameter_first_entry_complete.json',f.safe(g))
    records=[]
    for seed,sub,highres in [(9108401,'early_Z_lookup_dense_figures','early_energy_high_resolution_replay'),
                            (9108402,'early_z_refill_branches','early_energy_high_resolution_replay_seed9108402')]:
        source=WINDOW/sub/'runs'/f'early_z_refill_s{seed}'
        r=f.read(source/'result.json');assert r['job']['eta_m']==.005 and r['job']['tau_M_s']==1.
        source_end=r['end_s'];end=round(r['tracker']['entries'][1]['confirmation_s']+2,2)
        assert source_end>=end
        a=f.load(source);a=trim_display(a,end);m=f.analyze(a,r)
        snaps,event=select_states(a,m,r)
        a['display_snapshots']=snaps;a['event_order']=model_event_order(a,m,event)
        a['trajectory_time_window_s']=[0,end];a['stagger_close_stage_labels']=True
        qa_path=WINDOW/highres/'qa.json';qa=f.read(qa_path);assert qa['status']=='PASS'
        field_path=Path(qa['field_path'])
        assert hashlib.sha256(field_path.read_bytes()).hexdigest()==qa['field_sha256']
        raw=np.load(field_path,mmap_mode='r')
        assert np.array_equal(raw.reshape(-1,10,400).sum(1),a['field_1ms'][:len(raw)//10])
        a['early_field_0p1ms']=raw
        a['early_field_source']=dict(path=str(field_path),sha256=qa['field_sha256'],QA=str(qa_path),
                                    recording_window_s=[0,len(raw)/10000],same_full_trajectory=True)
        a['native_diagnostic']=str(WINDOW/highres/'figures/native_energy_resolution.png')
        a['E2_renderer']=draw_e2
        a['E2_semantics']='Model interictal contact recruitment order versus first-onset contact power change; original patient Fig3C below.'
        a['E2_display']=dict(top_left='Single actual model event, masked contact rank, 0 early / 1 late',
                            top_right='Original CAR 1–150 Hz log-power robust z; signed values retained',
                            bottom='Unmodified canonical patient Fig3C',
                            native_field='Separate original diagnostic, not contact interpolation',
                            interpolation='Gaussian contact display kernel, sigma 2.5 mm; no inferential role',
                            primary_contact_endpoint_changed=False)
        a['display_contract']=dict(user_revision='2026-09-14',states=[s['label'] for s in snaps],
            second_onset_shown_on_continuous_trace=True,second_high_is_not_a_numbered_state=True,
            display_end_after_second_confirmation_s=2,original_source_duration_s=source_end,
            operation='Display crop of an already complete trajectory; original simulation files preserved',
            Z_on=True,M_on=True,M_cleared_during_refill=False,
            state_label_scope='Ictal means model high-activity interval. Interictal HFO is the requested display label for a finite model event. Neither proves clinical HFO or sustained ictal oscillation.')
        folder=OUT/f'seed{seed}'
        f.render(a,r,m,folder/'figures',g)
        p=folder/'fig5_metadata.json';meta=f.read(p)
        meta.update(source_duration_s=source_end,analyzed_prefix_duration_s=end,source_run=str(source),
                    revision_producer=str(Path(__file__).resolve()),
                    revision_producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    verified_states_chronological=True,verified_original_field_population_count_conservation=True,
                    native_early_field_exact_aggregation=True)
        f.write(p,meta)
        (folder/'figures/README.md').write_text('### fig5.png / .pdf\n'
            '固定手放双核、Z与弱快M同时开启的同一条连续SNN轨迹；A至D展示Resting、Interictal HFO、Entry、Ictal、Recovery五个观察位置。'
            '第二次进入高活动仍保留，显示在确认后2秒结束；Z人工回填期间M及快状态连续。\n'
            'E2上排比较同一次轨迹的间期事件参与通道顺序和早期1–150Hz功率变化，下排原样使用正式Fig3C；F为完整20个M组合、各2个噪声种子的首次进入时间及180秒内进入比例。\n'
            '**关注点**：显示标签不构成临床HFO或持续发作振荡的验证；模型功率降低保留，电极插值不等同于原生神经元场。等待用户目视验收。\n')
        records.append(dict(seed=seed,source_duration_s=source_end,display_end_s=end,
                            first_onset_s=m['entries'][0]['onset_s'],second_onset_s=m['entries'][1]['onset_s'],
                            snapshots=snaps,figure=str(folder/'figures/fig5.png'),
                            energy_positive_contacts=meta['E2']['n_model_contacts_above_baseline'],
                            complete_F_cells=meta['complete_F_cells'],agent_visual_review='PENDING',human_review='PENDING'))
        print(f.safe(records[-1]),flush=True)
        del a,raw
    f.write(OUT/'summary.json',f.safe(dict(records=records,first_entry_endpoints=40,complete_F_cells=20)))


if __name__=='__main__':main()
