#!/usr/bin/env python3
"""Automatic paired timing analysis and continuous Fig5 with synchronized zooms."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plot_topic4_m_parameter_modes as f
import plot_topic4_fig5_resting_recovery as layout
from run_topic4_weaker_M_onset_pilot import OUT,WINDOW,SOURCE,ETAS,SEEDS


def sources():
    p=f.read(OUT/'protocol.json')
    for ref in p['references']:
        yield dict(name=f"baseline_s{ref['seed']}",eta_m=.005,seed=ref['seed'],
                   source=Path(ref['source']),reused=True)
    for job in p['jobs']:
        yield dict(name=job['name'],eta_m=job['eta_m'],seed=job['seed'],
                   source=OUT/'runs'/job['name'],reused=False)


def load_case(row):
    r=f.read(row['source']/'result.json');a=f.load(row['source'])
    # The existing baseline files have a longer tail, which is retained on disk.
    end=r['end_s']
    if len(r['tracker']['entries'])>=2:
        end=round(min(end,r['tracker']['entries'][1]['confirmation_s']+2),2)
    a=layout.trim_display(a,end)
    if row['reused']:
        high=WINDOW/('early_energy_high_resolution_replay' if row['seed']==SEEDS[0]
                     else 'early_energy_high_resolution_replay_seed9108402')
        qa=f.read(high/'qa.json');assert qa['status']=='PASS'
        raw=np.load(qa['field_path'],mmap_mode='r')
        a['native_diagnostic']=str(high/'figures/native_energy_resolution.png')
        a['early_field_source']=dict(path=qa['field_path'],QA=str(high/'qa.json'),sha256=qa['field_sha256'],same_full_trajectory=True)
    else:
        raw=a.pop('field_0p1ms')[:round(end*10000)]
        pop=a.pop('population_0p1ms')[:len(raw)]
        assert np.array_equal(pop.reshape(-1,10,2).sum(1),a['spikes_1ms'])
        assert np.array_equal(raw.sum(1),pop[:,0])
        a['native_diagnostic']=str(OUT/'candidates'/row['name']/'figures/native_energy.png')
        a['early_field_source']=dict(source=str(row['source']/'chunks'),same_full_trajectory=True,
            acquired_in_original_run=True,recording_window_s=[0,end],bin_ms=.1)
    assert np.array_equal(raw.reshape(-1,10,400).sum(1),a['field_1ms'][:len(raw)//10])
    a['early_field_0p1ms']=raw
    return a,r,f.analyze(a,r)


def measure(row,a,r,m):
    entries=m['entries'];rec=m['recoveries'];release=r['tracker']['release_s']
    first=entries[0] if entries else None;second=entries[1] if len(entries)>1 else None
    before=[v for v in m['events'] if not first or v['end_s']<=first['onset_s']]
    after=[v for v in m['events'] if rec and v['start_s']>=max(rec[0]['confirmation_s'],release or 0)
           and (not second or v['end_s']<=second['onset_s'])]
    quiet_fraction=None
    if rec:
        lo=max(rec[0]['confirmation_s'],release or 0);hi=second['onset_s'] if second else m['duration_s']
        rate=a['spikes_1ms'].reshape(-1,10,2).sum(1)[:,0]/320
        quiet_fraction=float((rate[round(lo*100):round(hi*100)]<5).mean()) if hi>lo else None
    return dict(name=row['name'],eta_m=row['eta_m'],seed=row['seed'],reused_baseline=row['reused'],
        status='COMPLETE',source=str(row['source']),source_duration_s=r['end_s'],display_duration_s=m['duration_s'],
        mode=m['mode'],first_onset_s=None if first is None else first['onset_s'],
        first_confirmation_s=None if first is None else first['confirmation_s'],
        second_onset_s=None if second is None else second['onset_s'],Z_release_s=release,
        release_to_second_onset_s=None if second is None or release is None else second['onset_s']-release,
        recovery_confirmation_s=rec[0]['confirmation_s'] if rec else None,
        recovery_confirmation_after_release_s=rec[0]['confirmation_s']-release if rec and release is not None else None,
        recovery_mechanism=rec[0]['mechanism'] if rec else None,
        finite_events_before=len(before),finite_events_after_recovery=len(after),
        finite_event_duration_before_s=[v['end_s']-v['start_s'] for v in before],
        finite_event_duration_after_s=[v['end_s']-v['start_s'] for v in after],postrecovery_quiet_fraction=quiet_fraction,
        first_onset_Z=float(np.interp(first['onset_s'],a['slow_time_ms']/1000,a['Z'][:,0])) if first else None,
        first_onset_M_feedback=float(np.interp(first['onset_s'],a['slow_time_ms']/1000,a['M'][:,0])*row['eta_m']) if first else None,
        preserves_finite_events_recovery_and_reentry=bool(first and second and rec and len(before)>=2 and len(after)>=2),
        scientific_limits='Fixed-topology paired-noise timing assay; high rate does not establish HFO/ictal oscillation; external Z recovery is not autonomous termination.')


def f_timing(fig,spec,g,job):
    gs=spec.subgridspec(2,1,height_ratios=[.15,1],hspace=.12)
    h=fig.add_subplot(gs[0]);h.axis('off');h.text(0,.4,'F  M feedback and entry time',fontsize=18,weight='bold')
    sub=gs[1].subgridspec(1,2,wspace=.4)
    for i,(field,title) in enumerate([('first_onset_s','First onset'),('release_to_second_onset_s','Release Z → next onset')]):
        ax=fig.add_subplot(sub[i])
        for seed,col in zip(SEEDS,['#267ba8','#cc7b2a']):
            vals=[]
            for eta in ETAS:
                row=next((v for v in g['f_records'] if v['eta_m']==eta and v['seed']==seed),None)
                vals.append(np.nan if row is None or row.get(field) is None else row[field])
            ax.plot(range(4),vals,'o-',color=col,lw=1.2,ms=5,label=f'Seed {SEEDS.index(seed)+1}')
        ax.set(xlim=(-.2,3.2),xticks=range(4),xticklabels=['.005','.0025','.001','0'],
               xlabel=r'$\eta_M$',ylabel='Time (s)' if i==0 else '',title=title)
        ax.set_ylim(bottom=0);ax.tick_params(labelsize=12);ax.title.set_fontsize(14)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if i==0:ax.legend(frameon=False,fontsize=11,loc='upper right')


def states_and_zooms(a,m,r):
    try:snaps,event=layout.select_states(a,m,r)
    except AssertionError:
        old=f.snapshots(m,r)
        rate=a['spikes_1ms'].reshape(-1,10,2).sum(1)[:,0]/320
        first=m['entries'][0]['onset_s'] if m['entries'] else m['duration_s']
        quiet=[(max(l,100),h) for l,h in f.spans(rate<5) if h-max(l,100)>=5 and h*.01<first]
        pre=[v for v in m['events'] if v['end_s']<first]
        event=pre[-1] if pre else None
        ts=[(quiet[0][0]+2.5)*.01 if quiet else None,event['peak_s'] if event else None]+[v['time_s'] for v in old[1:4]]
        snaps=[dict(number=i+1,time_s=t,label=label,color=col) for i,(t,label,col) in enumerate(zip(ts,
            ['Resting','Interictal HFO','Entry','Ictal','Recovery'],['#616873','#267ba8','#dd871c','#ba263c','#248d78']))]
    a['display_snapshots']=snaps;a['trajectory_time_window_s']=[0,m['duration_s']]
    a['stagger_close_stage_labels']=True
    windows=[]
    if event:
        windows.append(dict(label='Interictal event · magnified',window_s=[max(0,event['start_s']-.08),min(m['duration_s'],event['end_s']+.08)],color='#267ba8'))
    if m['entries']:
        t=m['entries'][0]['onset_s']
        windows.append(dict(label='First transition · magnified',window_s=[max(0,t-.8),min(m['duration_s'],t+.6)],color='#dd871c'))
    a['AB_zoom_windows']=windows
    return event


def render_case(row,g):
    a,r,m=load_case(row);event=states_and_zooms(a,m,r)
    folder=OUT/'candidates'/row['name'];folder.mkdir(parents=True,exist_ok=True)
    a['F_renderer']=f_timing
    a['F_complete_cells']=sum(sum(v['eta_m']==eta for v in g['records'])==2 for eta in ETAS)
    a['F_semantics']='This four-eta paired pilot: first onset and time from Z release to next onset. Pending/absent endpoints are not plotted as zero. Not the previous two-dimensional M grid.'
    if event:
        try:
            a['event_order']=layout.model_event_order(a,m,event)
            if f.early(a,m)['status']=='MEASURED':
                a['E2_renderer']=layout.draw_e2
                a['E2_semantics']='Model event contact rank and same-run early contact power; unchanged canonical patient Fig3C below.'
                a['E2_display']=dict(top_left='Masked contact order of displayed finite event',top_right='Signed original CAR log-power robust z',
                    bottom='Original Fig3C',native='Separate diagnostic at0.1ms',primary_contact_endpoint_changed=False)
        except AssertionError as exc:
            a['E2_semantics']='Selected event order not estimable: '+str(exc)
    a['display_contract']=dict(states=[v['label'] for v in a['display_snapshots']],Z_on=True,M_observed=True,
        effective_M_feedback=row['eta_m']>0,M_cleared_during_refill=False,
        second_high_not_numbered=True,second_onset_tail_s=2,continuous_AB_and_synchronized_magnification=True,
        original_source_duration_s=r['end_s'],requested_ictal_and_HFO_labels_are_operational=True)
    f.render(a,r,m,folder/'figures',g)
    en=f.early(a,m)
    if en['status']=='MEASURED':
        fig,ax=plt.subplots(figsize=(5.7,5.1))
        from matplotlib.colors import TwoSlopeNorm
        im=ax.imshow(np.ma.masked_invalid(np.asarray(en['native_bandpower_change_db'],float).reshape(20,20)),
                     origin='lower',extent=[0,20,0,20],cmap='RdBu',norm=TwoSlopeNorm(vmin=-20,vcenter=0,vmax=20))
        ax.set(xlabel='x (mm)',ylabel='y (mm)',title='Native spikes · 0.1 ms')
        fig.colorbar(im,ax=ax,extend='both').set_label('1–150 Hz change (dB)')
        fig.tight_layout();fig.savefig(folder/'figures/native_energy.png',dpi=180,bbox_inches='tight');plt.close(fig)
    meta=f.read(folder/'fig5_metadata.json');meta.update(source_duration_s=r['end_s'],source_run=str(row['source']),
        revised_producer=str(Path(__file__).resolve()),observer_conservation_verified=True,
        controls_reused_without_additional_sample=row['reused'])
    f.write(folder/'fig5_metadata.json',meta)
    (folder/'figures/README.md').write_text('### fig5.png / .pdf\n'
        '同一连续SNN轨迹保留A/B/C总览，在A/B下增加间期事件和首次转变的同步放大窗。放大窗固定选取SCL9、SCL6、ICL11、ICL1四个电极，使用相同滤波样本与全程幅度尺度，raster仍为相同80个神经元。\n'
        'F显示本轮四档ηM、两个配对种子的首次进入和释放Z后再次进入时间；待完成或未出现的终点不填0。E2保留实测正负变化及原Fig3C。\n'
        '**关注点**：版式改善与动力学改变分开评价；高率标签及人工恢复不是临床振荡或自主终止的证据，待用户目视审阅。\n\n'
        '### native_energy.png\n同一轨迹0.1ms原生空间计数计算的早期1–150Hz功率变化，与电极插值图分开显示。\n'
        '**关注点**：保留增强和降低；不由平滑电极图推断原生神经元场。\n')
    if (folder/'figures/fig5_AB_magnified.png').exists():
        with (folder/'figures/README.md').open('a') as stream:
            stream.write('\n### fig5_AB_magnified.png / .pdf\n'
                '直接从完整Fig5的两个同步放大窗导出，方便在对话中看清单次事件与首次转变。电极读出与raster使用相同时间轴、原样样本和全程固定幅度尺度。\n'
                '**关注点**：这是同一条轨迹的放大视图，不增加新事件或新仿真；高率时带通信号变弱的现象保留。\n')


def main(skip_figures=False):
    OUT.mkdir(parents=True,exist_ok=True);records=[];pending=[];first_only=[]
    source_rows=list(sources())
    for row in source_rows:
        if not (row['source']/'result.json').exists():
            pending.append(row['name'])
            endpoint=f.committed_first_endpoint(row['source'])
            if endpoint['status']=='ESTABLISHED':
                first_only.append(dict(name=row['name'],eta_m=row['eta_m'],seed=row['seed'],
                    first_onset_s=endpoint['onset_s'],first_confirmation_s=endpoint['confirmation_s'],
                    release_to_second_onset_s=None,status='FIRST_ENDPOINT_ESTABLISHED_FOLLOWUP_PENDING',
                    endpoint_evidence=endpoint))
            continue
        a,r,m=load_case(row);records.append(measure(row,a,r,m));del a
    for row in records:
        base=next(v for v in records if v['eta_m']==.005 and v['seed']==row['seed'])
        for key in ['first_onset_s','release_to_second_onset_s','display_duration_s']:
            row['delta_'+key]=None if row[key] is None else row[key]-base[key]
    verdicts=[]
    for eta in ETAS[1:]:
        rr=[v for v in records if v['eta_m']==eta]
        status='PENDING'
        if len(rr)==2:
            preserves=all(v['preserves_finite_events_recovery_and_reentry'] for v in rr)
            faster=all(v['delta_first_onset_s'] is not None and v['delta_first_onset_s']<0 and
                       v['delta_release_to_second_onset_s'] is not None and v['delta_release_to_second_onset_s']<0 for v in rr)
            status='PAIRED_TIMING_DIRECTION_SUPPORTED' if preserves and faster else 'TIMING_OR_EVENT_RETENTION_TARGET_NOT_MET'
        verdicts.append(dict(eta_m=eta,status=status))
    for row in first_only:
        base=next(v for v in records if v['eta_m']==.005 and v['seed']==row['seed'])
        row['delta_first_onset_s']=None if row['first_onset_s'] is None else row['first_onset_s']-base['first_onset_s']
    report=dict(status='COMPLETE_PENDING_SCIENTIFIC_AND_VISUAL_REVIEW' if not pending else 'PARTIAL',
        records=records,pending=pending,verdicts=verdicts,independent_noise_seeds=2,fixed_networks=1,
        first_endpoint_only=first_only,established_first_endpoints=len(records)+len(first_only),
        reused_baselines=2,new_completed=len(records)-2,new_total=6,
        timing_direction_does_not_establish_burst_morphology=True,human_review='PENDING')
    f.write(OUT/'comparison.json',f.safe(report))
    if records:
        with (OUT/'paired_metrics.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    # Existing controls are drawn immediately; complete new cases join automatically.
    f_records=records+first_only
    g=dict(records=records,f_records=f_records,count=np.array([[sum(v['eta_m']==eta for v in records) for eta in ETAS]]))
    if not skip_figures:
        for row in source_rows:
            if (row['source']/'result.json').exists():render_case(row,g)
    lines=['# 弱M配对扫描：当前实际结果','',f'基线复用2条，新增完整随访完成{len(records)-2}/6；首次终点确定{len(f_records)}/8；固定网络1张、配对噪声2个。',
           '', '| ηM | seed | 首次onset(s) | 释放Z至再入(s) | 前/后有限事件 |','|---|---|---|---|---|']
    for v in records:
        show=lambda x:'未出现' if x is None else f'{x:.2f}'
        lines.append(f"| {v['eta_m']:g} | {v['seed']} | {show(v['first_onset_s'])} | {show(v['release_to_second_onset_s'])} | {v['finite_events_before']}/{v['finite_events_after_recovery']} |")
    if first_only:
        lines+=['','## 已确定首次终点，后续观察仍在运行','','| ηM | seed | 首次onset(s) | 相对同seed基线(s) |','|---|---|---|---|']
        for v in first_only:
            lines.append(f"| {v['eta_m']:g} | {v['seed']} | {show(v['first_onset_s'])} | {show(v['delta_first_onset_s'])} |")
    lines+=['','只有同一种子的前后配对才用于判断提速；当前未完成条件不能推断阴性。两种子方向一致也只是本底物的描述性结果，不是普遍单调规律。',
            '恢复来自外部补Z的轨迹与原生返回分别记载；调整M不构成HFO/持续振荡或患者早期能量匹配的验证。',
            '候选图沿用Resting、Interictal HFO、Entry、Ictal、Recovery五状态，缺失状态留空；连续总览与放大窗共用样本、幅度尺度与时间标记。']
    (OUT/'scientific_review.md').write_text('\n'.join(lines)+'\n')
    if not skip_figures:
        from plot_topic4_weaker_M_original_fig5 import render_original_package
        render_original_package()
    print(f.safe(dict(new_complete=len(records)-2,pending=pending,verdicts=verdicts)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--skip-figures',action='store_true');args=parser.parse_args()
    main(args.skip_figures)
