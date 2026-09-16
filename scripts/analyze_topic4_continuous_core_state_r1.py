#!/usr/bin/env python3
"""FIT-only first-round diagnostics; no fitting, re-ranking, or new simulations."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from src import topic4_initial_state_runtime as rt
from src.topic4_initial_state import off_diagonal_terms,block_matched_reference
from scripts.review_topic4_same_network_events import group_times

COLORS={0:'#2878b5',1:'#d34f37'}
ROUTES=['right_before_upper_and_left','upper_and_left_before_right','middle_before_both_ICL_ends']


def write_csv(path, rows):
    if not rows:
        path.write_text('status\nNO_ROWS\n');return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows([{k:('' if v is None else v) for k,v in rt.json_safe(r).items()} for r in rows])


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-incomplete',action='store_true')
    ap.add_argument('--output-root',type=Path,help='Analyze an explicitly supplied result directory')
    args=ap.parse_args()
    design=rt.read(ROOT/'config/topic4_continuous_core_state_r1.json');out=args.output_root or Path(design['output_root'])
    records=[];missing=[]
    for job in design['jobs']:
        p=out/'workers'/(job['id']+'.json')
        if not p.exists():missing.append(job['id']);continue
        rec=rt.read(p)
        if rec['design_sha256']!=rt.sha(ROOT/'config/topic4_continuous_core_state_r1.json'):
            raise RuntimeError('worker design mismatch')
        if rt.sha(p.with_suffix('.npz'))!=rec['arrays_sha256']:raise RuntimeError('worker arrays changed')
        records.append(rec)
    if missing and not args.allow_incomplete:raise RuntimeError(f'{len(missing)} workers incomplete')
    evaluator=rt.load_evaluator(design);objective=rt.load_objective(design)
    centers=np.asarray(rt.candidate_record(design)['node_field']['centers_mm'])
    patient_mu=evaluator.fit;patient_phi=objective.embedding(patient_mu)
    blocks=evaluator.blocks[evaluator.index['FIT']]
    pl=evaluator.fit_labels
    target={m:patient_phi[pl==m].mean(0) for m in range(2)}
    names=rt.load_observation_contract(design)['contact_names']
    patient_routes=[group_times(x,names) for x in patient_mu]
    patient={}
    for m in range(2):
        ids=np.flatnonzero(pl==m);patient[m]={'n':len(ids),'participation':np.isfinite(patient_mu[ids]).mean(0).tolist()}
        for key in ROUTES:
            v=[patient_routes[i][key] for i in ids if patient_routes[i][key] is not None]
            patient[m][key]={'n':len(v),'fraction':np.mean(v) if v else None}
    rt.write(out/'patient_fit_reference.json',dict(contact_names=names,modes=patient,scope='reused FIT only'))
    rows=[];events=[];conditional=[];participation=[];bins=[];audit=[];reference_cache={}
    edges=np.r_[-np.inf,design['analysis']['state_bins'],np.inf]
    for rec in records:
        job=rec['job'];jid=job['id']
        with np.load(out/'workers'/(jid+'.npz')) as ar:
            mu=ar['centroid_ms'];phi=ar['event_phi'];times=ar['event_time_ms'];labels=ar['event_mode'];support=ar['event_support']
            primary=np.asarray(ar['primary_event_indices'],int)
            windows=[('fixed',1500.,6000.)] if job['kind']=='fixed' else []
            if job['duration_ms']>6000:windows.append(('continuous',1500.,30000.))
            trtime=ar['trace_time_ms'];stz=ar['state_z'];dt=float(ar['state_dt_ms'])
            for e in rec['events']:
                idx=e['event_index'];row=dict(job=jid,kind=job['kind'],z_setting=job['z'],dynamics_seed=job['dynamics_seed'],**e)
                row['window_start_ms'],row['window_end_ms']=row.pop('window_ms')
                row['exclusion_reasons']='|'.join(row['exclusion_reasons'])
                row['distance_mode0'],row['distance_mode1']=row.pop('distance_modes')
                row.update(group_times(mu[idx],names))
                row.update({f'centroid_{name}_ms':v for name,v in zip(names,mu[idx])})
                events.append(row)
            for phase,lo,hi in windows:
                # Full analysis windows only; edge/cross-ramp events remain in all_events.csv.
                ids=np.array([i for i in primary if rec['events'][i]['window_ms'][0]>=lo and rec['events'][i]['window_ms'][1]<=min(hi,rec['actual_duration_ms'])],int)
                duration=max(0,min(hi,rec['actual_duration_ms'])-lo)/1000
                tr=(trtime>=lo)&(trtime<hi)
                row=dict(job=jid,phase=phase,kind=job['kind'],z=job['z'],dynamics_seed=job['dynamics_seed'],
                         physical_status=rec['physical_status'],window_complete=rec['actual_duration_ms']>=hi,
                         exposure_s=duration,n_events=len(ids),event_rate=len(ids)/duration if duration else None,
                         n_detected_all=rec['n_detected'],n_excluded_all=rec['n_detected']-rec['n_primary'],
                         mode0=int(np.sum(labels[ids]==0)),mode1=int(np.sum(labels[ids]==1)),
                         mode0_fraction=np.mean(labels[ids]==0) if len(ids) else None,
                         supported_fraction=np.mean(support[ids]==1) if len(ids) else None)
                for core in ['coreA','coreB']:
                    for cell in ['E','I']:
                        group=core+cell;ncell=rec['group_counts'][group]
                        row[group+'_rate_Hz']=ar['trace_'+group+'_spikes'][tr].sum()/ncell/(tr.sum()/1000) if tr.any() else None
                    ie=ar['trace_'+core+'E_mean_I_E'][tr].mean() if tr.any() else np.nan
                    ii=ar['trace_'+core+'E_mean_I_I'][tr].mean() if tr.any() else np.nan
                    row[core+'_E_mean_I_E']=ie;row[core+'_E_mean_I_I']=ii
                    row[core+'_E_balance']=(ie-ii)/(abs(ie)+abs(ii)+1e-12)
                rows.append(row)
                for m in range(2):
                    take=ids[labels[ids]==m];terms=off_diagonal_terms(phi[take],target[m])
                    ck=(m,len(take))
                    if ck not in reference_cache:
                        reference_cache[ck]=block_matched_reference(patient_phi[pl==m],blocks[pl==m],target[m],
                                                                   n_model=len(take),n_resamples=300,seed=891000+10000*m+len(take))
                    ref=reference_cache[ck]
                    c=dict(job=jid,phase=phase,kind=job['kind'],z=job['z'],mode=m,window_complete=row['window_complete'],
                           **terms,patient_Doff_q025=ref.get('D_off_quantiles',[None]*3)[0],
                           patient_Doff_q975=ref.get('D_off_quantiles',[None]*3)[2])
                    for key in ROUTES:
                        vals=[group_times(mu[i],names)[key] for i in take]
                        vals=[v for v in vals if v is not None]
                        c[key+'_n']=len(vals);c[key+'_fraction']=np.mean(vals) if vals else None
                    conditional.append(c)
                    for k,name in enumerate(names):
                        participation.append(dict(job=jid,phase=phase,z=job['z'],kind=job['kind'],mode=m,contact=name,n=len(take),
                                                  patient_fraction=patient[m]['participation'][k],
                                                  model_fraction=np.isfinite(mu[take,k]).mean() if len(take) else None))
                if phase=='continuous':
                    state_ids=np.arange(len(stz))*dt
                    valid_time=(state_ids>=lo)&(state_ids<min(hi,rec['actual_duration_ms']))
                    for k,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
                        exposure=float(np.sum(valid_time & (stz>=left)&(stz<right))*dt/1000)
                        take=[i for i in ids if left<=rec['events'][i]['z_at_detection_start']<right]
                        bins.append(dict(job=jid,kind=job['kind'],bin=k,interval=f'[{left},{right})',exposure_s=exposure,
                                         n_events=len(take),rate=len(take)/exposure if exposure else None,
                                         mode0=int(np.sum(labels[take]==0)),mode1=int(np.sum(labels[take]==1)),
                                         supported=int(np.sum(support[take]==1))))
        audit.append(dict(job=jid,**rec['state_audit']))
    write_csv(out/'run_summary.csv',rows);write_csv(out/'all_events.csv',events)
    write_csv(out/'conditional_propagation.csv',conditional);write_csv(out/'contact_participation.csv',participation)
    write_csv(out/'continuous_state_exposure.csv',bins);rt.write(out/'input_audit.json',audit)
    # Paired legacy-count identity must agree over identical simulated prefixes.
    pairing=[]
    for seed in sorted({j['dynamics_seed'] for j in design['jobs']}):
        rr=[r for r in records if r['job']['dynamics_seed']==seed]
        prefix=[r['state_audit']['legacy_first_6s_sha256'] for r in rr if r['actual_duration_ms']>=6000]
        long=[r['state_audit']['legacy_external_counts_sha256'] for r in rr if r['actual_duration_ms']>=30000]
        pairing.append(dict(seed=seed,n_full_6s=len(prefix),prefix_equal=len(set(prefix))<=1,
                            n_full_30s=len(long),long_equal=len(set(long))<=1))
    if any(not r['prefix_equal'] or not r['long_equal'] for r in pairing):raise RuntimeError('paired background changed')
    rt.write(out/'pairing_audit.json',pairing)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    fields=['mode0_fraction','event_rate','supported_fraction','coreAI_rate_Hz','coreBI_rate_Hz',None]
    titles=['M0 fraction (labels, not route recovery)','Primary event rate (/s)','Frozen observer support fraction',
            'Core A inhibitory firing (Hz)','Core B inhibitory firing (Hz)','Mode-conditional feature discrepancy']
    fixed=[r for r in rows if r['phase']=='fixed']
    for ax,field,title in zip(axes.flat,fields,titles):
        if field:
            for seed in sorted({r['dynamics_seed'] for r in fixed}):
                rr=sorted([r for r in fixed if r['dynamics_seed']==seed],key=lambda r:r['z'])
                ax.plot([r['z'] for r in rr],[r[field] if r['window_complete'] else np.nan for r in rr],'.-',alpha=.7,label=str(seed))
        else:
            for m in [0,1]:
                for z in [-1,-.5,0,.5,1]:
                    vals=[r['D_off'] for r in conditional if r['phase']=='fixed' and r['z']==z and r['mode']==m and r['D_off'] is not None and r['window_complete']]
                    ax.scatter([z+.025*(2*m-1)]*len(vals),vals,color=COLORS[m],s=22,label=f'M{m}' if z==-1 else None)
            ax.legend()
        ax.set(title=title,xlabel='Fixed continuous state z')
    if fixed:axes[0,0].legend(title='Noise replay',fontsize=8)
    fig.suptitle('Historical geometry | one graph | a=0.2 | 1.5–6 s | each point is a noise run')
    save(fig,figures/'state_response')
    ou_records=[r for r in records if r['job']['kind']=='ou']
    if ou_records:
        fig,axes=plt.subplots(len(ou_records),3,figsize=(14,3*len(ou_records)),squeeze=False,layout='constrained')
        for axrow,rec in zip(axes,ou_records):
            with np.load(out/'workers'/(rec['job']['id']+'.npz')) as ar:
                t=ar['trace_time_ms']/1000;dt=float(ar['state_dt_ms'])
                axrow[0].plot(np.arange(len(ar['state_z']))[::100]*dt/1000,ar['state_z'][::100],color='.25')
                for m in [0,1]:
                    ee=[e for e in rec['events'] if e['primary_eligible'] and e['mode']==m]
                    axrow[0].scatter([e['qualifying_start_ms']/1000 for e in ee],[e['z_at_detection_start'] for e in ee],color=COLORS[m],s=16,label=f'M{m}')
                for core,color in [('coreA','#c17c22'),('coreB','#168d93')]:
                    vals=ar['trace_'+core+'I_spikes'];step=50;nn=(len(vals)//step)*step
                    tt=t[:nn].reshape(-1,step).mean(1)
                    rate=vals[:nn].reshape(-1,step).sum(1)/(step/1000)/rec['group_counts'][core+'I']
                    axrow[1].plot(tt,rate,color=color,label=core+' I',lw=.8)
                env=ar['contact_envelope'];vmax=np.quantile(env, .999) if env.size else 1
                axrow[2].imshow(env,aspect='auto',origin='lower',extent=[0,env.shape[1]*2/1000,0,len(names)],cmap='magma',vmin=0,vmax=max(vmax,1e-10))
                axrow[2].set_yticks(np.arange(len(names))+.5,names,fontsize=6)
            axrow[0].set(title=rec['job']['id'],ylabel='z');axrow[0].legend(fontsize=7)
            axrow[1].set(title='Local I firing (50 ms bins)',ylabel='Hz');axrow[1].legend(fontsize=7)
            axrow[2].set_title('Complete contact firing-density envelope')
            for ax in axrow:ax.set_xlabel('Time (s)');ax.axvspan(0,1.5,color='.6',alpha=.12)
        save(fig,figures/'continuous_state')
        # Predetermined first OU replay, first primary event of each label.
        rep=sorted(ou_records,key=lambda r:r['job']['id'])[0]
        ee=[next((e for e in rep['events'] if e['primary_eligible'] and e['mode']==m and e['window_ms'][0]>=1500),None) for m in [0,1]]
        fig,axes=plt.subplots(2,5,figsize=(15,6),layout='constrained',gridspec_kw={'width_ratios':[1,1,1,1,2]})
        with np.load(out/'workers'/(rep['job']['id']+'.npz')) as ar:
            raw=ar['sheet_activity_counts'];env=ar['contact_envelope'];xy=ar['contact_xy_mm']
            for m,e in enumerate(ee):
                if e is None:
                    for ax in axes[m]:ax.set_axis_off()
                    axes[m,2].text(.5,.5,f'No qualifying M{m} example',ha='center');continue
                lo,hi=map(lambda x:round(x/2),e['window_ms']);burst=raw[lo:hi];vmax=max(1,float(burst.max()))
                for ax,offset in zip(axes[m,:4],[0,25,50,75]):
                    frame=min(len(burst)-1,offset);ax.imshow(burst[frame],origin='lower',extent=[0,20,0,20],vmin=0,vmax=vmax,cmap='magma')
                    for center,name,color in zip(centers,['A','B'],['#42d8db','#85da60']):
                        ax.plot(*center,'+',color=color,ms=7)
                        ax.text(center[0]+.4,center[1]+.4,name,color=color,fontsize=8)
                        if 'radius_mm' in rep:ax.add_patch(Circle(center,rep['radius_mm'],fill=False,ec=color,lw=.65))
                    ax.set(title=f'M{m}, window +{frame*2} ms',xticks=[],yticks=[])
                mass=env[:,lo:hi];axes[m,4].imshow(mass,origin='lower',aspect='auto',extent=[0,(hi-lo)*2,0,len(names)],cmap='magma',vmin=0,vmax=max(float(mass.max()),1e-10))
                axes[m,4].set_yticks(np.arange(len(names))+.5,names,fontsize=6)
                axes[m,4].set(title=f'First M{m}; z={e["z_at_detection_start"]:.2f}',xlabel='Window time (ms)')
        fig.suptitle('Predetermined examples; active E cells per 2 ms and contact envelope; + marks core A/B')
        save(fig,figures/'native_examples')
    descriptions={
        'state_response':'固定状态五点的逐噪声运行响应，含事件率、标签比例、局部 I 活动和条件特征误差。失败或不可估计项不补零。**关注点**：模式偏好与患者条件传播是否同时改善。',
        'continuous_state':'同一秒级 OU 状态方程的重演，展示状态、局部 I 放电与完整接触包络。标签颜色不是传播验收。**关注点**：状态访问与事件出现之间的关系。',
        'native_examples':'首个 OU 重演中各标签首个合格事件，固定时间帧展示原生活动和接触包络，未按好坏挑选。无事件时保留缺失说明。**关注点**：同一事件的局部参与及传播时序。'}
    (figures/'README.md').write_text('\n\n'.join('### '+name+'.png / '+name+'.pdf\n'+description
        for name,description in descriptions.items() if (figures/(name+'.png')).exists())+'\n')
    completed=len(records);runaway=sum(r['physical_status']=='RUNAWAY' for r in records)
    txt=['# 历史双 core × 连续慢状态：第一轮结果','',
         f'正式执行 {completed}/{len(design["jobs"])}；物理 runaway {runaway}；缺少输出 {len(missing)}。本轮未优化基底参数或状态方程。','',
         '基底为 v2_anchor_historical__baseline，topology=2511；不是 old_joint。状态幅度 0.2、OU tau=5 s；目标 I 群总期望输入率守恒。三个噪声重演不构成患者重复。','',
         '## 固定状态响应','',
         '|z|完整运行|事件数（各噪声）|M0比例（各噪声）|观测支持比例（各噪声）|',
         '|---|---:|---|---|---|']
    fmt=lambda x:'NA' if x is None or not np.isfinite(x) else f'{x:.3f}'
    for z in [-1,-.5,0,.5,1]:
        rr=sorted([r for r in fixed if r['z']==z],key=lambda r:r['dynamics_seed'])
        txt.append(f'|{z}|{sum(r["window_complete"] for r in rr)}/3|'+', '.join(str(r['n_events']) for r in rr)+'|'+', '.join(fmt(r['mode0_fraction']) for r in rr)+'|'+', '.join(fmt(r['supported_fraction']) for r in rr)+'|')
    txt+=['','固定状态表使用完整落在 1.5–6 s 的观察窗；全部检测和排除另见 all_events.csv。模式支持是冻结观察器的邻域读数，不等于恢复完整患者传播。条件误差 D_off、均值项 A、方差减项 B、患者匹配 N 的 FIT 块参考与区域先后比例见 conditional_propagation.csv。N<2 保持不可估计。','',
          '## 连续状态运行','',
          '|运行|物理状态|事件数|M0/M1|支持比例|','|---|---|---:|---|---:|']
    for r in rows:
        if r['phase']=='continuous':txt.append(f'|{r["job"]}|{r["physical_status"]}|{r["n_events"]}|{r["mode0"]}/{r["mode1"]}|{fmt(r["supported_fraction"])}|')
    txt+=['','continuous_state_exposure.csv 使用全部尾部状态区间、实际时间暴露和事件发生率，避免把时间占比混同于事件比例。有限 30 秒窗口不证明 OU 状态已充分遍历。','',
          '## 证据边界','',
          '本轮回答固定基底上的局部 I 输入重分配是否改变响应。五个状态点和三个噪声重复本身不能建立连续宽状态区域，不能把两个标签当成两类患者传播已恢复。必须把条件误差、参与及区域先后与模式比例一起审阅；缺少模式样本不是零误差。若仅比例变化，状态选择仍未解决传播质量。','',
          '状态由外部 OU 方程给定且在事件期间持续作用，不称纯初态因果效应、内源双稳态或已识别病理连接。状态加载空间图样与底层连接分开。没有自动增加强度、换几何、扩大搜索或进入论文主图。','',
          '图为诊断候选，尚待人工科学目视检查。输入背景配对见 pairing_audit.json；资格与来源见 qualification.json；全部配置见 design.json。']
    (out/'scientific_report.md').write_text('\n'.join(txt)+'\n')
    rt.write(out/'analysis_status.json',dict(status='COMPLETE' if not missing else 'PARTIAL',n_workers=completed,
                                          n_missing=len(missing),runaway=runaway,pairing=pairing,
                                          human_visual_review_pending=True,scientific_acceptance='NOT_ESTABLISHED_BY_AUTOMATED_REPORT'))
    print(json.dumps(dict(workers=completed,runaway=runaway,report=str(out/'scientific_report.md'))))

if __name__=='__main__':main()
